"""
run_experiment.py
===================
Paste this as the LAST cell in your Kaggle notebook (after all other files'
contents have been pasted into earlier cells, or after `%run` on each .py
file if you upload them as notebook data).

Runs every system (baselines + proposed hybrid framework) over N RCA100
cases, scores each with the official protocol + additional metrics, and
writes a results CSV to config.RESULTS_DIR for RQ1-RQ5 analysis.
"""

import argparse
import datetime as dt
import os
import traceback
import pandas as pd
from tqdm.auto import tqdm

import config
from config.args import build_parser, apply_overrides, print_effective_config
from data.loader import list_case_ids
from agents.factory import get_llm_client
from pipeline.pipeline import GraphRAGPipeline
from pipeline.baselines import BASELINE_REGISTRY
from evaluation.scoring import load_ground_truth, full_case_report
from data.loader import Case


def _current_model_name() -> str:
    """Resolves the actual model identifier for whichever backend is
    active, so it can be stamped onto every results.csv row (see below) --
    without this, merging chunk files from different backends/sessions
    (e.g. local Ollama qwen2.5:7b + Kaggle Gemma 4 12B) leaves no way to
    tell which row came from which model, purely from the CSV itself."""
    return {
        "ollama": config.LLM_MODEL_NAME,
        "gemini": config.GEMINI_MODEL_NAME,
        "kaggle": config.KAGGLE_MODEL_HANDLE,
    }.get(config.LLM_BACKEND, "unknown")


def run_all(n_cases: int = 10, start_case: int = 0, case_ids=None, systems=None, save_path=None,
            wandb_enabled: bool = False, wandb_project: str = "gama-rca", wandb_group: str = None,
            checkpoint_every: int = 5, resume: bool = True):
    """
    n_cases: how many cases from manifest.txt to evaluate (start small on
             Kaggle CPU/GPU time limits, e.g. 5-10, before a full 103-case run).
    start_case: 0-based offset into manifest.txt before taking n_cases --
             e.g. start_case=16, n_cases=4 covers cases 17-20 (1-indexed as
             people usually mean it). Ignored if case_ids is given.
    case_ids: explicit list of case IDs (e.g. ["t017", "t018"]) -- overrides
             n_cases/start_case entirely when provided.
    systems: list of system names to run; defaults to all 5.
    wandb_enabled: if True, logs each case+system result live to W&B as it
             completes, plus a final results table and summary means.
    wandb_group: use the SAME group name across parallel chunks (e.g. when
             splitting a 103-case run across several Kaggle notebooks) so
             they all show up together under one logical experiment in the
             W&B UI, while still being separate runs.
    checkpoint_every: write the results CSV (+ manifest) to disk after every
             N cases that had at least one new (case_id, system) pair
             actually run -- not just once at the very end. A full run (up
             to 5 systems x 103 cases, ~6 LLM calls/case for
             proposed_hybrid alone) can run for hours; previously the CSV
             was only written after EVERY case+system finished, so a Kaggle
             session timeout or crash lost the entire run's results with
             nothing recoverable from disk. Pass a very large number to get
             the old "save once at the end" behavior.
    resume: if True (default) and save_path already points to an existing
             CSV (e.g. from an interrupted earlier session), loads it first
             and skips any (case_id, system) pair that already has a
             non-errored row there, so a resumed run only does the
             remaining work instead of starting over. A pair whose row
             says "error" is NOT skipped -- it's dropped and retried, since
             the earlier failure might have been transient (network/API).
             To always start fresh, pass resume=False (the old file at
             save_path, if any, gets overwritten) or use a fresh save_path.
             IMPORTANT for resumability across separate Kaggle sessions:
             pass an explicit --save_path rather than relying on the
             timestamped default (which is different every invocation, so
             a later session would never find the earlier one's file).
    """
    systems = systems or ["direct_llm", "standard_rag", "graphrag_only",
                            "multi_agent_only", "proposed_hybrid"]
    if case_ids:
        case_ids = list(case_ids)
    else:
        all_ids = list_case_ids()
        case_ids = all_ids[start_case:start_case + n_cases]
    print(f"Running cases: {case_ids}")

    # Fixed up front (not just when saving at the end) so checkpointing and
    # resume both have one stable target path for the whole run.
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    save_path = save_path or f"{config.RESULTS_DIR}/{timestamp}_results.csv"

    rows = []
    done_pairs = set()   # {(case_id, system)} pairs with an existing non-errored row
    if resume and os.path.exists(save_path):
        try:
            prior_df = pd.read_csv(save_path)
            has_error_col = "error" in prior_df.columns
            n_dropped = 0
            for _, r in prior_df.iterrows():
                is_error = has_error_col and pd.notna(r.get("error"))
                if is_error:
                    n_dropped += 1
                    continue   # dropped here -- will be retried in the loop below
                rows.append(r.to_dict())
                done_pairs.add((r.get("case_id"), r.get("system")))
            print(f"[resume] loaded {len(rows)} previously-completed rows from {save_path} "
                  f"({n_dropped} errored rows dropped and will be retried)")
        except Exception as e:
            print(f"[resume] could not read existing {save_path} ({e}) -- starting fresh")

    wandb_run = None
    if wandb_enabled:
        import wandb
        from config.args import get_effective_config
        # WANDB_API_KEY is loaded from .env automatically (see config/__init__.py's
        # load_dotenv() call) -- log in with it here instead of requiring an
        # interactive `wandb.login()` beforehand. Falls back to wandb's own
        # credential discovery (already logged in, key set directly in the
        # shell environment, etc.) if it's not in .env.
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        wandb_run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=f"{case_ids[0]}-{case_ids[-1]}" if case_ids else "empty",
            config={**get_effective_config(), "systems": systems, "n_cases_requested": len(case_ids)},
        )

    llm = get_llm_client()
    hybrid_pipeline = GraphRAGPipeline(llm=llm)

    # Resolved once per run (backend/model don't change mid-run) and
    # stamped onto every row below -- see _current_model_name()'s docstring.
    llm_backend = config.LLM_BACKEND
    llm_model = _current_model_name()
    print(f"LLM backend: {llm_backend}  |  model: {llm_model}")

    # The two-stage RandomForest fault classifier (and its .pkl files) was
    # removed: it was trained on RCA100's own ground-truth labels, which is
    # invalid for evaluating genuine agentic reasoning even under
    # leave-one-out cross-validation. Fault typing now relies solely on
    # pipeline/zero_shot_matching.py (public taxonomy definitions only) and
    # the full 28-type taxonomy shown to the Coordinator -- no classifier
    # path check is needed here anymore.

    def _write_checkpoint():
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        return df

    cases_with_new_work = 0
    # tqdm gives the case-level "epoch" analog the user is used to from deep
    # learning training loops: a bar advancing one tick per case (not per
    # case+system, since systems-per-case varies with --systems/resume), an
    # ETA, and a postfix showing running mean final_score + error count so a
    # degenerate run (e.g. every case scoring ~0) is visible within the
    # first few ticks instead of only after the whole run finishes. Actual
    # per-case/system detail (scores, predicted vs. GT fault type, timing)
    # still prints via pbar.write() below so it doesn't get clobbered by the
    # bar redrawing itself.
    score_sum, score_n, error_n = 0.0, 0, 0
    case_pbar = tqdm(case_ids, desc="Cases", unit="case")
    for case_id in case_pbar:
        case_pbar.write(f"=== {case_id} ===")
        pending_systems = [s for s in systems if (case_id, s) not in done_pairs]
        if not pending_systems:
            case_pbar.write("  [skip] all requested systems already completed for this case (resume)")
            continue
        try:
            case = Case(case_id)
            gt = load_ground_truth(case_id, name_index=case.name_index)
        except Exception as e:
            case_pbar.write(f"  [skip] failed to load case/ground-truth: {e}")
            # keep an explicit error row so the case is not silently dropped from the denominator
            for _s in pending_systems:
                rows.append({"case_id": case_id, "system": _s, "error": f"load failed: {e}",
                             "llm_backend": llm_backend, "llm_model": llm_model})
            error_n += len(pending_systems)
            continue

        for system_name in pending_systems:
            case_pbar.write(f"  -> {system_name}")
            try:
                if system_name == "proposed_hybrid":
                    result = hybrid_pipeline.run(case_id)
                else:
                    fn = BASELINE_REGISTRY[system_name]
                    result = fn(case_id, llm)
                report = full_case_report(result, gt, case.topology, evidence_items=result.evidence_items)
                report["system"] = system_name
                report["llm_backend"] = llm_backend
                report["llm_model"] = llm_model
                rows.append(report)
                # Live per-case summary -- without this, a multi-hour run
                # (e.g. 103 cases x several minutes each) gives ZERO score
                # visibility until the entire run finishes; the only prior
                # feedback was "=== t030 ===" / "-> proposed_hybrid" with no
                # outcome shown, making it impossible to spot a degenerate
                # run (e.g. every case scoring 0) early without waiting hours.
                def _fmt(key, default=-1.0):
                    return report.get(key, default)
                case_pbar.write("     entity_loc={:.3f}  fault_id={:.3f}  fault_group={:.3f}  "
                      "reasoning={:.3f}  final={:.3f}  pred={} (gt={})  [{:.1f}s]".format(
                          _fmt("entity_localization"), _fmt("fault_identification"),
                          _fmt("fault_group_identification"), _fmt("reasoning_process"),
                          _fmt("final_score"), report.get("predicted_fault_type", "?"),
                          report.get("gt_fault_type", "?"), _fmt("total_pipeline_time_s", 0.0)))
                fs = report.get("final_score")
                if isinstance(fs, (int, float)):
                    score_sum += fs
                    score_n += 1
                case_pbar.set_postfix(avg_score=f"{score_sum / score_n:.3f}" if score_n else "n/a",
                                       errors=error_n)
                if wandb_run:
                    # Live per-case logging -- lets you watch scores/timing
                    # trend across a long run in the W&B dashboard instead
                    # of only seeing results after everything finishes.
                    wandb_run.log({k: v for k, v in report.items()
                                    if isinstance(v, (int, float, bool)) and not isinstance(v, str)})
            except Exception as e:
                case_pbar.write(f"     [error] {system_name} on {case_id}: {e}")
                traceback.print_exc()
                rows.append({"case_id": case_id, "system": system_name, "error": str(e),
                             "llm_backend": llm_backend, "llm_model": llm_model})
                error_n += 1
                case_pbar.set_postfix(avg_score=f"{score_sum / score_n:.3f}" if score_n else "n/a",
                                       errors=error_n)
                if wandb_run:
                    wandb_run.log({"case_error": 1})

        cases_with_new_work += 1
        if cases_with_new_work >= checkpoint_every:
            _write_checkpoint()
            cases_with_new_work = 0
            case_pbar.write(f"  [checkpoint] saved {len(rows)} rows so far to {save_path}")

    df = _write_checkpoint()
    print(f"\nSaved {len(df)} rows to {save_path}")

    # Companion JSON manifest, saved alongside the CSV -- captures run
    # parameters/config/environment that were previously only visible in
    # console output (lost unless redirected to a log file) or W&B (only if
    # --wandb was used). Without this, reproducing which exact config
    # produced a given results.csv meant manually cross-referencing log
    # files, which is fragile once several chunk files from different
    # sessions/backends get merged together.
    import json as _json
    from config.args import get_effective_config
    manifest = {
        "timestamp": timestamp,
        "results_csv_path": str(save_path),
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "systems": systems,
        "n_cases_requested": len(case_ids),
        "case_ids_requested": case_ids,
        "n_rows_written": len(df),
        "n_error_rows": int(df["error"].notna().sum()) if "error" in df.columns else 0,
        "effective_config": get_effective_config(),
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project if wandb_enabled else None,
        "wandb_group": wandb_group if wandb_enabled else None,
    }
    manifest_path = str(save_path).rsplit(".", 1)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        _json.dump(manifest, f, indent=2, default=str)
    print(f"Saved run manifest to {manifest_path}")

    if not df.empty and "final_score" in df.columns:
        summary = df.groupby("system")[["entity_localization", "fault_identification",
                                          "reasoning_process", "final_score"]].mean()
        print("\n=== Mean scores by system (RQ1-RQ3) ===")
        print(summary)

        if wandb_run:
            import wandb
            # Full results as a browsable table, plus summary means as
            # top-level metrics for cross-run comparison in the W&B UI.
            wandb_run.log({"results_table": wandb.Table(dataframe=df)})
            for system_name, row in summary.iterrows():
                for metric, value in row.items():
                    wandb_run.summary[f"{system_name}/{metric}_mean"] = value

    if wandb_run:
        wandb_run.finish()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cases", type=int, default=10)
    parser.add_argument("--start_case", type=int, default=0,
                         help="0-based offset into manifest.txt. E.g. --start_case 16 "
                              "--n_cases 4 covers the 17th-20th cases in manifest.txt.")
    parser.add_argument("--case_ids", nargs="+", default=None,
                         help="Explicit case IDs, e.g. --case_ids t017 t018 t019 t020. "
                              "Overrides --start_case/--n_cases if given.")
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--save_path", default=None,
                         help="Override the output CSV path (default: "
                              "graphrag_rca_work/results/YYYYMMDD_HHMM_results.csv)")
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true", default=False,
                         help="Log this run to Weights & Biases (requires `pip install wandb` "
                              "and `wandb login` first). Off by default.")
    parser.add_argument("--wandb_project", default="gama-rca",
                         help="W&B project name (default: gama-rca)")
    parser.add_argument("--wandb_group", default=None,
                         help="W&B group name -- use the SAME value across parallel chunks "
                              "(e.g. when splitting a 103-case run across several Kaggle "
                              "notebooks) so they all show up together in the W&B UI.")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                         help="Write the results CSV to disk after every N cases with new "
                              "work done (default: 5), not just once at the end -- so a "
                              "session timeout/crash loses at most this many cases' worth "
                              "of progress. Use a very large number to save only at the end.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                         help="Ignore any existing file at --save_path and start fresh "
                              "instead of resuming from it (default: resume if the file "
                              "exists). IMPORTANT: resuming across separate sessions only "
                              "works if you pass the SAME --save_path each time -- the "
                              "default path is timestamped differently on every invocation.")
    parser = build_parser(parser)   # adds --embedding-backend, --llm-backend, etc.
    args = parser.parse_args()
    apply_overrides(args)           # mutates config.* in place before anything reads it
    print_effective_config()
    run_all(n_cases=args.n_cases, start_case=args.start_case,
            case_ids=args.case_ids, systems=args.systems, save_path=args.save_path,
            wandb_enabled=args.wandb_enabled, wandb_project=args.wandb_project,
            wandb_group=args.wandb_group, checkpoint_every=args.checkpoint_every,
            resume=args.resume)