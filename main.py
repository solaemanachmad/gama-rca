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
import traceback
import pandas as pd

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
            wandb_enabled: bool = False, wandb_project: str = "gama-rca", wandb_group: str = None):
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
    """
    systems = systems or ["direct_llm", "standard_rag", "graphrag_only",
                            "multi_agent_only", "proposed_hybrid"]
    if case_ids:
        case_ids = list(case_ids)
    else:
        all_ids = list_case_ids()
        case_ids = all_ids[start_case:start_case + n_cases]
    print(f"Running cases: {case_ids}")

    wandb_run = None
    if wandb_enabled:
        import os
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

    # Upfront classifier path check -- shown BEFORE any case processing
    # starts, so a missing/misplaced .pkl is visible immediately rather
    # than only surfacing (even with the fixed warning) once the first
    # case reaches the classifier call.
    import os as _os
    _group_pkl = _os.path.join(config.WORK_DIR, "fault_group_classifier.pkl")
    _type_pkl = _os.path.join(config.WORK_DIR, "fault_type_classifier.pkl")
    print(f"Classifier check -- group: {_group_pkl} "
          f"({'FOUND' if _os.path.exists(_group_pkl) else 'NOT FOUND -- running without group narrowing'})")
    print(f"Classifier check -- type:  {_type_pkl} "
          f"({'FOUND' if _os.path.exists(_type_pkl) else 'NOT FOUND -- running without type narrowing'})")

    rows = []
    for case_id in case_ids:
        print(f"=== {case_id} ===")
        try:
            case = Case(case_id)
            gt = load_ground_truth(case_id, name_index=case.name_index)
        except Exception as e:
            print(f"  [skip] failed to load case/ground-truth: {e}")
            continue

        for system_name in systems:
            print(f"  -> {system_name}")
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
                print("     entity_loc={:.3f}  fault_id={:.3f}  fault_group={:.3f}  "
                      "reasoning={:.3f}  final={:.3f}  pred={} (gt={})  [{:.1f}s]".format(
                          _fmt("entity_localization"), _fmt("fault_identification"),
                          _fmt("fault_group_identification"), _fmt("reasoning_process"),
                          _fmt("final_score"), report.get("predicted_fault_type", "?"),
                          report.get("gt_fault_type", "?"), _fmt("total_pipeline_time_s", 0.0)))
                if wandb_run:
                    # Live per-case logging -- lets you watch scores/timing
                    # trend across a long run in the W&B dashboard instead
                    # of only seeing results after everything finishes.
                    wandb_run.log({k: v for k, v in report.items()
                                    if isinstance(v, (int, float, bool)) and not isinstance(v, str)})
            except Exception as e:
                print(f"     [error] {system_name} on {case_id}: {e}")
                traceback.print_exc()
                rows.append({"case_id": case_id, "system": system_name, "error": str(e),
                             "llm_backend": llm_backend, "llm_model": llm_model})
                if wandb_run:
                    wandb_run.log({"case_error": 1})

    df = pd.DataFrame(rows)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    save_path = save_path or f"{config.RESULTS_DIR}/{timestamp}_results.csv"
    df.to_csv(save_path, index=False)
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
    parser = build_parser(parser)   # adds --embedding-backend, --llm-backend, etc.
    args = parser.parse_args()
    apply_overrides(args)           # mutates config.* in place before anything reads it
    print_effective_config()
    run_all(n_cases=args.n_cases, start_case=args.start_case,
            case_ids=args.case_ids, systems=args.systems, save_path=args.save_path,
            wandb_enabled=args.wandb_enabled, wandb_project=args.wandb_project,
            wandb_group=args.wandb_group)