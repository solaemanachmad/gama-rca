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


def run_all(n_cases: int = 10, start_case: int = 0, case_ids=None, systems=None, save_path=None):
    """
    n_cases: how many cases from manifest.txt to evaluate (start small on
             Kaggle CPU/GPU time limits, e.g. 5-10, before a full 103-case run).
    start_case: 0-based offset into manifest.txt before taking n_cases --
             e.g. start_case=16, n_cases=4 covers cases 17-20 (1-indexed as
             people usually mean it). Ignored if case_ids is given.
    case_ids: explicit list of case IDs (e.g. ["t017", "t018"]) -- overrides
             n_cases/start_case entirely when provided.
    systems: list of system names to run; defaults to all 5.
    """
    systems = systems or ["direct_llm", "standard_rag", "graphrag_only",
                            "multi_agent_only", "proposed_hybrid"]
    if case_ids:
        case_ids = list(case_ids)
    else:
        all_ids = list_case_ids()
        case_ids = all_ids[start_case:start_case + n_cases]
    print(f"Running cases: {case_ids}")
    llm = get_llm_client()
    hybrid_pipeline = GraphRAGPipeline(llm=llm)

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
                rows.append(report)
            except Exception as e:
                print(f"     [error] {system_name} on {case_id}: {e}")
                traceback.print_exc()
                rows.append({"case_id": case_id, "system": system_name, "error": str(e)})

    df = pd.DataFrame(rows)
    save_path = save_path or f"{config.RESULTS_DIR}/results.csv"
    df.to_csv(save_path, index=False)
    print(f"\nSaved {len(df)} rows to {save_path}")

    if not df.empty and "final_score" in df.columns:
        summary = df.groupby("system")[["entity_localization", "fault_identification",
                                          "reasoning_process", "final_score"]].mean()
        print("\n=== Mean scores by system (RQ1-RQ3) ===")
        print(summary)

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
    parser = build_parser(parser)   # adds --embedding-backend, --llm-backend, etc.
    args = parser.parse_args()
    apply_overrides(args)           # mutates config.* in place before anything reads it
    print_effective_config()
    run_all(n_cases=args.n_cases, start_case=args.start_case,
            case_ids=args.case_ids, systems=args.systems)