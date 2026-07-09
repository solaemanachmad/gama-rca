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
from data_loader import list_case_ids
from llm_client import LLMClient
from pipeline import GraphRAGPipeline
from baselines import BASELINE_REGISTRY
from evaluation import load_ground_truth, full_case_report
from data_loader import Case


def run_all(n_cases: int = 10, systems=None, save_path=None):
    """
    n_cases: how many cases from manifest.txt to evaluate (start small on
             Kaggle CPU/GPU time limits, e.g. 5-10, before a full 103-case run).
    systems: list of system names to run; defaults to all 5.
    """
    systems = systems or ["direct_llm", "standard_rag", "graphrag_only",
                            "multi_agent_only", "proposed_hybrid"]
    case_ids = list_case_ids()[:n_cases]
    llm = LLMClient()
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
    parser.add_argument("--systems", nargs="+", default=None)
    args = parser.parse_args()
    run_all(n_cases=args.n_cases, systems=args.systems)
