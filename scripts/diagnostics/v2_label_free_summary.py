"""
v2_label_free_summary.py -- summarize a v2 run CSV (main.py output of hypothesis_rule / hypothesis_rca) WITHOUT any label-derived
column: no scores, no ground truth. Use it on dev-30 runs after the last label-informed change, so design checks stay label-free.

    python scripts/diagnostics/v2_label_free_summary.py RUN.csv
"""
import csv
import statistics
import sys
from collections import Counter

LABEL_COLS = {"gt_fault_type", "gt_target_entity_ids", "gt_fault_group"}
LABEL_PREFIX = ("entity_localization", "fault_identification", "reasoning_process", "final_score", "retrieval_", "rp_", "el_if",
                "rcl_", "ft_", "joint_", "fi_", "fault_group_identification", "explainability")


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    rows = [{k: v for k, v in r.items() if k not in LABEL_COLS and not k.startswith(LABEL_PREFIX)} for r in rows]
    err = [r["case_id"] for r in rows if (r.get("error") or "").strip()]
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    num = lambda k: [float(r[k]) for r in ok if r.get(k) not in (None, "")]
    med = lambda k: statistics.median(num(k)) if num(k) else None
    print(f"{len(rows)} cases, {len(err)} errors {err}")
    print("mode:", dict(Counter(r.get("v2_mode") for r in ok)))
    for k in ("v2_n_hypotheses", "v2_n_consistent", "v2_n_specific", "v2_n_diffuse", "total_pipeline_time_s", "total_calls",
              "v2_grounded_before", "v2_grounded_after"):
        if num(k):
            print(f"  median {k}: {med(k)}  [min {min(num(k))}, max {max(num(k))}]")
    if any(r.get("v2_choice_valid") not in (None, "") for r in ok):
        print("  LLM choice valid:", dict(Counter(r.get("v2_choice_valid") for r in ok)))
    print("  answered level:", dict(Counter("node" if r.get("v2_entity", "").startswith("cn-") else "service" for r in ok)))
    print("  answered type group:", dict(Counter(r.get("predicted_fault_group") for r in ok)))
    print("  answered types:", dict(Counter(r.get("predicted_fault_type") for r in ok).most_common(10)))


if __name__ == "__main__":
    main(sys.argv[1])
