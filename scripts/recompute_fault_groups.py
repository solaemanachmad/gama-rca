"""
recompute_fault_groups.py
===========================
Recomputes fault-group-level accuracy from an EXISTING results.csv --
no need to re-run main.py, since predicted_fault_type/gt_fault_type are
already saved per row. Uses the same FAULT_GROUPS mapping now built into
data/taxonomy.py, so results here will match what future runs produce
in the fault_group_identification column directly.

Usage:
    python scripts/recompute_fault_groups.py path/to/results.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.taxonomy import fault_group

path = sys.argv[1] if len(sys.argv) > 1 else "graphrag_rca_work/results/results.csv"
df = pd.read_csv(path)

if "predicted_fault_type" not in df.columns:
    raise SystemExit(
        f"{path} has no predicted_fault_type column -- it was produced before "
        f"full_case_report started saving raw predictions. Re-run main.py to "
        f"get a CSV this script can use."
    )

df["predicted_fault_group"] = df["predicted_fault_type"].apply(fault_group)
df["gt_fault_group"] = df["gt_fault_type"].apply(fault_group)
df["fault_group_match"] = (
    (df["predicted_fault_group"] == df["gt_fault_group"]) & (df["predicted_fault_group"] != "unknown")
).astype(float)

print("=== Fine-grained (28-type) vs Group-level (6-group) accuracy, by system ===")
summary = df.groupby("system").agg(
    fault_identification_mean=("fault_identification", "mean"),
    fault_group_accuracy=("fault_group_match", "mean"),
    n_cases=("case_id", "count"),
)
print(summary)

print("\n=== Per-case detail ===")
cols = ["case_id", "system", "predicted_fault_type", "predicted_fault_group",
        "gt_fault_type", "gt_fault_group", "fault_group_match"]
print(df[cols].to_string(index=False))

out_path = path.replace(".csv", "_with_groups.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved annotated copy to {out_path}")