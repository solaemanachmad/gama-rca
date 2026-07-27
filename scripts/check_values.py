"""
check_values.py
=================
inspect_case() only shows column NAMES. This shows actual VALUES, which is
what we need to diagnose why _container_name_ / service / entity_id aren't
resolving against name_index.

Usage:
    python check_values.py t003
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import pandas as pd
import config
from data.loader import Case

case_id = sys.argv[1] if len(sys.argv) > 1 else "t003"
case = Case(case_id)

print(f"=== name_index sample (topology-derived, {len(case.name_index)} entries) ===")
for k, v in list(case.name_index.items())[:15]:
    print(f"  {k!r} -> {v}")

print("\n=== logs.parquet: _container_name_ / _pod_name_ raw values ===")
logs_df = pd.read_parquet(f"{case.case_dir}/logs.parquet", columns=["_container_name_", "_pod_name_"])
print(logs_df["_container_name_"].value_counts(dropna=False).head(10))
print("\n_pod_name_ sample:")
print(logs_df["_pod_name_"].value_counts(dropna=False).head(10))

print("\n=== metrics.parquet: entity_id / entity_name / service / domain / entity_set raw values ===")
metrics_df = pd.read_parquet(
    f"{case.case_dir}/metrics.parquet",
    columns=["entity_id", "entity_name", "service", "domain", "entity_set"]
)
print("\ndomain value counts:")
print(metrics_df["domain"].value_counts(dropna=False))
print("\nentity_id null rate BY domain:")
print(metrics_df.groupby("domain", dropna=False)["entity_id"].apply(lambda s: s.isna().mean()))
print("\nentity_id sample (non-null):")
print(metrics_df.loc[metrics_df["entity_id"].notna(), "entity_id"].head(5).tolist())
print("\nentity_id sample (null rows) -- what entity_name/service look like instead:")
print(metrics_df.loc[metrics_df["entity_id"].isna(), ["entity_name", "service", "domain"]].head(10))