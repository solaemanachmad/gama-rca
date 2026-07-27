"""
check_metric_field.py
=======================
entity_name/service are empty for domain=="k8s" metric rows -- so the node
identity must be encoded somewhere else in this schema (most likely inside
the `metric` name itself, e.g. "node_cpu_usage{node=...}", or via
`metric_set_id` cross-referencing another structure). This prints raw
samples to find it.

Usage:
    python check_metric_field.py t003
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import pandas as pd
from data.loader import Case

case_id = sys.argv[1] if len(sys.argv) > 1 else "t003"
case = Case(case_id)

df = pd.read_parquet(f"{case.case_dir}/metrics.parquet")

print("=== full row sample, domain == 'k8s' AND entity_set == 'k8s.node' ===")
node_rows = df[(df["domain"] == "k8s") & (df["entity_set"] == "k8s.node")]
with pd.option_context("display.max_colwidth", 200):
    print(node_rows.head(10).to_string())

print("\n=== unique `metric` names for entity_set == 'k8s.node' ===")
print(node_rows["metric"].unique()[:30])

print("\n=== unique metric_set_id for entity_set == 'k8s.node' (sample) ===")
print(node_rows["metric_set_id"].unique()[:10])

print("\n=== does the GT node's host_name appear literally inside `metric` text anywhere? ===")
gt_hostname = "cn-hongkong.10.0.1.69"   # host_name of GT node in t003, from topology dump
hits = df[df["metric"].astype(str).str.contains(gt_hostname, na=False)]
print(f"rows where metric column contains {gt_hostname!r}: {len(hits)}")

print("\n=== does host_name appear in ANY column via a raw string search across the row? ===")
sample = node_rows.head(3)
for col in sample.columns:
    print(f"  {col}: {sample[col].tolist()}")