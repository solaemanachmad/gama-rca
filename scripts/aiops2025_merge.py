#!/usr/bin/env python3
"""aiops2025_merge.py -- merges parallel shard outputs (CSV, JSONL) into a unified dataset.

Usage:
    python scripts/aiops2025_merge.py --inputs "results/shards/*.jsonl" --out results/aiops2025/core
    python scripts/aiops2025_merge.py --inputs results/shards/ --out results/aiops2025/core --gt /path/to/groundtruth.jsonl
"""
import argparse
import glob
import json
import os
import sys
from typing import List

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_jsonl_shards(patterns: List[str]) -> List[dict]:
    """Reads all JSONL records matching pattern and deduplicates by uuid."""
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(glob.glob(os.path.join(pat, "*.jsonl")))
        else:
            files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print(f"Warning: No JSONL files found matching: {patterns}", file=sys.stderr)
        return []

    records = {}
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                uuid = item.get("uuid")
                if uuid and uuid not in records:
                    records[uuid] = item
    print(f"Loaded {len(records)} unique JSONL records from {len(files)} files.")
    return list(records.values())


def load_csv_shards(patterns: List[str]) -> pd.DataFrame:
    """Reads all CSV shard tables and deduplicates by case_id."""
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(glob.glob(os.path.join(pat, "*.csv")))
        else:
            files.extend(glob.glob(pat))
    # Exclude scored or check files
    files = [f for f in sorted(set(files)) if not f.endswith(("_scored.csv", "_check.csv", "_diagnostic.csv"))]
    if not files:
        print(f"Warning: No CSV files found matching: {patterns}", file=sys.stderr)
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    if "case_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["case_id"], keep="first")
    print(f"Loaded {len(merged)} unique CSV rows from {len(files)} files.")
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge parallel AIOps2025 shards")
    ap.add_argument("--inputs", nargs="+", required=True, help="paths, globs, or directory of shard outputs")
    ap.add_argument("--out", default="results/aiops2025/core", help="output directory for merged results")
    ap.add_argument("--name", default="summary", help="base output name (default: summary)")
    ap.add_argument("--gt", default=None, help="optional path to groundtruth.jsonl for evaluation")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    jsonl_records = load_jsonl_shards(a.inputs)
    csv_df = load_csv_shards(a.inputs)

    out_jsonl = os.path.join(a.out, f"{a.name}.jsonl")
    out_csv = os.path.join(a.out, f"{a.name}.csv")

    if jsonl_records:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for item in jsonl_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved merged JSONL -> {out_jsonl}")

    if not csv_df.empty:
        csv_df.to_csv(out_csv, index=False)
        print(f"Saved merged CSV   -> {out_csv}")

    if a.gt and os.path.exists(a.gt) and not csv_df.empty:
        from scripts.aiops2025_score import parse_groundtruth, evaluate_predictions
        gt_map = parse_groundtruth(a.gt)
        scored_df = evaluate_predictions(csv_df.to_dict("records"), gt_map)
        if not scored_df.empty:
            out_scored = os.path.join(a.out, f"{a.name}_scored.csv")
            scored_df.to_csv(out_scored, index=False)
            print("\n============================================================")
            print("         AIOps2025 Merged Empirical Evaluation              ")
            print("============================================================")
            print(f"Total evaluated cases : {len(scored_df)}")
            print(f"Localization Acc (LA) : {scored_df['loc_score'].mean():.4f}")
            print(f"Type Accuracy     (TA): {scored_df['type_score'].mean():.4f}")
            print(f"Final Score           : {scored_df['final_score'].mean():.4f}")
            print("============================================================")
            print(f"Saved scored breakdown -> {out_scored}")


if __name__ == "__main__":
    main()
