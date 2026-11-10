#!/usr/bin/env python3
"""aiops2025_score.py -- official empirical scoring for AIOps2025 predictions.
Scores predictions from a submission .jsonl or results .csv against groundtruth.jsonl.

Usage:
    python scripts/aiops2025_score.py --pred results/AIOPS_GAMA_core.jsonl --gt data/aiops2025/groundtruth.jsonl
    python scripts/aiops2025_score.py --pred results/AIOPS_GAMA_core.csv --gt data/aiops2025/groundtruth.jsonl --out results/scored.csv
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_groundtruth(path: str) -> Dict[str, dict]:
    """Reads groundtruth.jsonl into a dictionary keyed by case uuid."""
    gt_map = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uuid = str(row.get("uuid") or row.get("case_id") or "")
            if uuid:
                gt_map[uuid] = row
    return gt_map


def parse_predictions(path: str) -> List[dict]:
    """Reads predictions from .jsonl or .csv."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")

    preds = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    preds.append(json.loads(line))
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            preds.append({
                "uuid": str(r.get("case_id") or r.get("uuid")),
                "component": str(r.get("component") or ""),
                "fault_type": str(r.get("fault_type") or ""),
                "reason": str(r.get("reason") or ""),
                "confidence": float(r.get("confidence", 0.0) or 0.0),
                "reasoning_trace": r.get("reasoning_chain") or [],
            })
    return preds


def service_of(entity: str) -> str:
    """Strips pod replica suffix (e.g. 'checkoutservice-2' -> 'checkoutservice')."""
    return re.sub(r"-\d+$", "", str(entity or "")).strip()


def score_localization(pred: str, gt: str) -> float:
    """Scores component localization: 1.0 if exact match or parent service match."""
    p = str(pred or "").strip().lower()
    g = str(gt or "").strip().lower()
    if not p or not g:
        return 0.0
    if p == g:
        return 1.0
    if service_of(p) == service_of(g):
        return 1.0
    return 0.0


def score_type(pred_type: str, pred_reason: str, gt_type: str, gt_reason: str) -> float:
    """Scores fault type: keyword match on normalized phrases."""
    p_typ = str(pred_type or "").replace("-", " ").replace("_", " ").lower().strip()
    p_res = str(pred_reason or "").replace("-", " ").replace("_", " ").lower().strip()
    g_typ = str(gt_type or "").replace("-", " ").replace("_", " ").lower().strip()
    g_res = str(gt_reason or "").replace("-", " ").replace("_", " ").lower().strip()

    target = g_typ or g_res
    if not target:
        return 0.0

    # Direct equality
    if p_typ and p_typ == g_typ:
        return 1.0

    # Keyword match in the first 20 words of reason (AIOps2025 competition protocol)
    p_words = set(p_res.split()[:20])
    target_words = target.split()
    if all(w in p_words for w in target_words if len(w) > 2):
        return 1.0

    return 0.0


def evaluate_predictions(preds: List[dict], gt_map: Dict[str, dict]) -> pd.DataFrame:
    """Computes per-case scores and returns a summary DataFrame."""
    rows = []
    for p in preds:
        uuid = str(p.get("uuid") or p.get("case_id") or "")
        gt = gt_map.get(uuid)
        if not gt:
            continue

        gt_comp = str(gt.get("component") or gt.get("entity") or "")
        gt_type = str(gt.get("fault_type") or gt.get("type") or gt.get("reason") or "")
        pred_comp = str(p.get("component") or "")
        pred_type = str(p.get("fault_type") or "")
        pred_reason = str(p.get("reason") or "")

        loc_score = score_localization(pred_comp, gt_comp)
        type_score = score_type(pred_type, pred_reason, gt_type, str(gt.get("reason", "")))
        
        # AIOps2025 standard weights (0.5 Localization + 0.5 Type)
        final_score = round(0.5 * loc_score + 0.5 * type_score, 4)

        rows.append({
            "case_id": uuid,
            "pred_component": pred_comp,
            "gt_component": gt_comp,
            "loc_score": loc_score,
            "pred_type": pred_type,
            "gt_type": gt_type,
            "type_score": type_score,
            "final_score": final_score,
            "confidence": p.get("confidence", 0.0),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Empirical scorer for AIOps2025 benchmark.")
    parser.add_argument("--pred", required=True, help="Path to predictions .jsonl or .csv file")
    parser.add_argument("--gt", required=True, help="Path to groundtruth.jsonl file")
    parser.add_argument("--out", default=None, help="Optional output path for scored CSV summary")
    args = parser.parse_args()

    gt_map = parse_groundtruth(args.gt)
    preds = parse_predictions(args.pred)

    if not preds:
        print(f"No predictions found in {args.pred}", file=sys.stderr)
        sys.exit(1)

    scored_df = evaluate_predictions(preds, gt_map)
    if scored_df.empty:
        print(f"Warning: No matching case IDs found between predictions and ground truth.", file=sys.stderr)
        sys.exit(0)

    print("\n=== AIOps2025 Empirical Evaluation Summary ===")
    print(f"Cases Evaluated: {len(scored_df)} / {len(preds)}")
    print(f"Localization Accuracy (LA): {scored_df['loc_score'].mean():.4f}")
    print(f"Type Accuracy (TA):         {scored_df['type_score'].mean():.4f}")
    print(f"Final Score:                {scored_df['final_score'].mean():.4f}")
    print("=" * 46)

    print("\nPer-Case Breakdown:")
    show_cols = ["case_id", "pred_component", "gt_component", "loc_score", "pred_type", "gt_type", "type_score", "final_score"]
    print(scored_df[show_cols].to_string(index=False))

    if args.out:
        scored_df.to_csv(args.out, index=False)
        print(f"\nSaved per-case score table to: {args.out}")


if __name__ == "__main__":
    main()
