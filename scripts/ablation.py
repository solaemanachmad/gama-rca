#!/usr/bin/env python3
"""Ablation evaluation for GAMARCA modules.

Runs leave-one-out and progressive module ablations across test scenarios.
Usage:
    python scripts/ablation.py          # Fast ablation check (< 1 sec)
    python scripts/ablation.py --full   # Detailed module breakdown
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_gama import SCENARIOS, gama_snapshot


def eval_spec(spec: str) -> List[dict]:
    """Evaluate one configuration across scenarios."""
    out = []
    for sc in SCENARIOS:
        s = gama_snapshot(sc, spec)
        out.append({
            "scenario": sc.__name__.replace("sc_", ""),
            "entity": s["entity"][0] if s["entity"] else "-",
            "type": s["type"] or "-",
            "conf": float(s["confidence"]),
            "steps": len(s["chain"] or []),
        })
    return out


def run_ablation() -> List[Tuple[str, str, float, int]]:
    """Run core leave-one-out ablation suite."""
    specs = [
        ("core", "Full Core (M1-M6)"),
        ("no_mm", "w/o M1 (cross-modal)"),
        ("no_trace", "w/o M2 (trace-origin)"),
        ("no_typing", "w/o M3 (mechanism typing)"),
        ("no_load", "w/o M4 (load explain-away)"),
        ("no_text", "w/o M5 (text sensor)"),
        ("no_chain", "w/o M6 (causal chain)"),
        ("base", "Base model (M0 only)"),
        ("hybrid", "Hybrid (+ bounded LLM)"),
    ]
    rows = []
    for spec, desc in specs:
        res = eval_spec(spec)
        mean_conf = sum(r["conf"] for r in res) / len(res)
        mean_steps = sum(r["steps"] for r in res) // len(res)
        rows.append((spec, desc, mean_conf, mean_steps))
    return rows


def print_table(rows: List[Tuple[str, str, float, int]]) -> None:
    """Print markdown ablation table."""
    print("\n# GAMARCA Module Ablation Study\n")
    print("| Switch Spec | Description | Posterior P(e*) | Chain Steps |")
    print("|:---|:---|:---:|:---:|")
    base_conf = rows[0][2]
    for spec, desc, conf, steps in rows:
        delta = conf - base_conf
        diff_str = f"({delta:+.3f})" if abs(delta) > 1e-4 else " [ref]"
        print(f"| `{spec}` | {desc} | {conf:.3f} {diff_str} | {steps} |")
    print("\n*Note: Posterior P(e*) reflects internal Bayesian confidence, NOT ground-truth accuracy.*")
    print("*On 103 real cases (Table 5 in paper), removing M2 drops Final score by -5.6 points (overconfidence).*")


def main() -> int:
    t0 = time.time()
    rows = run_ablation()
    print_table(rows)
    print(f"Completed in {time.time() - t0:.2f}s.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
