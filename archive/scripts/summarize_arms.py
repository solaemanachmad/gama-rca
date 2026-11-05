"""Summarize one or more run CSVs (one per arm) -- eval only.

Usage: python scripts/summarize_arms.py A0=A0_dev_r1.csv A1=A1_dev_r1.csv [--ref A0] [--boot 10000]

Prints per arm: n, final_score_chain, EL, FI, RP(chain), strict RCL / FT / Joint (benchmark-team style,
binary), tokens per case and LLM calls per case. Paired bootstrap (over cases) of each arm minus --ref
for final_score_chain and joint_strict. Old CSVs without the strict columns fall back to
rcl_strict = (entity_localization == 1.0)."""
import argparse
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    if "rcl_strict" not in df:
        df["rcl_strict"] = (df["entity_localization"] >= 0.999).astype(float)
    if "ft_binary" not in df:
        df["ft_binary"] = df["fault_identification"]
    if "joint_strict" not in df:
        df["joint_strict"] = df["rcl_strict"] * df["ft_binary"]
    if "final_score_chain" not in df:
        df["final_score_chain"] = df["final_score"]
    return df.set_index("case_id")


def boot(a, b, n, seed=0):
    d = (a - b).dropna().values
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+")
ap.add_argument("--ref", default=None)
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
D = {}
for item in a.arms:
    name, path = item.split("=", 1)
    D[name] = load(path)
cols = ["final_score_chain", "entity_localization", "fault_identification", "reasoning_process_chain",
        "rcl_strict", "ft_binary", "joint_strict", "total_tokens", "total_calls"]
rows = []
for k, df in D.items():
    r = {"arm": k, "n": len(df)}
    for c in cols:
        r[c] = df[c].mean() if c in df else float("nan")
    rows.append(r)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
ref = a.ref or next(iter(D))
print(f"\nPaired bootstrap vs {ref} (mean diff [95% CI]):")
for k, df in D.items():
    if k == ref:
        continue
    common = df.index.intersection(D[ref].index)
    for c in ("final_score_chain", "joint_strict", "ft_binary", "rcl_strict"):
        m, lo, hi = boot(df.loc[common, c], D[ref].loc[common, c], a.boot)
        print(f"  {k:6s} {c:18s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(common)}")
