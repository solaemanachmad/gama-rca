#!/usr/bin/env python3
"""analyze.py -- statistical analysis of benchmark runs (PREREGISTRATION_V5.md).
Reads run outputs (the *_all.zip files, CSVs, or results folders); scores are already in
them (evaluation/scoring.py); nothing is re-run.

    python scripts/analyze.py --arm A=A_odd.csv,A_even.csv --arm GAMA_CORE="GAMA_CORE_7B_shard*_all.zip" \
        --arm GAMA_CORE_NO_TRACE="GAMA_CORE_NO_TRACE_7B_shard*_all.zip" ... --out report.md --json report.json

Arm labels follow the arm names (A, GAMA_CORE, GAMA_LLM, GAMA_CORE_NO_<M>, EVIDENCE_LLM_FINAL, BELIEF_LLM, DIRECT_LLM); a backbone is appended
after '@' (GAMA_LLM@llama31_8b). Comparison families are built from the labels present (see FAMILIES); --compare adds more.

Metrics (x100): Final = final_score_nodes (headline; RP = 0.5 chain-node match + 0.5 checkpoint hit rate), EL, FI (exact 28-class),
RP, group FI; secondary Final_step / RP_step (step-anchored checkpoints); counts of exact entity / exact type / joint.
Efficiency: seconds per case, LLM calls, tokens (total; input / output when logged, else from the trace's per-call records).
Statistics: 95% percentile bootstrap CIs (5,000 resamples, seed 0); paired Wilcoxon signed-rank on Final (one-sided where the
hypothesis is directional), Holm within each family; paired bootstrap CI of the mean difference; better / worse / equal counts;
exact McNemar (two-sided) on exact entity and exact type. Subsets: odd (development), even (validation), all. Error rows score 0 and
stay in the denominator; cases never run are reported as missing and left out of paired tests.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

METRICS = [("Final", "final_score_nodes"), ("EL", "entity_localization"), ("FI", "fault_identification_exact"),
           ("RP", "reasoning_process_nodes"), ("Group", "fault_group_identification"), ("Final_step", "final_score_step"),
           ("RP_step", "reasoning_process_step")]
OPTIONAL = [("narrator_chain_nodes", "rp_chain_nodes_llm"), ("narrator_checkpoint_step", "rp_checkpoint_hit_step_llm")]  # M7 narrative (H3)
COUNTS = [("entity_exact", "rcl_exact"), ("type_exact", "ft_binary"), ("joint", "joint_exact")]
B, SEED = 5000, 0


# ---------------------------------------------------------------- loading
def _read_any(path):
    """-> (DataFrame of rows, list of per-call trace records)."""
    frames, calls = [], []
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".csv") and "_trace/" not in n:
                    frames.append(pd.read_csv(io.BytesIO(z.read(n))))
                elif n.endswith("llm_calls.jsonl"):
                    calls += [json.loads(x) for x in z.read(n).decode("utf-8", "replace").splitlines() if x.strip()]
    elif os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "*.csv")):
            frames.append(pd.read_csv(f))
        for f in glob.glob(os.path.join(path, "**", "llm_calls.jsonl"), recursive=True):
            calls += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
    else:
        frames.append(pd.read_csv(path))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), calls


def load_arm(spec):
    paths = []
    for part in spec.split(","):
        hits = sorted(glob.glob(part.strip()))
        paths += hits or [part.strip()]
    frames, calls = [], []
    for p in paths:
        d, c = _read_any(p)
        frames.append(d)
        calls += c
    d = pd.concat(frames, ignore_index=True)
    d = d[d["case_id"].notna()].drop_duplicates("case_id", keep="last").set_index("case_id")   # resume: last row wins
    aliases = {
        "final_score_nodes": "final_score",
        "entity_localization": "entity_loc",
        "fault_identification_exact": "fault_type",
        "reasoning_process_nodes": "reasoning",
        "fault_group_identification": "fault_group",
    }
    for canon, clean in aliases.items():
        if canon not in d or d[canon].isna().all():
            if clean in d:
                d[canon] = d[clean]
    if "rcl_exact" not in d and "pred_entity" in d and "gt_entity" in d:
        d["rcl_exact"] = ((d["pred_entity"].notna()) & (d["pred_entity"] == d["gt_entity"])).astype(float)
    if "ft_binary" not in d and "fault_type" in d:
        d["ft_binary"] = d["fault_type"]
    if "joint_exact" not in d and "rcl_exact" in d and "ft_binary" in d:
        d["joint_exact"] = d["rcl_exact"] * d["ft_binary"]
    err = d["error"].fillna("").astype(str).str.strip().ne("") if "error" in d else pd.Series(False, index=d.index)
    for _, col in METRICS + COUNTS + OPTIONAL:
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan   # NaN = not scored in that run (older column)
        d.loc[err, col] = 0.0
    d["_error"] = err
    if calls and ("input_tokens" not in d or d["input_tokens"].isna().all()):
        per = defaultdict(lambda: [0, 0, 0])             # tokens from the per-call trace (runs before input/output logging)
        for c in calls:
            cid = c.get("case_id")
            if cid is None:
                continue
            per[cid][0] += int(c.get("input_tokens") or 0)
            per[cid][1] += int(c.get("output_tokens") or 0)
            per[cid][2] += int(c.get("tokens_total_delta") or 0)
        if any(v[0] for v in per.values()):
            d["input_tokens"] = [per[i][0] for i in d.index]
            d["output_tokens"] = [per[i][1] for i in d.index]
    return d


def subset(ids, name):
    num = lambda c: int(re.sub(r"\D", "", c) or 0)
    if name == "odd":
        return [c for c in ids if num(c) % 2 == 1]
    if name == "even":
        return [c for c in ids if num(c) % 2 == 0]
    return list(ids)


# ---------------------------------------------------------------- statistics
def boot_mean(x, b=B, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def wilcoxon(a, b, alternative):
    from scipy.stats import wilcoxon as w
    d = np.asarray(a, float) - np.asarray(b, float)
    if not np.any(np.abs(d) > 1e-12):
        return 1.0
    return float(w(d, alternative=alternative).pvalue)


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a) >= 0.999, np.asarray(b) >= 0.999
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [None] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = run
    return out


# ---------------------------------------------------------------- tables
def summary(d, ids):
    r = {"n": len(ids)}
    x = d.loc[ids]
    for name, col in METRICS:
        v = 100 * x[col].astype(float).dropna()
        r[name] = round(float(v.mean()), 1) if len(v) else math.nan
        if 0 < len(v) < len(x):
            r[name + "_n"] = int(len(v))                 # metric missing for part of the cases (older runs)
        if name in ("Final", "Final_step"):
            lo, hi = boot_mean(v)
            r[name + "_ci"] = (round(lo, 1), round(hi, 1))
    for name, col in COUNTS:
        r[name] = int((x[col] >= 0.999).sum()) if col in x else None
    for name, col in OPTIONAL:
        v = 100 * x[col].astype(float).dropna()
        if len(v):
            r[name] = round(float(v.mean()), 1)
    r["errors"] = int(x["_error"].sum())
    for name, col in (("sec_per_case", "total_pipeline_time_s"), ("llm_calls", "total_calls"), ("tokens", "total_tokens"),
                      ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"), ("max_input_tokens", "max_input_tokens")):
        if col in x and pd.to_numeric(x[col], errors="coerce").notna().any():
            v = pd.to_numeric(x[col], errors="coerce")
            r[name] = round(float(v.mean()), 1)
            if name in ("tokens", "sec_per_case"):
                r[name + "_median"] = round(float(v.median()), 1)
    return r


def compare(da, db, ids, alternative):
    a, b = 100 * da.loc[ids, "final_score_nodes"].values, 100 * db.loc[ids, "final_score_nodes"].values
    diff = a - b
    lo, hi = boot_mean(diff)
    out = {"n": len(ids), "diff": round(float(diff.mean()), 1), "diff_ci": (round(lo, 1), round(hi, 1)),
           "better": int((diff > 1e-9).sum()), "worse": int((diff < -1e-9).sum()), "equal": int((np.abs(diff) <= 1e-9).sum()),
           "p": wilcoxon(a, b, alternative), "alternative": alternative}
    sa, sb = 100 * da.loc[ids, "final_score_step"].astype(float), 100 * db.loc[ids, "final_score_step"].astype(float)
    ok = sa.notna().values & sb.notna().values                   # Final_step: comparable across chain sources (amendment)
    if ok.any():
        ds = sa.values[ok] - sb.values[ok]
        lo2, hi2 = boot_mean(ds)
        out["step"] = {"n": int(ok.sum()), "diff": round(float(ds.mean()), 1), "diff_ci": (round(lo2, 1), round(hi2, 1)),
                       "p": wilcoxon(sa.values[ok], sb.values[ok], alternative)}
    for name, col in (("entity_exact", "rcl_exact"), ("type_exact", "ft_binary")):
        n10, n01, p = mcnemar(da.loc[ids, col].values, db.loc[ids, col].values)
        out[name] = {"only_first": n10, "only_second": n01, "mcnemar_p": round(p, 4)}
    return out


def breakdown(d, ids):
    from pipeline.fault_calibrated import LEVEL
    from data.taxonomy import fault_group
    x = d.loc[ids].copy()
    gt = x["gt_fault_type"].fillna("").astype(str) if "gt_fault_type" in x else pd.Series("", index=x.index)
    x["_level"] = [LEVEL[t] if LEVEL[t] != "any" else "service" for t in gt]
    x["_group"] = [fault_group(t) for t in gt]
    out = {}
    for key in ("_level", "_group"):
        for g, part in x.groupby(key):
            out[f"{key[1:]}={g}"] = {"n": len(part), "EL": round(100 * part["entity_localization"].mean(), 1),
                                     "FI": round(100 * part["fault_identification_exact"].mean(), 1),
                                     "Final": round(100 * part["final_score_nodes"].mean(), 1)}
    return out


def families(labels):
    """Pre-registered comparison families (Analysis plan v5). Each item: (first, second, alternative)."""
    fam = defaultdict(list)
    for lab in labels:
        base, _, bb = lab.partition("@")
        sfx = f"@{bb}" if bb else ""
        if base == "GAMA_CORE" and "A" in labels:
            fam["main: core vs configuration A"].append((lab, "A", "greater"))
        if base.startswith("GAMA_CORE_NO_") and "GAMA_CORE" in labels:
            fam["ablation: leave one module out"].append(("GAMA_CORE", lab, "two-sided"))
        if base == "GAMA_LLM":
            if "GAMA_CORE" in labels:
                fam[f"llm{sfx}: what the LLM adds"].append((lab, "GAMA_CORE", "two-sided"))
            for other in ("EVIDENCE_LLM_FINAL", "DIRECT_LLM"):
                if other + sfx in labels:
                    fam[f"llm{sfx}: structure vs LLM-decides"].append((lab, other + sfx, "greater"))
        if base == "BELIEF_LLM":                                    # computed belief + an LLM that decides freely
            if "GAMA_CORE" in labels:
                fam[f"belief{sfx}: does a free LLM improve the belief?"].append((lab, "GAMA_CORE", "two-sided"))
            if "EVIDENCE_LLM_FINAL" + sfx in labels:
                fam[f"belief{sfx}: does the belief help the LLM?"].append((lab, "EVIDENCE_LLM_FINAL" + sfx, "greater"))
            if "GAMA_LLM" + sfx in labels:
                fam[f"belief{sfx}: free vs bounded LLM"].append((lab, "GAMA_LLM" + sfx, "two-sided"))
    gl = [l for l in labels if l.startswith("GAMA_LLM@")] or [l for l in labels if l == "GAMA_LLM"]
    for g in gl:                                                    # H4: GAMA + small LLM >= EVIDENCE_LLM with a large model
        for e in [l for l in labels if l.startswith("EVIDENCE_LLM_FINAL@") and l.split("@")[1] not in g]:
            fam["H4: structure beats scale"].append((g, e, "greater"))
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="LABEL=path[,path...] (zip / csv / folder; globs allowed)")
    ap.add_argument("--compare", action="append", default=[], help="FAMILY:FIRST>SECOND or FAMILY:FIRST-SECOND")
    ap.add_argument("--subsets", default="all")   # the paper reports all 103 cases; even / odd stay available
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        lab, _, paths = spec.partition("=")
        arms[lab] = load_arm(paths)
    fam = families(list(arms))
    for c in a.compare:
        name, _, pair = c.partition(":")
        sep = ">" if ">" in pair else "-"
        x, y = pair.split(sep)
        fam[name].append((x, y, "greater" if sep == ">" else "two-sided"))
    rep = {"arms": {}, "families": {}, "breakdown": {}, "missing": {}}
    universe = sorted(set().union(*[set(d.index) for d in arms.values()]))
    for lab, d in arms.items():
        rep["arms"][lab] = {s: summary(d, [i for i in subset(d.index, s)]) for s in a.subsets.split(",")}
        rep["missing"][lab] = sorted(set(universe) - set(d.index))
        rep["breakdown"][lab] = breakdown(d, list(d.index))
    for name, items in fam.items():
        for s in a.subsets.split(","):
            res = []
            for x, y, alt in items:
                ids = [i for i in subset(sorted(set(arms[x].index) & set(arms[y].index)), s)]
                if ids:
                    res.append({"first": x, "second": y, **compare(arms[x], arms[y], ids, alt)})
            for r, ph in zip(res, holm([r["p"] for r in res])):
                r["p_holm"] = ph
            rep["families"][f"{name} [{s}]"] = res
    lines = ["# GAMA-RCA analysis (PREREGISTRATION_V5.md, Analysis plan v5)", ""]
    for s in a.subsets.split(","):
        lines += [f"## Arms -- {s}", "", "| arm | n | Final [95% CI] | EL | FI | RP | Group | Final_step | RP_step | entity / type / joint "
                  "| s/case | calls | tokens (in / out) |", "|" + "---|" * 13]
        for lab in arms:
            r = rep["arms"][lab][s]
            if not r["n"]:
                continue
            tok = f"{r.get('tokens', '-')} ({r.get('input_tokens', '-')} / {r.get('output_tokens', '-')})"
            lines.append(f"| {lab} | {r['n']} | {r['Final']} [{r['Final_ci'][0]}, {r['Final_ci'][1]}] | {r['EL']} | {r['FI']} | "
                         f"{r['RP']} | {r['Group']} | {r['Final_step']}{'*' if 'Final_step_n' in r else ''} | "
                         f"{r['RP_step']}{'*' if 'RP_step_n' in r else ''} | {r['entity_exact']} / {r['type_exact']} / "
                         f"{r['joint']} | {r.get('sec_per_case', '-')} | {r.get('llm_calls', '-')} | {tok} |")
        lines.append("* = scored only for part of the cases (column absent in older runs)\n")
    for name, res in rep["families"].items():
        if not res:
            continue
        lines += [f"## {name}", "", "| first | second | n | diff [95% CI] | better / worse / equal | p | p (Holm) | entity exact "
                  "(McNemar p) | type exact (McNemar p) | Final_step diff [95% CI], p |", "|" + "---|" * 10]
        for r in res:
            lines.append(f"| {r['first']} | {r['second']} | {r['n']} | {r['diff']:+} [{r['diff_ci'][0]}, {r['diff_ci'][1]}] | "
                         f"{r['better']} / {r['worse']} / {r['equal']} | {r['p']:.3g} ({r['alternative']}) | {r['p_holm']:.3g} | "
                         f"{r['entity_exact']['only_first']}:{r['entity_exact']['only_second']} ({r['entity_exact']['mcnemar_p']}) | "
                         f"{r['type_exact']['only_first']}:{r['type_exact']['only_second']} ({r['type_exact']['mcnemar_p']}) | "
                         + (f"{r['step']['diff']:+} [{r['step']['diff_ci'][0]}, {r['step']['diff_ci'][1]}], {r['step']['p']:.3g} |"
                            if r.get("step") else "- |"))
        lines.append("")
    lines += ["## Breakdown (all cases of each arm)", ""]
    for lab, br in rep["breakdown"].items():
        lines.append(f"- **{lab}**: " + "; ".join(f"{k} (n={v['n']}) EL {v['EL']} FI {v['FI']} Final {v['Final']}" for k, v in br.items()))
    for lab, miss in rep["missing"].items():
        if miss:
            lines.append(f"\nMissing in {lab}: {len(miss)} cases ({', '.join(miss[:12])}{' ...' if len(miss) > 12 else ''})")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
