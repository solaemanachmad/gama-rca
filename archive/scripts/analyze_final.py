"""
analyze_final.py -- the pre-registered analysis of PREREGISTRATION_V3.md, fixed BEFORE the final runs (reads the run CSVs, which contain
the scores). Joint accuracy = exact entity (rcl_exact: the first predicted entity id is a target id) AND fault type (ft_binary).

    python scripts/analyze_final.py V3=V3_final.csv V3_RAWLOG=V3_RAWLOG_final.csv V2=V2_final.csv V3_B0=V3_B0_final.csv V3_RULE=V3_RULE_final.csv DIRECT=DIRECT_LLM_final.csv

Prints: per system joint / entity / type / group accuracy, RP (chain), Final, LLM calls and time per case, abstain-like low-confidence
share; confirmatory tests H1 (V3 > V2), H2 (V3 > V3_B0) and H3 (V3 > V3_RAWLOG, Amendment 2) on joint accuracy, one-sided exact McNemar with Holm correction;
exploratory paired tests; bootstrap 95% CIs of paired differences; breakdown by level (node / service) and by semantic fault group.
Error rows count as wrong (they stay in the denominator of 103).
"""
import csv
import math
import random
import sys

CONFIRMATORY = [("H1", "V3", "V2"), ("H2", "V3", "V3_B0"), ("H3", "V3", "V3_RAWLOG")]   # H3: Amendment 2
EXPLORATORY = [("V3", "V3_RULE"), ("V3", "DIRECT"), ("V2", "DIRECT"), ("V3_B0", "V3_RULE"), ("V3_RAWLOG", "V3_B0")]


def f(r, k):
    try:
        return float(r.get(k) or 0.0)
    except ValueError:
        return 0.0


def load(path):
    rows = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r.get("case_id"):
            rows[r["case_id"]] = r
    return rows


def outcome(r, key):
    if r is None or (r.get("error") or "").strip():
        return 0
    if key == "joint":
        return int(f(r, "rcl_exact") * f(r, "ft_binary") > 0)
    if key == "entity":
        return int(f(r, "rcl_exact") > 0)
    if key == "type":
        return int(f(r, "ft_binary") > 0)
    if key == "group":
        return int(f(r, "fault_group_identification") > 0)
    raise KeyError(key)


def mcnemar_one_sided(a, b):
    """P(at least n_ab successes | n = n_ab + n_ba, p = 0.5): A better than B."""
    nab = sum(1 for x, y in zip(a, b) if x and not y)
    nba = sum(1 for x, y in zip(a, b) if y and not x)
    n = nab + nba
    p = sum(math.comb(n, k) for k in range(nab, n + 1)) / 2 ** n if n else 1.0
    return nab, nba, p


def mcnemar_two_sided(a, b):
    nab, nba, _ = mcnemar_one_sided(a, b)
    n = nab + nba
    k = min(nab, nba)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n) if n else 1.0
    return nab, nba, p


def boot_ci(a, b, n=10000, seed=0):
    rnd = random.Random(seed)
    m = len(a)
    d = []
    for _ in range(n):
        idx = [rnd.randrange(m) for _ in range(m)]
        d.append(sum(a[i] - b[i] for i in idx) / m)
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def main(args):
    runs = dict(a.split("=", 1) for a in args)
    data = {k: load(v) for k, v in runs.items()}
    cases = sorted(set().union(*[set(d) for d in data.values()]))
    print(f"{len(cases)} cases in the union of runs")
    level = {}
    group = {}
    for d in data.values():
        for c, r in d.items():
            if r.get("gt_fault_group"):
                group.setdefault(c, r["gt_fault_group"])
            if r.get("gt_fault_type"):
                level.setdefault(c, "node" if any(x in r["gt_fault_type"] for x in ("nodeCpuHigh", "nodeMemoryOOM", "nodeDown")) else "service")
    out = {}
    print(f"\n{'system':10s} {'joint':>6s} {'entity':>6s} {'type':>6s} {'group':>6s} {'RPchain':>7s} {'Final':>6s} {'calls':>5s} {'sec':>6s} {'errors':>6s}")
    for k, d in data.items():
        rs = [d.get(c) for c in cases]
        res = {m: [outcome(r, m) for r in rs] for m in ("joint", "entity", "type", "group")}
        out[k] = res
        ok = [r for r in rs if r is not None and not (r.get("error") or "").strip()]
        mean = lambda key: sum(f(r, key) for r in ok) / max(len(ok), 1)
        print(f"{k:10s} " + " ".join(f"{100 * sum(res[m]) / len(cases):6.1f}" for m in ("joint", "entity", "type", "group"))
              + f" {100 * mean('reasoning_process_chain'):7.1f} {100 * mean('final_score'):6.1f} {mean('total_calls'):5.1f}"
              f" {mean('total_pipeline_time_s'):6.0f} {len(cases) - len(ok):6d}")
    print("\nCONFIRMATORY (joint accuracy, one-sided exact McNemar, Holm over the confirmatory tests present):")
    tests = []
    for h, a, b in CONFIRMATORY:
        if a in out and b in out:
            nab, nba, p = mcnemar_one_sided(out[a]["joint"], out[b]["joint"])
            tests.append([h, a, b, nab, nba, p])
    tests.sort(key=lambda t: t[5])
    m = len(tests)
    running = 0.0
    for i, t in enumerate(tests):
        adj = min(1.0, max(running, (m - i) * t[5]))
        running = adj
        lo, hi = boot_ci(out[t[1]]["joint"], out[t[2]]["joint"])
        print(f"  {t[0]}: {t[1]} > {t[2]}  only-{t[1]}={t[3]} only-{t[2]}={t[4]}  p={t[5]:.4f}  Holm p={adj:.4f}  "
              f"diff 95% CI [{100 * lo:.1f}, {100 * hi:.1f}] points")
    print("\nEXPLORATORY (two-sided exact McNemar, no correction):")
    for a, b in EXPLORATORY:
        if a in out and b in out:
            for metric in ("joint", "entity", "type"):
                nab, nba, p = mcnemar_two_sided(out[a][metric], out[b][metric])
                print(f"  {a} vs {b} [{metric}]: only-{a}={nab} only-{b}={nba} p={p:.4f}")
    for name, key in (("level", level), ("fault group", group)):
        print(f"\nBY {name.upper()} (joint accuracy %, n):")
        cats = sorted(set(key.values()))
        for cat in cats:
            idx = [i for i, c in enumerate(cases) if key.get(c) == cat]
            print(f"  {cat:20s} n={len(idx):3d} " + " ".join(f"{k}={100 * sum(out[k]['joint'][i] for i in idx) / len(idx):5.1f}" for k in out))


if __name__ == "__main__":
    main(sys.argv[1:])
