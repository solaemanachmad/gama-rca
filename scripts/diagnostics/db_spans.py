"""db_spans.py -- v5 diagnostic of dependency-span detection (PREREGISTRATION_V5.md amendment 9). DEVELOPMENT (ODD) CASES ONLY.

Why db_slow does not hold in some cases: for every service (or the ones given with --service) it prints
  * the spans that LOOK like database / cache / queue calls under a loose pattern (jdbc, hikari, sql, select, mongo, redis, ...),
    grouped by spanName, with the kind fault_calibrated assigns them (db / redis / queue / other) and one attribute sample;
  * span counts and median / p90 durations per kind, baseline (before the alert window) vs window;
  * the test behind db_slow today  -- window DB spans vs window OTHER spans of the same service (Mann-Whitney, one-sided) --
    and, for comparison only, window DB spans vs BASELINE DB spans (the same operation before the incident).
No answer key is read. Even cases are refused (validation half).

Usage (Kaggle, RCA100_ROOT set):  ALERT_T0_MODE=window python scripts/diagnostics/db_spans.py t015 t085 [--service inventory]
"""
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from data.loader import Case
from pipeline.fault_calibrated import mwu_greater, ALPHA
from pipeline.fault_fingerprint import _owner, _naive, _DB_RE, _REDIS_RE, _QUEUE_RE
from pipeline.propagation_evidence import window_bounds, _node_name, _node_type

LOOSE = re.compile(r"jdbc|hikari|sql|select|insert|update|delete|query|mysql|postgres|mongo|db\.|database|repository|dao|"
                   r"redis|valkey|cache|kafka|rabbit|queue|messaging", re.I)


def kind_of(span, attrs):          # exactly the rule of fault_calibrated._compute
    return ("db" if _DB_RE.search(span) or ('"db.system"' in attrs and not _REDIS_RE.search(attrs)) else
            "redis" if _REDIS_RE.search(span) or _REDIS_RE.search(attrs) else
            "queue" if _QUEUE_RE.search(span) or _QUEUE_RE.search(attrs) else "other")


def q(v, p):
    return f"{np.percentile(v, p):9.1f}" if v else "        -"


def fmt_p(p):
    return "  n<3" if p is None else f"{p:.3g}" + (" *" if p < ALPHA else "")


def run(cid, only):
    case = Case(cid, cases_dir=config.CASES_DIR)
    topo = case.topology
    w0, w1 = window_bounds(case, _naive(case.alert.alert_timestamp))
    services = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) in ("apm.service", "apm.svc")} - {None})
    dur = defaultdict(lambda: defaultdict(list))          # (svc, phase) -> kind -> [ms]
    names = defaultdict(Counter)                          # svc -> (spanName, kind) -> n   (loose-matching spans only)
    sample, errs = {}, defaultdict(Counter)
    n_tr = 0
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        s = _owner(p.get("serviceName"), services)
        ts = _naive(o.timestamp)
        if not s or ts is None or ts > w1 or (only and s not in only):
            continue
        n_tr += 1
        phase = "win" if ts >= w0 else "base"
        attrs = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
        span = str(p.get("spanName") or "")
        k = kind_of(span, attrs)
        try:
            dur[(s, phase)][k].append(float(p.get("duration")) / 1e6)
        except (TypeError, ValueError):
            pass
        if str(p.get("statusCode")) == "2":
            errs[(s, phase)][k] += 1
        if k != "other" or LOOSE.search(span) or LOOSE.search(attrs):
            names[s][(span[:70], k)] += 1
            sample.setdefault((s, span[:70]), attrs[:240])
    print(f"################ {cid}  window {w0} .. {w1}  spans read: {n_tr}")
    for s in (sorted(only) if only else services):
        if not names[s] and not any(dur[(s, ph)] for ph in ("base", "win")):
            continue
        print(f"\n=== {s}")
        print(f"  {'kind':6s} {'phase':5s} {'n':>7s} {'err':>5s} {'median ms':>9s} {'p90 ms':>9s}")
        for k in ("db", "redis", "queue", "other"):
            for ph in ("base", "win"):
                v = dur[(s, ph)].get(k, [])
                if v:
                    print(f"  {k:6s} {ph:5s} {len(v):7d} {errs[(s, ph)][k]:5d} {q(v, 50)} {q(v, 90)}")
        for k in ("db", "redis", "queue"):
            wk, wo, bk = dur[(s, "win")].get(k, []), dur[(s, "win")].get("other", []), dur[(s, "base")].get(k, [])
            if wk or bk:
                print(f"  {k}_slow  today (win {k} > win other): p={fmt_p(mwu_greater(wk, wo))}   "
                      f"alternative (win {k} > base {k}): p={fmt_p(mwu_greater(wk, bk))}")
        if names[s]:
            print("  dependency-like span names (loose pattern) -> kind assigned:")
            for (nm, k), n in names[s].most_common(15):
                print(f"    {n:6d}  {k:6s}  {nm}")
                a = sample.get((s, nm), "")
                if a:
                    print(f"            attrs: {a}")
        else:
            print("  no dependency-like span names (no DB / cache / queue client spans recorded for this service)")
    print()


def main(argv):
    only = set()
    if "--service" in argv:
        i = argv.index("--service")
        only = set(argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    for cid in argv:
        if int(cid[1:]) % 2 == 0:
            print(f"{cid}: refused -- even cases are the validation half (PREREGISTRATION_V5.md)")
            continue
        run(cid, only)


if __name__ == "__main__":
    main(sys.argv[1:] or ["t015", "t085"])
