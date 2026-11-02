"""silent_dev.py -- v5 diagnostic for scaled-down / unreachable services (t033, t065, t081: replicaScaleDown). ODD CASES ONLY;
no answer key is read.

Per case: (1) per service, spans and metric samples in the history (one window length before the window) vs in the window, i.e.
does a scaled-down service go SILENT, and is that visible; (2) the failing spans whose message says the callee could not be reached
(trace_structure.unreachable_kind), grouped by (service, span name, callee resolved by trace_structure.callee_of, message), with one
attribute sample: can the evidence be moved to the service that could not be reached?

Usage (Kaggle):  !RCA100_ROOT=... ALERT_T0_MODE=window python scripts/diagnostics/silent_dev.py t033 t065 t081 > /kaggle/working/silent_odd.txt 2>&1
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(cases):
    import config
    from data.loader import Case
    from pipeline.fault_calibrated import compute_signal_db
    from pipeline.fault_fingerprint import _owner, _naive
    from pipeline.trace_structure import callee_of, unreachable_kind

    for cid in cases:
        if int(cid[1:]) % 2 == 0:
            print(f"{cid}: refused (even = validation half)")
            continue
        case = Case(cid, cases_dir=config.CASES_DIR)
        db = compute_signal_db(case, case.alert.alert_timestamp)
        services = list(db["services"])
        w0, w1 = db["w0"], db["w1"]
        h0 = w0 - (w1 - w0)
        spans, metr = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        groups, sample = Counter(), {}
        for o in case.observations.get("traces", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            s = _owner(p.get("serviceName"), services)
            if ts is None or not s or ts < h0 or ts > w1:
                continue
            spans[s][1 if ts >= w0 else 0] += 1
            if ts >= w0 and str(p.get("statusCode")) == "2":
                sp = {"service": s, "name": p.get("spanName"), "msg": str(p.get("statusMessage") or ""),
                      "attrs": p.get("attributes") if isinstance(p.get("attributes"), str) else ""}
                if unreachable_kind(sp):
                    key = (s, str(sp["name"])[:60], callee_of(sp, services) or "-", sp["msg"][:90])
                    groups[key] += 1
                    sample.setdefault(key, sp["attrs"][:300])
        for o in case.observations.get("metrics", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            s = _owner(str(p.get("entity_name") or p.get("service") or ""), services)
            if ts is None or not s or ts < h0 or ts > w1:
                continue
            metr[s][1 if ts >= w0 else 0] += 1
        print(f"################ {cid}  window {w0} .. {w1}")
        print(f"  {'service':18s} {'spans before':>12s} {'spans in':>9s} {'metrics before':>15s} {'metrics in':>11s}")
        for s in services:
            a, b = spans.get(s, [0, 0]), metr.get(s, [0, 0])
            flag = "  <- silent?" if (a[0] and not a[1]) or (b[0] and not b[1]) else ""
            print(f"  {s:18s} {a[0]:12d} {a[1]:9d} {b[0]:15d} {b[1]:11d}{flag}")
        print("  unreachable-callee failures: count | service | span | callee_of | message")
        for (s, nm, cal, msg), n in groups.most_common(12):
            print(f"    {n:6d} | {s} | {nm} | {cal} | {msg}")
            print(f"           attrs: {sample[(s, nm, cal, msg)]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["t033", "t065", "t081"])
