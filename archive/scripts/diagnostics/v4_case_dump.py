"""
v4_case_dump.py -- per-case evidence dump for v4 development on RCA100 (dev data). Reads only the case's telemetry (never answer_key/).
For each case: the evidence window; per node and per service, compact TIMELINES (5-minute medians from the first sample to the window
end, '|' marks the window start) of the metrics that the failing fault types depend on, the v4 calibrated p of each continuous signal,
and Kubernetes events that overlap the window with their first/last time and count (chronic vs new).

Usage (RCA100_ROOT set, ALERT_T0_MODE=window):  python scripts/diagnostics/v4_case_dump.py t008 t036 ...
"""
import json
import os
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from data.loader import Case, event_time_bounds, _parse_ts_iso
from pipeline.fault_calibrated import compute_signal_db, entity_signals
from pipeline.fault_fingerprint import _naive
from pipeline.fault_v4 import calibrate_v4

NODE_M = ("node_cpu_usage_rate", "node_memory_usage_rate", "node_ready_status", "node_pod_running_count")
SVC_M = ("cpu", "mem", "latency", "error", "workload", "deployment_available_replicas", "deployment_cpu_usage_vs_limits",
         "deployment_memory_usage_total", "arms_jvm_mem_used_bytes", "arms_jvm_gc_delta", "arms_jvm_gc_seconds_delta")


def fmt(v):
    if v is None:
        return "  .  "
    a = abs(v)
    return f"{v:.3g}" if a < 1e4 else f"{v:.2e}"


def timeline(pts, w0, w1, step_min=5):
    pts = sorted(pts)
    if not pts:
        return ""
    t0 = pts[0][0]
    bins = defaultdict(list)
    for t, v in pts:
        if t > w1:
            continue
        k = int((t - w0).total_seconds() // (60 * step_min))
        bins[k].append(v)
    lo = int((t0 - w0).total_seconds() // (60 * step_min))
    hi = int((w1 - w0).total_seconds() // (60 * step_min))
    cells = []
    for k in range(lo, hi + 1):
        if k == 0:
            cells.append("|")
        cells.append(fmt(median(bins[k])) if bins.get(k) else "  .  ")
    return " ".join(cells)


def main(ids):
    for cid in ids:
        case = Case(cid, cases_dir=config.CASES_DIR)
        db = compute_signal_db(case, case.alert.alert_timestamp)
        calibrate_v4(db)
        w0, w1 = db["w0"], db["w1"]
        print(f"\n################ {cid}  window {w0} .. {w1}  alert: {getattr(case.alert, 'alert_text', '')[:160]!r}")
        print("NODES (5-min medians, '|' = window start)")
        for n, pr in sorted(db["nodes"].items()):
            sig = {k: v for k, v in entity_signals(db, n).items()}
            ps = " ".join(f"{k}={v['p']:.3g}" if v["p"] is not None else k for k, v in sig.items())
            print(f"  {n}: {ps}")
            for m in NODE_M:
                if pr["series"].get(m):
                    print(f"     {m:28s} {timeline(pr['series'][m], w0, w1)}")
        print("SERVICES")
        for s, pr in sorted(db["services"].items()):
            sig = entity_signals(db, s)
            low = {k: v for k, v in sig.items() if v["p"] is None or v["p"] <= 0.2}
            print(f"  {s} (hosts {db['hosts'].get(s, [])}): " + " ".join(f"{k}={v['p']:.3g}" if v["p"] is not None else k for k, v in low.items()))
            for key, pts in sorted(pr["series"].items()):
                m = key.split(":")[-1]
                if m in SVC_M and ":" in key and (key.split(":")[1] == s or m.startswith("arms_jvm")):
                    print(f"     {key[:60]:60s} {timeline(pts, w0, w1)}")
        print("EVENTS overlapping the window (kind/name reason first..last count)")
        seen = defaultdict(int)
        for o in case.observations.get("events", []):
            p = o.payload or {}
            try:
                e = json.loads(p.get("eventId")) if isinstance(p.get("eventId"), str) else (p.get("eventId") or {})
            except Exception:
                continue
            f_, l_ = event_time_bounds(e)
            first, last = _naive(_parse_ts_iso(f_)), _naive(_parse_ts_iso(l_))
            if first is None or last is None or last < w0 or first > w1:
                continue
            io = e.get("involvedObject") or {}
            key = (io.get("kind"), io.get("name"), e.get("reason"), str(first)[:19], str(last)[:19])
            seen[key] += int(e.get("count") or 1)
        for (k, nme, r, f_, l_), c in sorted(seen.items(), key=lambda kv: kv[0][3])[:40]:
            print(f"  {k}/{nme} {r} {f_}..{l_} x{c}")


if __name__ == "__main__":
    main(sys.argv[1:])
