"""db_onset_dev.py -- v5 diagnostic: WHEN do a service's database calls slow down, compared with its CPU / memory / GC / threads?
ODD CASES ONLY. No answer key is read (the service is given on the command line).

Per 30 s bin from one window length before the alert window to its end, for the given service: number of DB spans, mean of the
slowest 10% of them, the share of the service's request time spent in DB calls (as in depshare), and a mark where both exceed the
history (B_SPAN window lengths before) by >= SLOW_CONTRAST. Then the metric onsets of the same service (as signal_onsets_dev.py).
Question for amendment 13: in the DB faults (t015, t021, t097) does the DB latency move BEFORE the CPU?

Usage (Kaggle):  !python scripts/diagnostics/db_onset_dev.py t015 t021 t097 --service inventory > /kaggle/working/db_onset_odd.txt
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BIN = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--service", default="inventory")
    ap.add_argument("--data", default="/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1")
    a = ap.parse_args()
    os.environ["RCA100_ROOT"] = a.data
    os.environ["ALERT_T0_MODE"] = "window"
    import config
    from data.loader import Case
    from pipeline import belief_diagnosis as bd
    from pipeline.dep_latency import dep_kind, tail90
    from pipeline.fault_calibrated import compute_signal_db, B_SPAN
    from pipeline.fault_fingerprint import _owner, _naive
    from pipeline.fault_v4 import calibrate_v4
    from pipeline.trace_structure import SLOW_CONTRAST

    for cid in a.cases:
        if int(cid[1:]) % 2 == 0:
            print(f"{cid}: refused (even = validation half)")
            continue
        case = Case(cid, cases_dir=config.CASES_DIR)
        db = compute_signal_db(case, case.alert.alert_timestamp)
        s = a.service
        if s not in db["services"]:
            print(f"{cid}: no service {s}")
            continue
        w0, w1 = db["w0"], db["w1"]
        L = (w1 - w0).total_seconds()
        h0 = w0.timestamp() - B_SPAN * L
        services = list(db["services"])
        dbd = defaultdict(list)                   # bin -> DB durations (ms)
        spans = []                                # (bin, spanId, parentId, ms) of the service's other spans
        ids = set()
        for o in case.observations.get("traces", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            if ts is None or ts > w1 or _owner(p.get("serviceName"), services) != s:
                continue
            t = ts.timestamp()
            if t < h0:
                continue
            try:
                ms = float(p.get("duration")) / 1e6
            except (TypeError, ValueError):
                continue
            b = int((t - w0.timestamp()) // BIN)  # negative = before the window
            attrs = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
            if dep_kind(str(p.get("spanName") or ""), attrs) == "db":
                dbd[b].append(ms)
            else:
                ids.add(p.get("spanId"))
                spans.append((b, p.get("parentSpanId"), ms))
        entry = defaultdict(float)
        for b, par, ms in spans:
            if not par or par not in ids:
                entry[b] += ms
        first_win = 0
        hist_bins = [b for b in dbd if b < -int(L // BIN)]          # history = before one window length ahead of w0
        hist = [x for b in hist_bins for x in dbd[b]]
        hist_share = sum(hist) / max(1e-9, sum(entry[b] for b in hist_bins))
        th = tail90(hist) if hist else None
        print(f"################ {cid}  service={s}  window {w0} .. {w1}  history: {len(hist)} DB spans, slowest-10% mean "
              f"{th if th is None else round(th, 3)} ms, DB share {hist_share:.0%}")
        print(f"  {'t(s)':>6s} {'n_db':>6s} {'slow10% ms':>11s} {'DB share':>9s}")
        onset = None
        for b in range(-int(L // BIN), int(L // BIN) + 1):
            v = dbd.get(b, [])
            if not v and not entry.get(b):
                continue
            tm = tail90(v) if v else 0.0
            sh = sum(v) / entry[b] if entry.get(b) else float("nan")
            hit = th is not None and v and tm >= (1 + SLOW_CONTRAST) * th and sh == sh and sh >= (1 + SLOW_CONTRAST) * hist_share
            if hit and onset is None:
                onset = b * BIN
            print(f"  {b * BIN:+6.0f} {len(v):6d} {tm:11.3g} {sh:9.0%} {'*' if hit else ''}")
        print(f"  DB onset (first marked bin): {onset}")
        calibrate_v4(db, peer=True, selfnorm=True)              # as DEFAULT_PARTS (score, peer, selfnorm)
        cal = (db.get("_cal") or {}).get(s, {})
        held = sorted((r["onset"] - w0.timestamp(), k) for k, r in cal.items()
                      if r.get("onset") is not None and r.get("p") is not None and r["p"] <= bd.ALPHA)
        print("  metric onsets: " + ", ".join(f"{k}@{t:+.0f}" for t, k in held) + "\n")


if __name__ == "__main__":
    main()
