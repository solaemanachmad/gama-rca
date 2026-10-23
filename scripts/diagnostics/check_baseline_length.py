"""
check_baseline_length.py -- how much NORMAL (pre-window) telemetry each case has. Label-free: reads only cases/, never answer_key/.

Why: the calibrated fingerprint layer compares the alert window with blocks of the same length cut from the baseline (conformal p-values).
The smallest attainable p-value is 1/(n+1) for n blocks, so a per-signal test at alpha = 0.05 needs n >= 19 blocks per entity. This script
reports n per case and modality, and the metric sampling interval, to decide between per-entity calibration and within-case pooled
calibration BEFORE the method is fixed.

Also prints the rule lines of the dataset's own README files (if present in RCA100_ROOT) so the usage rules can be checked against the method.

Usage (RCA100_ROOT set):  python scripts/diagnostics/check_baseline_length.py [--all] [t004 t005 ...]
"""
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config
from data.loader import load_alert_context, list_case_ids
from pipeline.propagation_evidence import window_bounds

DEV30 = ["t004", "t005", "t012", "t014", "t015", "t018", "t026", "t028", "t029", "t030", "t032", "t036", "t054", "t055", "t058",
         "t065", "t070", "t072", "t076", "t078", "t082", "t087", "t091", "t092", "t093", "t095", "t097", "t099", "t100", "t101"]


class _C:   # the minimal case object window_bounds needs
    def __init__(self, alert):
        self.alert = alert


def _ts(series, unit=None):
    t = pd.to_datetime(series, unit=unit, utc=True, errors="coerce") if unit else pd.to_datetime(series, utc=True, errors="coerce")
    return t.dropna().dt.tz_localize(None)


def readme_rules():
    root = config.DATASET_ROOT
    files = sorted(glob.glob(os.path.join(root, "*README*")) + glob.glob(os.path.join(root, "*", "*README*")))   # usage rules only, no labels
    keys = ("must", "not ", "never", "forbid", "only", "allowed", "prohibit", "license", "answer_key", "禁止", "不得", "不要")
    for f in files[:5]:
        try:
            lines = open(f, encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        hits = [l.strip() for l in lines if any(k in l.lower() for k in keys)]
        print(f"\n--- usage rules in {os.path.relpath(f, root)} ({len(hits)} lines)")
        for l in hits[:40]:
            print("  ", l[:200])


def main(argv):
    ids = list_case_ids() if "--all" in argv else ([a for a in argv if not a.startswith("--")] or DEV30)
    print("case  window(s) mode   | baseline minutes and n blocks per modality (metrics / traces / logs / events) | metric step(s) | samples/series")
    rows = []
    for cid in ids:
        cdir = os.path.join(config.CASES_DIR, cid)
        try:
            alert = load_alert_context(cdir, cid)
            at = alert.alert_timestamp.replace(tzinfo=None) if alert.alert_timestamp.tzinfo else alert.alert_timestamp
            w0, w1 = window_bounds(_C(alert), at)
            L = (w1 - w0).total_seconds()
            out = {"case": cid, "L": L}
            m = pd.read_parquet(os.path.join(cdir, config.FILE_METRICS), columns=["time", "entity_set", "entity_name", "metric"])
            m["t"] = pd.to_datetime(m["time"], unit="us", utc=True, errors="coerce").dt.tz_localize(None)
            mods = {"metrics": m["t"].dropna(),
                    "traces": _ts(pd.read_parquet(os.path.join(cdir, config.FILE_TRACES), columns=["startTime"])["startTime"], "ns"),
                    "logs": _ts(pd.read_parquet(os.path.join(cdir, config.FILE_LOGS), columns=["_time_"])["_time_"])}
            ev = pd.read_parquet(os.path.join(cdir, config.FILE_EVENTS))
            mods["events"] = _ts(ev["eventId"].astype(str).str.extract(r'"lastTimestamp":\s*"([^"]+)"')[0]) if "eventId" in ev else pd.Series([], dtype="datetime64[ns]")
            parts = []
            for k, t in mods.items():
                base = (w0 - t.min()).total_seconds() / 60 if len(t) else 0.0
                n = int(max(0, base * 60) // L) if L else 0
                out[f"{k}_base_min"], out[f"{k}_n"] = round(base, 1), n
                parts.append(f"{base:6.1f}m/{n:3d}")
            # metric sampling: median step and median number of baseline samples per (entity, metric) series
            b = m[m["t"] < w0].sort_values("t")
            g = b.groupby(["entity_set", "entity_name", "metric"])["t"]
            steps = g.apply(lambda s: s.diff().dt.total_seconds().median()).dropna()
            counts = g.size()
            out["metric_step_s"] = round(float(steps.median()), 1) if len(steps) else None
            out["metric_base_samples"] = int(counts.median()) if len(counts) else 0
            rows.append(out)
            print(f"{cid}  {L:6.0f} {getattr(config, 'ALERT_T0_MODE', 'legacy'):6s} | " + " | ".join(parts)
                  + f" | {out['metric_step_s']} | {out['metric_base_samples']}")
        except Exception as e:
            print(f"{cid}: error {e!r}")
    if rows:
        print("\nSUMMARY over", len(rows), "cases (median [min, max]):")
        for k in ("L", "metrics_n", "traces_n", "logs_n", "events_n", "metrics_base_min", "metric_step_s", "metric_base_samples"):
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                print(f"  {k:22s} {statistics.median(v):8.1f}  [{min(v)}, {max(v)}]")
        for k in ("metrics_n", "traces_n", "logs_n"):
            v = [r[k] for r in rows]
            print(f"  cases with {k} >= 19 (per-entity alpha=0.05 feasible): {sum(x >= 19 for x in v)}/{len(v)}")
    readme_rules()


if __name__ == "__main__":
    main(sys.argv[1:])
