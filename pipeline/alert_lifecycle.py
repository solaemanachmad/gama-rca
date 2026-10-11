"""Alert lifecycle summary (evidence extraction ONLY -- no decision is made here).

The alert stream repeats the same alert every 15-60 s (median 13 rows per case, up to 393). The only
information in those repeats is: when the alert first fired, how the monitored value moved while it kept
firing, and whether/when it recovered. This module run-length-encodes consecutive repeats into runs and
prints one short block.

Rules (uniform for every case, nothing tuned on ground truth):
  * time axis: T = the case's first alert row; offsets are seconds from it.
  * only rows up to the task's `alert_window.end` are used; the number of later rows is reported, not shown.
  * a run = consecutive rows with identical condition values (rounded to VALUE_DECIMALS); a RECOVERED row
    always starts its own run. No significance threshold, no filtering; only compression.
  * at most MAX_RUNS runs are printed (first FIRST_RUNS + last LAST_RUNS); omitted runs are counted and their
    value range is stated.
Uses the alert stream and the task contract only (no ground truth, no training)."""
from __future__ import annotations

import json
import statistics
from typing import Dict, List, Optional

import pandas as pd

VALUE_DECIMALS = 3
MAX_RUNS = 12
FIRST_RUNS = 5
LAST_RUNS = 6
RULE_CHARS = 220


def _j(x) -> dict:
    if isinstance(x, dict):
        return x
    try:
        v = json.loads(x) if isinstance(x, str) else {}
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _t(p: dict) -> Optional[float]:
    for k, unit in (("time_s", 1.0), ("timestamp", 1000.0)):
        try:
            v = float(p.get(k))
            if v == v:
                return v / unit
        except (TypeError, ValueError):
            continue
    return None


def _clock(sec: float) -> str:
    sec = int(round(sec))
    sign = "-" if sec < 0 else "+"
    sec = abs(sec)
    return f"T{sign}{sec // 60}:{sec % 60:02d}"


def _g(x) -> str:
    return f"{x:.{VALUE_DECIMALS}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def _conds(data: dict) -> tuple:
    out = []
    for d in data.get("detailValue") or []:
        if not isinstance(d, dict):
            continue
        try:
            out.append((round(float(d.get("value")), VALUE_DECIMALS),
                        round(float(d["yoyOldValue"]), VALUE_DECIMALS) if d.get("yoyOldValue") is not None else None,
                        round(float(d["yoyNewValue"]), VALUE_DECIMALS) if d.get("yoyNewValue") is not None else None))
        except (TypeError, ValueError):
            out.append((None, None, None))
    return tuple(out)


def _cond_text(c: tuple) -> str:
    parts = []
    for i, (v, old, new) in enumerate(c):
        s = f"c{i}={_g(v) if v is not None else 'n/a'}"
        if old is not None and new is not None:
            s += f" (yoy {_g(old)}->{_g(new)})"
        parts.append(s)
    return ", ".join(parts) or "no condition values"


def compute_alert_lifecycle(case) -> Dict:
    empty = {"text": "", "runs": 0}
    try:
        return _compute(case) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _compute(case) -> Optional[Dict]:
    obs = case.observations.get("alerts", [])
    rows = []
    for o in obs:
        p = o.payload or {}
        t = _t(p)
        if t is None:
            continue
        rows.append((t, p))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])  # stable: equal times keep file order
    t0 = rows[0][0]

    end = None
    try:
        end = pd.Timestamp(((case.alert.raw or {}).get("alert_window") or {}).get("end")).timestamp()
    except Exception:
        end = None
    used = [r for r in rows if end is None or r[0] <= end]
    n_after = len(rows) - len(used)
    if not used:
        return None

    first = used[0][1]
    ann = _j(first.get("annotations"))
    rule = " ".join(str(ann.get("_cms_rule_display_statement") or first.get("subject") or "").split())[:RULE_CHARS]
    ent = (case.alert.raw or {}).get("_resolved_entity_name") or ""
    sev = first.get("severity") or ""
    thr = []
    for d in _j(first.get("data")).get("detailValue") or []:
        if isinstance(d, dict) and d.get("thresholdValue") is not None:
            thr.append(f"c{d.get('idx', len(thr))} threshold {_g(d['thresholdValue'])}")

    # One tick = one evaluation time. The stream emits several identical rows per tick (and sometimes both a
    # PERSISTENT and a RECOVERED record in the same tick), so rows are first reduced to the distinct states of
    # their tick; consecutive ticks with the same set of states form a run.
    ticks: Dict[float, dict] = {}
    for t, p in used:
        status = str(p.get("status"))
        st = ticks.setdefault(t, {})
        st.setdefault((status == "RECOVERED", _conds(_j(p.get("data")))), set()).add(status)
    runs: List[dict] = []
    for t in sorted(ticks):
        key = tuple(sorted(ticks[t], key=repr))
        if runs and runs[-1]["key"] == key:
            r = runs[-1]
        else:
            r = {"key": key, "t": [], "st": {}}
            runs.append(r)
        r["t"].append(t)
        for names in ticks[t].values():
            for nm in names:
                r["st"][nm] = r["st"].get(nm, 0) + 1

    def run_line(r) -> str:
        ts = sorted(set(r["t"]))
        span = _clock(ts[0] - t0) + ("" if len(ts) == 1 else "-" + _clock(ts[-1] - t0))
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        every = f", every {int(round(statistics.median(gaps)))} s" if gaps else ""
        st = ", ".join(f"{k} x{v}" for k, v in r["st"].items())
        vals = " | ".join(_cond_text(c) for _, c in r["key"])
        return f"- {span}: {len(ts)} evaluation(s){every}; {st}; {vals}"

    shown, omitted = runs, []
    if len(runs) > MAX_RUNS:
        shown = runs[:FIRST_RUNS] + runs[-LAST_RUNS:]
        omitted = runs[FIRST_RUNS:-LAST_RUNS]

    lines = ["Alert lifecycle (the alert stream's repeats condensed; T = first alert firing; "
             "rows after the task window end are not used):",
             f"- rule: {rule}; entity: {ent}; severity: {sev}" + (f"; {', '.join(thr)}" if thr else "")]
    lines += [run_line(r) for r in shown[:FIRST_RUNS if omitted else len(shown)]]
    if omitted:
        vals = [c[0] for r in omitted for _, cs in r["key"] for c in cs if c and c[0] is not None]
        rng = f"; c-values {_g(min(vals))}..{_g(max(vals))}" if vals else ""
        lines.append(f"- ... {len(omitted)} further run(s) between {_clock(min(omitted[0]['t']) - t0)} and "
                     f"{_clock(max(omitted[-1]['t']) - t0)} omitted for length{rng}")
        lines += [run_line(r) for r in shown[FIRST_RUNS:]]

    last_t = max(ticks)
    names = sorted({n for ns in ticks[last_t].values() for n in ns})
    end_txt = (f"RECOVERED at {_clock(last_t - t0)}" if "RECOVERED" in names
               else f"still firing ({'/'.join(names)}) at the last evaluation, {_clock(last_t - t0)}")
    if "RECOVERED" in names and len(names) > 1:
        end_txt += f" (the same evaluation also carries {'/'.join(n for n in names if n != 'RECOVERED')} records)"
    lines.append(f"- state at the end of the window: {end_txt}"
                 + (f"; {n_after} later row(s) after the window end not used" if n_after else ""))
    text = "\n".join(lines)
    return {"text": text, "runs": len(runs), "rows_used": len(used), "rows_after_end": n_after}
