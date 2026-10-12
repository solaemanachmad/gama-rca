"""Arm X: time-aligned evidence strip (evidence extraction ONLY -- no decision, no significance filter, no threshold).

Design: claude/design-arm-X-interval-strip-2026-10-12.md; measurements: claude/audit-cross-modality-interval-2026-10-12.md.

All modalities share a time axis and entity names, so what the other modalities show in the same interval can be joined by NAME and
TIME without asking a model. The block shows, on one grid of 30 s bins from the window start (the native metric resolution):
  * per service of the alert neighbourhood (alert service, direct callers, callees <= 2 hops; chosen by structure, cap MAX_SVC):
    workload, error and latency as multiples of the series' own baseline median (baseline = all samples before the window),
    with the offsets of the minimum and maximum bin; span error share per bin where an export burst covers the bin;
  * the alert stream (evaluations per bin), K8s events on the neighbourhood's pods (first / last seen per bin and reason);
  * the part of the window covered by the log and trace export bursts. Outside it a missing record is an export artefact,
    not evidence of absence (audit: bursts cover a median of ~10% of the window).
Nothing is selected by how unusual it is: every bin of every shown series is printed; services beyond the cap are dropped in
structural order, never by value. Ground truth is not used.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from pipeline.export_bursts import (burst_ids, burst_spans, classify_bursts, drop_fragments, neighbourhood, off_label, sec)
from pipeline.propagation_evidence import _calls_graph, _node_name, window_bounds

BIN_S = 30.0          # native metric resolution (all 657 audited series have a 30 s interval)
MAX_SVC = 8           # services shown (structural order); observed neighbourhood size 2-12
MAX_CHARS = 3600      # uniform budget; services beyond it are dropped in structural order
MAX_REASONS = 8       # event reasons shown (alphabetical)
_SERIES = (("workload", "workload"), ("error", "error"), ("latency", "latency"))
_POD = re.compile(r"^(-(?=[a-z0-9]*\d)[a-z0-9]{8,10}-[a-z0-9]{5}|-\d+)$")   # deployment pod / statefulset suffix


def compute_interval_strip(case, alert_ts: Optional[dt.datetime], entry_entity_id: Optional[str] = None,
                           max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0, "stats": {}}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _to_sec(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    m = float(v.dropna().abs().median()) if v.notna().any() else 0.0
    div = 1e9 if m > 1e17 else 1e6 if m > 1e14 else 1e3 if m > 1e11 else 1.0
    return v / div


def _num(x: float) -> str:
    """short absolute number for the 'base' annotation."""
    if x != x:
        return "n/a"
    a = abs(x)
    if a >= 1e6:
        return f"{x / 1e6:.1f}M"
    if a >= 1e4:
        return f"{x / 1e3:.0f}k"
    if a >= 1e3:
        return f"{x / 1e3:.1f}k"
    if a >= 10:
        return f"{x:.0f}"
    if a >= 1:
        return f"{x:.1f}"
    return f"{x:.2g}"


def _mult(r: float) -> str:
    """multiple of the baseline median, 2 significant characters where possible."""
    if r != r:
        return "·"
    if r == 0:
        return "0"
    if r < 0.01:
        return "~0"
    if r < 0.1:
        return f"{r:.2f}"
    if r < 10:
        return f"{r:.1f}"
    if r < 1000:
        return f"{r:.0f}"
    return f"{r / 1000:.1f}k"


def _order(case, svc0, cand) -> List[str]:
    """alert service, direct callers, then callees by hop distance; ties by name (structure only)."""
    import networkx as nx
    topo = case.topology
    calls = _calls_graph(topo)
    dist = {svc0: 0}
    if svc0 in calls:
        dist.update(nx.single_source_shortest_path_length(calls, svc0, cutoff=2))
    callers = set(calls.predecessors(svc0)) if svc0 in calls else set()
    def key(e):
        nm = _node_name(topo, e)
        if e == svc0:
            return (0, 0, nm)
        if e in callers and e not in dist:
            return (1, 0, nm)
        return (2, dist.get(e, 9), nm)
    return [e for e in sorted(cand, key=key)]


def _bursts(times_sorted: np.ndarray, w0s: float, w1s: float):
    """(window-burst intervals clipped to the window, fallback flag)."""
    if times_sorted.size == 0:
        return [], False
    ids = burst_ids(times_sorted)
    span, _ = drop_fragments(burst_spans(times_sorted, ids), ids)
    win, _base, note = classify_bursts(span, w0s, w1s)
    iv = [(max(span[k][0], w0s), min(span[k][1], w1s)) for k in win if span[k][1] >= w0s and span[k][0] <= w1s]
    return sorted(iv), bool(note)


def _covered(mid: float, iv: List[Tuple[float, float]]) -> bool:
    return any(a <= mid <= b for a, b in iv)


def _compute(case, alert_ts, entry_entity_id, max_hops):
    if alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = window_bounds(case, at)
    w0s, w1s = sec(w0), sec(w1)
    n_bins = max(1, int(np.ceil((w1s - w0s) / BIN_S)))
    edges = w0s + BIN_S * np.arange(n_bins + 1)
    mids = edges[:-1] + BIN_S / 2
    labels = [off_label(e, w0s) for e in edges[:-1]]

    cand, svc0 = neighbourhood(case, entry_entity_id, max_hops)
    if not cand:
        return None
    topo = case.topology
    ordered = _order(case, svc0, cand)
    names = [(e, _node_name(topo, e)) for e in ordered]

    # ---- metrics ----
    fp = os.path.join(case.case_dir, config.FILE_METRICS)
    m = pq.read_table(fp, columns=["time", "entity_set", "entity_name", "metric", "value"]).to_pandas()
    m = m[(m["entity_set"] == "apm.service.legacy") & (m["entity_name"].isin({n for _, n in names}))]
    m["t"] = _to_sec(m["time"])
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    m = m.dropna(subset=["t", "value"])
    m = m[m["t"] <= w1s]

    # ---- trace export coverage and span error share per service and bin ----
    tfp = os.path.join(case.case_dir, config.FILE_TRACES)
    tr = pq.read_table(tfp, columns=["startTime", "serviceName", "statusCode"]).to_pandas()
    tt = _to_sec(tr["startTime"]).values
    order = np.argsort(tt)
    t_sorted = tt[order][~np.isnan(tt[order])]
    tr_iv, tr_fallback = _bursts(t_sorted, w0s, w1s)
    tr["t"] = tt
    trw = tr[(tr["t"] >= w0s) & (tr["t"] < w0s + n_bins * BIN_S)]
    # ---- log export coverage (time column only) ----
    lfp = os.path.join(case.case_dir, config.FILE_LOGS)
    lg = pq.read_table(lfp, columns=["_time_"]).to_pandas()
    lt = np.sort((pd.to_datetime(lg["_time_"], utc=True, errors="coerce").astype("int64") / 1e9).dropna().values)
    del lg
    lg_iv, lg_fallback = _bursts(lt, w0s, w1s)

    def cov_len(iv):
        return sum(max(0.0, b - a) for a, b in iv) / (w1s - w0s) if w1s > w0s else 0.0

    blocks: List[str] = []
    shown = 0
    for e, nm in names[:MAX_SVC]:
        g = m[m["entity_name"] == nm]
        if g.empty:
            continue
        lines = [f"- {nm}" + (" (alert service)" if e == svc0 else "")]
        for key, label in _SERIES:
            s = g[g["metric"] == key]
            if s.empty:
                continue
            ser = s.groupby("t")["value"].mean().sort_index()
            base = ser[ser.index < w0s]
            bm = float(base.median()) if len(base) else float("nan")
            win = ser[(ser.index >= w0s) & (ser.index < w0s + n_bins * BIN_S)]
            if win.empty:
                continue
            idx = np.clip(((win.index.values - w0s) // BIN_S).astype(int), 0, n_bins - 1)
            cell = [float("nan")] * n_bins
            acc: Dict[int, List[float]] = {}
            for i, v in zip(idx, win.values):
                acc.setdefault(int(i), []).append(float(v))
            for i, vs in acc.items():
                cell[i] = float(np.mean(vs))
            if bm == bm and bm != 0:
                vals = [c / bm if c == c else float("nan") for c in cell]
                txt = " ".join(_mult(v) for v in vals)
                basetxt = _num(bm)
            else:
                vals = cell
                txt = " ".join("·" if v != v else _num(v) for v in vals)
                basetxt = "0/none (absolute values)"
            fin = [(i, v) for i, v in enumerate(vals) if v == v]
            ext = ""
            if len(fin) > 1:
                imin = min(fin, key=lambda x: x[1])[0]
                imax = max(fin, key=lambda x: x[1])[0]
                ext = f" [min {labels[imin]}, max {labels[imax]}]"
            lines.append(f"  {label} (base {basetxt}): {txt}{ext}")
        # span error share per covered bin
        ts = trw[trw["serviceName"] == nm]
        if len(ts) and tr_iv:
            b = ((ts["t"].values - w0s) // BIN_S).astype(int)
            err = (ts["statusCode"].astype(str).values == "2")
            parts = []
            for i in range(n_bins):
                if _covered(mids[i], tr_iv):
                    sel = b == i
                    if sel.any():
                        parts.append(f"{labels[i]} {100.0 * err[sel].mean():.0f}% of {int(sel.sum())}")
            if parts:
                lines.append("  span error share (covered bins only): " + ", ".join(parts))
        if len(lines) == 1:
            continue
        cand_block = "\n".join(lines)
        if sum(len(x) + 1 for x in blocks) + len(cand_block) > MAX_CHARS:
            break
        blocks.append(cand_block)
        shown += 1

    # ---- alerts per bin ----
    afp = os.path.join(case.case_dir, config.FILE_ALERTS)
    al = pq.read_table(afp, columns=["time_s"]).to_pandas()
    ats = pd.to_numeric(al["time_s"], errors="coerce").dropna().values.astype(float)
    ats = ats[(ats >= w0s) & (ats < w0s + n_bins * BIN_S)]
    acounts = np.bincount(((ats - w0s) // BIN_S).astype(int), minlength=n_bins)[:n_bins] if ats.size else np.zeros(n_bins, int)

    # ---- events on the neighbourhood's pods ----
    ev = pq.read_table(os.path.join(case.case_dir, config.FILE_EVENTS), columns=["eventId"]).to_pandas()
    svc_names = sorted({n for _, n in names}, key=len, reverse=True)
    first_c: Dict[str, np.ndarray] = {}
    last_c: Dict[str, np.ndarray] = {}
    n_ev_other = 0
    for raw in ev["eventId"]:
        try:
            e = json.loads(raw)
        except Exception:
            continue
        obj = (e.get("involvedObject") or {}).get("name") or ""
        hit = None
        for sn in svc_names:
            if obj == sn or (obj.startswith(sn + "-") and _POD.match(obj[len(sn):])):
                hit = sn
                break
        try:
            f = pd.Timestamp(e.get("firstTimestamp")).tz_convert("UTC").timestamp()
            l = pd.Timestamp(e.get("lastTimestamp") or e.get("firstTimestamp")).tz_convert("UTC").timestamp()
        except Exception:
            continue
        in_f = w0s <= f < w0s + n_bins * BIN_S
        in_l = w0s <= l < w0s + n_bins * BIN_S
        if not (in_f or in_l):
            continue
        if hit is None:
            n_ev_other += 1
            continue
        rs = str(e.get("reason"))
        if in_f:
            first_c.setdefault(rs, np.zeros(n_bins, int))[int((f - w0s) // BIN_S)] += 1
        if in_l:
            last_c.setdefault(rs, np.zeros(n_bins, int))[int((l - w0s) // BIN_S)] += 1
    reasons = sorted(set(first_c) | set(last_c))

    out: List[str] = []
    out.append(
        f"Interval strip: the window [alert_window.start .. end] in {int(BIN_S)} s bins, one value per bin, T+m:ss = bin start "
        f"relative to the window start. Metric values are multiples of the series' own baseline median (baseline = all samples "
        f"before the window); '·' = no sample, '~0' = below 0.01. A service that never reports errors has no error row. "
        f"The log/trace exports cover only part of the window: outside the covered intervals a missing record is an export "
        f"artefact, NOT evidence of absence.")
    out.append("bins: " + " ".join(labels))
    out.extend(blocks)
    dropped = len([1 for _, nm in names if nm in set(m["entity_name"])]) - shown
    if dropped > 0:
        out.append(f"({dropped} further neighbourhood service(s) not shown: length budget, structural order)")
    out.append("alert stream, evaluations per bin: " + " ".join(str(int(x)) for x in acounts))
    for rs in reasons[:MAX_REASONS]:
        if rs in first_c:
            out.append(f"events on neighbourhood pods, {rs}, first seen per bin: " + " ".join(str(int(x)) for x in first_c[rs]))
        if rs in last_c:
            out.append(f"events on neighbourhood pods, {rs}, last seen per bin: " + " ".join(str(int(x)) for x in last_c[rs]))
    if len(reasons) > MAX_REASONS:
        out.append(f"({len(reasons) - MAX_REASONS} further event reason(s) not shown)")
    if n_ev_other:
        out.append(f"({n_ev_other} other event group(s) with first or last seen in the window belong to objects outside the neighbourhood)")

    def fmt(iv):
        return "none" if not iv else ", ".join(f"{off_label(x, w0s)}..{off_label(y, w0s)}" for x, y in iv)
    out.append(f"export coverage inside the window: logs {fmt(lg_iv)} ({100 * cov_len(lg_iv):.0f}% of window); "
               f"traces {fmt(tr_iv)} ({100 * cov_len(tr_iv):.0f}% of window)")
    text = "\n".join(out)
    stats = {"strip_services": shown, "strip_bins": n_bins, "strip_event_reasons": len(reasons),
             "strip_log_covered_frac": round(cov_len(lg_iv), 3), "strip_trace_covered_frac": round(cov_len(tr_iv), 3)}
    return {"text": text, "rows": shown, "stats": stats}
