"""Trace evidence per call edge and export burst (evidence extraction ONLY -- no decision, no significance filter).

The trace modality is the only one that records WHO CALLED WHOM and whether the call failed. This module rebuilds the
observed service->service calls from parent/child spans and reports, per edge, how the window burst(s) differ from the
earlier bursts. Because the export is made of bursts (see pipeline/export_bursts.py), counts and rates are artefacts; only
FRACTIONS inside a burst are shown:
  * edge share = the edge's spans / all spans of the burst,
  * edge error fraction (callee-side status error), median and p95 duration,
  * per service: error spans / its spans, and how many of its error spans have an erroring child span (the error may be
    inherited from a callee) versus none (a leaf error); both are facts counted from parent/child links, no ranking.
Orphan spans (parent id present but not found in the export) cannot be attached to an edge; their share is reported.
Edges come from the spans themselves, so edges that exist only in the traces are included; nothing is taken from the
topology except the choice of the neighbourhood (by NAME) that gets edge detail. Every other edge appears in one compact
line, so no edge is hidden. Uses traces + topology + task window only (no ground truth, no training)."""
from __future__ import annotations

import datetime as dt
import os
import statistics
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config
from pipeline.export_bursts import (BURST_GAP_S, burst_ids, burst_spans, classify_bursts, drop_fragments,
                                    neighbourhood, off_label, ms, pct, sec)
from pipeline.propagation_evidence import _node_name, window_bounds

MAX_EDGE_DETAIL = 14
MAX_SVC_DETAIL = 8
_MISSING = {"", "None", "none", "nan", "NaN", "null"}


def compute_trace_edges(case, alert_ts: Optional[dt.datetime], entry_entity_id: Optional[str] = None,
                        max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0, "stats": {}}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _med_rng(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return None, None, None
    return statistics.median(vals), min(vals), max(vals)


def _compute(case, alert_ts, entry_entity_id, max_hops):
    if alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = window_bounds(case, at)
    w0s, w1s = sec(w0), sec(w1)

    fp = os.path.join(case.case_dir, config.FILE_TRACES)
    d = pd.read_parquet(fp, columns=["spanId", "parentSpanId", "serviceName", "startTime", "duration", "statusCode"])
    d["t"] = pd.to_numeric(d["startTime"], errors="coerce") / 1e9
    d["dur"] = pd.to_numeric(d["duration"], errors="coerce") / 1e6
    d = d.dropna(subset=["t"]).sort_values("t", kind="mergesort").reset_index(drop=True)
    if d.empty:
        return None
    ids = burst_ids(d["t"].to_numpy())
    span, n_frag = drop_fragments(burst_spans(d["t"].to_numpy(), ids), ids)
    win, base, note = classify_bursts(span, w0s, w1s)
    if not win:
        return None
    d["burst"] = ids
    d = d[d["burst"].isin(set(win) | set(base))].copy()
    d["err"] = d["statusCode"].astype(str) == "2"
    svc_of = d.drop_duplicates("spanId").set_index("spanId")["serviceName"]
    d["par_svc"] = d["parentSpanId"].map(svc_of)
    pid = d["parentSpanId"].astype(str)
    has_parent_id = ~pid.isin(_MISSING)
    d["orphan"] = has_parent_id & d["par_svc"].isna()
    err_parents = set(d.loc[d["err"], "parentSpanId"].astype(str))
    d["err_child"] = d["err"] & d["spanId"].astype(str).isin(err_parents)
    cross = d["par_svc"].notna() & (d["par_svc"] != d["serviceName"])
    d["edge"] = np.where(cross, d["par_svc"].astype(str) + ">" + d["serviceName"].astype(str), "")

    in_win = d["burst"].isin(set(win))
    tot_b = d.groupby("burst").size()
    wtot = int(tot_b[[k for k in win if k in tot_b.index]].sum())
    nb = len(base)

    # ---- per edge ----
    e = d[d["edge"] != ""]
    ew = e[e["burst"].isin(set(win))].groupby("edge")
    w_stat = {}
    for key, g in ew:
        w_stat[key] = {"share": len(g) / wtot if wtot else None, "err": float(g["err"].mean()),
                       "p50": float(g["dur"].median()), "p95": float(g["dur"].quantile(0.95))}
    b_stat: Dict[str, Dict[str, list]] = {}
    for (key, k), g in e[e["burst"].isin(set(base))].groupby(["edge", "burst"]):
        s = b_stat.setdefault(key, {"share": [], "err": [], "p50": [], "p95": []})
        s["share"].append(len(g) / tot_b[k])
        s["err"].append(float(g["err"].mean()))
        s["p50"].append(float(g["dur"].median()))
        s["p95"].append(float(g["dur"].quantile(0.95)))
    edges = sorted(set(w_stat) | set(b_stat))

    # ---- per service ----
    sv_w = d[in_win].groupby("serviceName")
    sv_stat = {}
    for key, g in sv_w:
        ne = int(g["err"].sum())
        sv_stat[key] = {"err": float(g["err"].mean()), "n_err": ne,
                        "inh": (float(g["err_child"].sum() / ne) if ne else None)}
    sv_b: Dict[str, Dict[str, list]] = {}
    for (key, k), g in d[d["burst"].isin(set(base))].groupby(["serviceName", "burst"]):
        s = sv_b.setdefault(key, {"err": [], "inh": []})
        s["err"].append(float(g["err"].mean()))
        ne = int(g["err"].sum())
        s["inh"].append(float(g["err_child"].sum() / ne) if ne else None)
    services = sorted(set(sv_stat) | set(sv_b))
    orph_w = float(d.loc[in_win, "orphan"].mean()) if in_win.any() else None

    # ---- neighbourhood by name ----
    cand, svc0 = neighbourhood(case, entry_entity_id, max_hops)
    names = {_node_name(case.topology, x) for x in cand} if cand else set()
    alert_name = _node_name(case.topology, svc0) if svc0 else None

    def rng(m, lo, hi, f):
        return f" {f(m)}" if m is not None and lo == hi else (f" {f(m)} [{f(lo)}-{f(hi)}]" if m is not None else " n/a")

    def edge_line(k):
        w = w_stat.get(k)
        b = b_stat.get(k)
        bs = (lambda key, f: "base" + rng(*_med_rng(b[key]), f)) if b else (lambda key, f: "base none")
        if w is None:
            return f"- {k}: no calls in the window burst(s); earlier bursts: share {pct(_med_rng(b['share'])[0])}, error {pct(_med_rng(b['err'])[0])}"
        return (f"- {k}: {pct(w['share'])} of spans ({bs('share', pct)}); error {pct(w['err'])} ({bs('err', pct)}); "
                f"p50 {ms(w['p50'])} ({bs('p50', ms)}), p95 {ms(w['p95'])} ({bs('p95', ms)})")

    def touches(k):
        a, c = k.split(">", 1)
        return alert_name in (a, c)

    in_neigh = [k for k in edges if (k.split(">", 1)[0] in names and k.split(">", 1)[1] in names)] if names else []
    in_neigh.sort(key=lambda k: (not touches(k), k))
    detail_edges = in_neigh[:MAX_EDGE_DETAIL]
    rest_edges = [k for k in edges if k not in set(detail_edges)]

    def svc_line(s):
        w = sv_stat.get(s)
        b = sv_b.get(s)
        berr = _med_rng(b["err"]) if b else (None, None, None)
        binh = _med_rng(b["inh"]) if b else (None, None, None)
        if w is None:
            return f"- {s}: no spans in the window burst(s)"
        inh = "no error spans" if w["inh"] is None else (
            f"{pct(w['inh'])} of its error spans have an erroring child span, {pct(1 - w['inh'])} have none"
            f" (earlier bursts: {pct(binh[0])} with)")
        return f"- {s}: error spans {pct(w['err'])} (base {pct(berr[0])}); {inh}"

    detail_svcs = [s for s in services if s in names][:MAX_SVC_DETAIL] if names else []

    hdr = (f"Trace evidence per call edge. Spans are exported as bursts (up to ~100,000 spans, about 1 minute every ~5 "
           f"minutes), so span counts and rates carry no information; fractions inside a burst do. T = evidence-window "
           f"start. Window burst(s): {len(win)} "
           f"({', '.join(f'{off_label(span[k][0], w0s)}..{off_label(span[k][1], w0s)}' for k in win)}); earlier bursts: "
           f"{nb}{f' ({n_frag} tiny export fragments ignored)' if n_frag else ''}.{note} Edge = parent span in one service, child span in another (from the spans themselves); share = "
           f"edge spans / burst spans; error and durations are of the callee-side child span; base = median [range] over "
           f"earlier bursts. Orphan spans (parent not in the export): {pct(orph_w)} of window spans, not attached to any edge. "
           f"Nothing is filtered by significance.")
    out = [hdr]
    if detail_edges:
        out.append("Edges among the alert service's call-graph neighbourhood:")
        out.extend(edge_line(k) for k in detail_edges)
    if detail_svcs:
        out.append("Errors per service of the neighbourhood (inherited = has an erroring child span):")
        out.extend(svc_line(s) for s in detail_svcs)
    if rest_edges:
        def comp(k):
            w, b = w_stat.get(k), b_stat.get(k)
            return f"{k} {pct(w['err']) if w else 'none'}/{pct(_med_rng(b['err'])[0]) if b else 'none'}"
        out.append("Other edges (error fraction window/earlier bursts): " + "; ".join(comp(k) for k in rest_edges))
    text = "\n".join(out)
    return {"text": text, "rows": len(detail_edges) + len(rest_edges),
            "stats": {"trace_edge_bursts": len(span), "trace_edge_window_bursts": len(win),
                      "trace_edges_total": len(edges), "trace_edges_detail": len(detail_edges),
                      "trace_edge_orphan_frac": None if orph_w is None else round(orph_w, 4)}}
