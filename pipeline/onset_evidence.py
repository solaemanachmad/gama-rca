"""Onset / temporal-order evidence operator (evidence extraction ONLY).

Motivation: the other operators summarise the alert window with one number per signal (baseline -> window max).
That loses WHEN each signal started deviating, which is the main discriminator between a cause and its
propagation: the origin deviates first, downstream/upstream entities follow. This operator shows, for every
signal of the call-graph neighbourhood of the alert service (services: error, latency; hosting nodes: cpu, memory),
the robust z-score per minute over [alert-5min .. alert+1min] against the pre-window baseline.

No threshold, no filtering, no ranking, no "first mover" verdict: rows are listed in name order and the LLM reads
the series itself. Prompt size is bounded by a char budget (rows beyond it are counted, not shown).
z = (value - baseline median) / scale, scale = 1.4826*MAD; when the baseline is (near) flat, scale falls back to
5% of |median| (or 1e-9); integers, capped at +-99. Uses telemetry + topology + alert time only.
"""
from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from typing import Dict, Optional

from pipeline.propagation_evidence import _calls_graph
from pipeline.propagation_evidence import _node_name, _node_type

MINUTES = list(range(-5, 2))            # minute offsets relative to the alert
CHAR_BUDGET = 1800
SVC_METRICS = (("error", "err"), ("latency", "lat"))
NODE_METRICS = (("node_cpu_usage_rate", "cpu"), ("node_memory_usage_rate", "mem"))
LEGEND = ("Onset table: robust z-score per minute vs the pre-window baseline, minutes "
          f"{MINUTES[0]:+d}..{MINUTES[-1]:+d} relative to the alert (0 = alert minute; '.' = no sample). "
          "Compare WHEN each signal starts to deviate: an origin deviates before the entities it affects. "
          "Nothing is filtered.")


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _z(v, med, mad):
    scale = 1.4826 * mad
    if scale < 0.05 * abs(med):
        scale = 0.05 * abs(med)
    if scale <= 0:
        scale = 1e-9
    return max(-99, min(99, round((v - med) / scale)))


def _series_text(series, at):
    w0 = at - dt.timedelta(minutes=-MINUTES[0])
    base = [v for t, v in series if t < w0]
    if len(base) < 3:
        return None
    med = statistics.median(base)
    mad = statistics.median([abs(v - med) for v in base])
    buckets = defaultdict(list)
    for t, v in series:
        if t < w0:
            continue
        m = int((t - at).total_seconds() // 60)
        if MINUTES[0] <= m <= MINUTES[-1]:
            buckets[m].append(v)
    if not buckets:
        return None
    return " ".join(str(_z(max(buckets[m]), med, mad)) if m in buckets else "." for m in MINUTES)


def compute_onset_evidence(case, alert_ts: Optional[dt.datetime], entry_entity_id: Optional[str],
                           max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _compute(case, alert_ts, entry_entity_id, max_hops):
    from data.loader import find_service_ancestor
    import networkx as nx

    topo = case.topology
    if alert_ts is None or not entry_entity_id or entry_entity_id not in topo:
        return None
    at = _naive(alert_ts)
    svc0 = find_service_ancestor(entry_entity_id, topo) or entry_entity_id
    calls = _calls_graph(topo)
    svc_names = {_node_name(topo, svc0)}
    if svc0 in calls:
        for n in nx.single_source_shortest_path_length(calls, svc0, cutoff=max_hops):
            if _node_type(topo, n) == "apm.service":
                svc_names.add(_node_name(topo, n))
        for p in calls.predecessors(svc0):
            if _node_type(topo, p) == "apm.service":
                svc_names.add(_node_name(topo, p))
    node_names = set()
    for u, v, d in topo.edges(data=True):
        rel = d.get("relation")
        rt = getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)
        if rt == "contains" and _node_type(topo, u) == "k8s.node":
            pn = _node_name(topo, v)
            if any(pn.startswith(s) for s in svc_names if s):
                node_names.add(_node_name(topo, u))

    w1 = at + dt.timedelta(minutes=MINUTES[-1] + 1)
    groups = defaultdict(list)
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts >= w1 or p.get("value") is None:
            continue
        key = (p.get("entity_set"), p.get("entity_name"), p.get("metric"))
        if (key[0] == "apm.service.legacy" and key[1] in svc_names and key[2] in dict(SVC_METRICS)) or \
           (key[0] == "k8s.node" and key[1] in node_names and key[2] in dict(NODE_METRICS)):
            try:
                groups[key].append((ts, float(p["value"])))
            except (TypeError, ValueError):
                pass

    rows = []
    short = {**dict(SVC_METRICS), **dict(NODE_METRICS)}
    for (eset, name, metric), series in sorted(groups.items(), key=lambda kv: (kv[0][0] != "apm.service.legacy",
                                                                               kv[0][1], kv[0][2])):
        txt = _series_text(sorted(series), at)
        if txt is not None:
            rows.append(f"{'node ' if eset == 'k8s.node' else ''}{name} {short[metric]}: {txt}")
    if not rows:
        return None
    out, used, dropped = [], len(LEGEND), 0
    for r in rows:
        if used + len(r) + 1 > CHAR_BUDGET:
            dropped += 1
            continue
        out.append(r)
        used += len(r) + 1
    text = LEGEND + "\n" + "\n".join(out) + (f"\n(+{dropped} more rows omitted for length)" if dropped else "")
    return {"text": text, "rows": len(out)}
