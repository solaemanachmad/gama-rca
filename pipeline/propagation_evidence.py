"""Propagation evidence (evidence extraction ONLY -- no decision is made here).

Motivation (offline audit over all 103 RCA100 cases, GT used for analysis only):
an alert usually fires on the *victim* of a fault, not its origin. The root
cause is a downstream callee of the alert's service in ~47% of cases, the same
service in ~30%, a K8s node in ~15% and an upstream caller in ~4%. A
proximity prior ("closest to the alert", "most central") therefore cannot
localize the origin. This module hands the LLM the information it needs to
reason about *direction*:

  for every service in the alert service's call-graph neighbourhood
  (itself, downstream callees up to `max_hops`, direct upstream callers):
    relation to the alert service, span error rate in the alert window vs. the
    earlier baseline, p95 latency ratio, and which of its own callees are also
    erroring (so the LLM can tell "origin" from "relays an error it received").

No threshold is applied, nothing is filtered or ranked-out: every neighbourhood
service is reported and the agent decides. Features are computed from the
alert time and telemetry only (no ground truth, no training).
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

WINDOW_BEFORE_S = 300      # alert window: [alert-5min, alert+1min]
WINDOW_AFTER_S = 60


def _calls_graph(topology: nx.DiGraph) -> nx.DiGraph:
    g = nx.DiGraph()
    for u, v, d in topology.edges(data=True):
        rel = d.get("relation")
        rtype = getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)
        if rtype == "calls":
            g.add_edge(u, v)
    return g


def _node_type(topology: nx.DiGraph, eid: str) -> str:
    ent = topology.nodes[eid].get("entity") if eid in topology else None
    return getattr(ent, "entity_type", "") if ent else ""


def _node_name(topology: nx.DiGraph, eid: str) -> str:
    ent = topology.nodes[eid].get("entity") if eid in topology else None
    return (getattr(ent, "name", None) or eid) if ent else eid


def _p95(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def compute_propagation_evidence(case, alert_ts: Optional[dt.datetime],
                                 entry_entity_id: Optional[str],
                                 max_hops: int = 3) -> Dict:
    """Returns {"rows": [...], "text": str}. Empty rows if the alert has no
    entity, no timestamp, or the entity cannot be rolled up to a service."""
    from data.loader import find_service_ancestor

    topo = case.topology
    if not entry_entity_id or alert_ts is None or entry_entity_id not in topo:
        return {"rows": [], "text": ""}
    svc0 = find_service_ancestor(entry_entity_id, topo) or entry_entity_id
    calls = _calls_graph(topo)
    if svc0 not in calls:
        return {"rows": [], "text": ""}

    hop = {svc0: 0}
    for node, d in nx.single_source_shortest_path_length(calls, svc0, cutoff=max_hops).items():
        if d > 0 and _node_type(topo, node) == "apm.service":
            hop[node] = d
    upstream = {p for p in calls.predecessors(svc0) if _node_type(topo, p) == "apm.service"}

    at = alert_ts.replace(tzinfo=None) if alert_ts.tzinfo else alert_ts
    w0 = at - dt.timedelta(seconds=WINDOW_BEFORE_S)
    w1 = at + dt.timedelta(seconds=WINDOW_AFTER_S)

    cand = set(hop) | upstream
    win = defaultdict(lambda: [0, 0, []])    # spans, errors, durations
    base = defaultdict(lambda: [0, 0, []])
    # Trace observations resolve to k8s.service / apm.instance entity ids, not
    # the apm.service node, so attribute spans by the span's own serviceName.
    name_to_eid = {_node_name(topo, e): e for e in cand}
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        eid = name_to_eid.get(p.get("serviceName"))
        if eid is None or o.timestamp is None:
            continue
        ts = o.timestamp.replace(tzinfo=None) if o.timestamp.tzinfo else o.timestamp
        if ts > w1:
            continue
        bucket = win if ts >= w0 else base
        rec = bucket[eid]
        rec[0] += 1
        if p.get("statusCode") == "2":
            rec[1] += 1
        try:
            rec[2].append(float(p.get("duration")) / 1e6)
        except (TypeError, ValueError):
            pass

    def rate(rec):
        return rec[1] / rec[0] if rec[0] else None

    rows = []
    for eid in cand:
        w, b = win[eid], base[eid]
        wr, br = rate(w), rate(b)
        wp, bp = _p95(w[2]), _p95(b[2])
        rows.append({
            "entity_id": eid, "name": _node_name(topo, eid),
            "relation": ("alert service" if eid == svc0 else
                         f"downstream callee, {hop[eid]} hop(s)" if eid in hop else "direct upstream caller"),
            "hop": hop.get(eid, -1),
            "spans_window": w[0], "err_window": wr, "err_base": br,
            "p95_window_ms": wp, "p95_base_ms": bp,
            "lat_ratio": (wp / bp) if (wp and bp) else None,
        })
    err = {r["entity_id"]: (r["err_window"] or 0.0) - (r["err_base"] or 0.0) for r in rows}
    for r in rows:
        callees = [c for c in calls.successors(r["entity_id"]) if c in err]
        r["erroring_callees"] = sorted(_node_name(topo, c) for c in callees if err[c] > 0.02)
    rows.sort(key=lambda r: (r["hop"] if r["hop"] >= 0 else 99, -(err[r["entity_id"]])))

    def f(x, pct=False):
        return "n/a" if x is None else (f"{x*100:.1f}%" if pct else f"{x:.2f}")

    lines = ["Call-graph neighbourhood of the alert service (alert time window vs earlier baseline). "
             "The alert service may only be a VICTIM that relays an error from a dependency."]
    for r in rows:
        lines.append(
            f"- {r['name']} [{r['entity_id']}] ({r['relation']}): spans={r['spans_window']}, "
            f"error_rate window={f(r['err_window'], True)} baseline={f(r['err_base'], True)}, "
            f"p95_latency_ratio={f(r['lat_ratio'])}, erroring_callees={r['erroring_callees'] or 'none'}")
    return {"rows": rows, "text": "\n".join(lines)}
