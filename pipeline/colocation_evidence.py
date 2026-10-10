"""Node co-location evidence operator (evidence extraction ONLY).

Motivation (R5D, dev-30): K8s / Cloud root causes are the cases where the
pipeline fails completely (EL 0.11-0.19, FI 0). An operator separates "service
fault" from "host fault" by looking at WHO SHARES A NODE: if several services on
the same node degrade at the same time while the same services on other nodes do
not, the cause is the host; if one service degrades on every node it runs on,
the cause is the service itself. The topology snapshot gives the placement
(service -deployed_as-> deployment -manages-> pod <-contains- node); services
are spread over several nodes (many-to-many), so this must be shown as a table.

Output: one line per node (all nodes, name order): node cpu/memory/disk/running
pods (baseline median -> window max) and the hosted services with their error
and latency (baseline median -> window max); then one line per service listing
the nodes it runs on. No threshold, no ranking, nothing filtered by significance;
lines are only capped for prompt size. All judging is left to the LLM.
Uses telemetry + topology + alert time only (no ground truth, no training).
"""
from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict
from typing import Dict, Optional

import config
from pipeline.infra_evidence import change_text, CHANGE_LEGEND
from pipeline.propagation_evidence import (WINDOW_AFTER_S, WINDOW_BEFORE_S,
                                           _node_name, _node_type)

NODE_METRICS = (("node_cpu_usage_rate", "cpu"), ("node_memory_usage_rate", "mem"),
                ("node_disk_usage_rate", "disk"), ("node_pod_running_count", "pods"))
SVC_METRICS = (("error", "err"), ("latency", "lat"))
CHAR_BUDGET = {"node": 4200, "placement": 1300}


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _bm(s):
    """'b->m' plus the change factor when USE_CHANGE_FACTOR is on."""
    return f"{_fmt(s[0])}->{_fmt(s[1])}" + change_text(s[2], s[0], s[1], compact=True)


def _fmt(x):
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.3g}"


def _bw(series, w0, w1):
    """(baseline median, window max) or None."""
    base = [v for t, v in series if t < w0]
    win = [v for t, v in series if w0 <= t <= w1]
    if not win:
        return None
    return (statistics.median(base) if base else None, max(win), base)


def compute_colocation_evidence(case, alert_ts: Optional[dt.datetime]) -> Dict:
    empty = {"text": "", "rows": 0, "stats": {}}
    try:
        return _compute(case, alert_ts) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _rel_type(topo, u, v):
    rel = topo.edges[u, v].get("relation")
    return getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)


def _compute(case, alert_ts):
    if alert_ts is None:
        return None
    topo = case.topology
    at = _naive(alert_ts)
    w0, w1 = at - dt.timedelta(seconds=WINDOW_BEFORE_S), at + dt.timedelta(seconds=WINDOW_AFTER_S)

    pod_node, dep_pods, svc_deps = {}, defaultdict(list), defaultdict(list)
    for u, v in topo.edges:
        tu, tv = _node_type(topo, u), _node_type(topo, v)
        if tu == "k8s.node" and tv == "k8s.pod":
            pod_node[v] = u
        elif tu == "k8s.deployment" and tv == "k8s.pod":
            dep_pods[u].append(v)
        elif tu == "apm.service" and tv == "k8s.deployment":
            svc_deps[u].append(v)
    node_svcs, svc_nodes = defaultdict(set), {}
    for s, deps in svc_deps.items():
        nodes = {pod_node[p] for d in deps for p in dep_pods[d] if p in pod_node}
        svc_nodes[_node_name(topo, s)] = sorted(_node_name(topo, n) for n in nodes)
        for n in nodes:
            node_svcs[_node_name(topo, n)].add(_node_name(topo, s))
    if not node_svcs:
        return None

    want_node = {m for m, _ in NODE_METRICS}
    want_svc = {m for m, _ in SVC_METRICS}
    groups = defaultdict(list)
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        es, en, m = p.get("entity_set"), p.get("entity_name"), p.get("metric")
        if not ((es == "k8s.node" and m in want_node) or (es == "apm.service.legacy" and m in want_svc)):
            continue
        ts = _naive(o.timestamp)
        if ts is None or ts > w1 or p.get("value") is None:
            continue
        try:
            groups[(es, en, m)].append((ts, float(p["value"])))
        except (TypeError, ValueError):
            continue

    def short(n):
        return n.split(".", 1)[1] if "." in n else n

    node_lines, svc_status = [], {}
    all_nodes = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) == "k8s.node"})
    for nn in all_nodes:
        parts = []
        for metric, lab in NODE_METRICS:
            s = _bw(groups.get(("k8s.node", nn, metric), []), w0, w1)
            if s:
                parts.append(f"{lab} {_bm(s)}")
        hosted = []
        for svc in sorted(node_svcs.get(nn, [])):
            sp = []
            for metric, lab in SVC_METRICS:
                s = _bw(groups.get(("apm.service.legacy", svc, metric), []), w0, w1)
                if s:
                    sp.append(f"{lab} {_bm(s)}")
            hosted.append(f"{svc}[{', '.join(sp)}]" if sp else svc)
        node_lines.append(f"node {short(nn)}: {'; '.join(parts) if parts else 'no node metrics'}"
                          f" | hosts: {', '.join(hosted) if hosted else 'no mapped service'}")

    placement = [f"{svc} runs on {len(ns)} node(s): {', '.join(short(n) for n in ns)}"
                 for svc, ns in sorted(svc_nodes.items()) if ns]

    def cap(lines, budget):
        out, used = [], 0
        for ln in lines:
            if used + len(ln) + 1 > budget:
                break
            out.append(ln)
            used += len(ln) + 1
        return out

    shown_nodes = cap(node_lines, CHAR_BUDGET["node"])
    shown_place = cap(placement, CHAR_BUDGET["placement"])
    text = ("Node co-location evidence. Services are spread over several nodes. Values are "
            "baseline median -> alert-window max (cpu/mem/disk = usage rate; pods = running pods; "
            "err = error count; lat = latency)." + CHANGE_LEGEND().replace(" change =", " Brackets after a value: change =", 1) + " Nothing is filtered or ranked: judge yourself "
            "whether a node's own cpu/mem/disk/pods change while its hosted services degrade "
            "(host-level cause) or services degrade without any change on their nodes "
            "(service-level cause). Note: service err/lat are service-wide aggregates, so the "
            "same service shows the same values on every node that hosts it; only the node "
            "cpu/mem/disk/pods values are specific to one node.\n"
            "Per node:\n" + "\n".join(shown_nodes) + "\nPlacement per service:\n" + "\n".join(shown_place))
    return {"text": text, "rows": len(shown_nodes),
            "stats": {"nodes_total": len(all_nodes), "nodes_shown": len(shown_nodes),
                      "services_mapped": len(svc_nodes), "chars": len(text)}}
