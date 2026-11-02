"""Reasoning chain (output of the diagnosis): the topology path from the decided root to the alerted operation, one step per
entity -- cause at the root, propagation through the services in between, impact at the alerted operation. M6 'chain': every step
cites metric=value of its own target (golden signals: request, error, latency for services / operations; CPU, memory, disk for
nodes; at the cause step first the readings behind the root's holding evidence). The chain affects the reasoning-process score only.
Moved unchanged from pipeline/belief_rca.py (development arms import it from here)."""
from __future__ import annotations

from collections import defaultdict

from pipeline.propagation_evidence import _calls_graph, _node_name, _node_type


def callee_map(topo):
    out = defaultdict(list)
    for u, v in _calls_graph(topo).edges():
        if _node_type(topo, u) in ("apm.service", "apm.svc") and _node_type(topo, v) in ("apm.service", "apm.svc"):
            a, b = _node_name(topo, u), _node_name(topo, v)
            if a != b and b not in out[a]:
                out[a].append(b)
    return dict(out)


def _ids(topo):
    m = {}
    for n in topo.nodes:
        if _node_type(topo, n) in ("apm.service", "apm.svc", "k8s.node"):
            m.setdefault(_node_name(topo, n), n)
    return m


def call_path(callees, hosts, alert, root, held=None):
    """Topology path root -> ... -> alerted service (services along callee edges, reversed). For a node root, the path starts at the
    node and continues from the hosted service closest to the alerted service. [] when unknown.
    v5 'rpchain' (held = entities with a holding observation): among the shortest paths, the one whose intermediate services show the
    most symptoms (the propagation the evidence supports), instead of an arbitrary one."""
    import networkx as nx
    if not alert or not root:
        return []
    g = nx.DiGraph()
    for a, bs in callees.items():
        for b in bs:
            g.add_edge(a, b)
    starts = [root] if root not in sum(hosts.values(), []) else [s for s, ns in hosts.items() if root in ns]
    best = None
    for st in starts:
        if st == alert:
            p = [alert]
        elif alert in g and st in g and nx.has_path(g, alert, st):
            if held is not None:
                cands = [list(reversed(q)) for q in nx.all_shortest_paths(g, alert, st)]
                p = max(cands, key=lambda q: sum(1 for x in q[1:-1] if x in held))
            else:
                p = list(reversed(nx.shortest_path(g, alert, st)))
        else:
            continue
        if best is None or len(p) < len(best):
            best = p
    if best is None:
        return [root] if root == alert else [root, alert]
    return ([root] + best) if best[0] != root else best


RED_FAMILIES = ("request", "error", "latency")          # golden signals of a service / operation step (rate, errors, duration)
NODE_FAMILIES = ("cpu", "mem", "disk")
RESOURCE_WORDS = ("cpu", "mem", "disk", "thread", "gc")
STEP_METRICS = 4
import re as _re
_EV_READING = _re.compile(r"([A-Za-z_][\w.]*): baseline median=[-+0-9.eE]+, window (max|min)=([-+0-9.eE]+)")


def chain_step_metrics(case, w0, w1, entities, op_name):
    """v5 'redchain': for every entity on the chain (root, path services or node, alerted operation), up to STEP_METRICS of its own
    metrics of the golden-signal families (services / operations: request, error, latency; nodes: cpu, memory, disk), each as
    metric=value with the window extreme in the direction of its change from the baseline median, the largest changes first."""
    import math
    from pipeline.fault_fingerprint import _naive
    want = {e for e in entities if e}
    if op_name:
        want.add(op_name)
    series = defaultdict(list)
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        name = str(p.get("entity_name") or "")
        if name not in want or p.get("value") is None:
            continue
        ts = _naive(o.timestamp)
        if ts is None or ts > w1:
            continue
        try:
            series[(name, str(p.get("metric") or ""), str(p.get("entity_set") or ""))].append((ts, float(p["value"])))
        except (TypeError, ValueError):
            continue
    out = defaultdict(list)
    for (name, m, eset), pts in series.items():
        fams = NODE_FAMILIES if eset == "k8s.node" else RED_FAMILIES
        ml = m.lower()
        if not any(f in ml for f in fams):
            continue
        if eset != "k8s.node" and any(r in ml for r in RESOURCE_WORDS):   # e.g. deployment_cpu_usage_vs_requests is a CPU ratio,
            continue                                                       # not a request count
        base = sorted(v for t, v in pts if t < w0)
        win = [v for t, v in pts if w0 <= t <= w1]
        if not win or not base:
            continue
        b = base[len(base) // 2]
        hi, lo = max(win), min(win)
        v = hi if abs(hi - b) >= abs(lo - b) else lo
        score = abs(math.log((abs(v) + 1e-9) / (abs(b) + 1e-9)))
        out[name].append((score, m, float(f"{v:.6g}")))
    return {e: [(m, v) for _, m, v in sorted(out[e], reverse=True)[:STEP_METRICS]] for e in
            [x for x in list(dict.fromkeys(entities)) + ([op_name] if op_name else []) if x in out]}


def rule_chain(res, path, op_name, step_metrics=None):
    """Chain without an LLM: cause at the root with its strongest own evidence, propagation through the path's services with their
    own symptom evidence, impact at the alerted operation."""
    ev = {}
    for e in res.get("evidence", []):
        if e["holds"] and e["entity"] not in ev and e["signal"] != "alert":     # the alert itself is not evidence of a step
            ev[e["entity"]] = f"{e['signal']} ({e['ev']})" if e.get("ev") else e["signal"]
    root = res["entity"]
    sm = dict(step_metrics or {})
    if step_metrics is not None:                         # v5 'redchain' (amendment 21): the cause step also cites the readings behind
        mech = []                                        # the root's holding evidence (the mechanism metrics: GC, threads, replicas,
        for e in res.get("evidence", []):                # node CPU / memory ...), as metric=window extreme
            if e["entity"] == root and e["holds"]:
                mm_ = _EV_READING.search(str(e.get("ev") or ""))
                if mm_ and mm_.group(1) not in {m for m, _ in mech}:
                    mech.append((mm_.group(1), float(f"{float(mm_.group(3)):.6g}")))
        sm[root] = (mech + [x for x in sm.get(root, []) if x[0] not in {m for m, _ in mech}])[:STEP_METRICS + 3]
    cite = lambda e: (" | " + "; ".join(f"{m}={v}" for m, v in sm[e])) if sm.get(e) else ""
    chain = [f"cause: {root} {res['type']} " + ev.get(root, "") + cite(root)]
    for s_ in (path or [])[1:]:
        if s_ != root:
            chain.append(f"propagation: {s_} " + ev.get(s_, "") + cite(s_))
    if op_name:
        chain.append(f"impact: {op_name}" + cite(op_name))
    return [c.strip() for c in chain]
