"""Infrastructure / APM metric evidence operators (evidence extraction ONLY).

Motivation: RCA100's own reasoning-process ground truth is expressed as
observability checkpoints <signal, comparator, numeric value> (99.5% numeric,
~72% on request count / latency / error count), and pod/node faults are only
visible in k8s metrics and events. Embedding retrieval over ~1M raw
observations (index capped at 10k) cannot surface these numbers, so the
operators below hand the LLM compact, *numeric* tables:

  * APM service rows (neighbourhood of the alert service): request count,
    error count, latency, cpu/mem/slow -- baseline median vs. window max/median.
  * APM operation rows for the alert service (request/error/latency).
  * JVM rows (gc / memory / thread pool) for neighbourhood services.
  * k8s.node rows for ALL nodes (cpu/mem/disk usage, ready status, pods) plus
    the services hosted on each node (node->pod edges).
  * k8s.deployment rows for neighbourhood services (replicas, cpu vs limits).
  * k8s events grouped by (object, reason): counts, first/last, window overlap.

No threshold is applied and nothing is removed: rows are only CAPPED for prompt
size and shown in a fixed (name) order, never ranked by a decision rule. The
LLM does all judging. Uses telemetry + alert time only (no GT, no training).
"""
from __future__ import annotations

import datetime as dt
import json
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

import config
from pipeline.propagation_evidence import (WINDOW_AFTER_S, WINDOW_BEFORE_S,
                                           _calls_graph, _node_name, _node_type)

# Display budgets (characters) per section. Prompt-size control only: the
# Coordinator prompt must fit the 7B model's GPU memory (a 16k-token prompt hit
# CUDA OOM on a 15 GB T4). Entities are listed in name order, never ranked.
CHAR_BUDGET = {"svc": 2600, "op": 1300, "jvm": 1200, "node": 2800, "dep": 1200, "event": 2200}


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _fmt(x):
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.4g}"


def _stat(series, w0, w1):
    base = [v for t, v in series if t < w0]
    win = [v for t, v in series if w0 <= t <= w1]
    if not win and not base:
        return None
    return {
        "base_med": statistics.median(base) if base else None,
        "base_max": max(base) if base else None,
        "base_vals": base,
        "win_med": statistics.median(win) if win else None,
        "win_max": max(win) if win else None,
        "win_min": min(win) if win else None,
        "n_win": len(win),
    }


def change_factor(base, peak):
    """max-in-window / baseline-median as a plain number (evidence formatting only: no threshold,
    no ranking). Returns 'x213', 'x1.0', 'new' (baseline 0, window >0) or '' if undefined."""
    if base is None or peak is None:
        return ""
    if base == 0:
        return "new" if peak != 0 else "x1"
    r = peak / base
    return f"x{r:.3g}" if r < 1000 else f"x{r:.0f}"


def change_stats(base_vals, peak):
    """Robust companions to the ratio (evidence formatting only: no threshold, no ranking).
    z    = (peak - median) / (1.4826 * MAD) over the baseline samples ('inf' if the baseline is
           perfectly flat and the peak differs, '0' if equal);
    rank = % of baseline samples strictly below the window max (bounded 0-100, never blows up);
    n    = number of baseline samples (how much to trust the baseline).
    Returns (z_txt, rank_txt, n) or None when undefined."""
    if not base_vals or peak is None:
        return None
    med = statistics.median(base_vals)
    mad = statistics.median([abs(v - med) for v in base_vals])
    sig = 1.4826 * mad
    if sig == 0:
        z = "0" if peak == med else ("+inf" if peak > med else "-inf")
    else:
        z = f"{(peak - med) / sig:+.3g}"
    rank = 100.0 * sum(1 for v in base_vals if v < peak) / len(base_vals)
    return z, f"{rank:.0f}%", len(base_vals)


def change_text(base_vals, base_med, peak, compact=False):
    """' change=x670 z=+24 rank=100% n=40' (or compact '(x670,z+24,r100%,n40)'); '' when off/undefined."""
    if not getattr(config, "USE_CHANGE_FACTOR", False):
        return ""
    cf = change_factor(base_med, peak)
    st = change_stats(base_vals, peak) if getattr(config, "USE_CHANGE_STATS", False) else None
    if compact:
        items = [cf] if cf else []
        if st:
            items += [f"z{st[0]}", f"r{st[1]}", f"n{st[2]}"]
        return f" ({','.join(items)})" if items else ""
    items = [f"change={cf}"] if cf else []
    if st:
        items += [f"z={st[0]}", f"rank={st[1]}", f"n={st[2]}"]
    return (" " + " ".join(items)) if items else ""


def CHANGE_LEGEND():
    """One-line explanation of the change columns (only when they are shown)."""
    if not getattr(config, "USE_CHANGE_FACTOR", False):
        return ""
    txt = (" change = window max / baseline median (a ratio; 'new' = baseline was 0)")
    if getattr(config, "USE_CHANGE_STATS", False):
        txt += ("; z = (window max - baseline median) / robust baseline spread (MAD), i.e. how many "
                "normal-variation units away; rank = % of baseline samples below the window max; "
                "n = number of baseline samples. A large ratio with a small z means a tiny, noisy "
                "baseline; a large z with ratio ~x1 means a very stable metric moved slightly")
    return txt + "."


def _row_text(label, metric, s):
    txt = f"{label} {metric}: base={_fmt(s['base_med'])} med={_fmt(s['win_med'])} max={_fmt(s['win_max'])}"
    txt += change_text(s.get("base_vals"), s["base_med"], s["win_max"])
    return txt


def compute_infra_evidence(case, alert_ts: Optional[dt.datetime],
                           entry_entity_id: Optional[str], max_hops: int = 2) -> Dict:
    """Returns {"rows": [...], "text": str, "observations": [Observation]}."""
    empty = {"rows": [], "text": "", "observations": []}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # defensive: evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _compute(case, alert_ts, entry_entity_id, max_hops):
    from data.loader import find_service_ancestor
    from schema import Observation

    topo = case.topology
    if alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = at - dt.timedelta(seconds=WINDOW_BEFORE_S), at + dt.timedelta(seconds=WINDOW_AFTER_S)

    # neighbourhood service names (alert service, callees <= max_hops, callers)
    svc_names = set()
    node_names_by_svc = defaultdict(set)
    if entry_entity_id and entry_entity_id in topo:
        svc0 = find_service_ancestor(entry_entity_id, topo) or entry_entity_id
        calls = _calls_graph(topo)
        import networkx as nx
        if svc0 in calls:
            for n in nx.single_source_shortest_path_length(calls, svc0, cutoff=max_hops):
                if _node_type(topo, n) == "apm.service":
                    svc_names.add(_node_name(topo, n))
            for p in calls.predecessors(svc0):
                if _node_type(topo, p) == "apm.service":
                    svc_names.add(_node_name(topo, p))
        svc_names.add(_node_name(topo, svc0))
        alert_svc = _node_name(topo, svc0)
    else:
        alert_svc = None

    # service -> hosting node names via node contains pod / pod hosts instance
    pod_node = {}
    for u, v, d in topo.edges(data=True):
        rel = d.get("relation")
        rt = getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)
        if rt == "contains" and _node_type(topo, u) == "k8s.node":
            pod_node[v] = _node_name(topo, u)
    node_services = defaultdict(set)
    for pod, node in pod_node.items():
        pn = _node_name(topo, pod)
        for s in svc_names:
            if s and pn.startswith(s):
                node_services[node].add(s)

    # group metrics: (entity_set, entity_name, metric) -> [(ts, value)]
    groups = defaultdict(list)
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts > w1 or p.get("value") is None:
            continue
        try:
            groups[(p.get("entity_set"), p.get("entity_name"), p.get("metric"))].append((ts, float(p["value"])))
        except (TypeError, ValueError):
            continue

    sections = {k: [] for k in ("svc", "op", "jvm", "node", "dep")}
    obs_out: List = []

    def add(kind, ename, metric, series, label):
        s = _stat(series, w0, w1)
        if s is None:
            return
        txt = _row_text(label, metric, s)
        sections[kind].append((ename, metric, txt))
        if s["win_max"] is not None:
            obs_out.append(Observation(entity_id=None, timestamp=at, modality="infra_synth",
                                       text=f"{ename} {metric}={_fmt(s['win_max'])} (window max); "
                                            f"baseline median={_fmt(s['base_med'])}",
                                       payload={}, source_file="infra_evidence"))

    for (eset, ename, metric), series in groups.items():
        if not ename:
            continue
        prefix = ename.split("::")[0]
        if eset == "apm.service.legacy" and ename in svc_names:
            add("svc", ename, metric, series, f"service {ename}")
        elif eset == "apm.operation" and alert_svc and prefix == alert_svc:
            add("op", ename, metric, series, f"operation {ename}")
        elif eset in ("apm.metric.jvm", "apm.metric.thread") and ename in svc_names:
            add("jvm", ename, metric, series, f"jvm {ename}")
        elif eset == "k8s.node":
            hosted = ", ".join(sorted(node_services.get(ename, []))) or "none of the listed services"
            add("node", ename, metric, series, f"node {ename} (hosts: {hosted})")
        elif eset == "k8s.deployment" and any(ename.startswith(s) or s in ename for s in svc_names):
            add("dep", ename, metric, series, f"deployment {ename}")

    # k8s events grouped by (object, reason)
    ev = defaultdict(lambda: {"count": 0, "first": None, "last": None, "msg": "", "type": ""})
    for o in case.observations.get("events", []):
        p = o.payload or {}
        raw = p.get("eventId")
        try:
            e = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            e = {}
        io = e.get("involvedObject") or {}
        obj = f"{io.get('kind', '?')}/{io.get('name') or p.get('pod_name') or p.get('hostname') or '?'}"
        g = ev[(obj, e.get("reason", ""))]
        g["count"] += int(e.get("count") or 1)
        for key, fld in (("first", "firstTimestamp"), ("last", "lastTimestamp")):
            t = e.get(fld)
            if t and (g[key] is None or (t < g[key] if key == "first" else t > g[key])):
                g[key] = t
        g["msg"] = g["msg"] or (e.get("message") or "")[:120]
        g["type"] = g["type"] or e.get("type", "")
    ev_lines = []
    for (obj, reason), g in sorted(ev.items(), key=lambda kv: (kv[1]["type"] != "Warning", kv[0])):
        ev_lines.append(f"event {obj} reason={reason} type={g['type']} count={g['count']} "
                        f"first={g['first']} last={g['last']} msg={g['msg'][:60]}")
        obs_out.append(Observation(entity_id=None, timestamp=at, modality="infra_synth",
                                   text=f"{obj} {reason} count={g['count']}", payload={},
                                   source_file="infra_evidence"))

    titles = {"svc": "APM service metrics (request count=workload, error count=error; base=baseline median, med/max=alert window)",
              "op": "APM operation metrics of the alert service",
              "jvm": "JVM / thread-pool metrics",
              "node": "K8s node metrics (all nodes; usage rates, ready status, pods)",
              "dep": "K8s deployment metrics (replicas, cpu vs limits)"}
    lines = [f"Numeric infrastructure/APM evidence. Window = [alert-{WINDOW_BEFORE_S//60}min, "
             f"alert+{WINDOW_AFTER_S}s]; baseline = samples before the window. Nothing is filtered by "
             f"significance: judge which signals are abnormal yourself." + CHANGE_LEGEND()]
    rows = []

    def emit(header, body_lines, budget):
        out, used, dropped = [], 0, 0
        for ln in body_lines:
            if used + len(ln) + 1 > budget:
                dropped += 1
                continue
            out.append(ln)
            used += len(ln) + 1
        if out:
            lines.append("\n" + header + ":")
            lines.extend(out)
            if dropped:
                lines.append(f"(+{dropped} more lines omitted for length)")
        return out

    for k in ("svc", "op", "jvm", "node", "dep"):
        by_entity = defaultdict(list)
        for ename, metric, txt in sorted(sections[k]):
            by_entity[ename].append((metric, txt))
        body = []
        for ename in sorted(by_entity):
            metrics = [m for m, _ in by_entity[ename]]
            parts = [t.split(" " + m + ": ", 1)[1] for m, t in by_entity[ename]]
            label = by_entity[ename][0][1].split(" " + metrics[0] + ":")[0]
            body.append("- " + label + " | " + "; ".join(f"{m} {p}" for m, p in zip(metrics, parts)))
        rows.extend(emit(titles[k], body, CHAR_BUDGET[k]))
    if ev_lines:
        rows.extend(emit("K8s events grouped by object and reason (last may be later than the incident)",
                         ["- " + t for t in ev_lines], CHAR_BUDGET["event"]))
    return {"rows": rows, "text": "\n".join(lines), "observations": obs_out}
