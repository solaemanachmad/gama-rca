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
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

import config
from data.loader import event_time_bounds, _parse_ts_iso
from pipeline.propagation_evidence import (WINDOW_AFTER_S, WINDOW_BEFORE_S, window_bounds,
                                           _calls_graph, _node_name, _node_type)

# Display budgets (characters) per section. Prompt-size control only: the
# Coordinator prompt must fit the 7B model's GPU memory (a 16k-token prompt hit
# CUDA OOM on a 15 GB T4). Entities are listed in name order, never ranked.
# Arm E helpers (flag USE_EVENT_DIGEST lives in config): time-labelled event digest, every group represented.
EVENT_DIGEST_NAMES = 6      # workload names listed per line; the rest is counted ("+n more")
EVENT_MSG_CHARS = 90
EVENT_RECURRING_S = 86400   # first occurrence older than this before the window start -> 'recurring since', no counter
_POD_SUFFIX = re.compile(r"-[a-z0-9]{8,10}-[a-z0-9]{5}$|-[a-z0-9]{5}$")


def _workload_name(kind: str, name: str) -> str:
    return _POD_SUFFIX.sub("", name) if kind == "Pod" else name


def _off(t, w0) -> str:
    """Offset from the window start: m:ss within an hour, then hours, then days (event clocks reach back weeks)."""
    if t is None:
        return "n/a"
    sec = int(round((_naive(t) - w0).total_seconds()))
    sign = "-" if sec < 0 else "+"
    a = abs(sec)
    if a >= 86400:
        return f"T{sign}{a / 86400:.1f}d"
    if a >= 3600:
        return f"T{sign}{a / 3600:.1f}h"
    return f"T{sign}{a // 60}:{a % 60:02d}"


def event_digest_lines(ev: dict, w0) -> List[str]:
    """ev: {(obj 'Kind/name', reason): {count, first, last, msg, type}} -> digest lines. One line per
    (type, kind, reason); workload names (pod hash suffix stripped) are NAMES, nothing is chosen by significance."""
    agg: dict = {}
    for (obj, reason), g in ev.items():
        kind, _, name = obj.partition("/")
        a = agg.setdefault((g["type"] or "?", kind, reason),
                           {"count": 0, "first": None, "last": None, "wl": {}, "msg": "", "groups": 0})
        a["count"] += g["count"]
        a["groups"] += 1
        wl = _workload_name(kind, name)
        a["wl"][wl] = a["wl"].get(wl, 0) + 1
        for key, t in (("first", g["first"]), ("last", g["last"])):
            t = _naive(_parse_ts_iso(t))   # event times are ISO strings
            if t is None:
                continue
            if a[key] is None or (t < a[key] if key == "first" else t > a[key]):
                a[key] = t
        a["msg"] = a["msg"] or g["msg"]
    # most recent activity first (Warning before other types); groups without any timestamp last; name as tie-break
    order = sorted(agg.items(), key=lambda kv: (kv[0][0] != "Warning", kv[1]["last"] is None,
                                                -(kv[1]["last"] - w0).total_seconds() if kv[1]["last"] else 0, kv[0]))
    n_warn = sum(a["groups"] for k, a in agg.items() if k[0] == "Warning")
    lines = [f"{len(ev)} event groups ({n_warn} Warning, {len(ev) - n_warn} other) condensed to {len(order)} lines; "
             f"T = window start (negative = before); Warning first, then most recent activity first. xN = number of "
             f"occurrences reported by Kubernetes, shown only when the event first occurred within {EVENT_RECURRING_S // 3600} h "
             f"before the window start; older ones are marked 'recurring since' (their counter is cumulative over days)"]
    for (typ, kind, reason), a in order:
        names = sorted(a["wl"])
        shown = ", ".join(n + (f"x{a['wl'][n]}" if a["wl"][n] > 1 else "") for n in names[:EVENT_DIGEST_NAMES])
        more = f" +{len(names) - EVENT_DIGEST_NAMES} more" if len(names) > EVENT_DIGEST_NAMES else ""
        msg = f" | e.g. {a['msg'][:EVENT_MSG_CHARS]}" if a["msg"] else ""
        old = a["first"] is not None and (w0 - a["first"]).total_seconds() > EVENT_RECURRING_S
        when = (f"recurring since {_off(a['first'], w0)}, last {_off(a['last'], w0)}" if old
                else f"x{a['count']}, first {_off(a['first'], w0)}, last {_off(a['last'], w0)}")
        lines.append(f"{typ} {kind} {reason} on {len(names)} name(s) [{shown}{more}]; {when}{msg}")
    return lines


CHAR_BUDGET = {"svc": 2600, "op": 1300, "jvm": 1200, "node": 2800, "dep": 1200, "event": 2200}
# Arm "direction evidence" (USE_DIRECTION_EVIDENCE=1): every row also carries the low side, so budgets grow with the rows.
CHAR_BUDGET_DIRECTION = {"svc": 3800, "op": 1900, "jvm": 1800, "node": 2800, "dep": 1200, "event": 2200}
# Metric order inside a service/operation line under that arm: traffic first, then errors, then the rest.
_METRIC_PRIORITY = ("workload", "request_count", "error", "error_count", "error_rate", "latency",
                    "avg_request_latency_seconds", "slow", "slow_count", "cpu", "mem")


def _metric_rank(metric):
    return _METRIC_PRIORITY.index(metric) if metric in _METRIC_PRIORITY else len(_METRIC_PRIORITY)


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
    if getattr(config, "USE_DIRECTION_EVIDENCE", False):
        txt += ("; 'low' repeats ratio/z/rank at the window MINIMUM (a drop shows there; the max-based columns "
                "cannot show one) and 'median change' is the window median over the baseline median (a sustained shift)")
    return txt + "."


def low_side_text(s):
    """' | low: min=.. change=x0.01 z=-9.5 rank=0%; median change=x0.01' -- the same ratio/z/rank as the max side, taken
    at the window MINIMUM, plus the ratio of the window MEDIAN (a sustained shift). Evidence formatting only: no
    threshold, no ranking, nothing filtered. The max-based columns cannot show a drop."""
    low = s.get("win_min")
    if low is None:
        return ""
    cf = change_factor(s["base_med"], low)
    st = change_stats(s.get("base_vals"), low) if getattr(config, "USE_CHANGE_STATS", False) else None
    items = [f"min={_fmt(low)}"]
    if cf:
        items.append(f"change={cf}")
    if st:
        items += [f"z={st[0]}", f"rank={st[1]}"]
    med_cf = change_factor(s["base_med"], s.get("win_med"))
    txt = " | low: " + " ".join(items)
    return txt + (f"; median change={med_cf}" if med_cf else "")


def _row_text(label, metric, s):
    txt = f"{label} {metric}: base={_fmt(s['base_med'])} med={_fmt(s['win_med'])} max={_fmt(s['win_max'])}"
    txt += change_text(s.get("base_vals"), s["base_med"], s["win_max"])
    if getattr(config, "USE_DIRECTION_EVIDENCE", False) and getattr(config, "USE_CHANGE_FACTOR", False):
        txt += low_side_text(s)
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
    w0, w1 = window_bounds(case, at)

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
        _f, _l = event_time_bounds(e)
        for key, t in (("first", _f), ("last", _l)):
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
    _win_txt = (f"[alert_window.start, alert_window.end] ({int((w1 - w0).total_seconds())}s)"
                if getattr(config, "ALERT_T0_MODE", "legacy") == "window"
                else f"[alert-{WINDOW_BEFORE_S//60}min, alert+{WINDOW_AFTER_S}s]")
    lines = [f"Numeric infrastructure/APM evidence. Window = {_win_txt}; baseline = samples before the window. Nothing is filtered by "
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
        _rows = (sorted(sections[k], key=lambda r: (r[0], _metric_rank(r[1]), r[1]))
                 if getattr(config, "USE_DIRECTION_EVIDENCE", False) else sorted(sections[k]))
        for ename, metric, txt in _rows:
            by_entity[ename].append((metric, txt))
        body = []
        for ename in sorted(by_entity):
            metrics = [m for m, _ in by_entity[ename]]
            parts = [t.split(" " + m + ": ", 1)[1] for m, t in by_entity[ename]]
            label = by_entity[ename][0][1].split(" " + metrics[0] + ":")[0]
            body.append("- " + label + " | " + "; ".join(f"{m} {p}" for m, p in zip(metrics, parts)))
        rows.extend(emit(titles[k], body, (CHAR_BUDGET_DIRECTION if getattr(config, "USE_DIRECTION_EVIDENCE", False) else CHAR_BUDGET)[k]))
    if ev_lines and getattr(config, "USE_EVENT_DIGEST", False):
        _dl = event_digest_lines(ev, w0)
        lines.append("\nK8s events, condensed (no group omitted):")
        lines.extend("- " + t for t in _dl)
        rows.extend(_dl)
    elif ev_lines:
        rows.extend(emit("K8s events grouped by object and reason (last may be later than the incident)",
                         ["- " + t for t in ev_lines], CHAR_BUDGET["event"]))
    return {"rows": rows, "text": "\n".join(lines), "observations": obs_out}
