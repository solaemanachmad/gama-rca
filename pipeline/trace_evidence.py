"""Trace evidence (evidence extraction ONLY -- no decision is made here).

Why: metric/log tables do not separate fault types that differ in WHAT the failing calls say, e.g.
rateLimiting (HTTP 429, or 403 with throttling wording) vs httpError5xx vs nullPointerException vs
codeDefect. Error spans carry exactly that: the HTTP status code attribute and the status message.

For every service in the call-graph neighbourhood of the alert service (itself, downstream callees up to
`max_hops`, direct upstream callers -- the neighbourhood comes from the topology) report, for the alert
window versus the earlier baseline:
  * span count and error-span count,
  * HTTP status-code counts (from the span's `http.status_code` attribute, whatever the OTel statusCode says),
  * the most frequent error messages (ids/numbers masked, counts window/baseline),
  * the slowest operations in the window next to the baseline p95 of the same operation.
No threshold, no ranking by a decision rule, nothing filtered: rows are only capped for prompt size.
Uses telemetry + alert time only (no ground truth, no training)."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from typing import Dict, Optional

import networkx as nx

from pipeline.propagation_evidence import window_bounds, _calls_graph, _node_name, _node_type, _p95

CHAR_BUDGET = 3800
LINE_CAP = 500
_HTTP_RE = re.compile(r'"http\.status_code":"?(\d{3})')
_ID_RE = re.compile(r"\d{4}-\d\d-\d\dT[\d:.+Z-]+|\b(?=[A-Z0-9]*\d)[A-Z0-9]{8,}\b|\b\d{4,}\b")


def _mask(msg: str, cap: int = 110) -> str:
    msg = re.sub(r"\s+", " ", str(msg or "")).strip()
    msg = _ID_RE.sub("<id>", msg)
    return msg[:cap]


def _fmt_ms(x):
    return "n/a" if x is None else f"{x:.0f}ms"


def compute_trace_evidence(case, alert_ts: Optional[dt.datetime],
                           entry_entity_id: Optional[str], max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _compute(case, alert_ts, entry_entity_id, max_hops):
    from data.loader import find_service_ancestor

    topo = case.topology
    if alert_ts is None or not entry_entity_id or entry_entity_id not in topo:
        return None
    svc0 = find_service_ancestor(entry_entity_id, topo) or entry_entity_id
    calls = _calls_graph(topo)
    if svc0 not in calls:
        return None
    hop = {svc0: 0}
    for node, d in nx.single_source_shortest_path_length(calls, svc0, cutoff=max_hops).items():
        if d > 0 and _node_type(topo, node) == "apm.service":
            hop[node] = d
    upstream = {p for p in calls.predecessors(svc0) if _node_type(topo, p) == "apm.service"}
    cand = set(hop) | upstream
    name_of = {e: _node_name(topo, e) for e in cand}
    by_name = {n: e for e, n in name_of.items()}

    at = alert_ts.replace(tzinfo=None) if alert_ts.tzinfo else alert_ts
    w0, w1 = window_bounds(case, at)

    S = defaultdict(lambda: {"win_n": 0, "win_err": 0, "base_n": 0, "base_err": 0,
                             "http_w": Counter(), "http_b": Counter(),
                             "msg_w": Counter(), "msg_b": Counter(),
                             "op_w": defaultdict(float), "op_b": defaultdict(list)})
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        eid = by_name.get(p.get("serviceName"))
        if eid is None or o.timestamp is None:
            continue
        ts = o.timestamp.replace(tzinfo=None) if o.timestamp.tzinfo else o.timestamp
        if ts > w1:
            continue
        in_win = ts >= w0
        s = S[eid]
        is_err = str(p.get("statusCode")) == "2"
        s["win_n" if in_win else "base_n"] += 1
        if is_err:
            s["win_err" if in_win else "base_err"] += 1
        attrs = p.get("attributes")
        m = _HTTP_RE.search(attrs) if isinstance(attrs, str) else None
        if m and (in_win or is_err):
            (s["http_w"] if in_win else s["http_b"])[m.group(1)] += 1
        if is_err:
            key = _mask(p.get("statusMessage")) or (f"(no message; http {m.group(1)})" if m else "(no message)")
            (s["msg_w"] if in_win else s["msg_b"])[key] += 1
        try:
            dur = float(p.get("duration")) / 1e6
        except (TypeError, ValueError):
            continue
        op = str(p.get("spanName"))[:70]
        if in_win:
            s["op_w"][op] = max(s["op_w"][op], dur)
        else:
            s["op_b"][op].append(dur)

    def codes(c):
        return ", ".join(f"{k}:{v}" for k, v in sorted(c.items())) or "none"

    lines = ["Trace evidence for the call-graph neighbourhood of the alert service (alert window vs earlier "
             "baseline; HTTP codes from span attributes, error messages with ids masked; nothing is "
             "filtered: judge what the failing calls actually say)."]
    order = sorted(cand, key=lambda e: (hop.get(e, 99), name_of[e]))
    for eid in order:
        s = S.get(eid)
        rel = ("alert service" if eid == svc0 else
               f"downstream {hop[eid]} hop(s)" if eid in hop else "direct upstream caller")
        if not s or (s["win_n"] + s["base_n"]) == 0:
            lines.append(f"- {name_of[eid]} ({rel}): no spans")
            continue
        msgs = "; ".join(f'"{m}" {c}/{s["msg_b"].get(m, 0)}' for m, c in s["msg_w"].most_common(3)) or "none"
        slow = sorted(s["op_w"].items(), key=lambda kv: -kv[1])[:2]
        slow_txt = "; ".join(
            f"{op} max={d:.0f}ms (baseline p95={_fmt_ms(_p95(s['op_b'].get(op, [])))})"
            for op, d in slow) or "none"
        line = (f"- {name_of[eid]} ({rel}): spans win={s['win_n']} base={s['base_n']}; errors win={s['win_err']} "
                f"base={s['base_err']}; http win[{codes(s['http_w'])}] base-errors[{codes(s['http_b'])}]; "
                f"error msgs (win/base): {msgs}; slowest: {slow_txt}")
        lines.append(line[:LINE_CAP])
    text, used = [], 0
    for ln in lines:
        if used + len(ln) + 1 > CHAR_BUDGET:
            text.append("... (further services omitted for prompt size)")
            break
        text.append(ln)
        used += len(ln) + 1
    return {"text": "\n".join(text), "rows": len(order)}
