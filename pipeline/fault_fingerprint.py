"""Fault fingerprints (arm FP; FAULT_FINGERPRINT = show | shortlist | rule). Evidence-to-label layer for the fault type.

Why: dev-30 shows the entity right in 13/30 cases but the type right in only 3 of those 13. The Coordinator and the typing call choose freely
among 28 labels, which a 7B model does poorly. This module checks, for every service and node, a FIXED list of discriminating signals
(the ones that separate similar fault types, per the type definitions in data/taxonomy.py and the public PoS catalog) and maps them to the
fault types whose KEY signal is present (fault-type fingerprints). It is a hierarchical shortlist: level (node vs service) -> types whose key
signal is present -> ranked by present supporting signals minus contradicting ones.

Leakage rules (same as the rest of the repo): uses only telemetry and the alert time of the case. Thresholds are domain conventions set
BEFORE any run (constants below) and are never tuned on RCA100 labels; the rule table is written from the type DEFINITIONS, not from cases.
No label frequency or prior from RCA100 is used.

Modes (config.FAULT_FINGERPRINT):
  show      -- the signal profile and the candidate list are shown to the typing call as evidence; any of the 28 types may be chosen.
  shortlist -- the typing call must choose among the top-K candidates (constrained; invalid answer -> top candidate). No candidate -> the
               ordinary full-taxonomy typing call.
  rule      -- no LLM: the top candidate is the answer (ablation: does the LLM add anything over the rules?).
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from pipeline.propagation_evidence import window_bounds, _node_name, _node_type

# --- a-priori thresholds (fixed before any run; never tuned on RCA100) ---------------------------------------------------------
RISE_RATIO = 2.0        # metric: window max >= 2x baseline median ...
RISE_Z = 3.0            # ... and >= 3 robust SDs above it (MAD-based); a flat baseline counts as infinitely many SDs
STRONG_RATIO = 5.0      # 'strong' rise (e.g. full GC vs ordinary GC activity)
DROP_RATIO = 0.5        # window min <= half the baseline median (replicas, ready status, running pods)
HIGH_RATE = 0.8         # usage-rate metrics (cpu/mem/disk): >= 0.8 on a 0-1 scale, >= 80 on a percent scale
TEXT_MIN = 3            # text/trace signal: >= 3 matching lines or spans in the window ...
TEXT_RATE = 2.0         # ... at >= 2x the baseline rate per second (or no baseline match at all)
SLOW_RATIO = 2.0        # dependency spans: window max >= 2x baseline p95
SLOW_ABS_MS = 1000.0    # ... or >= 1 s when there is no baseline span
EVENT_LOOKBACK_S = 600  # K8s events count when their last occurrence is at most 10 min before the window start (or later)
TOP_K = 3

_METRIC_FAMILIES = [    # first match wins; matched on the lower-case metric name
    ("throttle", r"throttl"), ("gc", r"gc"), ("thread", r"thread"), ("error", r"error|exception|fail"),
    ("latency", r"latency|(^|_)rt($|_)|duration|response_time|elapsed"), ("replica", r"replica"), ("restart", r"restart"),
    ("ready", r"ready"), ("pods", r"pod.*(running|count)|running.*pod"), ("disk", r"disk|(^|_)io($|_)|iops|(^|_)fs_"),
    ("cpu", r"cpu"), ("mem", r"mem|heap"), ("request", r"request|qps|workload|(^|_)count$"),   # RCA100 APM: request count is exported as 'workload'

]
_METRIC_FAMILIES = [(f, re.compile(p)) for f, p in _METRIC_FAMILIES]
_RATE_NAME = re.compile(r"rate|usage|util|percent|ratio")

_TEXT_FAMILIES = [
    ("npe", r"nullpointer|null pointer|nonetype|nil pointer|\bnpe\b|cannot read propert(?:y|ies) of (?:null|undefined)"),
    ("ratelimit", r"rate.?limit|throttl|too many requests|quota exceeded|\b429\b"),
    ("oom", r"outofmemory|out of memory|\boom\b|heap space|memory limit"),
    ("gc", r"full gc|garbage collect|stop-the-world|gc overhead"),
    ("thread", r"rejectedexecution|thread ?pool|pool (?:is )?exhausted|no (?:available|free) thread|too many threads"),
    ("timeout", r"timeout|timed out|deadline exceeded"),
    ("refused", r"connection refused|econnrefused|connect(?:ion)? (?:failed|error)|connection reset|unavailable"),
    ("redis", r"redis|valkey"),
    ("dns", r"no such host|name resolution|could not resolve|unknown host|getaddrinfo|nxdomain|\bdns\b"),
    ("sql", r"\bsql\b|jdbc|postgres|mysql|database|\bquery\b"),
    ("queue", r"kafka|consumer lag|backlog|queue (?:is )?full|rebalanc"),
    ("cache", r"cache miss|\bcache\b"),
    ("lb", r"no healthy upstream|upstream connect error|no endpoints|endpoints? not (?:found|available)|load ?balanc"),
    ("netpol", r"network ?policy|i/o timeout|no route to host"),
    ("error", r"\berror\b|exception|\bfail"),
]
_TEXT_FAMILIES = [(f, re.compile(p, re.I)) for f, p in _TEXT_FAMILIES]
_TEXT_ANY = re.compile("|".join(p.pattern for _, p in _TEXT_FAMILIES), re.I)

_EVENT_FAMILIES = [
    ("oom", r"oom|outofmemory"), ("crashloop", r"backoff|crashloop"), ("unhealthy", r"unhealthy|probe failed"),
    ("unschedulable", r"failedscheduling|unschedulable|insufficient"), ("scale_down", r"scaled down|scale.* to 0\b|successfuldelete"),
    ("restart", r"killing|restart"), ("node_notready", r"notready|nodeunreachable|rebooted|node.*not ready"), ("evicted", r"evict"),
]
_EVENT_FAMILIES = [(f, re.compile(p, re.I)) for f, p in _EVENT_FAMILIES]
_HTTP_RE = re.compile(r'"http\.(?:response\.)?status_code":"?(\d{3})')
_DB_RE = re.compile(r'"db\.system"|^\s*(?:select|insert|update|delete)\b|\bsql\b|\bquery\b|postgres|mysql', re.I)
_REDIS_RE = re.compile(r"redis|valkey", re.I)
_QUEUE_RE = re.compile(r'kafka|"messaging\.system"', re.I)


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and getattr(ts, "tzinfo", None) else ts


def _fmt(x):
    return "n/a" if x is None else (f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}")


# ------------------------------------------------------------------------------------------------------------------------------
# 1. signal database: per service / node, computed once per case (cheap: one pass per modality)
# ------------------------------------------------------------------------------------------------------------------------------
def compute_signal_db(case, alert_ts) -> Dict:
    """Returns {"services": {name: profile}, "nodes": {name: profile}, "hosts": {svc: [node,...]}, "window_s": float} or {} on failure."""
    try:
        return _compute_db(case, alert_ts)
    except Exception as e:  # evidence must never break the pipeline
        return {"error": repr(e)}


def _owner(name: str, services: List[str]) -> Optional[str]:
    """Service owning an entity/pod/container name: exact, 'svc::op', or 'svc-<suffix>' (longest service name wins)."""
    if not name:
        return None
    name = str(name)
    best = None
    for s in services:
        if name == s or name.split("::")[0] == s or name.startswith(s + "-"):
            if best is None or len(s) > len(best):
                best = s
    return best


def _new_profile():
    return {"metrics": defaultdict(list), "ops": defaultdict(list), "text_w": Counter(), "text_b": Counter(),
            "trace": {"win_n": 0, "base_n": 0, "win_err": 0, "base_err": 0, "http_w": Counter(), "http_b": Counter(),
                      "dep": defaultdict(lambda: {"w": [], "b": [], "err_w": 0, "err_b": 0})},
            "events": defaultdict(lambda: {"count": 0, "reasons": Counter()})}


def _series_stat(series, w0, w1):
    base = [v for t, v in series if t < w0]
    win = [v for t, v in series if w0 <= t <= w1]
    if not win:
        return None
    bmed = statistics.median(base) if base else None
    wmax, wmin = max(win), min(win)
    z = None
    if base:
        mad = statistics.median([abs(v - bmed) for v in base])
        sig = 1.4826 * mad
        z = (float("inf") if wmax > bmed else 0.0) if sig == 0 else (wmax - bmed) / sig
    return {"base": bmed, "max": wmax, "min": wmin, "z": z}


def _classify(metric: str, st: dict) -> dict:
    b, mx, mn, z = st["base"], st["max"], st["min"], st["z"]
    ratio = (mx / b) if (b not in (None, 0)) else None
    rise = bool(b is not None and ((b == 0 and mx > 0) or (ratio is not None and ratio >= RISE_RATIO and (z is None or z >= RISE_Z))))
    strong = bool(b is not None and ((b == 0 and mx > 0) or (ratio is not None and ratio >= STRONG_RATIO and (z is None or z >= RISE_Z))))
    high = bool(_RATE_NAME.search(metric) and mx is not None and ((mx <= 1.0 and mx >= HIGH_RATE) or (mx > 1.0 and mx >= HIGH_RATE * 100 and mx <= 100)))
    drop = bool(b not in (None, 0) and mn is not None and mn / b <= DROP_RATIO)
    return {"rise": rise, "strong": strong, "high": high, "drop": drop,
            "txt": f"{metric} base={_fmt(b)} max={_fmt(mx)} min={_fmt(mn)}" + (f" (x{ratio:.3g})" if ratio else (" (new)" if b == 0 and mx > 0 else ""))}


def _family(metric: str) -> Optional[str]:
    m = (metric or "").lower()
    for f, rx in _METRIC_FAMILIES:
        if rx.search(m):
            return f
    return None


def _compute_db(case, alert_ts):
    topo = case.topology
    if alert_ts is None:
        return {}
    at = _naive(alert_ts)
    w0, w1 = window_bounds(case, at)
    services = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) in ("apm.service", "apm.svc")} - {None})
    nodes = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) == "k8s.node"} - {None})
    svc = {s: _new_profile() for s in services}
    nod = {n: _new_profile() for n in nodes}

    # hosting nodes: node contains pod, pod named after a service
    hosts = defaultdict(set)
    for u, v, d in topo.edges(data=True):
        rel = d.get("relation")
        rt = getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)
        if rt == "contains" and _node_type(topo, u) == "k8s.node":
            s = _owner(_node_name(topo, v), services)
            if s:
                hosts[s].add(_node_name(topo, u))

    t_min = w0
    # metrics
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts > w1 or p.get("value") is None:
            continue
        try:
            val = float(p["value"])
        except (TypeError, ValueError):
            continue
        eset, ename, metric = str(p.get("entity_set") or ""), str(p.get("entity_name") or p.get("service") or ""), str(p.get("metric") or "")
        if eset == "k8s.node":
            if ename in nod:
                nod[ename]["metrics"][metric].append((ts, val))
            continue
        s = _owner(ename, services)
        if not s:
            continue
        if eset == "apm.operation":
            svc[s]["ops"][(ename, metric)].append((ts, val))
        else:
            svc[s]["metrics"][f"{eset}:{metric}"].append((ts, val))

    # traces
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        s = _owner(p.get("serviceName"), services)
        ts = _naive(o.timestamp)
        if not s or ts is None or ts > w1:
            continue
        t_min = min(t_min, ts)
        win = ts >= w0
        tr = svc[s]["trace"]
        is_err = str(p.get("statusCode")) == "2"
        tr["win_n" if win else "base_n"] += 1
        if is_err:
            tr["win_err" if win else "base_err"] += 1
        attrs = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
        m = _HTTP_RE.search(attrs)
        if m:
            (tr["http_w"] if win else tr["http_b"])[m.group(1)] += 1
        msg = str(p.get("statusMessage") or "")
        if is_err and msg and _TEXT_ANY.search(msg):
            for f, rx in _TEXT_FAMILIES:
                if rx.search(msg):
                    (svc[s]["text_w"] if win else svc[s]["text_b"])[f] += 1
        span = str(p.get("spanName") or "")
        kind = ("db" if _DB_RE.search(span) or '"db.system"' in attrs and not _REDIS_RE.search(attrs) else
                "redis" if _REDIS_RE.search(span) or _REDIS_RE.search(attrs) else
                "queue" if _QUEUE_RE.search(span) or _QUEUE_RE.search(attrs) else None)
        if kind:
            try:
                dur = float(p.get("duration")) / 1e6
            except (TypeError, ValueError):
                dur = None
            dep = tr["dep"][kind]
            if dur is not None:
                dep["w" if win else "b"].append(dur)
            if is_err:
                dep["err_w" if win else "err_b"] += 1

    # logs
    for o in case.observations.get("logs", []):
        p = o.payload or {}
        content = p.get("content") or ""
        ts = _naive(o.timestamp)
        if ts is None or ts > w1:
            continue
        t_min = min(t_min, ts)
        if not content or not _TEXT_ANY.search(content):
            continue
        s = _owner(p.get("_container_name_"), services)
        if not s:
            continue
        win = ts >= w0
        for f, rx in _TEXT_FAMILIES:
            if rx.search(content):
                (svc[s]["text_w"] if win else svc[s]["text_b"])[f] += 1

    # K8s events
    from data.loader import event_time_bounds, _parse_ts_iso
    for o in case.observations.get("events", []):
        p = o.payload or {}
        raw = p.get("eventId")
        try:
            e = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            e = {}
        io = e.get("involvedObject") or {}
        kind, name = io.get("kind", ""), io.get("name") or p.get("pod_name") or p.get("hostname") or ""
        _f, _l = event_time_bounds(e)
        last, first = _naive(_parse_ts_iso(_l)), _naive(_parse_ts_iso(_f))
        if last is not None and (last - w0).total_seconds() < -EVENT_LOOKBACK_S:
            continue
        if first is not None and first > w1:
            continue
        txt = f"{e.get('reason', '')} {e.get('message', '')}"
        if kind == "Node" or name in nod:
            prof = nod.get(name)
        else:
            s = _owner(name, services)
            prof = svc.get(s) if s else None
        if prof is None:
            continue
        for f, rx in _EVENT_FAMILIES:
            if rx.search(txt):
                g = prof["events"][f]
                g["count"] += int(e.get("count") or 1)
                g["reasons"][str(e.get("reason", ""))] += 1

    return {"services": svc, "nodes": nod, "hosts": {k: sorted(v) for k, v in hosts.items()},
            "w0": w0, "w1": w1, "window_s": max(1.0, (w1 - w0).total_seconds()), "base_s": max(1.0, (w0 - t_min).total_seconds())}


# ------------------------------------------------------------------------------------------------------------------------------
# 2. signals of one origin entity
# ------------------------------------------------------------------------------------------------------------------------------
def _metric_families(prof, w0, w1) -> Dict[str, List[dict]]:
    fam = defaultdict(list)
    for metric, series in prof["metrics"].items():
        st = _series_stat(series, w0, w1)
        f = _family(metric.split(":")[-1])
        if st and f:
            c = _classify(metric, st)
            c["metric"] = metric
            fam[f].append(c)
    return fam


def _any(fam, f, key):
    hits = [c for c in fam.get(f, []) if c[key]]
    return hits[0]["txt"] if hits else None


def _text_rise(prof, f, db):
    w, b = prof["text_w"].get(f, 0), prof["text_b"].get(f, 0)
    if w < TEXT_MIN:
        return None
    rw, rb = w / db["window_s"], b / db["base_s"]
    if b == 0 or rw >= TEXT_RATE * rb:
        return f"{w} matching log/span lines in window (baseline {b})"
    return None


def _slow(dep):
    if not dep["w"]:
        return None
    mx = max(dep["w"])
    b = sorted(dep["b"])
    p95 = b[min(len(b) - 1, int(0.95 * len(b)))] if b else None
    if (p95 and mx >= SLOW_RATIO * p95) or (p95 is None and mx >= SLOW_ABS_MS):
        return f"span max={mx:.0f}ms baseline p95={_fmt(p95)}ms"
    return None


def origin_signals(db: Dict, origin: str) -> Dict:
    """{"kind": "service"|"node"|"unknown", "signals": {name: evidence text}, "host_nodes": [...]}; only PRESENT signals are listed."""
    if not db or "services" not in db or not origin:
        return {"kind": "unknown", "signals": {}, "host_nodes": []}
    w0, w1 = db["w0"], db["w1"]
    sig, sig_node = {}, {}
    services, nodes = db["services"], db["nodes"]
    o = origin.split("::")[0]
    if o in nodes:
        kind, node_list, prof = "node", [o], None
    else:
        s = o if o in services else _owner(o, list(services))
        if s is None:
            return {"kind": "unknown", "signals": {}, "host_nodes": []}
        kind, prof, node_list = "service", services[s], db["hosts"].get(s, [])

    # node-level signals (the origin node, or the nodes hosting the origin service)
    for n in node_list:
        nf = _metric_families(nodes[n], w0, w1)
        for name, f in (("node_cpu_high", "cpu"), ("node_mem_high", "mem"), ("node_disk_high", "disk")):
            t = _any(nf, f, "high") or _any(nf, f, "rise")
            if t and name not in sig:
                sig[name] = f"node {n}: {t}"
                sig_node[name] = n
        t = _any(nf, "ready", "drop") or _any(nf, "pods", "drop")
        ev = nodes[n]["events"]
        if t or ev.get("node_notready", {}).get("count"):
            sig_node.setdefault("node_ready_drop", n)
            sig.setdefault("node_ready_drop", f"node {n}: " + (t or f"events {dict(ev['node_notready']['reasons'])}"))
        if ev.get("oom", {}).get("count"):
            sig_node.setdefault("node_oom_event", n)
            sig.setdefault("node_oom_event", f"node {n}: events {dict(ev['oom']['reasons'])}")
        if ev.get("evicted", {}).get("count"):
            sig.setdefault("ev_evicted", f"node {n}: events {dict(ev['evicted']['reasons'])}")
    if kind == "node":   # pods on the node killed for memory -> node OOM signal
        oom_svcs = [s for s, hs in db["hosts"].items() if node_list[0] in hs and services[s]["events"].get("oom", {}).get("count")]
        if len(oom_svcs) >= 2:
            sig.setdefault("node_oom_event", f"OOM events on pods of {', '.join(oom_svcs)} hosted on {node_list[0]}")

    if prof is not None:
        sf = _metric_families(prof, w0, w1)
        for name, f, key in (("svc_cpu_rise", "cpu", "rise"), ("svc_mem_rise", "mem", "rise"), ("svc_gc_rise", "gc", "rise"),
                             ("svc_gc_strong", "gc", "strong"), ("svc_thread_rise", "thread", "rise"), ("svc_disk_rise", "disk", "rise"),
                             ("svc_throttle", "throttle", "rise"), ("svc_request_rise", "request", "rise"),
                             ("svc_latency_rise", "latency", "rise")):
            t = _any(sf, f, key) or (_any(sf, f, "high") if f in ("cpu", "mem") else None)
            if t:
                sig[name] = t
        t = _any(sf, "replica", "drop") or _any(sf, "pods", "drop")
        if t:
            sig["replica_drop"] = t
        # operation hotspot: request count rises on a minority of the service's operations
        ops = defaultdict(list)
        for (op, metric), series in prof["ops"].items():
            if _family(metric) == "request":
                st = _series_stat(series, w0, w1)
                if st:
                    ops[op].append(_classify(metric, st))
        rising = [op for op, cs in ops.items() if any(c["rise"] for c in cs)]
        if rising and len(ops) >= 3 and len(rising) <= max(1, len(ops) // 3):
            sig["op_hotspot"] = f"request count rises on {len(rising)} of {len(ops)} operations: {', '.join(sorted(rising)[:2])}"
        ev = prof["events"]
        for f in ("oom", "crashloop", "unhealthy", "unschedulable", "scale_down", "evicted"):
            if ev.get(f, {}).get("count"):
                sig[f"ev_{f}"] = f"events {dict(ev[f]['reasons'])} count={ev[f]['count']}"
        if ev.get("restart", {}).get("count", 0) >= 2:
            sig["ev_restart"] = f"events {dict(ev['restart']['reasons'])} count={ev['restart']['count']}"
        tr = prof["trace"]
        c5 = sum(v for k, v in tr["http_w"].items() if k.startswith("5"))
        b5 = sum(v for k, v in tr["http_b"].items() if k.startswith("5"))
        if c5 >= TEXT_MIN and c5 / db["window_s"] >= TEXT_RATE * b5 / db["base_s"]:
            sig["http5xx"] = f"HTTP 5xx in window={c5} baseline={b5} ({dict(tr['http_w'])})"
        for code in ("429", "403", "503"):
            if tr["http_w"].get(code, 0) >= TEXT_MIN and tr["http_w"][code] / db["window_s"] >= TEXT_RATE * tr["http_b"].get(code, 0) / db["base_s"]:
                sig[f"http{code}"] = f"HTTP {code} in window={tr['http_w'][code]} baseline={tr['http_b'].get(code, 0)}"
        if tr["win_err"] >= TEXT_MIN and tr["win_err"] / db["window_s"] >= TEXT_RATE * tr["base_err"] / db["base_s"]:
            sig["err_spans_rise"] = f"error spans window={tr['win_err']}/{tr['win_n']} baseline={tr['base_err']}/{tr['base_n']}"
        for f, _ in _TEXT_FAMILIES:
            t = _text_rise(prof, f, db)
            if t:
                sig[f"txt_{f}"] = t
        for k in ("db", "redis", "queue"):
            dep = tr["dep"].get(k)
            if not dep:
                continue
            t = _slow(dep)
            if t:
                sig[f"{k}_slow"] = f"{k} {t}"
            if dep["err_w"] >= TEXT_MIN and dep["err_w"] / db["window_s"] >= TEXT_RATE * dep["err_b"] / db["base_s"]:
                sig[f"{k}_err"] = f"{k} error spans window={dep['err_w']} baseline={dep['err_b']}"
        if sig.get("redis_err") or (sig.get("txt_redis") and (sig.get("txt_refused") or sig.get("txt_timeout"))):
            sig["redis_fail"] = sig.get("redis_err") or f"{sig['txt_redis']} with connection errors"
        lifecycle = any(sig.get(k) for k in ("ev_crashloop", "ev_restart", "ev_scale_down", "ev_oom", "ev_unhealthy", "replica_drop"))
        if (sig.get("txt_timeout") or sig.get("txt_refused")) and not lifecycle:
            sig["net_block"] = "calls time out or are refused while the pods show no restart, crash or scaling event"
    return {"kind": kind, "signals": sig, "host_nodes": node_list, "signal_node": sig_node}


# ------------------------------------------------------------------------------------------------------------------------------
# 3. fingerprint table (written from the type definitions; key = needed, sup = supporting, con = contradicting)
# ------------------------------------------------------------------------------------------------------------------------------
NODE_TYPES = {"F026-nodeCpuHigh", "F031-nodeMemoryOOM", "F001-nodeDown", "F025-diskIOHigh"}
FALLBACK_TYPES = {"F014-httpError5xx", "F011-codeDefect"}     # generic: key signal counts 1 instead of 2
FINGERPRINTS = {   # order = specificity, used only to break ties
    "F001-nodeDown":                 dict(key=["node_ready_drop"], sup=["txt_refused", "ev_unhealthy"], con=[]),
    "F031-nodeMemoryOOM":            dict(key=["node_oom_event"], sup=["node_mem_high", "ev_oom", "ev_evicted"], con=[]),
    "F026-nodeCpuHigh":              dict(key=["node_cpu_high"], sup=["svc_latency_rise"], con=["svc_request_rise"]),
    "F025-diskIOHigh":               dict(key=["node_disk_high", "svc_disk_rise"], sup=["svc_latency_rise"], con=[]),
    "F051-podPendingUnschedulable":  dict(key=["ev_unschedulable"], sup=["replica_drop"], con=[]),
    "F050-podCrashLoop":             dict(key=["ev_crashloop"], sup=["ev_restart"], con=[]),
    "F039-resourceLimitMisconfig":   dict(key=["ev_oom", "svc_throttle"], sup=["ev_restart", "ev_crashloop", "svc_mem_rise"], con=["node_oom_event"]),
    "F052-podRestartFlapping":       dict(key=["ev_unhealthy", "ev_restart"], sup=[], con=["ev_crashloop", "ev_oom"]),
    "F036-replicaScaleDown":         dict(key=["ev_scale_down", "replica_drop"], sup=["txt_refused", "http5xx", "http503"], con=["ev_crashloop", "ev_unschedulable"]),
    "F023-nullPointerException":     dict(key=["txt_npe"], sup=["http5xx", "err_spans_rise"], con=[]),
    "F016-rateLimiting":             dict(key=["http429", "txt_ratelimit"], sup=["http403"], con=[]),
    "F057-dnsResolutionFailure":     dict(key=["txt_dns"], sup=["err_spans_rise"], con=[]),
    "F029-redisUnavailable":         dict(key=["redis_fail"], sup=["txt_refused", "txt_timeout"], con=[]),
    "F009-cacheBreakdown":           dict(key=["txt_cache", "redis_slow"], sup=["db_slow", "svc_request_rise"], con=["redis_fail"]),
    "F010-slowSQL":                  dict(key=["db_slow"], sup=["txt_sql"], con=["txt_timeout"]),
    "F018-dbNetworkLatency":         dict(key=["db_slow"], sup=["txt_timeout", "db_err"], con=["txt_sql"]),
    "F005-messageQueueBacklog":      dict(key=["queue_slow", "queue_err", "txt_queue"], sup=["svc_latency_rise"], con=[]),
    "F022-fullGC":                   dict(key=["svc_gc_strong"], sup=["txt_gc", "svc_latency_rise"], con=["ev_oom"]),
    "F007-memoryPressure":           dict(key=["svc_mem_rise", "svc_gc_rise"], sup=["txt_oom"], con=["ev_oom", "svc_gc_strong"]),
    "F002-threadExhaustion":         dict(key=["svc_thread_rise", "txt_thread"], sup=["txt_timeout", "svc_latency_rise"], con=[]),
    "F012-cpuDeadLoop":              dict(key=["svc_cpu_rise"], sup=[], con=["node_cpu_high", "svc_request_rise"]),
    "F034-cpuFullLoad":              dict(key=["svc_cpu_rise"], sup=["svc_latency_rise"], con=["node_cpu_high", "svc_request_rise"]),
    "F004-trafficHotspot":           dict(key=["op_hotspot"], sup=["svc_request_rise"], con=[]),
    "F006-trafficSurge":             dict(key=["svc_request_rise"], sup=["svc_latency_rise", "svc_cpu_rise"], con=["op_hotspot"]),
    "F056-networkPolicyIsolation":   dict(key=["txt_netpol", "net_block"], sup=["err_spans_rise"], con=["txt_dns"]),
    "F020-loadBalancerFailure":      dict(key=["txt_lb", "http503"], sup=["http5xx"], con=["ev_scale_down", "replica_drop"]),
    "F014-httpError5xx":             dict(key=["http5xx"], sup=["err_spans_rise"], con=[]),
    "F011-codeDefect":               dict(key=["err_spans_rise", "txt_error"], sup=[], con=[]),
}
_ORDER = {t: i for i, t in enumerate(FINGERPRINTS)}


def rank_candidates(osig: Dict, top_k: int = TOP_K) -> List[Dict]:
    """Types whose key signal is present, ranked by 2*key (1 for the generic fallbacks) + supporting - 2*contradicting.
    Level: a node origin keeps only node-level types; a service origin keeps node-level types only through its hosting nodes' signals."""
    sig, kind = osig.get("signals", {}), osig.get("kind")
    out = []
    for t, fp in FINGERPRINTS.items():
        if kind == "node" and t not in NODE_TYPES:
            continue
        keys = [k for k in fp["key"] if sig.get(k)]
        if not keys:
            continue
        sups = [k for k in fp["sup"] if sig.get(k)]
        cons = [k for k in fp["con"] if sig.get(k)]
        score = (1 if t in FALLBACK_TYPES else 2) * len(keys) + len(sups) - 2 * len(cons)
        out.append({"type": t, "score": score, "key": keys, "sup": sups, "con": cons})
    out.sort(key=lambda c: (-c["score"], _ORDER[c["type"]]))
    return out[:top_k] if top_k else out


def profile_text(origin: str, osig: Dict, cands: List[Dict]) -> str:
    sig = osig.get("signals", {})
    from data.taxonomy import fault_group
    lines = [f"Code-checked signal profile of {origin} ({osig.get('kind')}"
             + (f"; hosting node(s): {', '.join(osig['host_nodes'])}" if osig.get("host_nodes") and osig.get("kind") == "service" else "")
             + "). Fixed thresholds set in advance: a metric RISES when its window max is >= 2x the baseline median and >= 3 robust SDs above "
               "it; a text signal needs >= 3 matching log lines or error spans in the window at >= 2x the baseline rate."]
    if sig:
        lines.append("Signals present:")
        lines.extend(f"- {k}: {v}" for k, v in sorted(sig.items()))
    else:
        lines.append("Signals present: none of the checked signals.")
    if cands:
        lines.append("Candidate fault types (each has its key signal present; types whose key signal is absent are not listed):")
        for i, c in enumerate(cands, 1):
            lines.append(f"{i}. {c['type']} [{fault_group(c['type'])}] key: {', '.join(c['key'])}"
                         + (f"; supporting: {', '.join(c['sup'])}" if c["sup"] else "")
                         + (f"; contradicting: {', '.join(c['con'])}" if c["con"] else ""))
    return "\n".join(lines)
