"""Calibrated fault fingerprints (FP_SCORING=calibrated, the default of the FP arms). Label-free statistical evidence for the fault type.

Method (paper, Section "Evidence-to-label layer"; Algorithms 1-3):
  Evidence model. Two kinds of evidence about an entity e in the alert window W = [w0, w1]:
    * continuous signals (metric families, log-keyword rates). Score = one-sided Mann-Whitney U p-value of the window against the
      history right before it (minimum over the entity's series of that family). Because series drift and are autocorrelated, this score
      is NOT used as a probability: it is calibrated against an empirical null of the same score computed on pseudo-windows cut from the
      case's own baseline, pooled over the entities of the same kind in the case (Efron 2004; conformal calibration, Vovk et al. 2005).
      The signal holds when the calibrated p <= ALPHA. See _calibrate().
    * categorical observations that name a mechanism and need no baseline: K8s event reasons in W, HTTP status classes and
      error-message patterns of error spans in W, failing database/Redis/Kafka spans in W. RCA100 exports traces only inside the window
      (no trace baseline; scripts/diagnostics/check_baseline_length.py), so trace evidence can only be categorical, or compared within
      the window (a dependency's span durations against the same service's other spans: Mann-Whitney U).
  Hypotheses are pairs (entity, type): the origin named by the Coordinator and, for a service origin, the nodes hosting it. A type is
  admitted only for an entity of its own level, only when at least one KEY signal of its fingerprint holds (necessary condition), and only
  when no CONTRADICTING signal holds (exclusion). Admitted hypotheses are ordered by (number of categorical observations among key and
  supporting signals, Fisher statistic -2*sum(ln p) of the continuous ones): direct observations of a mechanism outrank symptoms.
  If nothing is admitted but error spans or HTTP 5xx are observed, the residual types (httpError5xx, codeDefect; defined as fallbacks in the
  taxonomy) are returned.

Parameters: ALPHA (0.05, conventional), TOP_K (3). No threshold is chosen by looking at labels; the fingerprint table below is written from
the type definitions (data/taxonomy.py FAULT_DEFINITIONS and the PoS catalog, MIT, paraphrased) and frozen before evaluation.
Uses only the case's own telemetry and alert time; never reads answer_key/.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from pipeline.propagation_evidence import window_bounds, _node_name, _node_type
from pipeline.fault_fingerprint import (_owner, _family, _naive, _fmt, _TEXT_FAMILIES, _TEXT_ANY, _EVENT_FAMILIES, _HTTP_RE,
                                        _DB_RE, _REDIS_RE, _QUEUE_RE)

ALPHA = 0.05
TOP_K = 3
LOG_BIN_S = 30          # log-keyword counts per 30 s bin (= the metric sampling step)
MIN_SAMPLES = 3


# ------------------------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------------------------
def mwu_greater(x: List[float], y: List[float]) -> Optional[float]:
    """One-sided Mann-Whitney U: p-value for 'x tends to be larger than y' (normal approximation, tie-corrected, continuity-corrected).
    None when either sample has < MIN_SAMPLES values."""
    n1, n2 = len(x), len(y)
    if n1 < MIN_SAMPLES or n2 < MIN_SAMPLES:
        return None
    allv = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks, i, ties = [0.0] * len(allv), 0, 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u1 - n1 * n2 / 2.0 - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ------------------------------------------------------------------------------------------------------------------------------
# Algorithm 1: evidence database (one pass per modality)
# ------------------------------------------------------------------------------------------------------------------------------
def compute_signal_db(case, alert_ts) -> Dict:
    try:
        return _compute(case, alert_ts)
    except Exception as e:  # evidence must never break the pipeline
        return {"error": repr(e)}


def _prof():
    return {"series": defaultdict(list), "ops": defaultdict(list), "log_bins": defaultdict(Counter),
            "http": Counter(), "msg": Counter(), "n_spans": 0, "n_err": 0,
            "dur": defaultdict(list), "dep_err": Counter(), "events": defaultdict(Counter)}


def _compute(case, alert_ts):
    topo = case.topology
    if alert_ts is None:
        return {}
    w0, w1 = window_bounds(case, _naive(alert_ts))
    services = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) in ("apm.service", "apm.svc")} - {None})
    nodes = sorted({_node_name(topo, n) for n in topo.nodes if _node_type(topo, n) == "k8s.node"} - {None})
    svc = {s: _prof() for s in services}
    nod = {n: _prof() for n in nodes}
    hosts = defaultdict(set)
    for u, v, d in topo.edges(data=True):
        rel = d.get("relation")
        rt = getattr(rel, "relation_type", None) or (rel if isinstance(rel, str) else None)
        if rt == "contains" and _node_type(topo, u) == "k8s.node":
            s = _owner(_node_name(topo, v), services)
            if s:
                hosts[s].add(_node_name(topo, u))

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
                nod[ename]["series"][metric].append((ts, val))
            continue
        s = _owner(ename, services)
        if not s:
            continue
        if eset == "apm.operation":
            svc[s]["ops"][(ename, metric)].append((ts, val))
        else:
            svc[s]["series"][f"{eset}:{ename}:{metric}"].append((ts, val))

    log_t0 = None
    for o in case.observations.get("logs", []):
        ts = _naive(o.timestamp)
        if ts is None or ts > w1:
            continue
        log_t0 = ts if log_t0 is None or ts < log_t0 else log_t0
        p = o.payload or {}
        content = p.get("content") or ""
        if not content or not _TEXT_ANY.search(content):
            continue
        s = _owner(p.get("_container_name_"), services)
        if not s:
            continue
        b = int((ts - w0).total_seconds() // LOG_BIN_S)       # negative = baseline bins
        for f, rx in _TEXT_FAMILIES:
            if rx.search(content):
                svc[s]["log_bins"][f][b] += 1

    for o in case.observations.get("traces", []):
        p = o.payload or {}
        s = _owner(p.get("serviceName"), services)
        ts = _naive(o.timestamp)
        if not s or ts is None or ts < w0 or ts > w1:
            continue
        pr = svc[s]
        pr["n_spans"] += 1
        is_err = str(p.get("statusCode")) == "2"
        attrs = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
        span = str(p.get("spanName") or "")
        m = _HTTP_RE.search(attrs)
        if is_err:
            pr["n_err"] += 1
            if m:
                pr["http"][m.group(1)] += 1
            msg = str(p.get("statusMessage") or "")
            for f, rx in _TEXT_FAMILIES:
                if msg and rx.search(msg):
                    pr["msg"][f] += 1
        kind = ("db" if _DB_RE.search(span) or ('"db.system"' in attrs and not _REDIS_RE.search(attrs)) else
                "redis" if _REDIS_RE.search(span) or _REDIS_RE.search(attrs) else
                "queue" if _QUEUE_RE.search(span) or _QUEUE_RE.search(attrs) else "other")
        try:
            pr["dur"][kind].append(float(p.get("duration")) / 1e6)
        except (TypeError, ValueError):
            pass
        if is_err and kind != "other":
            pr["dep_err"][kind] += 1

    from data.loader import event_time_bounds, _parse_ts_iso
    for o in case.observations.get("events", []):
        p = o.payload or {}
        raw = p.get("eventId")
        try:
            e = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            e = {}
        _f, _l = event_time_bounds(e)
        first, last = _naive(_parse_ts_iso(_f)), _naive(_parse_ts_iso(_l))
        if first is None or last is None or last < w0 or first > w1:      # the event must overlap the window
            continue
        io = e.get("involvedObject") or {}
        kind, name = io.get("kind", ""), io.get("name") or p.get("pod_name") or p.get("hostname") or ""
        prof = nod.get(name) if (kind == "Node" or name in nod) else svc.get(_owner(name, services) or "")
        if prof is None:
            continue
        txt = f"{e.get('reason', '')} {e.get('message', '')}"
        for f, rx in _EVENT_FAMILIES:
            if rx.search(txt):
                prof["events"][f][str(e.get("reason", ""))] += int(e.get("count") or 1)

    return {"services": svc, "nodes": nod, "hosts": {k: sorted(v) for k, v in hosts.items()}, "w0": w0, "w1": w1,
            "log_base_bins": (int((w0 - log_t0).total_seconds() // LOG_BIN_S) if log_t0 and log_t0 < w0 else 0),
            "win_bins": max(1, int(math.ceil((w1 - w0).total_seconds() / LOG_BIN_S)))}


# ------------------------------------------------------------------------------------------------------------------------------
# tests per entity -> signals {name: {"kind": "cont"|"cat", "p": float|None, "ev": text}}
# ------------------------------------------------------------------------------------------------------------------------------
# Calibration (fixes the label-free check of f2e2bf4: plain rank-test p-values gave a pseudo-window false-alarm rate of ~30% instead of
# 5%, because metric series drift and are autocorrelated, so window and baseline samples are not exchangeable).
# Remedy: the rank-test p-value is used only as a SCORE. For every (entity kind, signal) the score of each entity is also computed on K
# pseudo-windows cut from the case's own baseline (no fault there: they precede the alert window); these scores, pooled over the entities
# of that kind in the same case, form the empirical null (Efron 2004; conformal calibration, Vovk et al. 2005). The calibrated p-value of
# the real window is (1 + #{null scores <= score}) / (N_null + 1). Every tested window (real or pseudo) is compared with the same length
# of history immediately before it (B_SPAN window lengths), so real and pseudo scores are on the same footing.
B_SPAN = 4              # history used by each rank test: the 4 window lengths (24 min) right before the tested window
K_PSEUDO = 5            # pseudo-windows per series: as many as the ~55 min metric baseline allows with B_SPAN = 4
LOG_B_SPAN, LOG_K = 1, 1  # logs have ~15 min of baseline (2.5 windows): 1 window of history, 1 pseudo-window

NODE_SIGNALS = [("node_cpu_rise", ("cpu",), "up"), ("node_mem_rise", ("mem",), "up"), ("node_disk_rise", ("disk",), "up"),
                ("node_ready_drop", ("ready", "pods"), "down")]
SVC_SIGNALS = [("svc_cpu_rise", ("cpu",), "up"), ("svc_mem_rise", ("mem",), "up"), ("svc_gc_rise", ("gc",), "up"),
               ("svc_thread_rise", ("thread",), "up"), ("svc_disk_rise", ("disk",), "up"), ("svc_throttle_rise", ("throttle",), "up"),
               ("svc_request_rise", ("request",), "up"), ("svc_latency_rise", ("latency",), "up"), ("svc_error_rise", ("error",), "up"),
               ("svc_restart_rise", ("restart",), "up"), ("replica_drop", ("replica", "pods", "ready"), "down")]


def _score(pts, a, b, span, direction, closed_end=False):
    """Rank-test p-value of the window [a, b) (or [a, b]) against the `span` seconds of history right before a."""
    base = [v for t, v in pts if a - span <= t < a]
    win = [v for t, v in pts if a <= t < b or (closed_end and t == b)]
    return mwu_greater(win, base) if direction == "up" else mwu_greater(base, win)


def _family_scores(series_map, families, direction, windows, span, name_of):
    """Signal score per window = min rank-test p over the entity's series of these families (None if no series)."""
    sel = [pts for key, pts in series_map.items() if _family(name_of(key)) in families]
    out = []
    for i, (a, b) in enumerate(windows):
        ps = [_score(pts, a, b, span, direction, closed_end=(i == 0)) for pts in sel]
        ps = [p for p in ps if p is not None]
        out.append(min(ps) if ps else None)
    return out, len(sel)


def _calibrate(db):
    """Fills db["_cal"][entity][signal] = {"p": calibrated p, "raw": raw score, "n_null": N, "pseudo": [scores], "n_series": m}."""
    if "_cal" in db:
        return db["_cal"]
    w0, w1 = db["w0"], db["w1"]
    L = w1 - w0
    span = B_SPAN * L
    wins = [(w0, w1)] + [(w0 - j * L, w0 - (j - 1) * L) for j in range(1, K_PSEUDO + 1)]
    metric_name = lambda k: k.split(":")[-1]
    raw = defaultdict(dict)
    for kind, group, sigs, name_of in (("node", "nodes", NODE_SIGNALS, metric_name), ("service", "services", SVC_SIGNALS, metric_name),
                                       ("op", "services", [("op_request_rise", ("request",), "up")], lambda k: k[1])):
        for ent, pr in db[group].items():
            smap = pr["ops"] if kind == "op" else pr["series"]
            for name, fams, d in sigs:
                sc, m = _family_scores(smap, fams, d, wins, span, name_of)
                if m and sc[0] is not None:
                    raw[ent][name] = {"kind_pool": (kind, name), "raw": sc[0], "pseudo": [x for x in sc[1:] if x is not None], "n_series": m}
    # log-keyword rates (30 s bins)
    nw = db["win_bins"]
    for ent, pr in db["services"].items():
        for f, _ in _TEXT_FAMILIES:
            bins = pr["log_bins"].get(f)
            if not bins:
                continue
            def binv(lo, hi):
                return [bins.get(i, 0) for i in range(lo, hi)]
            real = mwu_greater(binv(0, nw), binv(-LOG_B_SPAN * nw, 0)) if db["log_base_bins"] >= LOG_B_SPAN * nw else None
            pseudo = [mwu_greater(binv(-j * nw, -(j - 1) * nw), binv(-(j + LOG_B_SPAN) * nw, -j * nw)) for j in range(1, LOG_K + 1)
                      if db["log_base_bins"] >= (j + LOG_B_SPAN) * nw]
            if real is not None:
                raw[ent][f"log_{f}_rise"] = {"kind_pool": ("log", "log"), "raw": real, "pseudo": [x for x in pseudo if x is not None],
                                             "n_series": 1, "ev": f"'{f}' log lines in window={sum(binv(0, nw))}, previous window={sum(binv(-nw, 0))}"}
    null = defaultdict(list)
    for ent, sigs in raw.items():
        for name, r in sigs.items():
            null[r["kind_pool"]].extend(r["pseudo"])
    cal = defaultdict(dict)
    for ent, sigs in raw.items():
        for name, r in sigs.items():
            nl = null[r["kind_pool"]]
            r["n_null"] = len(nl)
            r["p"] = (1 + sum(1 for x in nl if x <= r["raw"])) / (len(nl) + 1)
            cal[ent][name] = r
    db["_cal"] = cal
    db["_null"] = null
    return cal


def entity_signals(db: Dict, ent: str) -> Dict:
    """Signals of one entity (a service or a node name). Continuous signals carry their calibrated p-value whether or not they hold;
    categorical signals are listed only when observed."""
    cal = _calibrate(db)
    sig = {}
    for name, r in cal.get(ent, {}).items():
        ev = r.get("ev") or f"rank-test score={r['raw']:.3g} over {r['n_series']} series"
        sig[name] = {"kind": "cont", "p": r["p"], "raw": r["raw"],
                     "ev": f"{ev} (calibrated p={r['p']:.3g} against {r['n_null']} pseudo-window scores)"}
    if ent in db["nodes"]:
        pr = db["nodes"][ent]
        for f, name in (("node_notready", "ev_node_notready"), ("oom", "ev_node_oom"), ("evicted", "ev_evicted")):
            if pr["events"].get(f):
                sig[name] = {"kind": "cat", "p": None, "ev": f"events {dict(pr['events'][f])}"}
        return sig
    pr = db["services"][ent]
    # categorical: events, status codes, error messages, failing dependency spans (window only)
    for f in ("oom", "crashloop", "unhealthy", "unschedulable", "scale_down", "restart", "evicted"):
        if pr["events"].get(f):
            sig[f"ev_{f}"] = {"kind": "cat", "p": None, "ev": f"events {dict(pr['events'][f])}"}
    h = pr["http"]
    for code in ("429", "403", "503"):
        if h.get(code):
            sig[f"http{code}"] = {"kind": "cat", "p": None, "ev": f"{h[code]} error spans with HTTP {code} (of {pr['n_err']} error spans, {pr['n_spans']} spans)"}
    c5 = sum(v for k, v in h.items() if k.startswith("5"))
    if c5:
        sig["http5xx"] = {"kind": "cat", "p": None, "ev": f"{c5} error spans with HTTP 5xx {dict(h)}"}
    if pr["n_err"]:
        sig["error_spans"] = {"kind": "cat", "p": None, "ev": f"{pr['n_err']} error spans of {pr['n_spans']} in the window"}
    for f in ("npe", "ratelimit", "dns", "refused", "timeout", "redis", "oom", "gc", "thread", "queue", "lb", "netpol", "sql", "cache"):
        if pr["msg"].get(f):
            sig[f"msg_{f}"] = {"kind": "cat", "p": None, "ev": f"{pr['msg'][f]} error spans whose message matches '{f}'"}
    for k in ("db", "redis", "queue"):
        if pr["dep_err"].get(k):
            sig[f"{k}_err"] = {"kind": "cat", "p": None, "ev": f"{pr['dep_err'][k]} failing {k} spans"}
        d, other = pr["dur"].get(k, []), pr["dur"].get("other", [])
        p = mwu_greater(d, other)
        if p is not None:
            sig[f"{k}_slow"] = {"kind": "cont", "p": p, "ev": f"{k} spans max={_fmt(max(d))}ms vs the service's other spans median="
                                                                f"{_fmt(sorted(other)[len(other) // 2])}ms in the window (p={p:.3g})"}
    return sig


def holds(s, alpha=ALPHA) -> bool:
    return bool(s) and (s["kind"] == "cat" or (s["p"] is not None and s["p"] <= alpha))


# ------------------------------------------------------------------------------------------------------------------------------
# fingerprint table: KEY (any one is necessary), SUP (supporting), CON (excluding). Written from the type definitions; frozen.
# Composite signals: "a&b" holds when both hold, "a|b" when either holds.
# ------------------------------------------------------------------------------------------------------------------------------
LEVEL = defaultdict(lambda: "service", {"F026-nodeCpuHigh": "node", "F031-nodeMemoryOOM": "node", "F001-nodeDown": "node",
                                        "F025-diskIOHigh": "any"})
RESIDUAL = ("F014-httpError5xx", "F011-codeDefect")
FINGERPRINTS = {
    # Cloud resource (node level)
    "F026-nodeCpuHigh":             dict(key=["node_cpu_rise"], sup=[], con=[]),
    "F031-nodeMemoryOOM":           dict(key=["ev_node_oom", "node_mem_rise&ev_evicted"], sup=["node_mem_rise", "ev_evicted"], con=[]),
    "F001-nodeDown":                dict(key=["ev_node_notready", "node_ready_drop"], sup=[], con=[]),
    "F025-diskIOHigh":              dict(key=["node_disk_rise", "svc_disk_rise"], sup=["svc_latency_rise"], con=[]),
    # K8s lifecycle
    "F051-podPendingUnschedulable": dict(key=["ev_unschedulable"], sup=["replica_drop"], con=[]),
    "F050-podCrashLoop":            dict(key=["ev_crashloop"], sup=["ev_restart", "svc_restart_rise"], con=[]),
    "F039-resourceLimitMisconfig":  dict(key=["ev_oom", "svc_throttle_rise"], sup=["ev_restart", "svc_mem_rise"], con=[]),
    "F052-podRestartFlapping":      dict(key=["ev_unhealthy", "ev_restart", "svc_restart_rise"], sup=[], con=["ev_crashloop", "ev_scale_down"]),
    "F036-replicaScaleDown":        dict(key=["ev_scale_down", "replica_drop"], sup=["http503", "msg_refused"], con=["ev_crashloop", "ev_unschedulable"]),
    "F056-networkPolicyIsolation":  dict(key=["msg_netpol", "msg_timeout|msg_refused"], sup=["error_spans"], con=["ev_restart", "ev_crashloop", "ev_scale_down", "msg_dns"]),
    "F057-dnsResolutionFailure":    dict(key=["msg_dns", "log_dns_rise"], sup=["error_spans"], con=[]),
    "F020-loadBalancerFailure":     dict(key=["msg_lb", "http503"], sup=["http5xx"], con=["ev_scale_down", "replica_drop"]),
    # Application logic
    "F023-nullPointerException":    dict(key=["msg_npe", "log_npe_rise"], sup=["http5xx", "error_spans"], con=[]),
    "F016-rateLimiting":            dict(key=["http429", "msg_ratelimit", "log_ratelimit_rise", "http403&msg_ratelimit"], sup=["http403"], con=[]),
    "F006-trafficSurge":            dict(key=["svc_request_rise"], sup=["svc_latency_rise", "svc_cpu_rise"], con=[]),
    "F004-trafficHotspot":          dict(key=["op_request_rise"], sup=[], con=["svc_request_rise"]),
    # Middleware & DB
    "F029-redisUnavailable":        dict(key=["redis_err", "msg_redis"], sup=["msg_refused", "msg_timeout"], con=[]),
    "F009-cacheBreakdown":          dict(key=["msg_cache", "redis_slow&db_slow"], sup=["db_slow", "svc_request_rise"], con=["redis_err"]),
    "F010-slowSQL":                 dict(key=["db_slow"], sup=["svc_latency_rise", "msg_sql"], con=["db_err"]),
    "F018-dbNetworkLatency":        dict(key=["db_slow", "db_err&msg_timeout"], sup=["svc_latency_rise", "msg_timeout"], con=[]),
    "F005-messageQueueBacklog":     dict(key=["queue_slow", "queue_err", "msg_queue", "log_queue_rise"], sup=["svc_latency_rise"], con=[]),
    # JVM runtime
    "F022-fullGC":                  dict(key=["svc_gc_rise"], sup=["log_gc_rise", "msg_gc", "svc_latency_rise"], con=["ev_oom"]),
    "F007-memoryPressure":          dict(key=["svc_mem_rise"], sup=["svc_gc_rise"], con=["ev_oom"]),
    "F002-threadExhaustion":        dict(key=["svc_thread_rise", "msg_thread", "log_thread_rise"], sup=["msg_timeout", "svc_latency_rise"], con=[]),
    # Resource & perf. (service level)
    "F034-cpuFullLoad":             dict(key=["svc_cpu_rise"], sup=["svc_latency_rise"], con=["svc_request_rise"]),
    "F012-cpuDeadLoop":             dict(key=["svc_cpu_rise"], sup=[], con=["svc_request_rise"]),
    # residual (fallback) types: only when nothing above is admitted
    "F014-httpError5xx":            dict(key=["http5xx"], sup=["error_spans", "svc_error_rise"], con=[]),
    "F011-codeDefect":              dict(key=["error_spans"], sup=["svc_error_rise"], con=[]),
}
_ORDER = {t: i for i, t in enumerate(FINGERPRINTS)}


def _holds_expr(expr, sig, alpha):
    if "&" in expr:
        return all(holds(sig.get(x), alpha) for x in expr.split("&"))
    if "|" in expr:
        return any(holds(sig.get(x), alpha) for x in expr.split("|"))
    return holds(sig.get(expr), alpha)


def _atoms(expr):
    return [x for part in expr.split("&") for x in part.split("|")]


# ------------------------------------------------------------------------------------------------------------------------------
# Algorithm 2: hypotheses (entity, type)
# ------------------------------------------------------------------------------------------------------------------------------
def origin_signals(db: Dict, origin: Optional[str]) -> Dict:
    """Same interface as pipeline.fault_fingerprint.origin_signals: {"kind", "signals", "host_nodes", "by_entity"}."""
    if not db or "services" not in db or not origin:
        return {"kind": "unknown", "signals": {}, "host_nodes": [], "by_entity": {}}
    o = str(origin).split("::")[0]
    if o in db["nodes"]:
        kind, ents, hosts = "node", [o], []
    else:
        s = o if o in db["services"] else _owner(o, list(db["services"]))
        if s is None:
            return {"kind": "unknown", "signals": {}, "host_nodes": [], "by_entity": {}}
        hosts = db["hosts"].get(s, [])
        kind, ents = "service", [s] + hosts
    by = {e: entity_signals(db, e) for e in ents}
    held = {k: v for k, v in by[ents[0]].items() if holds(v)}
    for n in hosts:
        held.update({f"{n}:{k}": v for k, v in by[n].items() if holds(v)})
    return {"kind": kind, "signals": {k: v["ev"] for k, v in held.items()}, "host_nodes": hosts, "by_entity": by, "origin": ents[0]}


def rank_candidates(osig: Dict, top_k: int = TOP_K, alpha: float = ALPHA) -> List[Dict]:
    by = osig.get("by_entity") or {}
    if not by:
        return []
    out = []
    for ent, sig in by.items():
        lvl = "node" if ent in osig.get("host_nodes", []) or osig.get("kind") == "node" else "service"
        for t, fp in FINGERPRINTS.items():
            if t in RESIDUAL or LEVEL[t] not in (lvl, "any"):
                continue
            keys = [k for k in fp["key"] if _holds_expr(k, sig, alpha)]
            if not keys or any(_holds_expr(c, sig, alpha) for c in fp["con"]):
                continue
            sups = [k for k in fp["sup"] if _holds_expr(k, sig, alpha)]
            atoms = {a for k in keys + sups for a in _atoms(k) if holds(sig.get(a), alpha)}   # only atoms that hold ("a|b": one may be absent)
            n_cat = sum(1 for a in atoms if sig[a]["kind"] == "cat")
            fisher = sum(-2.0 * math.log(max(sig[a]["p"], 1e-300)) for a in atoms if sig[a]["kind"] == "cont")
            out.append({"type": t, "entity": ent, "key": keys, "sup": sups, "con": [], "n_cat": n_cat, "fisher": round(fisher, 2),
                        "score": (n_cat, fisher)})
    if not out:   # residual types
        ent = osig.get("origin")
        sig = by.get(ent, {})
        for t in RESIDUAL:
            fp = FINGERPRINTS[t]
            keys = [k for k in fp["key"] if _holds_expr(k, sig, alpha)]
            if keys:
                sups = [k for k in fp["sup"] if _holds_expr(k, sig, alpha)]
                out.append({"type": t, "entity": ent, "key": keys, "sup": sups, "con": [], "n_cat": len(keys), "fisher": 0.0,
                            "score": (len(keys) + len(sups), 0.0), "residual": True})
    out.sort(key=lambda c: (-c["score"][0], -c["score"][1], _ORDER[c["type"]]))
    return out[:top_k] if top_k else out


def profile_text(origin: str, osig: Dict, cands: List[Dict]) -> str:
    from data.taxonomy import fault_group
    lines = [f"Statistical evidence profile of {origin} ({osig.get('kind')}"
             + (f"; hosting node(s): {', '.join(osig['host_nodes'])}" if osig.get("host_nodes") else "")
             + f"). Continuous signals: one-sided rank test of the alert window against the entity's own baseline, holds when p <= {ALPHA}. "
               "Categorical signals: event reasons, HTTP codes and error messages observed in the window."]
    sig = osig.get("signals") or {}
    lines.append("Signals that hold:" if sig else "Signals that hold: none of the checked signals.")
    lines.extend(f"- {k}: {v}" for k, v in sorted(sig.items()))
    if cands:
        lines.append("Admitted hypotheses (entity, fault type): a key signal holds and no excluding signal holds; ordered by the number of "
                     "categorical observations, then by the combined strength of the continuous ones:")
        for i, c in enumerate(cands, 1):
            lines.append(f"{i}. {c['type']} [{fault_group(c['type'])}] on {c['entity']}: key {', '.join(c['key'])}"
                         + (f"; supporting {', '.join(c['sup'])}" if c["sup"] else "") + (" (residual type)" if c.get("residual") else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------------------
# calibration check (label-free): how often does a continuous signal 'hold' when there is no fault?
# ------------------------------------------------------------------------------------------------------------------------------
def false_alarm_rate(db: Dict, alpha: float = ALPHA) -> Dict:
    """Label-free validity check. raw: share of pseudo-window-1 scores <= alpha (the uncalibrated rank test; drift inflates it).
    calibrated: pseudo-window 1 of every entity is calibrated against the null built from pseudo-windows 2..K only, and the share
    with calibrated p <= alpha is reported (should be close to alpha)."""
    if not db or "services" not in db:
        return {}
    cal = _calibrate(db)
    pools = defaultdict(list)
    for sigs in cal.values():
        for r in sigs.values():
            pools[r["kind_pool"]].extend(r["pseudo"][1:])
    n = raw_fired = cal_fired = 0
    for sigs in cal.values():
        for r in sigs.values():
            if not r["pseudo"]:
                continue
            x = r["pseudo"][0]
            nl = pools[r["kind_pool"]]
            n += 1
            raw_fired += x <= alpha
            if nl:
                cal_fired += (1 + sum(1 for y in nl if y <= x)) / (len(nl) + 1) <= alpha
    return {"series": n, "fired": cal_fired, "raw_fired": raw_fired, "rate": (cal_fired / n if n else None),
            "raw_rate": (raw_fired / n if n else None)}
