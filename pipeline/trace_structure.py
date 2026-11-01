"""v5 trace-structure evidence (part 'torigin', PREREGISTRATION_V5.md amendment 4). Label-free, dataset-agnostic: it uses only the
OpenTelemetry span fields traceId, spanId, parentSpanId, serviceName, duration and status (ERROR = statusCode 2).

Per-service counts summarise WHERE problems start inside the call trees, which per-service counts of error spans or latency cannot
(a failing or slow callee makes all its callers fail or slow down too):

  error origin   in every trace with an error span, the error spans none of whose child spans is in error are the deepest failures;
                 their services are where the errors start. A service is an ERROR ORIGIN (observation trace_err_origin) if it is an
                 origin in >= ORIGIN_SHARE of the error traces and in >= MIN_TRACES of them (TraceRCA-style attribution).
  slow origin    the self time of a span is its duration minus the time covered by its children (overlaps merged, never < 0). In
                 every trace, the service with the largest self time is where the time is spent. Traces are grouped by their root
                 operation; the slowest (>= SLOW_Q quantile, groups of >= MIN_GROUP traces) are compared with the rest of the group. A
                 service is a SLOW ORIGIN (trace_slow_origin) if its share among the slow traces exceeds its share among the others
                 by >= SLOW_CONTRAST, in >= MIN_TRACES slow traces. No baseline is needed: RCA100 exports traces for the window only.

Both are categorical observations on services that have spans in the window. In the model they are OWN evidence of the entity
(expected at the root and at services hosted on a root node; never treated as a propagated symptom), and their ABSENCE is
uninformative (faults without failing / slow spans exist). Constants fixed a priori."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

MIN_TRACES, ORIGIN_SHARE = 3, 0.3
SLOW_Q, MIN_GROUP, SLOW_CONTRAST = 0.9, 10, 0.2
# v2 ('torigin2', amendment 5): an origin share is unstable on a handful of traces -> absolute minimum counts; a NODE is supported by
# trace origins only when >= NODE_MIN_SERVICES of the services it hosts are origins (a node fault hits its co-located services together;
# one failing service on a node is that service's fault)
MIN_ERR_TRACES_V2, MIN_SLOW_TRACES_V2, NODE_MIN_SERVICES = 10, 5, 2
SIGNALS = ("trace_err_origin", "trace_slow_origin")


def _union_len(iv):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyse(spans: List[Dict], min_err: int = MIN_TRACES, min_slow: int = MIN_TRACES) -> Dict:
    """spans: dicts with traceId, spanId, parentSpanId, service, start (s), duration (s), error (bool). Returns per-service counts and
    the services that hold each signal, with a short evidence text."""
    by_trace = defaultdict(list)
    for s in spans:
        if s.get("traceId") and s.get("spanId") and s.get("service"):
            by_trace[s["traceId"]].append(s)
    err_traces, err_origin = 0, Counter()
    groups = defaultdict(list)                          # root operation -> [(trace duration, top self-time service)]
    for tid, ss in by_trace.items():
        ids = {s["spanId"]: s for s in ss}
        kids = defaultdict(list)
        for s in ss:
            p = s.get("parentSpanId")
            if p and p in ids:
                kids[p].append(s)
        if any(s["error"] for s in ss):
            err_traces += 1
            origins = {s["service"] for s in ss if s["error"] and not any(c["error"] for c in kids.get(s["spanId"], []))}
            for svc in origins:
                err_origin[svc] += 1
        self_t = Counter()
        for s in ss:
            st, du = s.get("start") or 0.0, max(s.get("duration") or 0.0, 0.0)
            cover = _union_len([(max(c.get("start") or 0.0, st), min((c.get("start") or 0.0) + max(c.get("duration") or 0.0, 0.0), st + du))
                                for c in kids.get(s["spanId"], []) if (c.get("start") or 0.0) < st + du])
            self_t[s["service"]] += max(du - cover, 0.0)
        roots = [s for s in ss if not (s.get("parentSpanId") and s["parentSpanId"] in ids)]
        if roots and self_t:
            r = max(roots, key=lambda x: x.get("duration") or 0.0)
            groups[(r["service"], r.get("name", ""))].append((r.get("duration") or 0.0, self_t.most_common(1)[0][0]))
    slow, rest = Counter(), Counter()
    n_slow = n_rest = 0
    for g in groups.values():
        if len(g) < MIN_GROUP:
            continue
        ds = sorted(d for d, _ in g)
        thr = ds[min(int(SLOW_Q * len(ds)), len(ds) - 1)]
        for d, svc in g:
            if d >= thr:
                slow[svc] += 1
                n_slow += 1
            else:
                rest[svc] += 1
                n_rest += 1
    out = {"n_traces": len(by_trace), "n_err_traces": err_traces, "n_slow": n_slow, "err_origin": dict(err_origin),
           "slow_origin": dict(slow), "holds": {}}
    for svc, c in err_origin.items():
        if err_traces and c >= min_err and c / err_traces >= ORIGIN_SHARE:
            out["holds"][(svc, "trace_err_origin")] = f"errors start here in {c} of {err_traces} failing traces"
    if n_slow and n_rest:
        for svc, c in slow.items():
            contrast = c / n_slow - rest.get(svc, 0) / n_rest
            if c >= min_slow and contrast >= SLOW_CONTRAST:
                out["holds"][(svc, "trace_slow_origin")] = (f"largest self time in {c} of {n_slow} slowest traces "
                                                            f"({100 * c / n_slow:.0f}% vs {100 * rest.get(svc, 0) / n_rest:.0f}% of the others)")
    return out


def origin_nodes(obs: List[Dict], hosts: Dict) -> set:
    """v2: nodes that host >= NODE_MIN_SERVICES services holding a trace origin."""
    from collections import Counter as _C
    origin_svcs = {o["entity"] for o in obs if o["signal"] in SIGNALS and o["holds"]}
    cnt = _C(n for s_ in origin_svcs for n in set(hosts.get(s_, ())))
    return {n for n, c in cnt.items() if c >= NODE_MIN_SERVICES}


def trace_observations(case, db, v2: bool = False) -> List[Dict]:
    """Window spans of the case -> observations for the services of db (holds True / False for every service with spans)."""
    from pipeline.fault_fingerprint import _owner, _naive
    from data.dataset_profile import TRACE_ERROR_STATUS
    services = list(db["services"])
    w0, w1 = db["w0"], db["w1"]
    spans, seen = [], set()
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts < w0 or ts > w1:
            continue
        svc = _owner(p.get("serviceName"), services)
        if not svc:
            continue
        try:
            du = float(p.get("duration") or 0.0) / 1e9         # nanoseconds -> seconds
        except (TypeError, ValueError):
            du = 0.0
        spans.append({"traceId": p.get("traceId"), "spanId": p.get("spanId"), "parentSpanId": p.get("parentSpanId"), "service": svc,
                      "name": p.get("spanName"), "start": ts.timestamp(), "duration": du, "error": str(p.get("statusCode")) == TRACE_ERROR_STATUS})
        seen.add(svc)
    res = analyse(spans, MIN_ERR_TRACES_V2, MIN_SLOW_TRACES_V2) if v2 else analyse(spans)
    db["_trace_structure"] = {k: v for k, v in res.items() if k != "holds"}
    obs = []
    for svc in sorted(seen):
        for sig in SIGNALS:
            ev = res["holds"].get((svc, sig), "")
            obs.append({"entity": svc, "signal": sig, "holds": bool(ev), "kind": "cat", "ev": ev})
    return obs
