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


_UNREACHABLE = None


def _norm(x: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def callee_of(span: Dict, services: List[str]) -> str:
    """v3 ('torigin3'): the service a CLIENT span calls, read from its name and attributes (gRPC 'pkg.ShippingService/GetQuote',
    server.address / peer.service / url.full ...): the longest known service name (normalised, 'service' suffix dropped) that occurs,
    other than the span's own service. '' if none."""
    text = _norm(span.get("name")) + " " + _norm(span.get("attrs"))
    best = ""
    for sv in services:
        key = _norm(sv).replace("service", "")
        if sv != span["service"] and len(key) >= 3 and key in text and len(key) > len(_norm(best)):
            best = sv
    return best


def named_target(msg: str, services: List[str], own: str) -> str:
    """v5 'msgtarget' (amendment 16): the service a failure message names as the endpoint that could not be reached -- the host of a
    URL ('http://quote:8080/getquote'), of 'dial tcp <host>:' or of a DNS lookup -- when that host is a known service other than the
    span's own (first DNS label; 'quote.default.svc.cluster.local' -> quote). '' if none (IP addresses name no service)."""
    import re
    hosts = re.findall(r"[a-z][a-z0-9+.-]*://([A-Za-z0-9._-]+)|dial tcp ([A-Za-z0-9._-]+):|lookup ([A-Za-z0-9._-]+)", msg or "")
    byname = {_norm(sv): sv for sv in services}
    for groups in hosts:
        for h in groups:
            sv = byname.get(_norm(h.split(".")[0])) if h else None
            if sv and sv != own:
                return sv
    return ""


def unreachable_kind(span: Dict) -> str:
    """'unreachable' if the failure message says the callee could not be reached (refused / unavailable / no such host / no
    endpoints), 'timeout' for timeouts, '' otherwise."""
    import re
    global _UNREACHABLE
    _UNREACHABLE = _UNREACHABLE or (re.compile(r"connection refused|econnrefused|unavailable|no such host|could not resolve|"
                                               r"name resolution|no healthy upstream|no endpoints|connection reset|"
                                               r"upstream connect error", re.I),
                                    re.compile(r"timeout|timed out|deadline exceeded", re.I))
    m = span.get("msg") or ""
    return "unreachable" if _UNREACHABLE[0].search(m) else ("timeout" if _UNREACHABLE[1].search(m) else "")


def analyse(spans: List[Dict], min_err: int = MIN_TRACES, min_slow: int = MIN_TRACES, services: List[str] = None,
            msgtarget: bool = False) -> Dict:
    """spans: dicts with traceId, spanId, parentSpanId, service, start (s), duration (s), error (bool). Returns per-service counts and
    the services that hold each signal, with a short evidence text."""
    by_trace = defaultdict(list)
    for s in spans:
        if s.get("traceId") and s.get("spanId") and s.get("service"):
            by_trace[s["traceId"]].append(s)
    err_traces, err_origin = 0, Counter()
    origin_spans = defaultdict(list)                    # service -> origin (deepest failing) spans, for 'omsg'
    origin_traces = defaultdict(set)                    # service -> traceIds where it is an origin
    relayed = Counter()                                  # v3: origins moved from a caller to the callee it could not reach
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
            deepest = [s for s in ss if s["error"] and not any(c["error"] for c in kids.get(s["spanId"], []))]
            owners = []
            for s in deepest:
                own = s["service"]
                if services is not None:                  # v3: a caller that could not reach / waited out its callee is not the origin
                    kind, ch = unreachable_kind(s), kids.get(s["spanId"], [])
                    named = named_target(s.get("msg"), services, s["service"]) if msgtarget and kind else ""
                    if named:                             # v5 'msgtarget': the message names the endpoint that failed
                        own = named
                    elif kind == "unreachable" and not ch:
                        own = callee_of(s, services) or own
                    elif kind == "timeout":
                        if ch:
                            own = max(ch, key=lambda c: c.get("duration") or 0.0)["service"]
                        else:
                            own = callee_of(s, services) or own
                    if own != s["service"]:
                        relayed[own] += 1
                owners.append((own, s))
            for svc in {o for o, _ in owners}:
                err_origin[svc] += 1
                origin_traces[svc].add(tid)
            for own, s in owners:
                origin_spans[own].append(s)
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
           "slow_origin": dict(slow), "holds": {}, "origin_spans": origin_spans, "origin_traces": origin_traces,
           "relayed_to_callee": dict(relayed)}
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


OMSG_SHARE = 0.3          # 'omsg': a category holds at an origin service if >= 30% of its origin spans / linked log lines carry it
_HEX32 = None


def origin_messages(res: Dict, logs=None) -> Dict:
    """v5 'omsg' (amendment 6): WHAT fails, read only where the failure starts. For every service that is an error origin, the status
    message and HTTP status code of its ORIGIN spans (deepest failing spans) and, when log lines carry the trace id of one of its origin
    traces, those log lines. Messages of callers that merely relay a callee's failure ('upstream failed: rate limit') are not counted
    at the caller. Returns {(service, signal): evidence} for omsg_<category> and ohttp<code> (429, 403, 503, 5xx)."""
    import re
    from pipeline.calibration.fingerprint import _TEXT_FAMILIES, _HTTP_RE
    global _HEX32
    _HEX32 = _HEX32 or re.compile(r"\b[0-9a-f]{32}\b", re.I)
    out = {}
    origins = {sv for (sv, sig) in res["holds"] if sig == "trace_err_origin"}
    tid2svc = {str(t).lower(): sv for sv in origins for t in res["origin_traces"].get(sv, ())}
    linked = defaultdict(list)
    for svc, line in (logs or []):
        for t in _HEX32.findall(line or ""):
            if tid2svc.get(t.lower()) == svc:
                linked[svc].append(line)
                break
    for svc in origins:
        spans = res["origin_spans"].get(svc, [])
        texts = [s.get("msg") or "" for s in spans] + linked.get(svc, [])
        n = max(len(texts), 1)
        cat = Counter()
        for t in texts:
            for f, rx in _TEXT_FAMILIES:
                if f != "error" and t and rx.search(t):
                    cat[f] += 1
        for f, c in cat.items():
            if c / n >= OMSG_SHARE:
                out[(svc, f"omsg_{f}")] = f"{c} of {n} origin spans / linked log lines mention '{f}'"
        codes = Counter()
        for s in spans:
            m = _HTTP_RE.search(s.get("attrs") or "")
            if m:
                c = m.group(1)
                codes[c] += 1
                if c.startswith("5"):
                    codes["5xx"] += 1
        for c in ("429", "403", "503", "5xx"):
            if len(spans) and codes[c] / len(spans) >= OMSG_SHARE:
                out[(svc, f"ohttp{c}")] = f"HTTP {c} in {codes[c]} of {len(spans)} origin spans"
    return out


def origin_nodes(obs: List[Dict], hosts: Dict) -> set:
    """v2: nodes that host >= NODE_MIN_SERVICES services holding a trace origin."""
    from collections import Counter as _C
    origin_svcs = {o["entity"] for o in obs if o["signal"] in SIGNALS and o["holds"]}
    cnt = _C(n for s_ in origin_svcs for n in set(hosts.get(s_, ())))
    return {n for n, c in cnt.items() if c >= NODE_MIN_SERVICES}


OMSG_SIGNALS = tuple([f"omsg_{f}" for f in ("npe", "ratelimit", "oom", "gc", "thread", "timeout", "refused", "redis", "dns", "sql",
                                                "queue", "cache", "lb", "netpol")] + ["ohttp429", "ohttp403", "ohttp503", "ohttp5xx"])


def trace_observations(case, db, v2: bool = False, omsg: bool = False, v3: bool = False, msgtarget: bool = False) -> List[Dict]:
    """Window spans of the case -> observations for the services of db (holds True / False for every service with spans)."""
    from pipeline.calibration.fingerprint import _owner, _naive
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
                      "name": p.get("spanName"), "start": ts.timestamp(), "duration": du, "error": str(p.get("statusCode")) == TRACE_ERROR_STATUS,
                      "msg": str(p.get("statusMessage") or ""), "attrs": p.get("attributes") if isinstance(p.get("attributes"), str) else ""})
        seen.add(svc)
    if v3:
        res = analyse(spans, MIN_ERR_TRACES_V2, MIN_SLOW_TRACES_V2, services=services, msgtarget=msgtarget)
    elif v2:
        res = analyse(spans, MIN_ERR_TRACES_V2, MIN_SLOW_TRACES_V2)
    else:
        res = analyse(spans)
    om = {}
    if omsg:
        logs = []
        for o in case.observations.get("logs", []):
            p = o.payload or {}
            ts = _naive(o.timestamp)
            if ts is None or ts < w0 or ts > w1:
                continue
            svc = _owner(p.get("_container_name_"), services)
            if svc:
                logs.append((svc, str(p.get("content") or "")))
        om = origin_messages(res, logs)
    db["_trace_structure"] = {k: v for k, v in res.items() if k not in ("holds", "origin_spans", "origin_traces")}
    db["_trace_structure"]["origin_messages"] = {f"{k[0]}:{k[1]}": v for k, v in om.items()}
    obs = []
    if msgtarget:                                        # a service that could not be reached may have no spans of its own at all
        seen |= {sv for sv, _ in res["holds"]} | {sv for sv, _ in om}
    for svc in sorted(seen):
        for sig in SIGNALS:
            ev = res["holds"].get((svc, sig), "")
            obs.append({"entity": svc, "signal": sig, "holds": bool(ev), "kind": "cat", "ev": ev})
        if omsg:
            for sig in OMSG_SIGNALS:
                ev = om.get((svc, sig), "")
                obs.append({"entity": svc, "signal": sig, "holds": bool(ev), "kind": "cat", "ev": ev})
    return obs
