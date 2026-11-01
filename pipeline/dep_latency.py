"""v5 dependency latency against the service's own history (part 'depbase', PREREGISTRATION_V5.md amendment 10). Label-free and
dataset-agnostic: it uses only OpenTelemetry span conventions (db.system / messaging.system attributes, client span names, and the
instrumentation scope of connection pools).

Before: <dep>_slow compared the window's dependency spans with the service's OTHER window spans. A database / cache / queue call is
much shorter than the server span that contains it, so the test rarely fires even when every query is 10x slower, and it measures
nothing about change. Now, per service and dependency kind (db, redis, queue):

  <dep>_slow holds  iff  window durations > history durations (one-sided Mann-Whitney, p <= ALPHA)
                    and  the window's slow tail exceeds the history's by >= SLOW_CONTRAST:  mean of slowest 10% (win) >= 1.2 x (history)

History = the same kind of span of the same service in the B_SPAN window lengths before the window (as for the metric tests). The
effect-size condition is needed because span counts are large (10^4): a rank test alone declares a 1% shift significant. Waiting for
a pooled connection (HikariCP / c3p0 / DBCP / Druid getConnection) is counted as a database call: when the database or the network to
it is slow, the pool runs dry and the wait grows. Constants: ALPHA, B_SPAN (metric tests) and SLOW_CONTRAST (trace_structure), no new.
Absence of dependency spans -> no observation (a service without such spans says nothing)."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from pipeline.fault_calibrated import mwu_greater, ALPHA, B_SPAN
from pipeline.fault_fingerprint import _DB_RE, _REDIS_RE, _QUEUE_RE
from pipeline.trace_structure import SLOW_CONTRAST

POOL_RE = re.compile(r'hikari|c3p0|dbcp|druid|getConnection', re.I)
KINDS = ("db", "redis", "queue")


def dep_kind(span: str, attrs: str) -> str:
    """The rule of fault_calibrated._compute, plus connection-pool spans as database calls."""
    if _DB_RE.search(span) or ('"db.system"' in attrs and not _REDIS_RE.search(attrs)) or POOL_RE.search(span) or POOL_RE.search(attrs):
        return "db"
    if _REDIS_RE.search(span) or _REDIS_RE.search(attrs):
        return "redis"
    if _QUEUE_RE.search(span) or _QUEUE_RE.search(attrs):
        return "queue"
    return "other"


def tail90(v: List[float]) -> float:
    """Mean of the slowest 10% (at least one value): smoother than the 0.9 quantile on durations recorded in 0.1 ms steps."""
    s = sorted(v, reverse=True)
    top = s[:max(1, len(s) // 10)]
    return sum(top) / len(top)


def slow_test(win: List[float], hist: List[float]):
    """(holds, p, tail_win, tail_hist) or None when either sample is too small for the rank test."""
    p = mwu_greater(win, hist)
    if p is None:
        return None
    qw, qh = tail90(win), tail90(hist)
    return p <= ALPHA and qw >= (1.0 + SLOW_CONTRAST) * qh, p, qw, qh


def share_rise(dep_win: float, entry_win: float, dep_hist: float, entry_hist: float):
    """v5 'depshare': (holds, share_win, share_hist). The share of the service's request time (sum of its entry spans) spent in
    calls of this dependency kind must rise by >= SLOW_CONTRAST. A slow dependency moves the extra time INTO the calls; a starved
    service (CPU, GC, threads) also stretches its calls a little but spends the extra time in itself, so the share does not rise."""
    if entry_win <= 0 or entry_hist <= 0:
        return None
    sw, sh = dep_win / entry_win, dep_hist / entry_hist
    return sw >= (1.0 + SLOW_CONTRAST) * sh, sw, sh


def dep_observations(case, db, share: bool = False) -> List[Dict]:
    from pipeline.fault_fingerprint import _owner, _naive
    services = list(db["services"])
    w0, w1 = db["w0"], db["w1"]
    h0 = w0 - B_SPAN * (w1 - w0)
    dur = defaultdict(lambda: ([], []))                     # (svc, kind) -> (history ms, window ms)
    plain = []                                              # 'depshare': (svc, phase, spanId, parentId, ms) of non-dependency spans
    ids = defaultdict(set)                                  # svc -> its span ids
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts < h0 or ts > w1:
            continue
        attrs = p.get("attributes") if isinstance(p.get("attributes"), str) else ""
        k = dep_kind(str(p.get("spanName") or ""), attrs)
        s = _owner(p.get("serviceName"), services)
        if not s:
            continue
        try:
            ms = float(p.get("duration")) / 1e6
        except (TypeError, ValueError):
            continue
        ph = 1 if ts >= w0 else 0
        if k == "other":
            if share:
                sid = p.get("spanId")
                ids[s].add(sid)
                plain.append((s, ph, p.get("parentSpanId"), ms))
            continue
        dur[(s, k)][ph].append(ms)
    entry = defaultdict(lambda: [0.0, 0.0])                 # svc -> request time (entry spans: parent not a span of the same service)
    for s, ph, par, ms in plain:
        if not par or par not in ids[s]:
            entry[s][ph] += ms
    obs = []
    for (s, k), (hist, win) in sorted(dur.items()):
        r = slow_test(win, hist)
        if r is None:
            continue
        holds, p, qw, qh = r
        ev = (f"{k} spans: slowest-10% mean {qw:.3g}ms in the window vs {qh:.3g}ms before it ({len(win)} vs {len(hist)} spans, "
              f"rank test p={p:.3g})")
        if share:
            sr = share_rise(sum(win), entry[s][1], sum(hist), entry[s][0])
            if sr is None:
                holds = False
                ev += "; no request spans to compare the time share"
            else:
                holds = holds and sr[0]
                ev += f"; share of request time in {k} calls {sr[1]:.0%} in the window vs {sr[2]:.0%} before"
        obs.append({"entity": s, "signal": f"{k}_slow", "holds": bool(holds), "kind": "cont", "p": p, "ev": ev})
    return obs
