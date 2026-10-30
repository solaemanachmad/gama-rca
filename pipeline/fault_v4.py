"""v4 evidence layer (design: claude/RCA_v4_design.md in the project). Replaces only the SCORE of v3's continuous signals; the empirical
calibration, the categorical observations and the signal names are those of pipeline/fault_calibrated.py.

Why: v3 scored a family by the minimum one-sided Mann-Whitney p over its series. A rank test saturates: any window whose values all
exceed the history gets the minimum p, whether the shift is a small drift or a 5x spike, and the pseudo-window null (cut from the case's
own baseline) contains drifts at that floor. After calibration a strong fault then looks like noise. v4 scores the SIZE of the shift:

  robust shift  z = (median(window) - median(history)) / scale,  scale = max(1.4826 * MAD(history), REL_FLOOR * |median(history)|, ABS_FLOOR)
  direction     'up' uses z, 'down' uses -z; the family score of an entity is the largest directional z over its TESTABLE series.
  testable      the series has >= MIN_COVERAGE of the expected history points and >= half of the expected window points (expected counts
                from the series' own sampling step); event-count families (errors, log keywords) are zero-filled per 30 s bin instead,
                because the exporter writes a sample only when the count is >= 1.
  excluded      configuration constants (JVM init/max bytes, pool sizes, desired replicas) and injector / platform workloads.
Calibration is unchanged in principle: the same score on K_PSEUDO pseudo-windows cut from the baseline, pooled per (entity kind, signal),
calibrated p = (1 + #{null >= z}) / (N + 1) (Efron 2004; conformal calibration). A pool smaller than MIN_POOL (resolution
1/(N+1) above alpha) is replaced by the pool of all signals of that entity kind; robust z is scale-free, so the pooled null is comparable. REL_FLOOR (5% of the baseline level) and ABS_FLOOR are
fixed a priori: a scale floor keeps a perfectly flat history (replicas, restarts) from producing an infinite z for a tiny change.

New signals (v4): svc_request_drop (request count falls: starvation, a silent or unreachable service), svc_avail_drop (deployment
availability / available replicas fall). Uses only the case's telemetry; never reads answer_key/."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, List, Optional, Tuple

from pipeline.fault_calibrated import B_SPAN, K_PSEUDO, LOG_B_SPAN, LOG_K, LOG_BIN_S, NODE_SIGNALS, SVC_SIGNALS, _fmt
from pipeline.fault_fingerprint import _family, _TEXT_FAMILIES

MIN_COVERAGE = 0.8
REL_FLOOR, ABS_FLOOR = 0.05, 1e-6
COUNT_FLOOR = 1.0                  # zero-filled event counts: one event per bin is the smallest meaningful scale
EXCLUDE_NAME = re.compile(r"^(l3reg-|chaosblade|chaos-)", re.I)            # fault-injection workloads (not present in production)
CONSTANT_METRIC = re.compile(r"_(init|max|committed)_bytes$|core_pool_size|max_pool_size|max_thread_count|desired_replicas", re.I)
COUNT_FAMILIES = {"error"}
MIN_POOL = 20                      # smallest null pool whose resolution 1/(N+1) is below alpha = 0.05
V4_SVC_SIGNALS = SVC_SIGNALS + [("svc_request_drop", ("request",), "down"), ("svc_avail_drop", ("avail",), "down")]
OP_SIGNALS = [("op_error_rise", ("error",), "up"), ("op_latency_rise", ("latency",), "up")]   # v4.2: the most affected operation
Z_ON = 3.0               # onset: first sample (from one window before w0) whose robust shift reaches 3 robust SDs, for 2 samples in a row


def family_v4(metric: str) -> Optional[str]:
    m = (metric or "").lower()
    if CONSTANT_METRIC.search(m):
        return None
    if "availab" in m:
        return "avail"
    if m in ("slow", "slow_count"):
        return "latency"
    return _family(m)


def _step(ts: List[float]) -> float:
    d = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    return d[len(d) // 2] if d else float(LOG_BIN_S)


def _scale(base: List[float], floor: float) -> Tuple[float, float]:
    mb = median(base)
    mad = median(abs(v - mb) for v in base)
    return mb, max(1.4826 * mad, REL_FLOOR * abs(mb), floor)


def shift_z(pts: List[Tuple[float, float]], a: float, b: float, span: float, direction: str, counts: bool = False,
            closed_end: bool = False) -> Optional[float]:
    """Directional robust shift of window [a, b) against [a - span, a); None if the series is not testable there. pts: (seconds, value),
    sorted by time."""
    if counts:                                              # zero-filled 30 s bins
        nb, nw = int(round(span / LOG_BIN_S)), max(1, int(round((b - a) / LOG_BIN_S)))
        bins = defaultdict(float)
        for t, v in pts:
            if a - span <= t < b or (closed_end and t == b):
                bins[min(int((t - a) // LOG_BIN_S), nw - 1)] += v
        base = [bins.get(i, 0.0) for i in range(-nb, 0)]
        win = [bins.get(i, 0.0) for i in range(0, nw)]
        floor = COUNT_FLOOR
    else:
        base = [v for t, v in pts if a - span <= t < a]
        win = [v for t, v in pts if a <= t < b or (closed_end and t == b)]
        step = _step([t for t, _ in pts])
        if len(base) < MIN_COVERAGE * span / step or len(win) < 0.5 * (b - a) / step or len(base) < 3 or not win:
            return None
        floor = ABS_FLOOR
    mb, sc = _scale(base, floor)
    z = (median(win) - mb) / sc
    return z if direction == "up" else -z


def _series_ok(name: str) -> bool:
    return not EXCLUDE_NAME.search(name or "")


def _family_z(series_map, fams, direction, windows, span, key_name, key_metric):
    sel = []
    for key, pts in series_map.items():
        if not _series_ok(key_name(key)):
            continue
        f = family_v4(key_metric(key))
        if f in fams:
            sel.append((key, sorted((t.timestamp(), v) for t, v in pts), f))
    out, desc, onset = [], None, None
    for i, (a, b) in enumerate(windows):
        zs = [(shift_z(pts, a.timestamp(), b.timestamp(), span, direction, counts=f in COUNT_FAMILIES, closed_end=(i == 0)), key, pts)
              for key, pts, f in sel]
        zs = [x for x in zs if x[0] is not None]
        out.append(max(x[0] for x in zs) if zs else None)
        if i == 0 and zs:
            z, key, pts = max(zs, key=lambda x: x[0])
            a_, b_ = a.timestamp(), b.timestamp()
            fam = family_v4(key_metric(key))
            onset = onset_time(pts, a_, b_, span, direction, counts=fam in COUNT_FAMILIES)
            base = sorted(v for t, v in pts if a_ - span <= t < a_)
            win = [v for t, v in pts if a_ <= t <= b_]
            if base and win:
                desc = (f"{key_metric(key)}: baseline median={_fmt(base[len(base) // 2])}, window "
                        + (f"max={_fmt(max(win))}" if direction == "up" else f"min={_fmt(min(win))}") + f" (robust shift {z:.2g})")
    return out, len(sel), desc, onset


def calibrate_v4(db, peer: bool = False, robust_null: bool = False, ops_level: bool = False, selfnorm: bool = False) -> Dict:
    """Fills db["_cal"] in the format of fault_calibrated._calibrate (so entity_signals() and the v3 belief code read it unchanged)."""
    w0, w1 = db["w0"], db["w1"]
    L = w1 - w0
    span = (B_SPAN * L).total_seconds()
    wins = [(w0, w1)] + [(w0 - j * L, w0 - (j - 1) * L) for j in range(1, K_PSEUDO + 1)]
    raw = defaultdict(dict)
    svc_key = (lambda k: k.split(":")[1] if k.count(":") >= 2 else k, lambda k: k.split(":")[-1])
    node_key = (lambda k: "", lambda k: k)
    for kind, group, sigs, (kn, km) in (("node", "nodes", NODE_SIGNALS, node_key), ("service", "services", V4_SVC_SIGNALS, svc_key)):
        for ent, pr in db[group].items():
            for name, fams, d in sigs:
                sc, m, desc, on = _family_z(pr["series"], fams, d, wins, span, kn, km)
                if m and sc[0] is not None:
                    raw[ent][name] = {"kind_pool": (kind, name), "zs": list(sc), "n_series": m, "ev": desc, "onset": on}
    op_sigs = [("op_request_rise", ("request",), "up")] + (OP_SIGNALS if ops_level else [])
    for ent, pr in db["services"].items():                 # operation level: hot operation (trafficHotspot); v4.2: failing operation
        ops = {(k, m): v for (k, m), v in pr["ops"].items()}
        for name, fams, d in op_sigs:
            sc, m, desc, on = _family_z(ops, fams, d, wins, span, lambda k: k[0], lambda k: k[1])
            if m and sc[0] is not None:
                raw[ent][name] = {"kind_pool": ("op", name), "zs": list(sc), "n_series": m, "ev": desc, "onset": on}
    nw = db["win_bins"]                                     # log-keyword counts per 30 s bin (zero-filled), as in v3 but scored by size
    for ent, pr in db["services"].items():
        for f, _ in _TEXT_FAMILIES:
            bins = pr["log_bins"].get(f)
            if not bins:
                continue
            binv = lambda lo, hi: [float(bins.get(i, 0)) for i in range(lo, hi)]
            def z_of(win, base):
                mb, sc = _scale(base, COUNT_FLOOR)
                return (median(win) - mb) / sc
            if db["log_base_bins"] < LOG_B_SPAN * nw:
                continue
            real = z_of(binv(0, nw), binv(-LOG_B_SPAN * nw, 0))
            pseudo = [z_of(binv(-j * nw, -(j - 1) * nw), binv(-(j + LOG_B_SPAN) * nw, -j * nw)) for j in range(1, LOG_K + 1)
                      if db["log_base_bins"] >= (j + LOG_B_SPAN) * nw]
            raw[ent][f"log_{f}_rise"] = {"kind_pool": ("log", "log"), "zs": [real] + pseudo, "n_series": 1,
                                         "ev": f"'{f}' log lines in window={int(sum(binv(0, nw)))}, previous window={int(sum(binv(-nw, 0)))}"}
    if selfnorm:
        self_adjust(raw)
    if peer:
        peer_adjust(raw)
    for ent, sigs in raw.items():
        for r in sigs.values():
            r["z"], r["pseudo"] = r["zs"][0], [x for x in r["zs"][1:] if x is not None]
    null, kind_all = defaultdict(list), defaultdict(list)
    for ent, sigs in raw.items():
        for r in sigs.values():
            null[r["kind_pool"]].extend(r["pseudo"])
            kind_all[r["kind_pool"][0]].extend(r["pseudo"])
    cal = defaultdict(dict)
    for ent, sigs in raw.items():
        for name, r in sigs.items():
            nl = null[r["kind_pool"]]
            if len(nl) < MIN_POOL:                          # e.g. JVM series exist for one service only: a pool of 5 pseudo scores
                nl = kind_all[r["kind_pool"][0]]            # cannot reach alpha; robust z is scale-free, so pool over the entity kind
                r["pool"] = "kind"
            r["n_null"] = len(nl)
            r["raw"] = -r["z"]                              # v3 convention: smaller raw = more anomalous
            if robust_null and len(nl) >= MIN_POOL:
                # Efron's empirical null (central matching): the pseudo-window scores may contain earlier injected faults (their
                # tail is contaminated), so the null is the normal fitted to their CENTRE (median, MAD), not their full distribution
                mu = median(nl)
                sd = max(1.4826 * median(abs(x - mu) for x in nl), 1e-9)
                r["p"] = 0.5 * math.erfc((r["z"] - mu) / sd / math.sqrt(2.0))
                r["null"] = "central"
            else:
                r["p"] = (1 + sum(1 for x in nl if x >= r["z"])) / (len(nl) + 1)
            cal[ent][name] = r
    db["_cal"], db["_null"], db["_v4_peer"] = cal, null, peer
    return cal


# ------------------------------------------------------------------------------------------------------------------------------
# v4.1 (DEVLOG_V4.md, "Case timelines"): peer-relative scores, new-event rule, availability as a state
# ------------------------------------------------------------------------------------------------------------------------------
PEER_FLOOR = 1.0          # robust scale floor of the cross-sectional z distribution (one robust SD of a single series)
MIN_PEERS = 3
EVENT_ALPHA = 0.05


def peer_adjust(raw: Dict) -> None:
    """Common-mode rejection: in every window (real and pseudo alike), each entity's z is re-standardised against the z of the other
    entities of the same kind for the same signal: z' = (z - median) / max(1.4826 MAD, PEER_FLOOR). A load step or a shared drift moves
    all entities together and cancels; a fault at one entity stands out. Applied identically to real and pseudo windows, so the
    empirical calibration stays valid."""
    by_pool = defaultdict(list)
    for ent, sigs in raw.items():
        for name, r in sigs.items():
            by_pool[r["kind_pool"]].append(r)
    for pool, rs in by_pool.items():
        n_w = max(len(r["zs"]) for r in rs)
        for i in range(n_w):
            vals = [r["zs"][i] for r in rs if i < len(r["zs"]) and r["zs"][i] is not None]
            if len(vals) < MIN_PEERS:
                continue
            med = median(vals)
            sc = max(1.4826 * median(abs(v - med) for v in vals), PEER_FLOOR)
            for r in rs:
                if i < len(r["zs"]) and r["zs"][i] is not None:
                    r["zs"][i] = (r["zs"][i] - med) / sc


def new_events(case, db, horizon_s: float) -> None:
    """Replaces db's event counts by NEW occurrences only. Kubernetes events are cumulative objects (count, firstTimestamp,
    lastTimestamp, re-exported as they repeat). For each (object, reason): increments inside the window are tested against the
    object's own long-run rate before the window (one-sided Poisson test, EVENT_ALPHA); an event that has repeated at the same rate for
    weeks (e.g. EvictionThresholdMet x247,000) is not evidence of a fault starting now. An object first seen within the history horizon
    has a long-run rate of ~0, so any occurrence is new."""
    import json as _json
    from data.loader import event_time_bounds, _parse_ts_iso
    from pipeline.fault_fingerprint import _owner, _naive, _EVENT_FAMILIES
    from pipeline.log_preprocess import poisson_sf
    w0, w1 = db["w0"], db["w1"]
    services, nodes = list(db["services"]), db["nodes"]
    rows = defaultdict(list)
    for o in case.observations.get("events", []):
        p = o.payload or {}
        try:
            e = _json.loads(p.get("eventId")) if isinstance(p.get("eventId"), str) else (p.get("eventId") or {})
        except Exception:
            continue
        f_, l_ = event_time_bounds(e)
        first, last = _naive(_parse_ts_iso(f_)), _naive(_parse_ts_iso(l_))
        if first is None or last is None or last > w1:
            continue
        io = e.get("involvedObject") or {}
        key = (io.get("kind", ""), io.get("name") or p.get("pod_name") or p.get("hostname") or "", str(e.get("reason", "")))
        rows[key].append((last, int(e.get("count") or 1), first, f"{e.get('reason', '')} {e.get('message', '')}"))
    for pr in list(db["services"].values()) + list(db["nodes"].values()):
        pr["events"] = defaultdict(Counter)
    kept = 0
    for (kind, name, reason), rs in rows.items():
        rs.sort()
        first = min(r[2] for r in rs)
        win = [r for r in rs if r[0] >= w0]
        if not win:
            continue
        before = [r for r in rs if r[0] < w0]
        if before:
            c0, t0 = before[-1][1], before[-1][0]
        elif first >= w0 - _td(horizon_s):
            c0, t0 = 0, None                                 # first seen within the horizon: no occurrence in the history
        else:                                            # only window rows of an old object: the first one is the reference
            c0, t0 = win[0][1], win[0][0]
            win = win[1:]
            if not win:
                continue
        inc = max(r[1] for r in win) - c0
        if inc <= 0:
            continue
        if t0 is None:                                       # rate before the window: 0 occurrences over the history horizon
            age, span = horizon_s, (w1 - w0).total_seconds()
        else:                                                # long-run rate of this object, then the remaining window
            age, span = max((t0 - first).total_seconds(), 60.0), max((w1 - t0).total_seconds(), 1.0)
        lam = (c0 + 0.5) / age * span
        if poisson_sf(inc, lam) > EVENT_ALPHA:
            continue
        prof = nodes.get(name) if (kind == "Node" or name in nodes) else db["services"].get(_owner(name, services) or "")
        if prof is None:
            continue
        txt = win[-1][3]
        for f, rx in _EVENT_FAMILIES:
            if rx.search(txt):
                prof["events"][f][reason] += inc
                kept += 1
    db["_v4_new_events"] = kept


def unavailability(db) -> Dict[str, bool]:
    """Availability as a state (no history needed): a service is unavailable in the window when its deployment's available replicas
    fall below the desired replicas, or its availability rate falls below full, at any window sample; and it was NOT so for more than
    half of its own history samples (a chronic state is not a new fault). Series that exist only inside the window (sparse export)
    still count."""
    w0, w1 = db["w0"], db["w1"]
    out = {}
    for s, pr in db["services"].items():
        ser = {k.split(":")[-1]: sorted(v) for k, v in pr["series"].items() if k.startswith("k8s.deployment:")}
        avail, desired, rate = ser.get("deployment_available_replicas", []), ser.get("deployment_desired_replicas", []), \
            ser.get("deployment_availability_rate", [])
        if not (avail or rate):
            continue

        def bad_at(t):
            res = None
            if avail:
                a = [v for tt, v in avail if tt <= t][-1:] or [avail[0][1]]
                d = [v for tt, v in desired if tt <= t][-1:] or ([desired[0][1]] if desired else [])
                if d:
                    res = a[0] < d[0]
                elif a[0] == 0:
                    res = True
            if rate:
                r = [v for tt, v in rate if tt <= t][-1:] or [rate[0][1]]
                full = 1.0 if max(v for _, v in rate) <= 1.0 else 100.0
                res = bool(res) or r[0] < 0.999 * full
            return bool(res)
        times = sorted({t for t, _ in avail + rate})
        win_bad = any(bad_at(t) for t in times if w0 <= t <= w1)
        hist = [bad_at(t) for t in times if t < w0]
        chronic = bool(hist) and sum(hist) > 0.5 * len(hist)
        out[s] = win_bad and not chronic
    db["_v4_unavail"] = out
    return out


def _td(seconds):
    import datetime as _dt
    return _dt.timedelta(seconds=seconds)


def onset_time(pts, a: float, b: float, span: float, direction: str, counts: bool = False) -> Optional[float]:
    """v4.2: when the series' anomaly starts. History = [a - L - span, a - L) with L = b - a (it ends one window before the evidence
    window, so an anomaly that starts shortly before the alert window is still measured against clean data); the onset is the first
    sample in [a - L, b] whose directional robust shift reaches Z_ON and stays there for the next sample. None if it never does."""
    L = b - a
    h0, h1 = a - L - span, a - L
    if counts:
        bins = defaultdict(float)
        for t, v in pts:
            if h0 <= t <= b:
                bins[int((t - h1) // LOG_BIN_S)] += v
        n_h, n_s = int(round(span / LOG_BIN_S)), int(round(2 * L / LOG_BIN_S))
        hist = [bins.get(i, 0.0) for i in range(-n_h, 0)]
        seq = [(h1 + i * LOG_BIN_S, bins.get(i, 0.0)) for i in range(0, n_s + 1)]
        floor = COUNT_FLOOR
    else:
        hist = [v for t, v in pts if h0 <= t < h1]
        seq = [(t, v) for t, v in pts if h1 <= t <= b]
        floor = ABS_FLOOR
    if len(hist) < 3 or len(seq) < 2:
        return None
    mb, sc = _scale(hist, floor)
    zs = [((v - mb) / sc) * (1 if direction == "up" else -1) for _, v in seq]
    for i in range(len(seq) - 1):
        if zs[i] >= Z_ON and zs[i + 1] >= Z_ON:
            return seq[i][0]
    return None


SELF_FLOOR = 1.0


def self_adjust(raw: Dict) -> None:
    """v4.3 volatile-entity normalisation: each entity's z per signal is re-standardised against the same entity's own pseudo-window z
    (median, MAD, floor 1). An entity that is 'anomalous' in every window of its history (a noisy service) no longer outranks one that
    changed only now. Pseudo windows use leave-one-out statistics (the other pseudo windows), so real and pseudo scores are on the same
    footing for calibration. Needs >= 3 other pseudo scores; otherwise the score is left unchanged."""
    for sigs in raw.values():
        for r in sigs.values():
            zs = r["zs"]
            pseudo = [(i, z) for i, z in enumerate(zs) if i > 0 and z is not None]
            new = list(zs)
            for i, z in enumerate(zs):
                if z is None:
                    continue
                ref = [v for j, v in pseudo if j != i]
                if len(ref) < 3:
                    continue
                med = median(ref)
                sc = max(1.4826 * median(abs(v - med) for v in ref), SELF_FLOOR)
                new[i] = (z - med) / sc
            r["zs"] = new
