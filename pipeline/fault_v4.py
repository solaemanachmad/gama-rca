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

import re
from collections import defaultdict
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
    out, desc = [], None
    for i, (a, b) in enumerate(windows):
        zs = [(shift_z(pts, a.timestamp(), b.timestamp(), span, direction, counts=f in COUNT_FAMILIES, closed_end=(i == 0)), key, pts)
              for key, pts, f in sel]
        zs = [x for x in zs if x[0] is not None]
        out.append(max(x[0] for x in zs) if zs else None)
        if i == 0 and zs:
            z, key, pts = max(zs, key=lambda x: x[0])
            a_, b_ = a.timestamp(), b.timestamp()
            base = sorted(v for t, v in pts if a_ - span <= t < a_)
            win = [v for t, v in pts if a_ <= t <= b_]
            if base and win:
                desc = (f"{key_metric(key)}: baseline median={_fmt(base[len(base) // 2])}, window "
                        + (f"max={_fmt(max(win))}" if direction == "up" else f"min={_fmt(min(win))}") + f" (robust shift {z:.2g})")
    return out, len(sel), desc


def calibrate_v4(db) -> Dict:
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
                sc, m, desc = _family_z(pr["series"], fams, d, wins, span, kn, km)
                if m and sc[0] is not None:
                    raw[ent][name] = {"kind_pool": (kind, name), "z": sc[0], "pseudo": [x for x in sc[1:] if x is not None],
                                      "n_series": m, "ev": desc}
    for ent, pr in db["services"].items():                 # hot operation (trafficHotspot)
        ops = {(k, m): v for (k, m), v in pr["ops"].items()}
        sc, m, desc = _family_z(ops, ("request",), "up", wins, span, lambda k: k[0], lambda k: k[1])
        if m and sc[0] is not None:
            raw[ent]["op_request_rise"] = {"kind_pool": ("op", "op_request_rise"), "z": sc[0], "pseudo": [x for x in sc[1:] if x is not None],
                                           "n_series": m, "ev": desc}
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
            raw[ent][f"log_{f}_rise"] = {"kind_pool": ("log", "log"), "z": real, "pseudo": pseudo, "n_series": 1,
                                         "ev": f"'{f}' log lines in window={int(sum(binv(0, nw)))}, previous window={int(sum(binv(-nw, 0)))}"}
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
            r["p"] = (1 + sum(1 for x in nl if x >= r["z"])) / (len(nl) + 1)
            cal[ent][name] = r
    db["_cal"], db["_null"] = cal, null
    return cal
