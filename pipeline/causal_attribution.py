"""Causal anomaly attribution for entity localization (v2, flag V2_CAUSAL; prototype).

Idea: the topology gives the causal structure (a caller depends on its callees; a node hosts services), and each case has ~55 min of
normal metrics before the alert window. Per case and without labels we fit, on that baseline, a linear structural model of every
service's latency and error count on its parents (callees' latency / errors, the CPU of its hosting nodes, its own request count).
In the alert window the RESIDUAL of an entity is the part of its anomaly that its parents do not explain: a victim that is only
dragged along has a normal residual, the origin has an abnormal one. Residual scores are calibrated on pseudo-windows of the same
case (empirical null, as in fault_calibrated.py). Because the model is linear, the deviation of the alerted service decomposes
additively into contributions of every entity's own residual times its total causal effect (sum over paths of edge coefficients);
for a linear structural model this additive decomposition coincides with the Shapley attribution.

Uses only the case's telemetry, topology and alert; no labels, no training across cases. Parameters are fixed constants below.
"""
from __future__ import annotations

import math

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.fault_fingerprint import _family

STEP_S = 30          # metric sampling step of RCA100 (30 s); series are binned to it
K_PSEUDO = 5         # pseudo-windows for the empirical null
MIN_FIT = 20         # fewer complete baseline bins -> intercept-only model (the variable is still scored, without explaining away)
RIDGE = 1e-3         # tiny ridge on standardized parents (numerical stability only)
MAX_PATH = 8
ALPHA = 0.05


def _bins(pts, w0):
    acc = defaultdict(list)
    for t, v in pts:
        acc[int((t - w0).total_seconds() // STEP_S)].append(v)
    return {b: float(np.mean(v)) for b, v in acc.items()}


def build_variables(db) -> Tuple[Dict, Dict]:
    """Variables {(entity, kind): {bin: value}} for kind in latency / error / request (services) and cpu (nodes), plus the metric
    name chosen for each (the series of that family with most samples)."""
    w0 = db["w0"]
    var, names = {}, {}
    for s, pr in db["services"].items():
        best = {}
        for key, pts in pr["series"].items():
            eset, _, rest = key.partition(":")
            if eset != "apm.service.legacy":
                continue
            fam = _family(rest.split(":")[-1])
            if fam in ("latency", "error", "request") and (fam not in best or len(pts) > len(best[fam][1])):
                best[fam] = (key, pts)
        for fam, (key, pts) in best.items():
            var[(s, fam)] = _bins(pts, w0)
            names[(s, fam)] = key.split(":")[-1]
    for n, pr in db["nodes"].items():
        cands = [(k, p) for k, p in pr["series"].items() if _family(k) == "cpu"]
        if cands:
            key, pts = max(cands, key=lambda kp: len(kp[1]))
            var[(n, "cpu")] = _bins(pts, w0)
            names[(n, "cpu")] = key
    return var, names


def parents_of(target, var, callees, hosts):
    ent, kind = target
    ps = []
    if kind == "latency":
        ps = [(c, "latency") for c in callees.get(ent, ())] + [(n, "cpu") for n in hosts.get(ent, ())] + [(ent, "request")]
    elif kind == "error":
        ps = [(c, "error") for c in callees.get(ent, ())] + [(ent, "request")]
    return [p for p in ps if p in var and p != target]


def _fit(y, X):
    """Least squares on standardized columns with a tiny ridge. Returns a predictor giving, for the rows of a window, the predictions
    and the standard error of the window's MEAN prediction error, sigma * sqrt(1/m + h_bar) with h_bar = z_bar' A^-1 z_bar at the
    window's mean parent values: the parameter part does not shrink with the window length, so a window whose parents lie far
    outside the baseline range (extrapolation) is not mistaken for an unexplained anomaly."""
    n, k = X.shape
    if k == 0 or n < MIN_FIT:
        m = float(np.mean(y)) if n else 0.0
        sig = float(np.std(y)) if n > 1 else 1.0
        def pred(Xn):
            return np.full(len(Xn), m), max(sig, 1e-9) * math.sqrt(1.0 / max(len(Xn), 1) + 1.0 / max(n, 1))
        return pred, np.zeros(k)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((n, 1)), (X - mu) / sd])
    A = Z.T @ Z + RIDGE * n * np.diag([0.0] + [1.0] * k)
    Ainv = np.linalg.inv(A)
    b = Ainv @ (Z.T @ y)
    resid = y - Z @ b
    sig = max(float(np.sqrt(np.sum(resid ** 2) / max(n - k - 1, 1))), 1e-9)
    def pred(Xn):
        Zn = np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd])
        zb = Zn.mean(0)
        return Zn @ b, sig * math.sqrt(1.0 / max(len(Xn), 1) + float(zb @ Ainv @ zb))
    return pred, b[1:] / sd


def _rows(target, parents, var, bins):
    yv = var[target]
    rows, ys, used = [], [], []
    for b in bins:
        if b in yv and all(b in var[p] for p in parents):
            rows.append([var[p][b] for p in parents])
            ys.append(yv[b])
            used.append(b)
    return np.array(ys, float), np.array(rows, float).reshape(len(ys), len(parents)), used


def score_variable(target, parents, var, nw):
    """Scores of the real window (index 0) and K pseudo-windows, each fitted on the baseline WITHOUT the scored window.
    Score = mean residual of the window / standard error of that mean (a z-score; upward deviation positive)."""
    yv = var[target]
    base_bins = sorted(b for b in yv if b < 0)
    wins = [(0, nw)] + [(-j * nw, -(j - 1) * nw) for j in range(1, K_PSEUDO + 1)]
    scores, coef, resid0 = [], None, None
    for i, (a, b) in enumerate(wins):
        y, X, _ = _rows(target, parents, var, [x for x in base_bins if not (a <= x < b)])
        pred, beta = _fit(y, X)
        yw, Xw, _ = _rows(target, parents, var, range(a, b))
        if i == 0:
            coef = dict(zip(parents, beta))
        if len(yw) == 0:
            scores.append(None)
            continue
        mu, se = pred(Xw)
        if i == 0:
            resid0 = float(np.mean(yw - mu))
        scores.append(float(np.mean(yw - mu)) / se)
    return scores, coef, resid0


def attribute(db, callees: Dict[str, List[str]], hosts: Dict[str, List[str]], alert_svc: Optional[str], alpha: float = ALPHA) -> Dict:
    var, names = build_variables(db)
    nw = db.get("win_bins") or 12
    targets = [k for k in var if k[1] in ("latency", "error", "cpu")]
    res = {}
    for t in targets:
        ps = parents_of(t, var, callees, hosts)
        own, coef, r0 = score_variable(t, ps, var, nw)
        raw, _, _ = score_variable(t, [], var, nw)
        res[t] = {"parents": ps, "coef": coef or {}, "own": own, "raw": raw, "resid": r0 or 0.0}
    # empirical null per variable kind, pooled over the entities of the case (pseudo-windows only)
    null = defaultdict(list)
    for t, r in res.items():
        for key in ("own", "raw"):
            null[(t[1], key)].extend(x for x in r[key][1:] if x is not None)
    def cal(t, key):
        x = res[t][key][0]
        nl = null[(t[1], key)]
        return None if x is None or not nl else (1 + sum(1 for y in nl if y >= x)) / (len(nl) + 1)
    for t, r in res.items():
        r["p_own"], r["p_raw"] = cal(t, "own"), cal(t, "raw")
    # alert target: the alerted service's latency or error, whichever deviates more (raw score); no alert -> most deviating service var
    cand = [(alert_svc, k) for k in ("latency", "error") if (alert_svc, k) in res] if alert_svc else []
    if not cand:
        cand = [t for t in res if t[1] in ("latency", "error")]
    cand = [t for t in cand if res[t]["raw"][0] is not None]
    if not cand:
        return {"vars": res, "names": names, "target": None, "ranking": []}
    target = max(cand, key=lambda t: res[t]["raw"][0])
    # total effect of every variable on the target (sum over parent->child paths of coefficient products)
    children = defaultdict(list)
    for t, r in res.items():
        for p, c in r["coef"].items():
            children[p].append((t, c))
    def total_effect(u):
        tot, stack = 0.0, [(u, 1.0, {u})]
        while stack:
            node, w, seen = stack.pop()
            if node == target:
                tot += w
                continue
            if len(seen) > MAX_PATH:
                continue
            for ch, c in children.get(node, ()):
                if ch not in seen:
                    stack.append((ch, w * c, seen | {ch}))
        return tot
    contrib = defaultdict(float)
    detail = defaultdict(list)
    for t, r in res.items():
        te = total_effect(t)
        if te == 0.0:
            continue
        c = r["resid"] * te
        contrib[t[0]] += c
        detail[t[0]].append({"var": t[1], "metric": names.get(t), "effect": te, "contribution": c, "p_own": r["p_own"]})
    pos = {e: c for e, c in contrib.items() if c > 0}
    total = sum(pos.values()) or 1.0
    ranking = []
    for e, c in sorted(pos.items(), key=lambda kv: -kv[1]):
        p_own = min((d["p_own"] for d in detail[e] if d["p_own"] is not None), default=None)
        ranking.append({"entity": e, "share": c / total, "p_own": p_own, "own_anomalous": p_own is not None and p_own <= alpha,
                        "detail": detail[e]})
    return {"vars": res, "names": names, "target": target, "ranking": ranking}
