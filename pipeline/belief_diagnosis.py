"""v3: evidence-seeking diagnosis with computed belief states (design: claude/RCA_v3_design_belief_eig.md in the project).

The belief is a probability distribution over hypotheses h = (entity, fault type) plus h0 ("none of the modelled hypotheses"), updated
by Bayes' rule from observations. The LLM never writes the belief: it is used as a TEXT SENSOR (semantic probes: map an entity's error
messages / log lines / event messages onto a fixed category set) and as the NARRATOR of the final chain. Which probe to spend an LLM call
on is chosen by expected information gain (EIG); the loop stops when one hypothesis has posterior >= STOP_P, when no probe is worth
EIG_MIN bits, or when the probe budget is spent. Probes are never repeated, so the loop cannot trap.

Likelihood model (fixed a priori from the fingerprints in pipeline/fault_calibrated.py; no label is used):
  observation (entity e, signal s, holds?)  under  h = (e*, tau):
    e == e*:  s key of tau -> q (the key expressions are ALTERNATIVES: with n testable keys, each holds with q = 1-(1-THETA_K)^(1/n),
              so that P(at least one key holds | h) = THETA_K), supporting -> THETA_U, excluding -> ALPHA_C, unrelated -> background(s)
    e != e*:  symptom expected (latency/error/5xx at a service whose callees or hosting nodes include e*;
              node CPU at the hosts of e* for load-type faults)                      -> THETA_P, else background(s)
  e == e* and s is a symptom (latency / error rise, error spans, 5xx) not named by the fingerprint -> THETA_P (the root shows symptoms too)
  the ALERT is an observation: P(alert at service A | h) = THETA_P if e* is A or can explain A's symptoms (Expl(A)), else ALPHA_ALERT.
  background(s): calibrated continuous tests -> ALPHA (their measured false-alarm rate); categorical observations -> share of entities
  of the same level showing s in this case, shrunk towards ALPHA with BG_PRIOR_WEIGHT pseudo-entities (few entities have spans).
  Observations are conditionally independent given h (naive Bayes; stated assumption).
Testability: a hypothesis enters the space only if at least one of its key expressions is observable for its entity in this case (all
atoms of the expression were measured); a type whose key evidence cannot be observed cannot be confirmed or refuted, and would otherwise
collect only generic supporting evidence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from pipeline.fault_calibrated import FINGERPRINTS, LEVEL, RESIDUAL, entity_signals, _atoms

THETA_K, THETA_U, THETA_P, ALPHA_C, ALPHA = 0.8, 0.6, 0.6, 0.05, 0.05
STOP_P, EIG_MIN, PROBE_BUDGET = 0.8, 0.05, 5
H0_PRIOR = 0.1
RESIDUAL_PRIOR = 0.5   # httpError5xx / codeDefect are defined as fallbacks ("no more specific mechanism"): half the prior weight
BG_PRIOR_WEIGHT = 10   # background rate of a categorical observation: Beta prior with mean ALPHA and 10 pseudo-entities
SYMPTOMS = {"svc_latency_rise", "svc_error_rise", "error_spans", "http5xx", "http503"}   # propagated to callers
OWN_SYMPTOMS = {"svc_latency_rise", "svc_error_rise", "error_spans"}                    # expected at the root itself (not a status code:
                                                                                       # a rate-limited service answers 429, not 5xx)
ALPHA_ALERT = 0.01     # P(alert fires on A | the fault cannot reach A): alerts of the benchmark are real, false alerts are rare
LOAD_TYPES = {"F006-trafficSurge", "F004-trafficHotspot", "F034-cpuFullLoad", "F012-cpuDeadLoop"}
SVC_CAT = (["ev_oom", "ev_crashloop", "ev_unhealthy", "ev_unschedulable", "ev_scale_down", "ev_restart", "ev_evicted"],   # always checkable
           ["http429", "http403", "http503", "http5xx", "error_spans"]
           + [f"msg_{f}" for f in ("npe", "ratelimit", "dns", "refused", "timeout", "redis", "oom", "gc", "thread", "queue", "lb",
                                   "netpol", "sql", "cache")])                                                         # need spans
NODE_CAT = ["ev_node_notready", "ev_node_oom", "ev_evicted"]
CATS = ["npe", "ratelimit", "dns", "refused", "timeout", "redis", "oom", "gc", "thread", "queue", "lb", "netpol", "sql", "cache"]


# ------------------------------------------------------------------------------------------------------------------------------
# hypotheses, roles, topology
# ------------------------------------------------------------------------------------------------------------------------------
def roles_of(tau):
    fp = FINGERPRINTS[tau]
    r = {}
    for role, key in (("C", "con"), ("U", "sup"), ("K", "key")):          # key last: wins if an atom appears twice
        for expr in fp[key]:
            for a in _atoms(expr):
                r[a] = role
    return r


ROLES = {t: roles_of(t) for t in FINGERPRINTS}


def hypotheses(db) -> List[Dict]:
    H = []
    for ent in db["services"]:
        H += [{"entity": ent, "level": "service", "type": t} for t in FINGERPRINTS if LEVEL[t] in ("service", "any")]
    for ent in db["nodes"]:
        H += [{"entity": ent, "level": "node", "type": t} for t in FINGERPRINTS if LEVEL[t] in ("node", "any")]
    return H + [{"entity": None, "level": None, "type": None}]                # h0, always last


def key_exprs(tau):
    return FINGERPRINTS[tau]["key"]


def testable(H, obs) -> List[Dict]:
    seen = defaultdict(set)
    for o in obs:
        seen[o["entity"]].add(o["signal"])
    out = []
    for h in H:
        if h["type"] is None:
            out.append(h)
            continue
        n = sum(1 for k in key_exprs(h["type"]) if all(a in seen[h["entity"]] for a in _atoms(k)))
        if n:
            out.append({**h, "q": 1.0 - (1.0 - THETA_K) ** (1.0 / n), "n_keys": n})
    return out


def explainers(db, callees: Dict[str, List[str]]) -> Dict[str, set]:
    """Expl(e) = entities whose fault can show as a symptom at service e: its downstream services (callee closure) and the nodes
    hosting e or any of them."""
    g = nx.DiGraph()
    for a, bs in callees.items():
        for b in bs:
            g.add_edge(a, b)
    hosts = db.get("hosts", {})
    out = {}
    for s in db["services"]:
        down = (nx.descendants(g, s) if s in g else set()) - {s}
        nodes = set(hosts.get(s, []))
        for d in down:
            nodes |= set(hosts.get(d, []))
        out[s] = down | nodes
    return out


# ------------------------------------------------------------------------------------------------------------------------------
# observations from the calibrated evidence (all CPU, cost 0)
# ------------------------------------------------------------------------------------------------------------------------------
def observations(db, alpha=ALPHA) -> List[Dict]:
    obs = []
    for ent in list(db["services"]) + list(db["nodes"]):
        node = ent in db["nodes"]
        sig = entity_signals(db, ent)
        for s, v in sig.items():
            if v["kind"] == "cont" and v.get("p") is not None:
                obs.append({"entity": ent, "signal": s, "holds": v["p"] <= alpha, "kind": "cont", "ev": v["ev"]})
        if node:
            cats = NODE_CAT
        else:
            pr = db["services"][ent]
            cats = SVC_CAT[0] + (SVC_CAT[1] if pr.get("n_spans") else [])
            cats += [f"{k}_err" for k in ("db", "redis", "queue") if pr["dur"].get(k)]
        for s in cats:
            obs.append({"entity": ent, "signal": s, "holds": s in sig, "kind": "cat", "ev": sig[s]["ev"] if s in sig else ""})
    return obs


def background(obs, db) -> Dict:
    """P(s holds | no fault at its entity): ALPHA for calibrated tests; for categorical observations the Laplace-smoothed share of
    entities of the same level showing it in this case, shrunk towards ALPHA (Beta prior, BG_PRIOR_WEIGHT pseudo-entities)."""
    cnt = defaultdict(lambda: [0, 0])
    for o in obs:
        if o["kind"] == "cat":
            c = cnt[(o["signal"], o["entity"] in db["nodes"])]
            c[0] += o["holds"]
            c[1] += 1
    return {k: (h + BG_PRIOR_WEIGHT * ALPHA) / (n + BG_PRIOR_WEIGHT) for k, (h, n) in cnt.items()}


def p_holds(o, h, bg, expl, hosts, db):
    s, e = o["signal"], o["entity"]
    if o["kind"] == "alert":
        if h["type"] is None:
            return ALPHA_ALERT
        return THETA_P if (h["entity"] == e or h["entity"] in expl.get(e, ())) else ALPHA_ALERT
    base = ALPHA if o["kind"] == "cont" else bg.get((s, e in db["nodes"]), ALPHA)
    if h["type"] is None:
        return base
    if e == h["entity"]:
        role = ROLES[h["type"]].get(s)
        if role is None and s in OWN_SYMPTOMS:
            return max(THETA_P, base)
        return {"K": h.get("q", THETA_K), "U": THETA_U, "C": ALPHA_C}.get(role, base)
    if s in SYMPTOMS and e in expl and h["entity"] in expl[e]:
        return max(THETA_P, base)
    if s == "node_cpu_rise" and h["type"] in LOAD_TYPES and e in hosts.get(h["entity"], []):
        return max(THETA_P, base)
    return base


def log_likelihood(obs, H, bg, expl, db) -> np.ndarray:
    hosts = db.get("hosts", {})
    ll = np.zeros(len(H))
    for o in obs:
        for i, h in enumerate(H):
            p = p_holds(o, h, bg, expl, hosts, db)
            ll[i] += math.log(p if o["holds"] else 1.0 - p)
    return ll


def prior(H) -> np.ndarray:
    w = np.array([RESIDUAL_PRIOR if h["type"] in RESIDUAL else 1.0 for h in H[:-1]])
    pr = np.append((1 - H0_PRIOR) * w / max(w.sum(), 1e-12), H0_PRIOR)
    return np.log(pr)


def posterior(logp: np.ndarray) -> np.ndarray:
    x = np.exp(logp - logp.max())
    return x / x.sum()


def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


# ------------------------------------------------------------------------------------------------------------------------------
# semantic probes (LLM text sensor): single-choice category per probed entity
# ------------------------------------------------------------------------------------------------------------------------------
def probe_dist(h, ent) -> np.ndarray:
    """P(category | h) for a probe of entity ent; categories CATS + ['none']."""
    k = len(CATS) + 1
    bgd = np.full(k, 0.5 / len(CATS))
    bgd[-1] = 0.5
    if h["type"] is None or h["entity"] != ent:
        return bgd
    r = ROLES[h["type"]]
    key = [i for i, c in enumerate(CATS) if r.get(f"msg_{c}") == "K" or r.get(f"log_{c}_rise") == "K"]
    sup = [i for i, c in enumerate(CATS) if r.get(f"msg_{c}") == "U" or r.get(f"log_{c}_rise") == "U"]
    mass_idx, theta = (key, THETA_K) if key else ((sup, THETA_U) if sup else (None, None))
    if mass_idx is None:
        return bgd
    d = np.zeros(k)
    d[mass_idx] = theta / len(mass_idx)
    rest = [i for i in range(k) if i not in mass_idx]
    d[-1] += (1 - theta) * 0.5
    others = [i for i in rest if i != k - 1]
    d[others] += (1 - theta) * 0.5 / max(len(others), 1)
    return d / d.sum()


def eig_probe(post: np.ndarray, H, ent) -> float:
    M = np.array([probe_dist(h, ent) for h in H])            # |H| x K
    pc = post @ M                                             # predictive distribution of the outcome
    h_now = entropy(post)
    exp_h = 0.0
    for c in range(M.shape[1]):
        if pc[c] <= 0:
            continue
        pp = post * M[:, c]
        exp_h += pc[c] * entropy(pp / pp.sum())
    return h_now - exp_h


def update_probe(logp: np.ndarray, H, ent, category: str) -> np.ndarray:
    c = CATS.index(category) if category in CATS else len(CATS)
    return logp + np.log(np.array([probe_dist(h, ent)[c] for h in H]))


# ------------------------------------------------------------------------------------------------------------------------------
# the loop
# ------------------------------------------------------------------------------------------------------------------------------
def diagnose_belief(db, callees, texts: Dict[str, List[str]], sensor=None, budget: int = PROBE_BUDGET, top_m: int = 8,
                    alert: str = None) -> Dict:
    """sensor(entity, texts) -> category (one of CATS or 'none'); None = no LLM (budget 0). Returns the answer, the posterior summary and
    the evidence log (observation, log Bayes factor of the leading hypothesis vs h0, belief before/after for probes)."""
    obs = observations(db)
    H = testable(hypotheses(db), obs)
    bg = background(obs, db)
    if alert and alert in db["services"]:
        obs.append({"entity": alert, "signal": "alert", "holds": True, "kind": "alert", "ev": "the alert fired on this service"})
    expl = explainers(db, callees)
    logp = prior(H) + log_likelihood(obs, H, bg, expl, db)
    log, probed = [], set()
    steps = 0
    while sensor is not None and steps < budget:
        post = posterior(logp)
        if post.max() >= STOP_P:
            break
        lead = np.argsort(-post)[:top_m]
        cands = {H[i]["entity"] for i in lead if H[i]["entity"] and texts.get(H[i]["entity"])} - probed
        if not cands:
            break
        gains = {e: eig_probe(post, H, e) for e in cands}
        e, g = max(gains.items(), key=lambda kv: kv[1])
        if g < EIG_MIN:
            break
        cat = sensor(e, texts[e])
        before = float(post.max())
        logp = update_probe(logp, H, e, cat)
        probed.add(e)
        steps += 1
        log.append({"probe": e, "eig_bits": round(g, 3), "category": cat, "max_belief_before": round(before, 3),
                    "max_belief_after": round(float(posterior(logp).max()), 3)})
    post = posterior(logp)
    order = np.argsort(-post)
    best = next(i for i in order if H[i]["type"] is not None)
    top = [{"entity": H[i]["entity"], "type": H[i]["type"], "p": round(float(post[i]), 4)} for i in order[:5]]
    # evidence for the answer: observations whose likelihood ratio (best vs h0) is largest
    hb = H[best]
    hosts = db.get("hosts", {})
    contrib = []
    for o in obs:
        p1, p0 = p_holds(o, hb, bg, expl, hosts, db), p_holds(o, H[-1], bg, expl, hosts, db)
        lr = math.log((p1 if o["holds"] else 1 - p1) / (p0 if o["holds"] else 1 - p0))
        if abs(lr) > 0.5:
            contrib.append({"entity": o["entity"], "signal": o["signal"], "holds": o["holds"], "log_bf": round(lr, 2), "ev": o["ev"]})
    contrib.sort(key=lambda x: -abs(x["log_bf"]))
    return {"entity": hb["entity"], "level": hb["level"], "type": hb["type"], "p": float(post[best]), "p_h0": float(post[-1]),
            "entropy_bits": round(entropy(post), 3), "top": top, "probes": log, "n_obs": len(obs), "n_hyp": len(H),
            "evidence": contrib[:12]}
