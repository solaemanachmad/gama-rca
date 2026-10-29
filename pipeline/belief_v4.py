"""v4 diagnosis (design: claude/RCA_v4_design.md in the project). Same Bayesian engine as v3 (pipeline/belief_diagnosis.py: likelihoods,
background rates, topology explain-away, the alert as an observation, EIG text probes); four changes, each switchable for ablation via
the `parts` argument (environment variable V4_PARTS, default all):

  score    continuous signals scored by the size of the shift (pipeline/fault_v4.py) instead of a saturating rank test
  open     open-world localisation: every entity also carries the hypothesis "anomaly of unspecified type" (key = any of its continuous
           signals; the keys are alternatives as in v3), so an entity whose anomaly matches no fingerprint can still be localised
  twostage the answer is chosen in two stages: entity = argmax of the marginal posterior P(e) (h0 excluded: an alert was raised, so an
           answer is required), then type = argmax P(type | e) over the specific types; prior mass is split equally over entities
           first, then over their hypotheses, so that an entity's localisation does not depend on how many types it can host
  fp       fingerprint additions derived from the taxonomy definitions (data/taxonomy.py), fixed before any v4 run:
             replicaScaleDown  + key svc_avail_drop (availability / available replicas fall), + sup svc_request_drop
             memoryPressure    + sup svc_cpu_rise (garbage collection under memory pressure burns CPU)
             cpuDeadLoop       + sup svc_request_drop (a pinned core starves the process: fewer requests complete)
v4.1 parts (DEVLOG_V4.md "Case timelines"; mechanisms seen in raw timelines of dev cases):
  peer     common-mode rejection: z re-standardised against the same signal of the other entities of the kind, per window
  rnull    Efron's empirical null by central matching (median / MAD of the pseudo-window scores): the history contains earlier injected
           faults, whose extreme scores would otherwise sit in the null
  prop     an absent symptom at an entity the hypothesis could reach is uninformative (partial propagation: some replicas, callers)
  events   Kubernetes events count only when their occurrence rate in the window exceeds their own long-run rate (Poisson test)
  avail    availability as a state: available replicas < desired, or availability < full, in the window and not chronically
v4.2 parts (ideas 1-3 of the development plan):
  onset    temporal precedence: a root starts no later than the entities it explains (onset = first sample at >= 3 robust SDs, twice)
  em       propagation probability estimated per case by EM (replaces the all-or-nothing choice between v4.0 and 'prop')
  ops      operation-level evidence: the most affected operation's error count / rate and latency (the alert is an apm.operation)
'rnull' is opt-in (it hurt in dev run 2); V4_PARTS=all means DEFAULT_PARTS.
Nothing here reads labels."""
from __future__ import annotations

import copy
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from pipeline import belief_diagnosis as bd
from pipeline.fault_calibrated import FINGERPRINTS, LEVEL, RESIDUAL, NODE_SIGNALS, _atoms, entity_signals
from pipeline.fault_v4 import calibrate_v4, V4_SVC_SIGNALS, new_events, unavailability, B_SPAN

ALL_PARTS = ("score", "open", "twostage", "fp", "peer", "rnull", "prop", "events", "avail", "onset", "em", "ops")
DEFAULT_PARTS = tuple(p for p in ALL_PARTS if p != "rnull")   # rnull hurt on every measure in dev run 2 (DEVLOG_V4.md); opt-in only
THETA_T, ONSET_TOL = 0.9, 60.0     # v4.2 onset: P(an entity the root explains starts no earlier than the root - 60 s | h)
PI_MIN, PI_MAX, EM_ITERS = 0.06, 0.95, 30
OP_CONT = ["op_error_rise", "op_latency_rise"]
ANOM = "ANOMALY"
SVC_CONT = [s for s, _, _ in V4_SVC_SIGNALS] + ["op_request_rise"]
NODE_CONT = [s for s, _, _ in NODE_SIGNALS]


def parts_from_env() -> set:
    v = os.environ.get("V4_PARTS", "all")
    return set(DEFAULT_PARTS) if v in ("all", "") else {p.strip() for p in v.split(",") if p.strip()}


def fingerprints_v4(parts) -> Dict:
    fp = copy.deepcopy(FINGERPRINTS)
    if "fp" in parts:
        fp["F036-replicaScaleDown"]["key"].append("svc_avail_drop")
        fp["F036-replicaScaleDown"]["sup"].append("svc_request_drop")
        fp["F007-memoryPressure"]["sup"].append("svc_cpu_rise")
        fp["F012-cpuDeadLoop"]["sup"].append("svc_request_drop")
    if "avail" in parts:
        fp["F036-replicaScaleDown"]["key"].append("svc_unavailable")
    if "ops" in parts:                                   # a failing operation (error count / rate of one endpoint)
        fp["F014-httpError5xx"]["key"].append("op_error_rise")
        for t in ("F011-codeDefect", "F023-nullPointerException", "F016-rateLimiting"):
            fp[t]["sup"].append("op_error_rise")
    return fp


class Model:
    """Hypotheses, roles and likelihoods for one fingerprint table (v3 functions, parametrised)."""

    def __init__(self, parts):
        self.parts = set(parts)
        self.fp = fingerprints_v4(self.parts)
        self.roles = {}
        for t, f in self.fp.items():
            r = {}
            for role, key in (("C", "con"), ("U", "sup"), ("K", "key")):
                for expr in f[key]:
                    for a in _atoms(expr):
                        r[a] = role
            self.roles[t] = r

    def keys(self, h):
        if h["type"] == ANOM:
            return NODE_CONT if h["level"] == "node" else SVC_CONT + (OP_CONT if "ops" in self.parts else [])
        return self.fp[h["type"]]["key"]

    @property
    def symptoms(self):
        return bd.SYMPTOMS | (set(OP_CONT) if "ops" in self.parts else set())

    @property
    def own_symptoms(self):
        return bd.OWN_SYMPTOMS | (set(OP_CONT) if "ops" in self.parts else set())

    def role(self, h, s):
        if h["type"] == ANOM:
            return "K" if s in self.keys(h) else None
        return self.roles[h["type"]].get(s)

    def hypotheses(self, db) -> List[Dict]:
        H = []
        for ent in db["services"]:
            H += [{"entity": ent, "level": "service", "type": t} for t in self.fp if LEVEL[t] in ("service", "any")]
            if "open" in self.parts:
                H.append({"entity": ent, "level": "service", "type": ANOM})
        for ent in db["nodes"]:
            H += [{"entity": ent, "level": "node", "type": t} for t in self.fp if LEVEL[t] in ("node", "any")]
            if "open" in self.parts:
                H.append({"entity": ent, "level": "node", "type": ANOM})
        return H + [{"entity": None, "level": None, "type": None}]

    def testable(self, H, obs):
        seen = defaultdict(set)
        for o in obs:
            seen[o["entity"]].add(o["signal"])
        out = []
        for h in H:
            if h["type"] is None:
                out.append(h)
                continue
            n = sum(1 for k in self.keys(h) if all(a in seen[h["entity"]] for a in _atoms(k)))
            if n:
                out.append({**h, "q": 1.0 - (1.0 - bd.THETA_K) ** (1.0 / n), "n_keys": n})
        return out

    def p_holds(self, o, h, bg, expl, hosts, db):
        s, e = o["signal"], o["entity"]
        if o["kind"] == "alert":
            if h["type"] is None:
                return bd.ALPHA_ALERT
            return bd.THETA_P if (h["entity"] == e or h["entity"] in expl.get(e, ())) else bd.ALPHA_ALERT
        base = bd.ALPHA if o["kind"] == "cont" else bg.get((s, e in db["nodes"]), bd.ALPHA)
        if h["type"] is None:
            return base
        if e == h["entity"]:
            role = self.role(h, s)
            if role is None and s in self.own_symptoms:
                return max(bd.THETA_P, base)
            return {"K": h.get("q", bd.THETA_K), "U": bd.THETA_U, "C": bd.ALPHA_C}.get(role, base)
        if s in self.symptoms and e in expl and h["entity"] in expl[e]:
            return max(bd.THETA_P, base)
        if s == "node_cpu_rise" and h["type"] in bd.LOAD_TYPES and e in hosts.get(h["entity"], []):
            return max(bd.THETA_P, base)
        return base

    def propagated(self, o, h, expl, hosts):
        s, e = o["signal"], o["entity"]
        if h["type"] is None or e == h["entity"]:
            return False
        return (s in self.symptoms and e in expl and h["entity"] in expl[e]) or \
               (s == "node_cpu_rise" and h["type"] in bd.LOAD_TYPES and e in hosts.get(h["entity"], []))

    def log_likelihood(self, obs, H, bg, expl, db):
        hosts = db.get("hosts", {})
        ll = np.zeros(len(H))
        for o in obs:
            for i, h in enumerate(H):
                p = self.p_holds(o, h, bg, expl, hosts, db)
                if "prop" in self.parts and not o["holds"] and self.propagated(o, h, expl, hosts):
                    p = self.p_holds(o, H[-1], bg, expl, hosts, db)      # partial propagation: an absent symptom is uninformative
                ll[i] += math.log(p if o["holds"] else 1.0 - p)
        return ll

    def ll_parts(self, obs, H, bg, expl, db):
        """For the EM part: log-likelihood WITHOUT the propagated-symptom observations, and per hypothesis the number of propagated
        symptoms that hold / are absent."""
        hosts = db.get("hosts", {})
        ll, n_hold, n_abs = np.zeros(len(H)), np.zeros(len(H)), np.zeros(len(H))
        for o in obs:
            for i, h in enumerate(H):
                if self.propagated(o, h, expl, hosts):
                    if o["holds"]:
                        n_hold[i] += 1
                    else:
                        n_abs[i] += 1
                    continue
                p = self.p_holds(o, h, bg, expl, hosts, db)
                ll[i] += math.log(p if o["holds"] else 1.0 - p)
        return ll, n_hold, n_abs

    def prior(self, H):
        w = np.array([bd.RESIDUAL_PRIOR if h["type"] in RESIDUAL else 1.0 for h in H[:-1]])
        if "twostage" in self.parts:                          # equal mass per entity, then per hypothesis of that entity
            tot = defaultdict(float)
            for h, x in zip(H[:-1], w):
                tot[h["entity"]] += x
            n_ent = len(tot)
            w = np.array([x / tot[h["entity"]] / n_ent for h, x in zip(H[:-1], w)])
        pr = np.append((1 - bd.H0_PRIOR) * w / max(w.sum(), 1e-12), bd.H0_PRIOR)
        return np.log(pr)


def _probe_dist(model, h, ent):
    if h["type"] == ANOM:
        return bd.probe_dist({"type": None, "entity": None}, ent)
    return bd.probe_dist(h, ent)


def _fallback_type(model, ent, level, sig):
    """Entity chosen, but only the unspecified-anomaly hypothesis was testable: the specific type of the entity's level whose key
    evidence is strongest (categorical first, then the smallest calibrated p); residual types last."""
    best, best_score = None, None
    for t, f in model.fp.items():
        if LEVEL[t] not in (level, "any"):
            continue
        sc = None
        for k in f["key"]:
            for a in _atoms(k):
                v = sig.get(a)
                if not v:
                    continue
                s = (1, 0.0) if v["kind"] == "cat" else (0, -(v["p"] if v["p"] is not None else 1.0))
                sc = s if sc is None or s > sc else sc
        if sc is None:
            continue
        sc = (0 if t in RESIDUAL else 1,) + sc
        if best_score is None or sc > best_score:
            best, best_score = t, sc
    return best or ("F026-nodeCpuHigh" if level == "node" else "F011-codeDefect")


def diagnose_v4(db, callees, texts, sensor=None, budget=bd.PROBE_BUDGET, top_m=8, alert=None, parts=None, case=None) -> Dict:
    parts = set(parts) if parts is not None else parts_from_env()
    model = Model(parts)
    if "events" in parts and case is not None:
        new_events(case, db, B_SPAN * (db["w1"] - db["w0"]).total_seconds())
    if "score" in parts:
        calibrate_v4(db, peer="peer" in parts, robust_null="rnull" in parts, ops_level="ops" in parts)
    obs = bd.observations(db)
    if "avail" in parts:
        for s_, bad in unavailability(db).items():
            obs.append({"entity": s_, "signal": "svc_unavailable", "holds": bad, "kind": "cat",
                        "ev": "deployment available replicas below desired / availability below full in the window" if bad else ""})
    H = model.testable(model.hypotheses(db), obs)
    bg = bd.background(obs, db)
    if alert and alert in db["services"]:
        obs.append({"entity": alert, "signal": "alert", "holds": True, "kind": "alert", "ev": "the alert fired on this service"})
    expl = bd.explainers(db, callees)
    pi = None
    if "em" in parts:
        # propagation probability pi (P(a symptom shows at an entity the root can reach)) is a per-case nuisance parameter, estimated by
        # EM from the case itself (no labels): E-step = posterior over hypotheses; M-step = posterior-weighted share of reachable
        # symptoms that hold. A wide hypothesis then pays for every reachable entity that stays quiet, at the rate this case shows.
        ll0, nh, na = model.ll_parts(obs, H, bg, expl, db)
        pr_ = model.prior(H)
        pi = 0.5
        for _ in range(EM_ITERS):
            lp = pr_ + ll0 + nh * math.log(pi) + na * math.log(1 - pi)
            w = bd.posterior(lp)
            den = float((w * (nh + na)).sum())
            new_pi = min(max(float((w * nh).sum()) / den, PI_MIN), PI_MAX) if den > 0 else pi
            if abs(new_pi - pi) < 1e-4:
                pi = new_pi
                break
            pi = new_pi
        logp = pr_ + ll0 + nh * math.log(pi) + na * math.log(1 - pi)
    else:
        logp = model.prior(H) + model.log_likelihood(obs, H, bg, expl, db)
    onsets = {}
    if "onset" in parts:
        cal = db.get("_cal", {})
        for ent, sigs in cal.items():
            ts = [r["onset"] for r in sigs.values() if r.get("onset") is not None and r.get("p") is not None and r["p"] <= bd.ALPHA]
            if ts:
                onsets[ent] = min(ts)
        logp = logp + onset_loglik(H, onsets, expl)
    log, probed, steps = [], set(), 0
    while sensor is not None and steps < budget:
        post = bd.posterior(logp)
        if post.max() >= bd.STOP_P:
            break
        lead = np.argsort(-post)[:top_m]
        cands = {H[i]["entity"] for i in lead if H[i]["entity"] and texts.get(H[i]["entity"])} - probed
        if not cands:
            break
        gains = {}
        for e in cands:
            M = np.array([_probe_dist(model, h, e) for h in H])
            pc = post @ M
            exp_h = sum(pc[c] * bd.entropy((post * M[:, c]) / (post * M[:, c]).sum()) for c in range(M.shape[1]) if pc[c] > 0)
            gains[e] = bd.entropy(post) - exp_h
        e, g = max(gains.items(), key=lambda kv: kv[1])
        if g < bd.EIG_MIN:
            break
        cat = sensor(e, texts[e])
        c = bd.CATS.index(cat) if cat in bd.CATS else len(bd.CATS)
        before = float(post.max())
        logp = logp + np.log(np.array([_probe_dist(model, h, e)[c] for h in H]))
        probed.add(e)
        steps += 1
        log.append({"probe": e, "eig_bits": round(g, 3), "category": cat, "max_belief_before": round(before, 3),
                    "max_belief_after": round(float(bd.posterior(logp).max()), 3)})
    post = bd.posterior(logp)
    order = np.argsort(-post)
    if "twostage" in parts:
        marg = defaultdict(float)
        for h, p in zip(H[:-1], post[:-1]):
            marg[h["entity"]] += float(p)
        ent = max(marg, key=marg.get)
        own = [i for i in order if H[i]["entity"] == ent and H[i]["type"] not in (None, ANOM)]
        level = next(h["level"] for h in H if h["entity"] == ent)
        if own:
            best = own[0]
            typ, p_type = H[best]["type"], float(post[best]) / max(marg[ent], 1e-12)
        else:
            best = next(i for i in order if H[i]["entity"] == ent)
            typ, p_type = _fallback_type(model, ent, level, entity_signals(db, ent)), None
        p_ent = marg[ent]
    else:
        best = next(i for i in order if H[i]["type"] not in (None, ANOM))
        ent, level, typ = H[best]["entity"], H[best]["level"], H[best]["type"]
        p_ent, p_type = float(post[best]), None
    top = [{"entity": H[i]["entity"], "type": H[i]["type"], "p": round(float(post[i]), 4)} for i in order[:5]]
    hb = {**H[best], "type": typ if typ in model.fp else H[best]["type"]}
    if hb["type"] == ANOM:
        hb = H[best]
    hosts = db.get("hosts", {})
    contrib = []
    for o in obs:
        p1, p0 = model.p_holds(o, hb, bg, expl, hosts, db), model.p_holds(o, H[-1], bg, expl, hosts, db)
        lr = math.log((p1 if o["holds"] else 1 - p1) / (p0 if o["holds"] else 1 - p0))
        if abs(lr) > 0.5:
            contrib.append({"entity": o["entity"], "signal": o["signal"], "holds": o["holds"], "log_bf": round(lr, 2), "ev": o["ev"]})
    contrib.sort(key=lambda x: -abs(x["log_bf"]))
    return {"entity": ent, "level": level, "type": typ, "p": p_ent, "p_type": p_type, "p_h0": float(post[-1]),
            "entropy_bits": round(bd.entropy(post), 3), "top": top, "probes": log, "n_obs": len(obs), "n_hyp": len(H),
            "evidence": contrib[:12], "parts": sorted(parts), "pi": pi,
            "onset_root": onsets.get(ent), "n_onsets": len(onsets)}


def onset_loglik(H, onsets, expl) -> np.ndarray:
    """v4.2 temporal precedence: a root cause starts no later than what it causes. For h = (e*, .) with a measured onset of e*, every
    service e that e* can explain (e* in Expl(e)) and that has its own onset contributes P(onset_e >= onset_e* - ONSET_TOL | h) = THETA_T
    (else 1 - THETA_T), against 0.5 when nothing is assumed about the order. No onset at e* (silent or unmeasured root) = neutral."""
    out = np.zeros(len(H))
    for i, h in enumerate(H):
        r = h["entity"]
        if h["type"] is None or r not in onsets:
            continue
        for e, t in onsets.items():
            if e == r or r not in expl.get(e, ()):
                continue
            ok = t >= onsets[r] - ONSET_TOL
            out[i] += math.log((THETA_T if ok else 1 - THETA_T) / 0.5)
    return out
