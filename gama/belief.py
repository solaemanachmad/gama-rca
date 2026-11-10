"""Stage 2 -- Bayesian belief over root-cause hypotheses (entity x fault type).

Hypotheses: every (service, type) and (node, type) of the taxonomy, an open "anomaly of unspecified type" per entity, and h0 (no
fault). Fingerprint roles per type and signal (from the taxonomy definitions, data/taxonomy.py; pipeline/fault_calibrated.FINGERPRINTS):
K key (one must hold), U support, M possible consequence (absence uninformative), C contradiction. Likelihood of an observation:
THETA_K / THETA_U / ALPHA_C for the roles at the root, THETA_P for a symptom the root explains along the topology (callers, hosts),
the case's own background rate otherwise (P1: calibrate, do not threshold). Propagation probability pi is estimated per case by EM.

Terms added by modules (gama/config.py):
  M1 mm      log(THETA_M / beta_M) when the type's mechanism evidence at the root holds in >= 2 modalities
  M2 trace   error / slow origins are own evidence of the root (or of a node hosting >= 2 origin services)
  M3 typing  a node's CPU rise is explained by any root whose own CPU rises; error origins expected only for error types
  M4 load    load that came from upstream (request, latency, CPU rises, and the host CPU they heat) is explained away (P2)
  always     onset precedence: a root starts no later than what it explains (THETA_T)
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from pipeline import belief_diagnosis as bd
from data.dataset_profile import PROFILE
from pipeline.fault_calibrated import FINGERPRINTS, LEVEL, RESIDUAL, NODE_SIGNALS, _atoms
from pipeline.fault import SVC_SIGNALS as V4_SVC_SIGNALS
from pipeline.trace_structure import SIGNALS as TRACE_SIGNALS
from gama.config import Modules

LOAD_SIGNALS = ("svc_request_rise", "op_request_rise", "svc_cpu_rise")
THETA_M, MM_B0, MM_W = 0.6, 0.1, 10          # M1: P(>=2 modalities corroborate | true root); background floor and prior weight
THETA_T, ONSET_TOL = 0.9, 60.0               # onset: P(an explained entity starts no earlier than the root - 60 s | h)
PI_MIN, PI_MAX, EM_ITERS = 0.06, 0.95, 30
ANOM = "ANOMALY"
SVC_CONT = [s for s, _, _ in V4_SVC_SIGNALS] + ["op_request_rise"]
NODE_CONT = [s for s, _, _ in NODE_SIGNALS]
MECHANISM_SIGNALS = ("svc_cpu_rise", "svc_mem_rise", "svc_gc_rise", "svc_thread_rise")   # the resources a process consumes


def fingerprints(m: Modules) -> Dict:
    """Fingerprint table: the taxonomy fingerprints + additions fixed before any v4 run (M0) + origin-message roles (M2)."""
    fp = copy.deepcopy(FINGERPRINTS)
    if PROFILE == "rca100":
        _rca100_additions(fp)
    if m.trace:                                   # the same message / status evidence read at the origin spans
        for t, f in fp.items():
            atoms = {a for k in f.get("key", []) + f.get("sup", []) for a in _atoms(k)}
            extra = [("omsg_" + a[4:]) for a in atoms if a.startswith("msg_")]
            extra += [("omsg_" + a[4:-5]) for a in atoms if a.startswith("log_") and a.endswith("_rise")]
            extra += [("o" + a) for a in atoms if a in ("http429", "http403", "http503", "http5xx")]
            if extra:
                f["may"] = list(dict.fromkeys(f.get("may", []) + sorted(set(extra))))
    return fp


def _rca100_additions(fp):
    """M0 additions to the RCA100 taxonomy fingerprints, fixed before any v4 run."""
    fp["F036-replicaScaleDown"]["key"].append("svc_avail_drop")
    fp["F036-replicaScaleDown"]["sup"].append("svc_request_drop")
    fp["F007-memoryPressure"]["sup"].append("svc_cpu_rise")
    fp["F036-replicaScaleDown"]["key"].append("svc_unavailable")
    # restarts, probe failures and OOM kills are often CONSEQUENCES of resource exhaustion (role M); an OOM kill no longer excludes
    # memory pressure / full GC; an abnormal memory rise no longer supports a limit set too low for normal usage
    for t in ("F007-memoryPressure", "F022-fullGC"):
        fp[t]["con"] = [c for c in fp[t]["con"] if c != "ev_oom"]
    fp["F007-memoryPressure"]["may"] = ["ev_oom", "ev_restart", "ev_unhealthy"]
    fp["F022-fullGC"]["may"] = ["ev_oom", "ev_unhealthy", "ev_restart"]
    for t in ("F034-cpuFullLoad", "F012-cpuDeadLoop", "F002-threadExhaustion"):
        fp[t]["may"] = ["ev_unhealthy", "ev_restart"]
    fp["F039-resourceLimitMisconfig"]["sup"] = [x for x in fp["F039-resourceLimitMisconfig"]["sup"] if x != "svc_mem_rise"]


class Model:
    """Hypotheses, fingerprint roles and likelihoods."""

    def __init__(self, m: Modules):
        self.m = m
        self.origin_nodes = set()        # M2: nodes hosting >= 2 origin services (set per case)
        self.cpu_up = set()              # M3: entities whose svc_cpu_rise holds (set per case)
        self.members = {}                # pod -> service, when the case defines memberships (AIOps2025 A4; RCA100: empty)
        self.fp = fingerprints(m)
        self.roles = {}
        for t, f in self.fp.items():
            r = {}
            for role, key in (("M", "may"), ("C", "con"), ("U", "sup"), ("K", "key")):
                for expr in f.get(key, []):
                    for a in _atoms(expr):
                        r[a] = role
            self.roles[t] = r

    symptoms = bd.SYMPTOMS | {"svc_request_rise"}       # a request rise the root can reach = retry amplification
    own_symptoms = bd.OWN_SYMPTOMS

    def keys(self, h):
        if h["type"] == ANOM:
            return NODE_CONT if h["level"] == "node" else SVC_CONT
        return self.fp[h["type"]]["key"]

    def role(self, h, s):
        if h["type"] == ANOM:
            return "K" if s in self.keys(h) else None
        return self.roles[h["type"]].get(s)

    def hypotheses(self, db) -> List[Dict]:
        H = []
        for ent in db["services"]:
            H += [{"entity": ent, "level": "service", "type": t} for t in self.fp if LEVEL[t] in ("service", "any")]
            H.append({"entity": ent, "level": "service", "type": ANOM})
        for ent in db["nodes"]:
            H += [{"entity": ent, "level": "node", "type": t} for t in self.fp if LEVEL[t] in ("node", "any")]
            H.append({"entity": ent, "level": "node", "type": ANOM})
        return H + [{"entity": None, "level": None, "type": None}]

    def testable(self, H, obs):
        """Keep the hypotheses with at least one key observed; q = per-key probability so that P(some key holds) = THETA_K."""
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
        if (e, s) in db.get("_load_explained", ()):              # M4: load that came from upstream
            base = max(base, bd.THETA_P)
        if s in TRACE_SIGNALS:                                   # M2: own evidence of the root (or of its node)
            return max(bd.THETA_P, base) if self.trace_own(o, h, hosts) else base
        if h["type"] is None:
            return base
        if self.members and h["type"] != ANOM and self.role(h, s) in ("K", "U"):
            if self.members.get(e) == h["entity"]:           # A4: a service-level fault shows its mechanism in every pod
                return bd.THETA_P
            if self.members.get(h["entity"]) == e:           # A4: a pod-level fault shows in its service's aggregate
                return max(bd.THETA_P, base)
        if e == h["entity"]:
            role = self.role(h, s)
            if role is None and s in self.own_symptoms:
                return max(bd.THETA_P, base)
            return {"K": h.get("q", bd.THETA_K), "U": bd.THETA_U, "C": bd.ALPHA_C, "M": bd.THETA_U}.get(role, base)
        if s in self.symptoms and e in expl and h["entity"] in expl[e]:
            return max(bd.THETA_P, base)
        if s == "node_cpu_rise" and self.host_cpu(h) and e in hosts.get(h["entity"], []):
            return max(bd.THETA_P, base)
        return base

    def host_cpu(self, h):
        return h["type"] in bd.LOAD_TYPES or (self.m.typing and h["type"] is not None and h["entity"] in self.cpu_up)

    def error_type(self, t):
        """M3: the type's fingerprint names an error signal."""
        if t not in self.fp:
            return True
        f = self.fp[t]
        atoms = {a for k in f.get("key", []) + f.get("sup", []) for a in _atoms(k)}
        return any(a.startswith(("http", "msg_")) or a in ("error_spans", "svc_error_rise", "svc_err_rate_rise")
                   or a.endswith("_err") for a in atoms)

    def trace_own(self, o, h, hosts):
        if h["type"] is None:
            return False
        if o["entity"] == h["entity"] or self.members.get(o["entity"]) == h["entity"]:   # A4: a pod's origin is its service's
            if self.m.typing and o["signal"] == "trace_err_origin" and h["type"] != ANOM:
                return self.error_type(h["type"])
            return True
        if h["entity"] not in (hosts or {}).get(o["entity"], ()):
            return False
        return not self.m.trace or h["entity"] in self.origin_nodes      # a node needs >= 2 origin services

    def may(self, o, h, hosts=None):
        if o["signal"] in TRACE_SIGNALS:                         # absence of a trace origin is uninformative
            return self.trace_own(o, h, hosts)
        if (self.members and h["type"] not in (None, ANOM) and self.members.get(h["entity"]) == o["entity"]
                and self.role(h, o["signal"]) in ("K", "U")):
            return True                                          # A4: absence in the service aggregate is uninformative
        return h["type"] not in (None, ANOM) and o["entity"] == h["entity"] and self.role(h, o["signal"]) == "M"

    def propagated(self, o, h, expl, hosts):
        s, e = o["signal"], o["entity"]
        if h["type"] is None or e == h["entity"]:
            return False
        return (s in self.symptoms and e in expl and h["entity"] in expl[e]) or \
               (s == "node_cpu_rise" and self.host_cpu(h) and e in hosts.get(h["entity"], []))

    def ll_parts(self, obs, H, bg, expl, db):
        """Log-likelihood WITHOUT the propagated-symptom observations, and per hypothesis the number of propagated symptoms that
        hold / are absent (the EM sufficient statistics)."""
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
                if not o["holds"] and self.may(o, h, hosts):
                    p = self.p_holds(o, H[-1], bg, expl, hosts, db)
                ll[i] += math.log(p if o["holds"] else 1.0 - p)
        return ll, n_hold, n_abs

    def prior(self, H):
        """Equal mass per entity, then per hypothesis of that entity (residual types down-weighted); h0 fixed."""
        w = np.array([bd.RESIDUAL_PRIOR if h["type"] in RESIDUAL else 1.0 for h in H[:-1]])
        tot = defaultdict(float)
        for h, x in zip(H[:-1], w):
            tot[h["entity"]] += x
        n_ent = len(tot)
        w = np.array([x / tot[h["entity"]] / n_ent for h, x in zip(H[:-1], w)])
        pr = np.append((1 - bd.H0_PRIOR) * w / max(w.sum(), 1e-12), bd.H0_PRIOR)
        return np.log(pr)


def infer(model: Model, db, callees, obs, alert=None) -> Dict:
    """Posterior log-probabilities over the testable hypotheses (before text probes). Mutates obs (alert) and db (_load_explained)."""
    m = model.m
    model.members = dict(db.get("memberships") or {})
    if m.typing:
        model.cpu_up = {o["entity"] for o in obs if o["signal"] == "svc_cpu_rise" and o["holds"]}
    H = model.testable(model.hypotheses(db), obs)
    bg = bd.background(obs, db)
    if alert and alert in db["services"]:
        obs.append({"entity": alert, "signal": "alert", "holds": True, "kind": "alert", "ev": "the alert fired on this service"})
    expl = bd.explainers(db, callees)
    db["_load_explained"] = (load_explained(obs, callees, signals=LOAD_SIGNALS + ("svc_latency_rise",), hosts=db.get("hosts", {}))
                             if m.load else set())
    extra, mm_beta, mm_k = np.zeros(len(H)), None, None
    if m.mm:
        extra, mm_beta, mm_k = mm_loglik(model, H, obs)
    # EM for pi = P(a symptom shows at an entity the root can reach): E-step posterior over hypotheses, M-step posterior-weighted
    # share of reachable symptoms that hold. A wide hypothesis pays for every reachable entity that stays quiet, at this case's rate.
    ll0, nh, na = model.ll_parts(obs, H, bg, expl, db)
    pr_ = model.prior(H) + extra
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
    onsets = {}
    lx = db["_load_explained"] if m.load else ()
    for ent, sigs in db.get("_cal", {}).items():
        ts = [r["onset"] for k, r in sigs.items() if r.get("onset") is not None and r.get("p") is not None and r["p"] <= bd.ALPHA
              and (ent, k) not in lx]                     # M4: load that came from upstream does not date the entity's fault
        if ts:
            onsets[ent] = min(ts)
    logp = logp + onset_loglik(H, onsets, expl)
    return {"H": H, "bg": bg, "expl": expl, "logp": logp, "pi": pi, "onsets": onsets, "mm_beta": mm_beta, "mm_k": mm_k}


def onset_loglik(H, onsets, expl) -> np.ndarray:
    """Temporal precedence: a root cause starts no later than what it causes. For h = (e*, .) with a measured onset of e*, every
    entity e that e* can explain and that has its own onset contributes THETA_T if onset_e >= onset_e* - ONSET_TOL, else 1 - THETA_T,
    against 0.5 when nothing is assumed about the order. No onset at e* (silent or unmeasured root) = neutral."""
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


def first_movers(sigs: Dict):
    """(signals with the earliest onset, holding signals without a measured onset, any onset?) among the MECHANISM_SIGNALS of one
    entity's calibrated signals; ties at the sampling resolution are all first movers."""
    on = {k: r["onset"] for k, r in sigs.items() if k in MECHANISM_SIGNALS and r.get("onset") is not None
          and r.get("p") is not None and r["p"] <= bd.ALPHA}
    non = {k for k, r in sigs.items() if k in MECHANISM_SIGNALS and r.get("onset") is None
           and r.get("p") is not None and r["p"] <= bd.ALPHA}
    if not on:
        return set(), non, False
    t0 = min(on.values())
    return {k for k, t in on.items() if t == t0}, non, True


def sigorder_loglik(H, cal: Dict, model) -> np.ndarray:
    """M3 intra-entity signal order: within the root, the cause moves before its consequences (memory leak -> GC -> CPU; CPU
    saturation -> GC / threads later). For h = (e*, type) whose key names a MECHANISM_SIGNAL, when one of them has a measured onset at
    e*: THETA_T if a key of the type is a first mover, neutral if the key holds without a measured onset, 1 - THETA_T otherwise."""
    out = np.zeros(len(H))
    cache = {}
    for i, h in enumerate(H):
        if h["type"] in (None, ANOM) or h.get("level") != "service":
            continue
        keys = {a for k in model.fp.get(h["type"], {}).get("key", []) for a in _atoms(k)} & set(MECHANISM_SIGNALS)
        if not keys:
            continue
        e = h["entity"]
        if e not in cache:
            cache[e] = first_movers(cal.get(e, {}))
        first, non, any_on = cache[e]
        if not any_on:
            continue
        p = THETA_T if keys & first else (0.5 if keys & non else 1 - THETA_T)
        out[i] = math.log(p / 0.5)
    return out


def modality(signal: str) -> str:
    """Telemetry modality of an observation: events = Kubernetes events; traces = status codes and span outcomes; logs = log-template
    rises and log / span messages; everything else is a metric (incl. availability state)."""
    if signal.startswith("ev_"):
        return "events"
    if signal.startswith(("log_", "msg_")) or signal == "message":
        return "logs"
    if signal.startswith(("http", "ohttp", "omsg_", "trace_")) or signal == "error_spans" or signal.split("_")[0] in ("db", "redis", "queue"):
        return "traces"
    return "metrics"


def mm_loglik(model, H, obs):
    """M1 cross-modal corroboration: (log-lik vector, beta_M, number of modalities per hypothesis). Noise of a volatile entity tends
    to stay in one modality; a real fault leaves aligned evidence in several. beta_M = share of the case's specific hypotheses with
    such corroboration (shrunk, floor MM_B0); absence is uninformative."""
    hold = defaultdict(set)
    for o in obs:
        if o["holds"] and o["kind"] != "alert":
            hold[o["entity"]].add(o["signal"])
    k = np.zeros(len(H))
    spec = []
    for i, h in enumerate(H):
        if h["type"] in (None, ANOM):
            continue
        spec.append(i)
        mods = {modality(sg) for sg in hold.get(h["entity"], ()) if sg not in bd.SYMPTOMS and model.role(h, sg) in ("K", "U", "M")}
        k[i] = len(mods)
    n2 = sum(1 for i in spec if k[i] >= 2)
    beta = max((n2 + MM_W * MM_B0) / (len(spec) + MM_W), MM_B0)
    out = np.where(k >= 2, math.log(THETA_M / beta), 0.0)
    return out, beta, k


def load_explained(obs, callees, signals=None, hosts=None) -> set:
    """M4: {(entity, signal)} for the load signals of services whose request count rises together with a caller's (load flows
    downstream). With hosts: the CPU rise of the nodes hosting a service whose CPU rise is load-explained is load-explained too."""
    signals = signals or LOAD_SIGNALS
    rise = {o["entity"] for o in obs if o["holds"] and o["signal"] in ("svc_request_rise", "op_request_rise")}
    callers = defaultdict(set)
    for a, bs in callees.items():
        for b_ in bs:
            callers[b_].add(a)
    out = set()
    for e in rise:
        if callers.get(e, set()) & rise:
            out |= {(e, sg) for sg in signals}
    if hosts is not None:
        cpu = {o["entity"] for o in obs if o["holds"] and o["signal"] == "svc_cpu_rise"}
        for e in {e for e, sg in out if sg == "svc_cpu_rise"} & cpu:
            out |= {(n, "node_cpu_rise") for n in hosts.get(e, [])}
    return out
