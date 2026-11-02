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
v4.3 parts:
  selfnorm volatile-entity normalisation: z re-standardised against the entity's own pseudo-window z (leave-one-out)
  llmtype  LLM typing stage (pipeline/llm_typing.py): after the entity is chosen, the LLM picks its fault type from all types of the
           entity's level, given the entity's evidence summary and the statistical ranking; accepted only with a valid type and a
           verbatim quote of the summary
v4.4 parts (consequence-aware evidence; DEVLOG_V4.md "Dev run 4"):
  conseq   restart / probe-failure / OOM-kill events can be explained (support) by the resource types that cause them; an OOM kill no
           longer excludes memory pressure / full GC
  retry    a request-count rise at a service the root can reach is a propagated symptom (retry amplification after failures), not
           only the key of trafficSurge at that service
v5 parts (PREREGISTRATION_V5.md amendment 1; opt-in until validated on the even cases):
  mm       multimodal corroboration: a hypothesis (e*, type) whose MECHANISM evidence at e* (key / support / may signals of the type,
           generic symptoms excluded) holds in >= 2 different modalities (metrics, events, traces, logs) gains log(THETA_M / beta_M);
           beta_M = share of the case's specific hypotheses with such corroboration (Beta-shrunk, floor MM_B0). Absence is uninformative.
           Rationale: noise of a volatile entity tends to stay in one modality; a real fault leaves aligned traces in several.
  loadx    load flows downstream: a request-count rise (and the CPU rise that comes with it) at a service is EXPECTED, not
           evidence of a fault there, when the service's own request count rises AND one of its callers' request count rises too
           (the extra load comes from upstream, e.g. the user-facing entry). Such observations get background THETA_P instead of
           ALPHA under every hypothesis whose fingerprint does not name them for that entity (amendment 2, from the odd-case traces:
           recommendation / ad show request + CPU rises in 29-36 of 52 incidents).
  ratio    relational signals (amendment 3): CPU per request, error rate, and request share relative to the callers; cpuFullLoad /
           cpuDeadLoop are keyed by CPU per request and trafficSurge by the request share (raw rises become support)
  torigin  trace structure (amendment 4, pipeline/trace_structure.py): error origin (deepest failing spans) and slow origin (largest
           self time in the slowest traces) per service, as OWN evidence of a root at that service or at the node hosting it; absence
           is uninformative; never a propagated symptom
  torigin2 trace structure v2 (amendment 5): as torigin, with minimum counts (10 failing / 5 slow traces) and a node supported only
           when >= 2 of its hosted services are origins (tier decision: one failing service on a node is a service fault)
  torigin3 trace structure v3 (amendment 7) = torigin2 + a deepest failing CLIENT span that could not reach its callee (refused /
           unavailable / no such host, no child span) or timed out waiting for it is attributed to the callee (named in the span
           name / attributes, or the longest child for a timeout), not to the caller
  hostcpu  (amendment 8) the CPU of a node rises with the CPU of the services it hosts: a node_cpu_rise at a host of e* is explained
           by ANY hypothesis at e* whose own svc_cpu_rise holds, not only by the load types (which gained up to one THETA_P factor per
           host and outranked memory / GC / application faults that also burn CPU)
  errclass (amendment 8) an ERROR origin at e* is expected only for types whose fingerprint names an error signal (status codes,
           error-span messages, error spans / rates, failing dependency calls); for resource types (CPU, memory, disk, traffic) it is
           unrelated evidence (background). Slow origins stay expected for every type.
  tiebreak (amendment 8) exactly tied types are resolved by fingerprint order (the general type before its special case: cpuFullLoad
           before cpuDeadLoop) through a stable sort, instead of an arbitrary one
  omsg     origin messages (amendment 6; needs torigin or torigin2): the status message / HTTP code of a service's ORIGIN spans and
           the log lines linked to its origin traces by trace id -> omsg_<category>, ohttp<code>; role 'may' of every type whose
           fingerprint names the same message category or status code (present: support; absent: uninformative)
  cpuspec  parsimony between the two CPU types (amendment 6): cpuDeadLoop is the SPECIFIC case (a loop starving its own process), so
           it needs the starvation evidence: key = CPU rise AND request drop; a CPU rise alone is cpuFullLoad
  xcard    LLM typing shows, per candidate type, its expected cross-modal signature (from the fingerprint, each signal tagged with its
           modality) and tags every evidence line with its modality.
'rnull' and 'ops' are opt-in (they did not help in dev runs 2 and 3); V4_PARTS=all means DEFAULT_PARTS.
Nothing here reads labels.  depconseq (amendment 9) for dependency-latency types (slowSQL, dbNetworkLatency, redisUnavailable, cacheBreakdown,
           messageQueueBacklog) the caller's svc_thread_rise / svc_cpu_rise / svc_gc_rise are 'may' consequences (threads blocked
           on the slow dependency), so these JVM signals no longer count only for resource types
  depbase  (amendment 10) db_slow / redis_slow / queue_slow test the service's dependency spans against the same spans before the
           window (rank test + upper-quantile effect), connection-pool waits count as database calls (pipeline/dep_latency.py)
  depshare (amendment 11) with depbase: <dep>_slow also needs the share of the service's request time spent in those calls to rise
           (a starved service stretches its calls too, but spends the extra time in itself)
  sigorder (amendment 12) inside the root, the key mechanism signal of the type (CPU / memory / GC / threads) must be among the
           first to move (onset), else the type is penalised (THETA_T, as the between-entity onset rule)
  sigorder2 (amendment 13) the same rule applied only when typing the decided root (the entity marginals are unchanged)
  loadx2   (amendment 15) loadx + the latency rise that load causes + the CPU rise of the hosts of a load-explained service; load-explained
           signals do not date an entity's onset
  msgtarget (amendment 16) with torigin3: a failure whose message names the unreachable endpoint (URL / dial / lookup host) is
           relayed to THAT service (a caller reports its callee's failure to reach a deeper service)
"""
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

ALL_PARTS = ("score", "open", "twostage", "fp", "peer", "rnull", "prop", "events", "avail", "onset", "em", "ops", "selfnorm", "llmtype",
             "conseq", "retry", "mm", "xcard", "loadx", "ratio", "torigin", "torigin2", "omsg", "cpuspec", "torigin3", "hostcpu",
             "errclass", "tiebreak", "depconseq", "depbase", "depshare", "sigorder", "sigorder2", "loadx2", "msgtarget")
V5_PARTS = ("mm", "xcard", "loadx", "ratio", "torigin", "torigin2", "omsg", "cpuspec", "torigin3", "hostcpu", "errclass", "tiebreak",
            "depconseq", "depbase", "depshare", "sigorder", "sigorder2", "loadx2", "msgtarget")
DEP_LATENCY_TYPES = ("F010-slowSQL", "F018-dbNetworkLatency", "F029-redisUnavailable", "F009-cacheBreakdown",
                     "F005-messageQueueBacklog")                         # v5 'depconseq': cause = a slow / failing dependency
DEP_CONSEQUENCES = ("svc_thread_rise", "svc_cpu_rise", "svc_gc_rise")
DEFAULT_PARTS = tuple(p for p in ALL_PARTS if p not in ("rnull", "ops") + V5_PARTS)      # rnull/ops did not help; v5 parts opt-in
from pipeline.trace_structure import SIGNALS as TRACE_SIGNALS        # v5 'torigin' (pipeline/trace_structure.py)
LOAD_SIGNALS = ("svc_request_rise", "op_request_rise", "svc_cpu_rise")
THETA_M, MM_B0, MM_W = 0.6, 0.1, 10          # v5 'mm': P(>=2 modalities corroborate | true root); background floor and prior weight
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
        # (v4.0 also added svc_request_drop as support of cpuDeadLoop; removed in v4.3: a request drop accompanies many faults
        #  (memory pressure, DB latency, full load) and made cpuDeadLoop win 17 of 34 entity-right / type-wrong dev cases)
    if "avail" in parts:
        fp["F036-replicaScaleDown"]["key"].append("svc_unavailable")
    if "conseq" in parts:
        # v4.4: restarts, probe failures and OOM kills are often CONSEQUENCES of resource exhaustion. A resource hypothesis may explain
        # them (support), and an OOM kill no longer excludes memory pressure / full GC (the taxonomy: memory pressure "causing ...
        # eviction risk"; an OOM kill is where pressure ends).
        # Role 'may' (M): explained when present (THETA_U), uninformative when absent; a plain support would cost every case without
        # a restart log((1 - THETA_U) / (1 - ALPHA)).
        for t in ("F007-memoryPressure", "F022-fullGC"):
            fp[t]["con"] = [c for c in fp[t]["con"] if c != "ev_oom"]
        fp["F007-memoryPressure"]["may"] = ["ev_oom", "ev_restart", "ev_unhealthy"]
        fp["F022-fullGC"]["may"] = ["ev_oom", "ev_unhealthy", "ev_restart"]
        for t in ("F034-cpuFullLoad", "F012-cpuDeadLoop", "F002-threadExhaustion"):
            fp[t]["may"] = ["ev_unhealthy", "ev_restart"]
        # resourceLimitMisconfig = OOM kills / throttling at a limit set too low for NORMAL usage: an abnormal memory rise is what
        # memory pressure predicts, so it no longer supports the misconfiguration (it tied the two types whenever memory grew)
        fp["F039-resourceLimitMisconfig"]["sup"] = [x for x in fp["F039-resourceLimitMisconfig"]["sup"] if x != "svc_mem_rise"]
    if "ratio" in parts:
        # v5 relational signals (pipeline/fault_v4.ratio_signals): a CPU fault burns more CPU PER REQUEST; a traffic surge at a service
        # is more load than its callers send it; the raw rises remain as support
        for t in ("F034-cpuFullLoad", "F012-cpuDeadLoop"):
            fp[t]["key"] = ["svc_cpu_per_req_rise"]
            fp[t]["sup"] = list(dict.fromkeys(fp[t]["sup"] + ["svc_cpu_rise"]))
        fp["F006-trafficSurge"]["key"] = ["svc_req_share_rise"]
        fp["F006-trafficSurge"]["sup"] = list(dict.fromkeys(fp["F006-trafficSurge"]["sup"] + ["svc_request_rise"]))
        for t in ("F014-httpError5xx", "F011-codeDefect"):
            fp[t]["sup"] = list(dict.fromkeys(fp[t]["sup"] + ["svc_err_rate_rise"]))
    if "omsg" in parts:                                  # v5: the same evidence read at the origin spans
        for t, f in fp.items():
            atoms = {a for k in f.get("key", []) + f.get("sup", []) for a in _atoms(k)}
            extra = [("omsg_" + a[4:]) for a in atoms if a.startswith("msg_")]
            extra += [("omsg_" + a[4:-5]) for a in atoms if a.startswith("log_") and a.endswith("_rise")]
            extra += [("o" + a) for a in atoms if a in ("http429", "http403", "http503", "http5xx")]
            if extra:
                f["may"] = list(dict.fromkeys(f.get("may", []) + sorted(set(extra))))
    if "depconseq" in parts:                             # v5: threads blocked on a slow dependency raise thread count, CPU and GC
        for t in DEP_LATENCY_TYPES:                      # of the caller -> consequences ('may'), not evidence for resource types only
            if t in fp:
                fp[t]["may"] = list(dict.fromkeys(fp[t].get("may", []) + list(DEP_CONSEQUENCES)))
    if "cpuspec" in parts:                               # v5: deadLoop needs starvation evidence; plain CPU rise = fullLoad
        fp["F012-cpuDeadLoop"]["key"] = ["svc_cpu_rise&svc_request_drop"]
    if "ops" in parts:                                   # a failing operation (error count / rate of one endpoint)
        fp["F014-httpError5xx"]["key"].append("op_error_rise")
        for t in ("F011-codeDefect", "F023-nullPointerException", "F016-rateLimiting"):
            fp[t]["sup"].append("op_error_rise")
    return fp


class Model:
    """Hypotheses, roles and likelihoods for one fingerprint table (v3 functions, parametrised)."""

    def __init__(self, parts):
        self.parts = set(parts)
        self.origin_nodes = set()                                # v5 'torigin2' (set per case in diagnose_v4)
        self.cpu_up = set()                                      # v5 'hostcpu': entities whose svc_cpu_rise holds (per case)
        self.fp = fingerprints_v4(self.parts)
        self.roles = {}
        for t, f in self.fp.items():
            r = {}
            for role, key in (("M", "may"), ("C", "con"), ("U", "sup"), ("K", "key")):
                for expr in f.get(key, []):
                    for a in _atoms(expr):
                        r[a] = role
            self.roles[t] = r

    def keys(self, h):
        if h["type"] == ANOM:
            return NODE_CONT if h["level"] == "node" else SVC_CONT + (OP_CONT if "ops" in self.parts else [])
        return self.fp[h["type"]]["key"]

    @property
    def symptoms(self):
        return bd.SYMPTOMS | (set(OP_CONT) if "ops" in self.parts else set()) | ({"svc_request_rise"} if "retry" in self.parts else set())

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
        if (e, s) in db.get("_load_explained", ()):              # v5 'loadx': load that came from upstream
            base = max(base, bd.THETA_P)
        if s in TRACE_SIGNALS:                                   # v5 'torigin': own evidence of the root (or of its node)
            return max(bd.THETA_P, base) if self.trace_own(o, h, hosts) else base
        if h["type"] is None:
            return base
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
        return h["type"] in bd.LOAD_TYPES or ("hostcpu" in self.parts and h["type"] is not None and h["entity"] in self.cpu_up)

    def error_type(self, t):
        """v5 'errclass': the type's fingerprint names an error signal."""
        if t not in self.fp:
            return True
        f = self.fp[t]
        atoms = {a for k in f.get("key", []) + f.get("sup", []) for a in _atoms(k)}
        return any(a.startswith(("http", "msg_")) or a in ("error_spans", "svc_error_rise", "svc_err_rate_rise")
                   or a.endswith("_err") for a in atoms)

    def trace_own(self, o, h, hosts):
        if h["type"] is None:
            return False
        if o["entity"] == h["entity"]:
            if "errclass" in self.parts and o["signal"] == "trace_err_origin" and h["type"] != ANOM:
                return self.error_type(h["type"])
            return True
        if h["entity"] not in (hosts or {}).get(o["entity"], ()):
            return False
        return not ({"torigin2", "torigin3"} & self.parts) or h["entity"] in self.origin_nodes   # v2/v3: a node needs >= 2 origin services

    def may(self, o, h, hosts=None):
        if o["signal"] in TRACE_SIGNALS:                         # absence of a trace origin is uninformative
            return self.trace_own(o, h, hosts)
        return h["type"] not in (None, ANOM) and o["entity"] == h["entity"] and self.role(h, o["signal"]) == "M"

    def propagated(self, o, h, expl, hosts):
        s, e = o["signal"], o["entity"]
        if h["type"] is None or e == h["entity"]:
            return False
        return (s in self.symptoms and e in expl and h["entity"] in expl[e]) or \
               (s == "node_cpu_rise" and self.host_cpu(h) and e in hosts.get(h["entity"], []))

    def log_likelihood(self, obs, H, bg, expl, db):
        hosts = db.get("hosts", {})
        ll = np.zeros(len(H))
        for o in obs:
            for i, h in enumerate(H):
                p = self.p_holds(o, h, bg, expl, hosts, db)
                if not o["holds"] and (("prop" in self.parts and self.propagated(o, h, expl, hosts)) or self.may(o, h, hosts)):
                    p = self.p_holds(o, H[-1], bg, expl, hosts, db)      # partial propagation / possible consequence: absence is uninformative
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
                if not o["holds"] and self.may(o, h, hosts):
                    p = self.p_holds(o, H[-1], bg, expl, hosts, db)
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


def diagnose_v4(db, callees, texts, sensor=None, budget=bd.PROBE_BUDGET, top_m=8, alert=None, parts=None, case=None,
                debug=None) -> Dict:
    parts = set(parts) if parts is not None else parts_from_env()
    model = Model(parts)
    if "events" in parts and case is not None:
        new_events(case, db, B_SPAN * (db["w1"] - db["w0"]).total_seconds())
    if "score" in parts:
        calibrate_v4(db, peer="peer" in parts, robust_null="rnull" in parts, ops_level="ops" in parts, selfnorm="selfnorm" in parts,
                     ratio="ratio" in parts, callees=callees)
    obs = bd.observations(db)
    if "avail" in parts:
        for s_, bad in unavailability(db).items():
            obs.append({"entity": s_, "signal": "svc_unavailable", "holds": bad, "kind": "cat",
                        "ev": "deployment available replicas below desired / availability below full in the window" if bad else ""})
    if ("torigin" in parts or "torigin2" in parts or "torigin3" in parts) and case is not None:
        from pipeline.trace_structure import trace_observations, origin_nodes
        obs += trace_observations(case, db, v2="torigin2" in parts or "torigin3" in parts, omsg="omsg" in parts,
                                  v3="torigin3" in parts, msgtarget="msgtarget" in parts)
        if "torigin2" in parts or "torigin3" in parts:
            model.origin_nodes = origin_nodes(obs, db.get("hosts", {}))
    if "depbase" in parts and case is not None:            # v5: dependency latency against the service's own history
        from pipeline.dep_latency import dep_observations
        dep = dep_observations(case, db, share="depshare" in parts)
        obs = [o for o in obs if o["signal"] not in ("db_slow", "redis_slow", "queue_slow")] + dep
        db["_dep_override"] = {}                         # the same values for entity_signals() (LLM typing, fallback type)
        for o in dep:
            db["_dep_override"].setdefault(o["entity"], {})[o["signal"]] = {"kind": "cont", "p": o["p"] if o["holds"] else 1.0,
                                                                            "ev": o["ev"]}
    if "hostcpu" in parts:
        model.cpu_up = {o["entity"] for o in obs if o["signal"] == "svc_cpu_rise" and o["holds"]}
    H = model.testable(model.hypotheses(db), obs)
    bg = bd.background(obs, db)
    if alert and alert in db["services"]:
        obs.append({"entity": alert, "signal": "alert", "holds": True, "kind": "alert", "ev": "the alert fired on this service"})
    expl = bd.explainers(db, callees)
    if "loadx2" in parts:                                 # v5 amendment 15: load and what load causes
        db["_load_explained"] = load_explained(obs, callees, signals=LOAD_SIGNALS + ("svc_latency_rise",),
                                               hosts=db.get("hosts", {}))
    else:
        db["_load_explained"] = load_explained(obs, callees) if "loadx" in parts else set()
    extra, mm_beta, mm_k = np.zeros(len(H)), None, None
    if "mm" in parts:
        extra, mm_beta, mm_k = mm_loglik(model, H, obs)
    pi = None
    if "em" in parts:
        # propagation probability pi (P(a symptom shows at an entity the root can reach)) is a per-case nuisance parameter, estimated by
        # EM from the case itself (no labels): E-step = posterior over hypotheses; M-step = posterior-weighted share of reachable
        # symptoms that hold. A wide hypothesis then pays for every reachable entity that stays quiet, at the rate this case shows.
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
    else:
        logp = model.prior(H) + extra + model.log_likelihood(obs, H, bg, expl, db)
    onsets = {}
    if "onset" in parts:
        cal = db.get("_cal", {})
        lx = db["_load_explained"] if "loadx2" in parts else ()
        for ent, sigs in cal.items():
            ts = [r["onset"] for k, r in sigs.items() if r.get("onset") is not None and r.get("p") is not None and r["p"] <= bd.ALPHA
                  and (ent, k) not in lx]                 # loadx2: load that came from upstream does not date the entity's fault
            if ts:
                onsets[ent] = min(ts)
        logp = logp + onset_loglik(H, onsets, expl)
    if "sigorder" in parts:                              # v5: inside the root, the mechanism of the fault moves first
        logp = logp + sigorder_loglik(H, db.get("_cal", {}), model)
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
    if debug is not None:                                # diagnostics only (scripts/diagnostics/entity_errors_dev.py)
        debug.update(model=model, obs=obs, bg=bg, expl=expl, H=H, logp=logp, pi=pi)
    order = np.argsort(-post, kind="stable") if "tiebreak" in parts else np.argsort(-post)
    if "twostage" in parts:
        marg = defaultdict(float)
        for h, p in zip(H[:-1], post[:-1]):
            marg[h["entity"]] += float(p)
        ent = max(marg, key=marg.get)
        tpost, torder = post, order
        if "sigorder2" in parts:                         # v5 (amendment 13): the order rule types the root, it does not move it
            so = sigorder_loglik(H, db.get("_cal", {}), model)
            idx = [i for i in range(len(H) - 1) if H[i]["entity"] == ent]
            w = post[idx] * np.exp(so[idx])
            if w.sum() > 0:
                tpost = post.copy()
                tpost[idx] = w / w.sum() * marg[ent]
                torder = np.argsort(-tpost, kind="stable") if "tiebreak" in parts else np.argsort(-tpost)
        own = [i for i in torder if H[i]["entity"] == ent and H[i]["type"] not in (None, ANOM)]
        level = next(h["level"] for h in H if h["entity"] == ent)
        type_rank = [(H[i]["type"], round(float(tpost[i]) / max(marg[ent], 1e-12), 4)) for i in own]
        if own:
            best = own[0]
            typ, p_type = H[best]["type"], float(tpost[best]) / max(marg[ent], 1e-12)
        else:
            best = next(i for i in order if H[i]["entity"] == ent)
            typ, p_type = _fallback_type(model, ent, level, entity_signals(db, ent)), None
        p_ent = marg[ent]
    else:
        best = next(i for i in order if H[i]["type"] not in (None, ANOM))
        ent, level, typ = H[best]["entity"], H[best]["level"], H[best]["type"]
        p_ent, p_type = float(post[best]), None
        type_rank = [(H[i]["type"], round(float(post[i]), 4)) for i in order if H[i]["entity"] == ent and H[i]["type"] not in (None, ANOM)]
    top = [{"entity": H[i]["entity"], "type": H[i]["type"], "p": round(float(post[i]), 4)} for i in order[:5]]
    hb = {**H[best], "type": typ if typ in model.fp else H[best]["type"]}
    if hb["type"] == ANOM:
        hb = H[best]
    hosts = db.get("hosts", {})
    contrib = []
    for o in obs:
        if ({"torigin", "torigin2", "torigin3"} & set(parts)) and not o["holds"] and (("prop" in parts and model.propagated(o, hb, expl, hosts))
                                                       or model.may(o, hb, hosts)):
            continue    # v5: an uninformative absence (as in the likelihood) is not listed as evidence; A (frozen) is unchanged
        p1, p0 = model.p_holds(o, hb, bg, expl, hosts, db), model.p_holds(o, H[-1], bg, expl, hosts, db)
        lr = math.log((p1 if o["holds"] else 1 - p1) / (p0 if o["holds"] else 1 - p0))
        if abs(lr) > 0.5:
            contrib.append({"entity": o["entity"], "signal": o["signal"], "holds": o["holds"], "log_bf": round(lr, 2), "ev": o["ev"]})
    contrib.sort(key=lambda x: -abs(x["log_bf"]))
    return {"entity": ent, "level": level, "type": typ, "p": p_ent, "p_type": p_type, "p_h0": float(post[-1]),
            "entropy_bits": round(bd.entropy(post), 3), "top": top, "probes": log, "n_obs": len(obs), "n_hyp": len(H),
            "evidence": contrib[:12], "parts": sorted(parts), "pi": pi, "type_rank": type_rank[:8],
            "candidate_types": [t for t in model.fp if LEVEL[t] in (level, "any")],
            "onset_root": onsets.get(ent), "n_onsets": len(onsets), "onsets": onsets,
            "load_explained": sorted({e for e, _ in db["_load_explained"]}),
            "trace_structure": db.get("_trace_structure"),
            "mm_beta": mm_beta, "mm_k": int(mm_k[best]) if mm_k is not None else None,
            "mm_n_corroborated": int((mm_k >= 2).sum()) if mm_k is not None else None,
            "holding": [{"entity": o["entity"], "signal": o["signal"], "ev": o["ev"]} for o in obs if o["holds"]]}


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


MECHANISM_SIGNALS = ("svc_cpu_rise", "svc_mem_rise", "svc_gc_rise", "svc_thread_rise")   # the resources a process consumes


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
    """v5 'sigorder' (amendment 12): within the root, the cause moves before its consequences. CPU, memory, GC and thread faults of
    one process raise the same four signals; which one moved FIRST tells them apart (memory leak -> GC -> CPU; CPU saturation ->
    GC / threads later). For h = (e*, type) whose key names one of the MECHANISM_SIGNALS, and when at least one of them has a
    measured onset at e*: THETA_T if a key of the type is a first mover, neutral (0.5) if the key holds without a measured onset
    (a gradual change: order unknown), 1 - THETA_T otherwise (the key moved later, or not at all). Other types: neutral. Same
    constant and resolution as the between-entity onset rule; no new constants."""
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
    """Telemetry modality of an observation (v5 'mm'): events = Kubernetes events; traces = status codes and span outcomes;
    logs = log-template rises and log / span messages; everything else is a metric (incl. availability state)."""
    if signal.startswith("ev_"):
        return "events"
    if signal.startswith(("log_", "msg_")) or signal == "message":
        return "logs"
    if signal.startswith(("http", "ohttp", "omsg_", "trace_")) or signal == "error_spans" or signal.split("_")[0] in ("db", "redis", "queue"):
        return "traces"
    return "metrics"


def mm_loglik(model, H, obs):
    """v5 multimodal corroboration term (see the module docstring, part 'mm'). Returns (log-lik vector, beta_M, k per hypothesis)."""
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
    """v5 'loadx': {(service, signal)} for the load signals of services whose request count rises together with a caller's.
    'loadx2' (amendment 15): the signals include the latency rise that load causes, and the CPU rise of the NODES hosting a service
    whose CPU rise is load-explained (the load heats the host too) is load-explained as well."""
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
