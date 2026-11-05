"""Stage 3 -- decision: (optional) text probes chosen by expected information gain, then a two-stage answer.

  probes  (M7, with an LLM sensor) while no hypothesis reaches STOP_P, read the log text of the entity whose category answer is
          expected to reduce the entropy most (EIG >= EIG_MIN, at most PROBE_BUDGET probes); the answer is a likelihood term
  entity  e* = argmax of the marginal posterior P(e) (h0 excluded: an alert fired, an answer is required)
  type    argmax P(type | e*); M3 re-weights the types of e* by the intra-entity signal order (it types the root, it does not move
          it) and resolves exact ties general-before-special (fingerprint order, stable sort)
The evidence list (log Bayes factors of the decided hypothesis against h0) is what the chain and the LLM stages cite.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict

import numpy as np

from pipeline import belief_diagnosis as bd
from pipeline.fault_calibrated import LEVEL, RESIDUAL, FALLBACK, _atoms, entity_signals
from gama import evidence
from gama.belief import ANOM, Model, infer, sigorder_loglik
from gama.config import Modules


def _probe_dist(h, ent):
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
    return best or FALLBACK["node" if level == "node" else "service"]


def _probes(model, H, logp, texts, sensor, budget, top_m):
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
            M = np.array([_probe_dist(h, e) for h in H])
            pc = post @ M
            exp_h = sum(pc[c] * bd.entropy((post * M[:, c]) / (post * M[:, c]).sum()) for c in range(M.shape[1]) if pc[c] > 0)
            gains[e] = bd.entropy(post) - exp_h
        e, g = max(gains.items(), key=lambda kv: kv[1])
        if g < bd.EIG_MIN:
            break
        cat = sensor(e, texts[e])
        c = bd.CATS.index(cat) if cat in bd.CATS else len(bd.CATS)
        before = float(post.max())
        logp = logp + np.log(np.array([_probe_dist(h, e)[c] for h in H]))
        probed.add(e)
        steps += 1
        log.append({"probe": e, "eig_bits": round(g, 3), "category": cat, "max_belief_before": round(before, 3),
                    "max_belief_after": round(float(bd.posterior(logp).max()), 3)})
    return logp, log


def diagnose(db, callees, texts, m: Modules, sensor=None, budget=bd.PROBE_BUDGET, top_m=8, alert=None, case=None,
             debug=None) -> Dict:
    model = Model(m)
    obs = evidence.observations(db, callees, case, m, model)                           # stage 1
    b = infer(model, db, callees, obs, alert=alert)                                    # stage 2
    H, bg, expl = b["H"], b["bg"], b["expl"]
    logp, log = _probes(model, H, b["logp"], texts, sensor, budget, top_m)             # stage 4 sensor (M7)
    post = bd.posterior(logp)
    if debug is not None:
        debug.update(model=model, obs=obs, bg=bg, expl=expl, H=H, logp=logp, pi=b["pi"])
    order = np.argsort(-post, kind="stable") if m.typing else np.argsort(-post)
    marg = defaultdict(float)                                                          # stage 3: entity, then type
    for h, p in zip(H[:-1], post[:-1]):
        marg[h["entity"]] += float(p)
    ent = max(marg, key=marg.get)
    tpost, torder = post, order
    if m.typing:                                                                       # M3: the signal order types the root
        so = sigorder_loglik(H, db.get("_cal", {}), model)
        idx = [i for i in range(len(H) - 1) if H[i]["entity"] == ent]
        w = post[idx] * np.exp(so[idx])
        if w.sum() > 0:
            tpost = post.copy()
            tpost[idx] = w / w.sum() * marg[ent]
            torder = np.argsort(-tpost, kind="stable")
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
    top = [{"entity": H[i]["entity"], "type": H[i]["type"], "p": round(float(post[i]), 4)} for i in order[:5]]
    entity_rank = []                                                                   # top entities, each with its top types
    for e in sorted(marg, key=lambda x: -marg[x])[:5]:
        ix = [i for i in range(len(H) - 1) if H[i]["entity"] == e and H[i]["type"] not in (None, ANOM)]
        ix.sort(key=lambda i: -float(post[i]))
        entity_rank.append({"entity": e, "level": H[ix[0]]["level"] if ix else next(h["level"] for h in H if h["entity"] == e),
                            "p": round(float(marg[e]), 4),
                            "types": [(H[i]["type"], round(float(post[i]) / max(marg[e], 1e-12), 3)) for i in ix[:3]]})
    hb = {**H[best], "type": typ if typ in model.fp else H[best]["type"]}
    if hb["type"] == ANOM:
        hb = H[best]
    hosts = db.get("hosts", {})
    contrib = []
    for o in obs:
        if m.trace and not o["holds"] and (model.propagated(o, hb, expl, hosts) or model.may(o, hb, hosts)):
            continue                                       # an uninformative absence (as in the likelihood) is not evidence
        p1, p0 = model.p_holds(o, hb, bg, expl, hosts, db), model.p_holds(o, H[-1], bg, expl, hosts, db)
        lr = math.log((p1 if o["holds"] else 1 - p1) / (p0 if o["holds"] else 1 - p0))
        if abs(lr) > 0.5:
            contrib.append({"entity": o["entity"], "signal": o["signal"], "holds": o["holds"], "log_bf": round(lr, 2), "ev": o["ev"]})
    contrib.sort(key=lambda x: -abs(x["log_bf"]))
    mm_k, onsets = b["mm_k"], b["onsets"]
    return {"entity": ent, "level": level, "type": typ, "p": p_ent, "p_type": p_type, "p_h0": float(post[-1]),
            "entropy_bits": round(bd.entropy(post), 3), "top": top, "probes": log, "n_obs": len(obs), "n_hyp": len(H),
            "evidence": contrib[:12], "parts": sorted(m.legacy_parts()), "modules": m.spec(), "pi": b["pi"],
            "type_rank": type_rank[:8], "entity_rank": entity_rank, "candidate_types": [t for t in model.fp if LEVEL[t] in (level, "any")],
            "onset_root": onsets.get(ent), "n_onsets": len(onsets), "onsets": onsets,
            "load_explained": sorted({e for e, _ in db["_load_explained"]}),
            "trace_structure": db.get("_trace_structure"),
            "mm_beta": b["mm_beta"], "mm_k": int(mm_k[best]) if mm_k is not None else None,
            "mm_n_corroborated": int((mm_k >= 2).sum()) if mm_k is not None else None,
            "holding": [{"entity": o["entity"], "signal": o["signal"], "ev": o["ev"]} for o in obs if o["holds"]],
            "_model": model}
