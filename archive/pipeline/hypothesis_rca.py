"""Hypothesis-constrained RCA (v2; research design: claude/GAMA-RCA_v2_research_design.md in the project).

    Stage 1  evidence map     calibrated continuous signals + categorical observations for EVERY service and node of the case
                              (pipeline/fault_calibrated.py: rank-test scores calibrated on pseudo-windows of the case's own baseline)
    Stage 2  hypotheses       (entity, type) admitted when a key signal of the type holds on the entity, no excluding signal holds,
                              and the type's level matches the entity's level (FINGERPRINTS in fault_calibrated.py)
    Stage 3  topology         a hypothesis must be able to explain the alert: a service downstream of the alerted service (faults
                              propagate callee -> caller) or a node hosting such a service. Root rule of graph-based RCA: prefer an
                              entity none of whose dependencies (callees, hosting nodes) carries a specific admitted hypothesis.
                              Ranking: (consistent, specific, root, #categorical, Fisher strength, depth).
    Stage 4  LLM              chooses ONE of the top-K hypotheses (constrained) and writes cause -> propagation -> impact with values
                              copied from the evidence; the verifier checks the quoted numbers (one repair call).
    Fallback                  no admitted hypothesis: one free LLM diagnosis over the same evidence, flagged 'abstain'.

Systems: hypothesis_rca (Stages 1-4) and hypothesis_rule (Stages 1-3 only: top hypothesis, no LLM; chain written from the signals).
Ablation flags (config): V2_NO_TOPO, V2_NO_EXCLUDE, V2_UNCALIBRATED, V2_NO_VERIFY. All parameters are fixed in config, none chosen
by looking at labels. Uses only the case's telemetry, topology and alert; never reads answer_key/.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

import config
from data.loader import Case, find_service_ancestor
from data.taxonomy import FAULT_DEFINITIONS, fault_group, taxonomy_prompt_block, knowledge_prompt_block_pos
from pipeline.fault_calibrated import (compute_signal_db, entity_signals, FINGERPRINTS, LEVEL, RESIDUAL, _holds_expr, _atoms,
                                       holds, _ORDER)
from pipeline.propagation_evidence import compute_propagation_evidence, _calls_graph, _node_name, _node_type
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.trace_evidence import compute_trace_evidence
from schema import RCAResult, EvidenceItem

SELECT_SYSTEM = ("You are a site-reliability engineer. Code has checked the telemetry and admitted a few root-cause hypotheses, each with "
                 "the evidence that admitted it. You choose the hypothesis that best explains the alert and write the causal chain. "
                 "Respond ONLY with valid JSON.")
FALLBACK_SYSTEM = ("You are a site-reliability engineer performing root-cause analysis on microservice telemetry. Respond ONLY with "
                   "valid JSON.")
CHAIN_RULES = ("Each chain step is 'step_type: target signal=value ...' with step_type in cause / propagation / impact; 'target' is a "
               "service, operation, node or pod NAME from the evidence; copy every value exactly as the evidence shows it (absolute "
               "values in their own units: request count, latency, error count, cpu/memory usage, replicas, event reasons and counts).")


# ------------------------------------------------------------------------------------------------------------------------------
# Stages 1-3
# ------------------------------------------------------------------------------------------------------------------------------
def _entities(topo):
    svc, nod = {}, {}
    for n in topo.nodes:
        t = _node_type(topo, n)
        if t in ("apm.service", "apm.svc"):
            svc.setdefault(_node_name(topo, n), n)
        elif t == "k8s.node":
            nod.setdefault(_node_name(topo, n), n)
    return svc, nod


def admitted(sig: Dict, ent: str, level: str, alpha: float) -> List[Dict]:
    out = []
    for t, fp in FINGERPRINTS.items():
        if LEVEL[t] not in (level, "any"):
            continue
        keys = [k for k in fp["key"] if _holds_expr(k, sig, alpha)]
        if not keys:
            continue
        if not config.V2_NO_EXCLUDE and any(_holds_expr(c, sig, alpha) for c in fp["con"]):
            continue
        sups = [k for k in fp["sup"] if _holds_expr(k, sig, alpha)]
        atoms = {a for k in keys + sups for a in _atoms(k) if holds(sig.get(a), alpha)}
        n_cat = sum(1 for a in atoms if sig[a]["kind"] == "cat")
        fisher = sum(-2.0 * math.log(max(sig[a]["p"], 1e-300)) for a in atoms if sig[a]["kind"] == "cont")
        out.append({"entity": ent, "level": level, "type": t, "key": keys, "sup": sups, "n_cat": n_cat, "fisher": round(fisher, 2),
                    "residual": t in RESIDUAL, "signals": {a: sig[a]["ev"] for a in sorted(atoms)}})
    return out


def hypotheses(case, db, alert_svc: Optional[str]) -> Dict:
    """All admitted (entity, type) hypotheses of the case with topology features, ranked."""
    alpha = config.FP_ALPHA
    topo = case.topology
    svc_ids, node_ids = _entities(topo)
    calls = _calls_graph(topo)
    name_calls = nx.DiGraph()
    for u, v in calls.edges():
        if _node_type(topo, u) in ("apm.service", "apm.svc") and _node_type(topo, v) in ("apm.service", "apm.svc"):
            name_calls.add_edge(_node_name(topo, u), _node_name(topo, v))
    hosted = defaultdict(set)
    for s, ns in db.get("hosts", {}).items():
        for n in ns:
            hosted[n].add(s)
    if alert_svc and alert_svc in name_calls:
        down = {alert_svc} | nx.descendants(name_calls, alert_svc)
        depth = nx.single_source_shortest_path_length(name_calls, alert_svc)
    else:
        down, depth = ({alert_svc} if alert_svc else None), ({alert_svc: 0} if alert_svc else {})

    sigs, H = {}, []
    for ent in list(db["services"]) + list(db["nodes"]):
        level = "node" if ent in db["nodes"] else "service"
        sigs[ent] = entity_signals(db, ent)
        if config.V2_UNCALIBRATED:
            for s in sigs[ent].values():
                if s["kind"] == "cont" and "raw" in s:
                    s["p"] = s["raw"]
        H.extend(admitted(sigs[ent], ent, level, alpha))
    # v2 dev round 1, node locality: a node-level fault concerns ONE node (definitions of nodeCpuHigh / nodeMemoryOOM / nodeDown /
    # diskIOHigh). When the same node-level type is admitted on two or more nodes at once, the rise is shared (typically load from
    # the services they host), so those node hypotheses are marked diffuse: they stay listed but neither rank as specific nor make
    # their hosted services non-root.
    per_type = defaultdict(int)
    for h in H:
        if h["level"] == "node":
            per_type[h["type"]] += 1
    for h in H:
        h["diffuse"] = h["level"] == "node" and per_type[h["type"]] >= 2 and not config.V2_NO_LOCALITY
    specific = {h["entity"] for h in H if not h["residual"] and not h["diffuse"]}
    for h in H:
        e = h["entity"]
        if h["level"] == "node":
            served = hosted.get(e, set())
            h["consistent"] = True if down is None else bool(served & down)
            h["depth"] = 1 + min((depth[s] for s in served if s in depth), default=99)
            deps = set()
        else:
            h["consistent"] = True if down is None else e in down
            h["depth"] = depth.get(e, 99)
            deps = (set(name_calls.successors(e)) if e in name_calls else set()) | set(db.get("hosts", {}).get(e, []))
        h["root"] = not (deps & specific)
        if config.V2_NO_TOPO:
            h["consistent"], h["root"] = True, True
    H.sort(key=lambda h: (not h["consistent"], h["residual"] or h["diffuse"], not h["root"], -h["n_cat"], -h["fisher"],
                          -(h["depth"] if h["depth"] < 99 else -1), _ORDER[h["type"]]))
    return {"H": H, "signals": sigs, "ids": {**svc_ids, **node_ids}, "down": down}


# ------------------------------------------------------------------------------------------------------------------------------
# Stage 4
# ------------------------------------------------------------------------------------------------------------------------------
def _hyp_block(H):
    lines = []
    for i, h in enumerate(H, 1):
        pos = "root: no dependency carries a specific hypothesis" if h["root"] else "a dependency also carries a hypothesis"
        lines.append(f"H{i}. entity={h['entity']} ({h['level']}) type={h['type']} [{fault_group(h['type'])}]"
                     + (" (residual type)" if h["residual"] else "") + (" (same node signal on several nodes)" if h.get("diffuse") else "")
                     + f" | {pos}; hops from the alerted service: "
                     + (str(h["depth"]) if h["depth"] < 99 else "n/a"))
        lines.append(f"    definition: {FAULT_DEFINITIONS.get(h['type'], '')}")
        for k, v in h["signals"].items():
            lines.append(f"    - {k}: {v}")
    return "\n".join(lines)


def _evidence_text(ev):
    return "\n\n".join(t for t in (ev["prop"], ev["infra"], ev["trace"]) if t)


def _rule_chain(h, alert_svc):
    sig = "; ".join(f"{k}: {v}" for k, v in list(h["signals"].items())[:3])
    chain = [f"cause: {h['entity']} {h['type']} ({sig})"]
    if alert_svc and alert_svc != h["entity"]:
        chain += [f"propagation: {h['entity']} -> {alert_svc}", f"impact: {alert_svc} alert"]
    else:
        chain += [f"impact: {h['entity']} alert"]
    return chain


def _safe(fn, *a):
    try:
        return fn(*a) or {}
    except Exception as e:  # evidence tables must never break a run
        return {"text": "", "error": repr(e)}


def hypothesis_rca(case_id: str, llm, cases_dir: str = config.CASES_DIR, use_llm: bool = True) -> RCAResult:
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    return diagnose(case, llm, use_llm=use_llm, t0=t0)


def diagnose(case, llm, use_llm: bool = True, t0: Optional[float] = None) -> RCAResult:
    t0 = t0 or time.time()
    case_id = getattr(case, "case_id", "")
    if llm is not None:
        llm.reset_usage()
    at = case.alert.alert_timestamp
    entry = case.alert.entry_entity_id
    alert_svc_id = find_service_ancestor(entry, case.topology) if entry else None
    alert_svc = _node_name(case.topology, alert_svc_id) if alert_svc_id else None
    stats = {"v2_alert_service": alert_svc}
    t1 = time.time()
    db = compute_signal_db(case, at)
    if "error" in db or "services" not in db:
        raise RuntimeError(f"signal database failed: {db.get('error')}")
    hy = hypotheses(case, db, alert_svc)
    H = hy["H"]
    stats.update(v2_evidence_time_s=round(time.time() - t1, 1), v2_n_hypotheses=len(H),
                 v2_n_consistent=sum(h["consistent"] for h in H), v2_n_specific=sum(not h["residual"] for h in H),
                 v2_n_diffuse=sum(h.get("diffuse", False) for h in H))
    prop = _safe(compute_propagation_evidence, case, at, entry)
    infra = _safe(compute_infra_evidence, case, at, entry)
    trace = _safe(compute_trace_evidence, case, at, entry)
    ev = {"prop": prop.get("text") or "", "infra": infra.get("text") or "", "trace": trace.get("text") or ""}
    top = [h for h in H if h["consistent"]][: config.FP_TOP_K]
    stats["v2_top"] = "|".join(f"{h['entity']}:{h['type']}" for h in top)
    chosen, chain, conf, mode = None, [], 0.0, ""
    if top and not use_llm:
        chosen, chain, mode = top[0], _rule_chain(top[0], alert_svc), "rule"
    elif top:
        kp = knowledge_prompt_block_pos() + "\n\n" if config.USE_KNOWLEDGE_POS else ""
        prompt = (f"Alert: {case.alert.alert_text}\nAlerted service: {alert_svc or '(none given)'}\n\n"
                  f"Admitted hypotheses:\n{_hyp_block(top)}\n\nEvidence tables:\n{_evidence_text(ev)}\n\n{kp}"
                  "Task: choose the ONE hypothesis whose entity and fault type best explain the alert, then write the causal chain "
                  "from that entity to the alerted service. " + CHAIN_RULES + "\n"
                  'Respond as JSON: {"choice": "H<n>", "reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."], '
                  '"confidence": <float 0-1>}')
        res = llm.generate_json(prompt, system=SELECT_SYSTEM)
        m = re.search(r"H?\s*(\d+)", str(res.get("choice", ""))) if isinstance(res, dict) else None
        k = int(m.group(1)) - 1 if m else -1
        stats["v2_choice_valid"] = 0 <= k < len(top)
        chosen = top[k] if 0 <= k < len(top) else top[0]
        chain = [str(x) for x in (res.get("reasoning_chain") or [])] if isinstance(res, dict) else []
        conf = float(res.get("confidence", 0) or 0) if isinstance(res, dict) else 0.0
        chain = chain or _rule_chain(chosen, alert_svc)
        mode = "select"
        stats["v2_prompt_chars"] = len(prompt)
    else:
        mode = "abstain"
        if use_llm:
            prompt = (f"Alert: {case.alert.alert_text}\nAlerted service: {alert_svc or '(none given)'}\n\n{_evidence_text(ev)}\n\n"
                      f"{taxonomy_prompt_block()}\n\nTask: write the root-cause chain first, then name the root-cause entity (the cause "
                      "step's target) and its fault type. " + CHAIN_RULES + "\n"
                      'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."], '
                      '"predicted_entity": "<name>", "predicted_fault_type": "<taxonomy slug>", "confidence": <float 0-1>}')
            res = llm.generate_json(prompt, system=FALLBACK_SYSTEM)
            res = res if isinstance(res, dict) else {}
            chain = [str(x) for x in (res.get("reasoning_chain") or [])]
            conf = float(res.get("confidence", 0) or 0)
            chosen = {"entity": str(res.get("predicted_entity") or alert_svc or ""), "type": str(res.get("predicted_fault_type") or "unknown")}
        else:
            chosen = {"entity": alert_svc or "", "type": "unknown"}
    stats["v2_mode"] = mode
    # verifier: quoted numbers must occur in the evidence the LLM saw (signals included); one repair call
    if use_llm and chain and not config.V2_NO_VERIFY:
        from agents.multi_agent import _evidence_pool, _grounding
        sig_txt = "\n".join(v for h in top for v in h["signals"].values())
        pool, _ = _evidence_pool({"propagation_text": ev["prop"], "infra_text": ev["infra"] + "\n" + sig_txt, "trace_text": ev["trace"]})
        g0, n0, bad = _grounding(chain, pool)
        stats.update(v2_grounded_before=round(g0, 3), v2_numbers=n0)
        if n0 and g0 < config.VERIFY_MIN_GROUNDED:
            fix = llm.generate_json(
                f"Evidence:\n{_evidence_text(ev)}\n{sig_txt}\n\nChain:\n" + "\n".join(f"- {s}" for s in chain)
                + f"\n\nThese quoted values do not occur in the evidence: {', '.join(bad[:12])}. Rewrite the chain with the same steps "
                  "and targets, copying every value exactly as the evidence shows it and dropping a signal no row shows.\n"
                  'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."]}', system=SELECT_SYSTEM)
            new = [str(x) for x in (fix.get("reasoning_chain") or [])] if isinstance(fix, dict) else []
            g1 = _grounding(new, pool)[0] if new else 0.0
            if new and g1 > g0:
                chain, g0 = new, g1
                stats["v2_repaired"] = True
        stats["v2_grounded_after"] = round(g0, 3)
    eid = hy["ids"].get(chosen["entity"]) or chosen["entity"]
    stats.update(total_pipeline_time_s=round(time.time() - t0, 1), v2_entity=chosen["entity"], v2_type=chosen["type"],
                 v2_chosen_rank=(top.index(chosen) + 1) if chosen in top else None,
                 v2_hypotheses_all="|".join(f"{h['entity']}:{h['type']}{'' if h['consistent'] else '(x)'}" for h in H[:15]))
    if llm is not None:
        stats.update(llm.usage_stats())
    items = [EvidenceItem(observation=o) for o in infra.get("observations", [])]
    return RCAResult(case_id=case_id, predicted_entity_ids=[eid] if eid else [], predicted_fault_type=chosen["type"],
                     reasoning_chain=chain, confidence=conf, agent_findings=[], retrieval_stats=stats, evidence_items=items)


def hypothesis_rule(case_id: str, llm, cases_dir: str = config.CASES_DIR) -> RCAResult:
    return hypothesis_rca(case_id, None, cases_dir, use_llm=False)
