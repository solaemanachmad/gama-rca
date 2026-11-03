"""Belief->LLM arm (PREREGISTRATION_V5.md, amendment of 2026-11-03): GAMA's computed belief given to an LLM that decides FREELY.

Same prompt as the Evidence->LLM baseline (pipeline/evidence_llm.build_prompt: the same holding observations per entity, call graph,
hosts, log templates, 28 fault types) plus ONE block: the statistical model's ranking (top-5 entities with P(root), each with its
top-3 fault types and P(type | entity), and the model's answer). The LLM may keep the answer or choose any other entity / type; no
shortlist, no quote check. One call per case; its chain is scored, as for Evidence->LLM.

The gradation of the cross-model study:
  Evidence->LLM  (evidence only, LLM decides)  <  Belief->LLM (evidence + computed belief, LLM decides)
  GAMA + LLM     (belief decides, LLM bounded) ;  GAMA (belief decides, no LLM)
Belief->LLM vs GAMA answers "does a capable LLM improve the computed belief or damage it?"; Belief->LLM vs Evidence->LLM answers "does
the computed belief help the LLM?". Nothing here reads answer_key/."""
from __future__ import annotations

import time

import config
from data.loader import Case, find_service_ancestor
from pipeline.propagation_evidence import _node_name
from schema import RCAResult
from gama import evidence
from gama.chain import callee_map, _ids
from gama.config import Modules
from gama.decide import diagnose


def belief_block(res) -> str:
    lines = ["Statistical diagnosis (a Bayesian model of the same evidence: every candidate root cause is an entity with a fault "
             "type, scored by how well it explains the evidence along the call graph and the hosting nodes; P(root) sums to 1 "
             "over the entities). Most probable root-cause entities:"]
    for k, r in enumerate(res.get("entity_rank") or [], 1):
        types = ", ".join(f"{t} {p:.2f}" for t, p in r["types"]) or "(no specific type)"
        lines.append(f"{k}. {r['entity']} ({r['level']}): P(root) = {r['p']:.2f}; most probable fault types given this entity: {types}")
    lines.append(f"The model's answer: root-cause entity = {res['entity']}, fault type = {res['type']}.")
    return "\n".join(lines)


def run_belief_llm(case, llm) -> RCAResult:
    from pipeline.evidence_llm import build_prompt, EVIDENCE_SYSTEM
    from pipeline.log_preprocess import preprocess_texts
    config.ALERT_T0_MODE = "window"
    m = Modules.parse("core")
    t0 = time.time()
    llm.reset_usage()
    at = case.alert.alert_timestamp
    entry = case.alert.entry_entity_id
    sid = find_service_ancestor(entry, case.topology) if entry else None
    alert = _node_name(case.topology, sid) if sid else None
    op_name = _node_name(case.topology, entry) if (entry and entry in case.topology) else None
    db = evidence.signal_db(case, at, m)
    if "services" not in db:
        raise RuntimeError(f"signal database failed: {db.get('error')}")
    callees = callee_map(case.topology)
    res = diagnose(db, callees, {}, m, alert=alert, case=case)
    res.pop("_model", None)
    ev = {}
    for o in res["holding"]:
        if o["signal"] != "alert":
            ev.setdefault(o["entity"], []).append(f"{o['signal']}: {o['ev']}" if o["ev"] else o["signal"])
    texts, _notes, _stats = preprocess_texts(case, db)
    block = belief_block(res)
    prompt = build_prompt(case.alert.alert_text, alert, op_name, callees, db.get("hosts", {}), ev, texts, belief_block=block)
    out = llm.generate_json(prompt, system=EVIDENCE_SYSTEM)
    out = out if isinstance(out, dict) else {}
    name = str(out.get("root_cause_entity", "")).strip()
    typ = str(out.get("predicted_fault_type", "unknown"))
    ids = _ids(case.topology)
    eid = ids.get(name)
    if eid is None:                                       # same tolerance as Evidence->LLM (case / whitespace only)
        low = {k.lower(): v for k, v in ids.items()}
        eid = low.get(name.lower(), name or None)
    stats = {"total_pipeline_time_s": round(time.time() - t0, 1), "evidence_prompt_chars": len(prompt), "evidence_entities": len(ev),
             "evidence_answer_entity": name, "belief_entity": res["entity"], "belief_type": res["type"],
             "belief_kept_entity": name == res["entity"], "belief_kept_type": typ == res["type"], **llm.usage_stats(),
             "_trace": {"alert_service": alert, "alert_operation": op_name, "belief_block": block, "evidence_by_entity": ev,
                        "answer": out}}
    return RCAResult(case_id=getattr(case, "case_id", ""), predicted_entity_ids=[eid] if eid else [], predicted_fault_type=typ,
                     reasoning_chain=[str(x) for x in (out.get("reasoning_chain") or [])],
                     confidence=float(out.get("confidence", 0.0) or 0.0), agent_findings=[], retrieval_stats=stats, evidence_items=[])


def belief_llm(case_id, llm, cases_dir=config.CASES_DIR):
    """main.py system 'belief_llm'."""
    return run_belief_llm(Case(case_id, cases_dir=cases_dir), llm)
