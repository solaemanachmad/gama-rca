"""Baseline EVIDENCE_LLM (v5 protocol, PREREGISTRATION_V5.md): the LLM DECIDES from the SAME calibrated evidence that GAMA-RCA's
Bayesian model receives, without the model. It isolates the contribution of the Bayesian decision layer (fingerprints, topology
explain-away, EM propagation, onset, two-stage choice): GAMA-RCA (V4) vs EVIDENCE_LLM differ only in who decides.

Input to the LLM (one call): the alert, the alerted service and operation, the call graph and the hosting nodes, every observation that
HOLDS per entity (calibrated continuous signals with their values, new Kubernetes events, status codes, failing dependency spans,
availability), the significant log templates per entity (same preprocessing as the text sensor), and the 28 fault types.
Nothing here reads answer_key/."""
from __future__ import annotations

import time
from typing import Dict, List

from data.loader import Case, find_service_ancestor
from data.taxonomy import taxonomy_prompt_block
from pipeline import belief_diagnosis as bd
from pipeline.fault_calibrated import compute_signal_db
from pipeline.fault_v4 import calibrate_v4, new_events, unavailability, B_SPAN
from pipeline.propagation_evidence import _node_name
from schema import RCAResult

EVIDENCE_SYSTEM = ("You are a site-reliability engineer performing root cause analysis of a microservice incident from summarised, "
                   "statistically tested evidence. Respond ONLY with valid JSON.")
MAX_ENT_LINES, MAX_TEXTS, MAX_ENTITIES = 12, 5, 30


def build_prompt(alert_text: str, alert: str, op_name: str, callees: Dict[str, List[str]], hosts: Dict[str, List[str]],
                 ev_by_entity: Dict[str, List[str]], texts: Dict[str, List[str]]) -> str:
    edges = [f"{a} -> {b}" for a, bs in sorted(callees.items()) for b in bs]
    host = [f"{s}: {', '.join(ns)}" for s, ns in sorted(hosts.items()) if ns]
    order = sorted(ev_by_entity, key=lambda e: -len(ev_by_entity[e]))[:MAX_ENTITIES]
    blocks = []
    for e in order:
        lines = [f"  - {l}" for l in ev_by_entity[e][:MAX_ENT_LINES]]
        lines += [f"  - log: {t}" for t in (texts or {}).get(e, [])[:MAX_TEXTS]]
        blocks.append(f"{e}:\n" + "\n".join(lines))
    return (f"Alert: {alert_text}\nAlerted service: {alert or '(none given)'}\nAlerted operation: {op_name or '(none given)'}\n\n"
            "Call graph (caller -> callee):\n" + ("\n".join(edges) or "(none)")
            + "\n\nServices and the nodes hosting them:\n" + ("\n".join(host) or "(none)")
            + "\n\nEvidence: for each entity, the signals that changed significantly in the incident window compared with the "
              "entity's own baseline (calibrated tests), new Kubernetes events, error status codes and failing dependency calls, "
              "and log templates that are new or clearly more frequent than before the window. Entities not listed showed no "
              "significant change.\n" + ("\n".join(blocks) or "(no significant evidence)")
            + f"\n\n{taxonomy_prompt_block()}\n\n"
              "Task: identify the ONE root-cause entity (a service or a node, name exactly as written above) and its fault type "
              "(one id from the list), and write the causal chain from the root cause to the alerted operation as "
              "'cause: ...', 'propagation: ...', 'impact: ...' steps citing signal values from the evidence.\n"
              'Respond as JSON: {"root_cause_entity": "<name>", "predicted_fault_type": "<fault type id>", '
              '"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."], "confidence": <0-1>}')


def evidence_llm(case_id: str, llm, cases_dir=None) -> RCAResult:
    import config
    from pipeline.belief_rca import callee_map, _ids
    from pipeline.log_preprocess import preprocess_texts
    t0 = time.time()
    llm.reset_usage()
    case = Case(case_id, cases_dir=cases_dir or config.CASES_DIR)
    at = case.alert.alert_timestamp
    entry = case.alert.entry_entity_id
    sid = find_service_ancestor(entry, case.topology) if entry else None
    alert = _node_name(case.topology, sid) if sid else None
    op_name = _node_name(case.topology, entry) if (entry and entry in case.topology) else None
    db = compute_signal_db(case, at)
    if "services" not in db:
        raise RuntimeError(f"signal database failed: {db.get('error')}")
    # identical evidence preparation to belief_v4 with its default parts (events, score, peer, selfnorm, avail)
    new_events(case, db, B_SPAN * (db["w1"] - db["w0"]).total_seconds())
    calibrate_v4(db, peer=True, robust_null=False, ops_level=False, selfnorm=True)
    obs = bd.observations(db)
    for s_, bad in unavailability(db).items():
        obs.append({"entity": s_, "signal": "svc_unavailable", "holds": bad, "kind": "cat",
                    "ev": "deployment available replicas below desired / availability below full in the window" if bad else ""})
    ev = {}
    for o in obs:
        if o["holds"]:
            ev.setdefault(o["entity"], []).append(f"{o['signal']}: {o['ev']}" if o["ev"] else o["signal"])
    texts, _notes, _stats = preprocess_texts(case, db)
    callees = callee_map(case.topology)
    prompt = build_prompt(case.alert.alert_text, alert, op_name, callees, db.get("hosts", {}), ev, texts)
    out = llm.generate_json(prompt, system=EVIDENCE_SYSTEM)
    out = out if isinstance(out, dict) else {}
    name = str(out.get("root_cause_entity", "")).strip()
    ids = _ids(case.topology)
    eid = ids.get(name)
    if eid is None:                                   # tolerate case / whitespace differences only
        low = {k.lower(): v for k, v in ids.items()}
        eid = low.get(name.lower(), name or None)
    stats = {"total_pipeline_time_s": round(time.time() - t0, 1), "evidence_prompt_chars": len(prompt),
             "evidence_entities": len(ev), "evidence_answer_entity": name, **llm.usage_stats(),
             "_trace": {"alert_service": alert, "alert_operation": op_name, "evidence_by_entity": ev, "answer": out}}
    return RCAResult(case_id=case_id, predicted_entity_ids=[eid] if eid else [],
                     predicted_fault_type=str(out.get("predicted_fault_type", "unknown")),
                     reasoning_chain=[str(x) for x in (out.get("reasoning_chain") or [])],
                     confidence=float(out.get("confidence", 0.0) or 0.0), agent_findings=[], retrieval_stats=stats, evidence_items=[])
