"""v3 systems: belief_rca (sensor budget 5 + narrator), belief_rca_b0 (no sensor, narrator), belief_rule (no LLM at all).
See pipeline/belief_diagnosis.py for the method. Uses only the case's telemetry, topology and alert; never reads answer_key/."""
from __future__ import annotations

import time
from collections import defaultdict

import config
from data.loader import Case, find_service_ancestor
from data.taxonomy import FAULT_DEFINITIONS
from pipeline.belief_diagnosis import diagnose_belief, PROBE_BUDGET
from pipeline.fault_calibrated import compute_signal_db
from pipeline.propagation_evidence import compute_propagation_evidence, _calls_graph, _node_name, _node_type
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.trace_evidence import compute_trace_evidence
from schema import RCAResult, EvidenceItem

NARRATOR_SYSTEM = ("You are a site-reliability engineer. The diagnosis has already been decided from the evidence; you write the causal "
                   "chain that the evidence supports. Respond ONLY with valid JSON.")


def callee_map(topo):
    out = defaultdict(list)
    for u, v in _calls_graph(topo).edges():
        if _node_type(topo, u) in ("apm.service", "apm.svc") and _node_type(topo, v) in ("apm.service", "apm.svc"):
            a, b = _node_name(topo, u), _node_name(topo, v)
            if a != b and b not in out[a]:
                out[a].append(b)
    return dict(out)


def _ids(topo):
    m = {}
    for n in topo.nodes:
        if _node_type(topo, n) in ("apm.service", "apm.svc", "k8s.node"):
            m.setdefault(_node_name(topo, n), n)
    return m


def _safe(fn, *a):
    try:
        return fn(*a) or {}
    except Exception as e:
        return {"text": "", "error": repr(e)}


def run_case(case, llm, budget: int, narrate: bool) -> RCAResult:
    t0 = time.time()
    if llm is not None:
        llm.reset_usage()
    at = case.alert.alert_timestamp
    entry = case.alert.entry_entity_id
    sid = find_service_ancestor(entry, case.topology) if entry else None
    alert = _node_name(case.topology, sid) if sid else None
    db = compute_signal_db(case, at)
    if "services" not in db:
        raise RuntimeError(f"signal database failed: {db.get('error')}")
    callees = callee_map(case.topology)
    texts, sensor_log, sensor = {}, [], None
    if llm is not None and budget > 0:
        from pipeline.semantic_probes import collect_texts, make_sensor
        texts = collect_texts(case, db)
        sensor = make_sensor(llm, sensor_log)
    res = diagnose_belief(db, callees, texts, sensor=sensor, budget=budget, alert=alert)
    stats = {"v3_alert_service": alert, "v3_answer_p": round(res["p"], 4), "v3_p_h0": round(res["p_h0"], 4),
             "v3_entropy_bits": res["entropy_bits"], "v3_n_obs": res["n_obs"], "v3_n_hyp": res["n_hyp"], "v3_probes": len(res["probes"]),
             "v3_probe_log": "|".join(f"{p['probe']}:{p['category']}:{p['eig_bits']}" for p in res["probes"]),
             "v3_sensor_quote_ok": sum(1 for s in sensor_log if s["quote_ok"]), "v3_top": "|".join(f"{t['entity']}:{t['type']}:{t['p']}" for t in res["top"]),
             "v3_budget": budget}
    ev_lines = [f"- {e['entity']} {e['signal']} {'holds' if e['holds'] else 'absent'} (log Bayes factor {e['log_bf']}): {e['ev']}"
                for e in res["evidence"]]
    chain = []
    prop = _safe(compute_propagation_evidence, case, at, entry)
    infra = _safe(compute_infra_evidence, case, at, entry)
    trace = _safe(compute_trace_evidence, case, at, entry)
    tables = "\n\n".join(t for t in (prop.get("text"), infra.get("text"), trace.get("text")) if t)
    if narrate and llm is not None:
        prompt = (f"Alert: {case.alert.alert_text}\nAlerted service: {alert or '(none given)'}\n\n"
                  f"Diagnosis (decided): root cause entity = {res['entity']} ({res['level']}), fault type = {res['type']}: "
                  f"{FAULT_DEFINITIONS.get(res['type'], '')}\n\nEvidence that decided it:\n" + "\n".join(ev_lines)
                  + ("\n\nText probes: " + "; ".join(f"{p['probe']} -> {p['category']}" for p in res["probes"]) if res["probes"] else "")
                  + f"\n\nEvidence tables:\n{tables}\n\n"
                  "Task: write the causal chain from the root cause to the alerted service. Each step is 'step_type: target signal=value "
                  "...' with step_type in cause / propagation / impact; 'target' is an entity NAME from the evidence; copy every value "
                  "exactly as the evidence shows it.\n"
                  'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."]}')
        out = llm.generate_json(prompt, system=NARRATOR_SYSTEM)
        chain = [str(x) for x in (out.get("reasoning_chain") or [])] if isinstance(out, dict) else []
        if chain:
            from agents.multi_agent import _evidence_pool, _grounding
            pool, _ = _evidence_pool({"propagation_text": prop.get("text") or "", "infra_text": (infra.get("text") or "") + "\n"
                                      + "\n".join(ev_lines), "trace_text": trace.get("text") or ""})
            g, n, _bad = _grounding(chain, pool)
            stats.update(v3_grounded=round(g, 3), v3_numbers=n)
        stats["v3_narrator_prompt_chars"] = len(prompt)
    if not chain:
        chain = [f"cause: {res['entity']} {res['type']}"] + [f"evidence: {e['entity']} {e['signal']}" for e in res["evidence"][:3]]
        if alert and alert != res["entity"]:
            chain.append(f"impact: {alert}")
    ids = _ids(case.topology)
    eid = ids.get(res["entity"], res["entity"])
    stats["total_pipeline_time_s"] = round(time.time() - t0, 1)
    if llm is not None:
        stats.update(llm.usage_stats())
    return RCAResult(case_id=getattr(case, "case_id", ""), predicted_entity_ids=[eid] if eid else [], predicted_fault_type=res["type"],
                     reasoning_chain=chain, confidence=res["p"], agent_findings=[], retrieval_stats=stats,
                     evidence_items=[EvidenceItem(observation=o) for o in infra.get("observations", [])])


def belief_rca(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), llm, PROBE_BUDGET, narrate=True)


def belief_rca_b0(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), llm, 0, narrate=True)


def belief_rule(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), None, 0, narrate=False)
