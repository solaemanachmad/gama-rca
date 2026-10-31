"""v3 systems: belief_rca (sensor budget 5 + narrator), belief_rca_b0 (no sensor, narrator), belief_rule (no LLM at all).
See pipeline/belief_diagnosis.py for the method. Uses only the case's telemetry, topology and alert; never reads answer_key/."""
from __future__ import annotations

import os
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


def run_case(case, llm, budget: int, narrate: bool, v4: bool = False) -> RCAResult:
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
    log_mode, pre_stats = os.environ.get("V3_LOG_MODE", "template"), {}
    if llm is not None and budget > 0:
        from pipeline.semantic_probes import collect_texts, make_sensor
        if log_mode == "raw":                                    # frozen v3 input (arm V3_RAWLOG)
            texts = collect_texts(case, db)
            sensor = make_sensor(llm, sensor_log)
        else:                                                    # Amendment 2: templates tested against the pre-window rate
            from pipeline.log_preprocess import preprocess_texts
            texts, notes, pre_stats = preprocess_texts(case, db)
            sensor = make_sensor(llm, sensor_log, notes)
    if v4:                                                       # v4 (pipeline/belief_v4.py; parts from V4_PARTS)
        from pipeline.belief_v4 import diagnose_v4
        res = diagnose_v4(db, callees, texts, sensor=sensor, budget=budget, alert=alert, case=case)
        if llm is not None and "llmtype" in res.get("parts", []):   # v4.3 LLM typing stage (pipeline/llm_typing.py)
            from pipeline.llm_typing import llm_type
            ttexts = texts
            if not ttexts:
                from pipeline.log_preprocess import preprocess_texts
                ttexts, _n, _s = preprocess_texts(case, db)
            ty = llm_type(llm, db, res, ttexts)
            stats_llmtype = {"v4_type_bayes": res["type"], "v4_type_llm": ty["llm_type"], "v4_type_llm_accepted": ty["accepted"],
                             "v4_type_llm_changed": ty["changed"], "v4_typing_prompt_chars": ty["prompt_chars"]}
            if ty["accepted"]:
                res["type"] = ty["llm_type"]
        else:
            stats_llmtype = {}
    else:
        stats_llmtype = {}
        res = diagnose_belief(db, callees, texts, sensor=sensor, budget=budget, alert=alert)
    stats = {"v3_alert_service": alert, "v3_answer_p": round(res["p"], 4), "v3_p_h0": round(res["p_h0"], 4),
             "v3_entropy_bits": res["entropy_bits"], "v3_n_obs": res["n_obs"], "v3_n_hyp": res["n_hyp"], "v3_probes": len(res["probes"]),
             "v3_probe_log": "|".join(f"{p['probe']}:{p['category']}:{p['eig_bits']}" for p in res["probes"]),
             "v3_sensor_quote_ok": sum(1 for s in sensor_log if s["quote_ok"]), "v3_top": "|".join(f"{t['entity']}:{t['type']}:{t['p']}" for t in res["top"]),
             "v3_budget": budget, "v4_parts": ",".join(res.get("parts", [])), "v4_p_type": res.get("p_type"), "v3_log_mode": log_mode if budget > 0 and llm is not None else "",
             "v3_text_entities": len(texts), "v3_sensor_lines": "|".join(str(x["n_lines"]) for x in sensor_log), **pre_stats,
             "v4_pi": round(res["pi"], 4) if res.get("pi") is not None else None, "v4_onset_root": res.get("onset_root"),
             "v4_n_onsets": res.get("n_onsets"), **stats_llmtype}
    ev_lines = [f"- {e['entity']} {e['signal']} {'holds' if e['holds'] else 'absent'} (log Bayes factor {e['log_bf']}): {e['ev']}"
                for e in res["evidence"]]
    chain = []
    prop = _safe(compute_propagation_evidence, case, at, entry)
    infra = _safe(compute_infra_evidence, case, at, entry)
    trace = _safe(compute_trace_evidence, case, at, entry)
    tables = "\n\n".join(t for t in (prop.get("text"), infra.get("text"), trace.get("text")) if t)
    op_name = _node_name(case.topology, entry) if (v4 and entry and entry in case.topology) else None
    path = call_path(callees, db.get("hosts", {}), alert, res["entity"]) if v4 else []
    if narrate and llm is not None and v4:
        prompt = (f"Alert: {case.alert.alert_text}\nAlerted service: {alert or '(none given)'}\nAlerted operation: {op_name or '(none given)'}\n"
                  f"Call path from the root cause to the alerted service (topology): {' -> '.join(path) if path else '(none)'}\n\n"
                  f"Diagnosis (decided): root cause entity = {res['entity']} ({res['level']}), fault type = {res['type']}: "
                  f"{FAULT_DEFINITIONS.get(res['type'], '')}\n\nEvidence that decided it:\n" + "\n".join(ev_lines)
                  + ("\n\nText probes: " + "; ".join(f"{p['probe']} -> {p['category']}" for p in res["probes"]) if res["probes"] else "")
                  + f"\n\nEvidence tables:\n{tables}\n\n"
                  "Task: write the causal chain from the root cause to the alerted operation, one step per entity of the call path: "
                  "'cause: <root cause entity> ...' first, then 'propagation: <service> ...' for each service between the root and the "
                  "alerted service, and last 'impact: <alerted operation name, exactly as given> ...'. 'target' names are copied from "
                  "the text above; write signals as metric_name=value using the window value the evidence shows (e.g. "
                  "node_cpu_usage_rate=0.95, error_count=412, latency=2.8), copied exactly; do not write test statistics, p-values or "
                  "ratios.\n"
                  'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."]}')
    elif narrate and llm is not None:
        prompt = (f"Alert: {case.alert.alert_text}\nAlerted service: {alert or '(none given)'}\n\n"
                  f"Diagnosis (decided): root cause entity = {res['entity']} ({res['level']}), fault type = {res['type']}: "
                  f"{FAULT_DEFINITIONS.get(res['type'], '')}\n\nEvidence that decided it:\n" + "\n".join(ev_lines)
                  + ("\n\nText probes: " + "; ".join(f"{p['probe']} -> {p['category']}" for p in res["probes"]) if res["probes"] else "")
                  + f"\n\nEvidence tables:\n{tables}\n\n"
                  "Task: write the causal chain from the root cause to the alerted service. Each step is 'step_type: target signal=value "
                  "...' with step_type in cause / propagation / impact; 'target' is an entity NAME from the evidence; write signals as "
                  "metric_name=value using the window value the evidence shows (e.g. node_cpu_usage_rate=0.95, error_count=412, "
                  "latency=2.8), copied exactly; do not write test statistics, p-values or ratios.\n"
                  'Respond as JSON: {"reasoning_chain": ["cause: ...", "propagation: ...", "impact: ..."]}')
    if narrate and llm is not None:          # (fix 2026-10-30: the v4 prompt was built but never sent; V4 GPU run 4 had no narrator)
        out = llm.generate_json(prompt, system=NARRATOR_SYSTEM)
        chain = [str(x) for x in (out.get("reasoning_chain") or [])] if isinstance(out, dict) else []
        if chain:
            from agents.multi_agent import _evidence_pool, _grounding
            pool, _ = _evidence_pool({"propagation_text": prop.get("text") or "", "infra_text": (infra.get("text") or "") + "\n"
                                      + "\n".join(ev_lines), "trace_text": trace.get("text") or ""})
            g, n, _bad = _grounding(chain, pool)
            stats.update(v3_grounded=round(g, 3), v3_numbers=n)
        stats["v3_narrator_prompt_chars"] = len(prompt)
    if not chain and v4:                                          # deterministic chain along the topology path (no LLM)
        chain = rule_chain(res, path, op_name)
    if not chain:
        chain = [f"cause: {res['entity']} {res['type']}"] + [f"evidence: {e['entity']} {e['signal']}" for e in res["evidence"][:3]]
        if alert and alert != res["entity"]:
            chain.append(f"impact: {alert}")
    stats["reasoning_chain"] = " || ".join(chain)                 # saved for the paper's appendix (propagation examples)
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


def belief_v4(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), llm, PROBE_BUDGET, narrate=True, v4=True)


def belief_v4_b0(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), llm, 0, narrate=True, v4=True)


def belief_v4_rule(case_id, llm, cases_dir=config.CASES_DIR):
    return run_case(Case(case_id, cases_dir=cases_dir), None, 0, narrate=False, v4=True)


def call_path(callees, hosts, alert, root):
    """Topology path root -> ... -> alerted service (services along callee edges, reversed). For a node root, the path starts at the
    node and continues from the hosted service closest to the alerted service. [] when unknown."""
    import networkx as nx
    if not alert or not root:
        return []
    g = nx.DiGraph()
    for a, bs in callees.items():
        for b in bs:
            g.add_edge(a, b)
    starts = [root] if root not in sum(hosts.values(), []) else [s for s, ns in hosts.items() if root in ns]
    best = None
    for st in starts:
        if st == alert:
            p = [alert]
        elif alert in g and st in g and nx.has_path(g, alert, st):
            p = list(reversed(nx.shortest_path(g, alert, st)))
        else:
            continue
        if best is None or len(p) < len(best):
            best = p
    if best is None:
        return [root] if root == alert else [root, alert]
    return ([root] + best) if best[0] != root else best


def rule_chain(res, path, op_name):
    """Chain without an LLM: cause at the root with its strongest own evidence, propagation through the path's services with their
    own symptom evidence, impact at the alerted operation."""
    ev = {}
    for e in res.get("evidence", []):
        if e["holds"] and e["entity"] not in ev and e["signal"] != "alert":     # the alert itself is not evidence of a step
            ev[e["entity"]] = f"{e['signal']} ({e['ev']})" if e.get("ev") else e["signal"]
    root = res["entity"]
    chain = [f"cause: {root} {res['type']} " + ev.get(root, "")]
    for s_ in (path or [])[1:]:
        if s_ != root:
            chain.append(f"propagation: {s_} " + ev.get(s_, ""))
    if op_name:
        chain.append(f"impact: {op_name}")
    return [c.strip() for c in chain]
