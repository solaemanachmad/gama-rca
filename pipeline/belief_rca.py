"""v3 systems: belief_rca (sensor budget 5 + narrator), belief_rca_b0 (no sensor, narrator), belief_rule (no LLM at all).
See pipeline/belief_diagnosis.py for the method. Uses only the case's telemetry, topology and alert; never reads answer_key/."""
from __future__ import annotations

import os
import time

import config
from data.loader import Case, find_service_ancestor
from data.taxonomy import FAULT_DEFINITIONS
from pipeline.belief_diagnosis import diagnose_belief, PROBE_BUDGET
from pipeline.fault_calibrated import compute_signal_db
from pipeline.propagation_evidence import compute_propagation_evidence, _node_name
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.trace_evidence import compute_trace_evidence
from schema import RCAResult, EvidenceItem
from gama.llm import NARRATOR_SYSTEM, narrator_prompt
from gama.chain import callee_map, _ids, call_path, chain_step_metrics, rule_chain   # moved to gama/chain.py (unchanged)


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
    typing_trail = None
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
            typing_trail = ty
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
             "v4_n_onsets": res.get("n_onsets"), "v5_mm_k": res.get("mm_k"), "v5_mm_beta": res.get("mm_beta"),
             "v5_mm_n_corroborated": res.get("mm_n_corroborated"), "v5_load_explained": "|".join(res.get("load_explained") or []),
             **({"v5_err_traces": (res.get("trace_structure") or {}).get("n_err_traces"),
                 "v5_err_origin": "|".join(f"{k}:{v}" for k, v in sorted(((res.get("trace_structure") or {}).get("err_origin") or {}).items(),
                                                                       key=lambda kv: -kv[1])[:3]),
                 "v5_relayed": "|".join(f"{k}:{v}" for k, v in ((res.get("trace_structure") or {}).get("relayed_to_callee") or {}).items()),
                 "v5_origin_msgs": "|".join(sorted(((res.get("trace_structure") or {}).get("origin_messages") or {}).keys())),
                 "v5_slow_origin": "|".join(f"{k}:{v}" for k, v in sorted(((res.get("trace_structure") or {}).get("slow_origin") or {}).items(),
                                                                        key=lambda kv: -kv[1])[:3])}
                if res.get("trace_structure") else {}),
             **stats_llmtype}
    ev_lines = [f"- {e['entity']} {e['signal']} {'holds' if e['holds'] else 'absent'} (log Bayes factor {e['log_bf']}): {e['ev']}"
                for e in res["evidence"]]
    chain = []
    prop = _safe(compute_propagation_evidence, case, at, entry)
    infra = _safe(compute_infra_evidence, case, at, entry)
    trace = _safe(compute_trace_evidence, case, at, entry)
    tables = "\n\n".join(t for t in (prop.get("text"), infra.get("text"), trace.get("text")) if t)
    op_name = _node_name(case.topology, entry) if (v4 and entry and entry in case.topology) else None
    rpchain = v4 and "rpchain" in (res.get("parts") or [])
    held = {o["entity"] for o in (res.get("holding") or [])} if rpchain else None
    path = call_path(callees, db.get("hosts", {}), alert, res["entity"], held=held) if v4 else []
    if rpchain:                                                   # v5 'rpchain' (amendment 19): evidence is also retrieved around the
        ids0 = _ids(case.topology)                                # decided root (its 1-hop neighbourhood), not only around the alert
        root_id = ids0.get(res["entity"])
        if root_id:
            extra = _safe(compute_infra_evidence, case, at, root_id, 1)
            seen_t = {o.text for o in infra.get("observations", [])}
            infra.setdefault("observations", [])
            infra["observations"] += [o for o in extra.get("observations", []) if o.text not in seen_t]
    step_metrics = None
    if v4 and "redchain" in (res.get("parts") or []):            # v5 'redchain' (amendment 20): each chain step cites the window
        step_metrics = chain_step_metrics(case, db["w0"], db["w1"], [res["entity"]] + list(path or []), op_name)   # values of its target
        if step_metrics:
            tables = (tables + "\n\nWindow values of the entities on the chain (cite them as metric=value):\n" + "\n".join(
                f"- {e}: " + "; ".join(f"{m}={v}" for m, v in vals) for e, vals in step_metrics.items())).strip()
    if narrate and llm is not None and v4:
        prompt = narrator_prompt(case, alert, op_name, path, res, ev_lines, tables)   # gama/llm.py (unchanged)
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
    if chain and v4 and "keepchain" in (res.get("parts") or []):   # v5 'keepchain' (amendment 22): the scored chain is the computed one
        stats["llm_chain"] = " || ".join(chain)                   # (path + per-step readings); the LLM's narrative is kept for the reader
        chain = []                                                # and scored separately (rp_*_llm columns)
    if not chain and v4:                                          # deterministic chain along the topology path (no LLM)
        chain = rule_chain(res, path, op_name, step_metrics)
    if not chain:
        chain = [f"cause: {res['entity']} {res['type']}"] + [f"evidence: {e['entity']} {e['signal']}" for e in res["evidence"][:3]]
        if alert and alert != res["entity"]:
            chain.append(f"impact: {alert}")
    stats["reasoning_chain"] = " || ".join(chain)                 # saved for the paper's appendix (propagation examples)
    stats["_trace"] = {                                           # decision trail (agents/trace.py); popped by main.py, never a column
        "alert_text": case.alert.alert_text, "alert_service": alert, "alert_operation": op_name, "parts": res.get("parts"),
        "n_observations": res.get("n_obs"), "n_hypotheses": res.get("n_hyp"), "holding_observations": res.get("holding"),
        "pi": res.get("pi"), "onsets": res.get("onsets"), "trace_structure": res.get("trace_structure"), "top_hypotheses": res.get("top"), "type_rank": res.get("type_rank"),
        "decided": {"entity": res.get("entity"), "level": res.get("level"), "type": res.get("type"), "p_entity": res.get("p"),
                    "p_type": res.get("p_type")},
        "evidence_by_bayes_factor": res.get("evidence"), "text_probes": res.get("probes"), "sensor_calls": sensor_log,
        "log_preprocessing": pre_stats, "typing": typing_trail, "call_path": path, "chain": chain,
        "grounding": {k: stats.get(k) for k in ("v3_grounded", "v3_numbers")}}
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
