"""GAMA-RCA on one case: telemetry, topology and the alert in; root entity, fault type and reasoning chain out. Never reads
answer_key/. Stages: 1 calibrated evidence (gama/evidence.py) -> 2 belief (gama/belief.py) -> 3 decision (gama/decide.py) ->
4 bounded LLM, optional (gama/llm.py); chain (gama/chain.py). Modules: gama/config.py.

    python main.py --systems gama ...            modules from GAMA_MODULES (default "core"; "full" = with the LLM)
"""
from __future__ import annotations

import time

import config
from data.loader import Case, find_service_ancestor
from pipeline.belief_diagnosis import PROBE_BUDGET
from pipeline.propagation_evidence import compute_propagation_evidence, _node_name
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.trace_evidence import compute_trace_evidence
from schema import RCAResult, EvidenceItem
from gama import evidence, llm as gllm
from gama.chain import callee_map, _ids, call_path, chain_step_metrics, rule_chain
from gama.config import Modules
from gama.decide import diagnose


def _safe(fn, *a):
    try:
        return fn(*a) or {}
    except Exception as e:
        return {"text": "", "error": repr(e)}


def run_case(case, llm, m: Modules) -> RCAResult:
    config.ALERT_T0_MODE = "window"                     # evidence window = the task's alert window [start, end]
    llm = llm if m.llm else None
    budget = PROBE_BUDGET if llm is not None else 0
    t0 = time.time()
    if llm is not None:
        llm.reset_usage()
    at = case.alert.alert_timestamp
    entry = case.alert.entry_entity_id
    sid = find_service_ancestor(entry, case.topology) if entry else None
    alert = _node_name(case.topology, sid) if sid else None
    db = evidence.signal_db(case, at, m)
    if "services" not in db:
        raise RuntimeError(f"signal database failed: {db.get('error')}")
    callees = callee_map(case.topology)
    texts, sensor_log, sensor, pre_stats = {}, [], None, {}
    if llm is not None:
        texts, sensor, pre_stats = gllm.texts_and_sensor(case, db, llm, sensor_log)
    res = diagnose(db, callees, texts, m, sensor=sensor, budget=budget, alert=alert, case=case)
    model = res.pop("_model")
    typing_trail, stats_llmtype = None, {}
    if llm is not None:
        ty = gllm.type_root(llm, db, res, texts, case, model)
        typing_trail = ty
        stats_llmtype = {"v4_type_bayes": res["type"], "v4_type_llm": ty["llm_type"], "v4_type_llm_accepted": ty["accepted"],
                         "v4_type_llm_changed": ty["changed"], "v4_typing_prompt_chars": ty["prompt_chars"]}
        if ty["accepted"]:
            res["type"] = ty["llm_type"]
    ts_ = res.get("trace_structure") or {}
    stats = {"v3_alert_service": alert, "v3_answer_p": round(res["p"], 4), "v3_p_h0": round(res["p_h0"], 4),
             "v3_entropy_bits": res["entropy_bits"], "v3_n_obs": res["n_obs"], "v3_n_hyp": res["n_hyp"], "v3_probes": len(res["probes"]),
             "v3_probe_log": "|".join(f"{p['probe']}:{p['category']}:{p['eig_bits']}" for p in res["probes"]),
             "v3_sensor_quote_ok": sum(1 for s in sensor_log if s["quote_ok"]),
             "v3_top": "|".join(f"{t['entity']}:{t['type']}:{t['p']}" for t in res["top"]),
             "v3_budget": budget, "v4_parts": ",".join(res.get("parts", [])), "gama_modules": res["modules"],
             "v4_p_type": res.get("p_type"), "v3_log_mode": "template" if llm is not None else "",
             "v3_text_entities": len(texts), "v3_sensor_lines": "|".join(str(x["n_lines"]) for x in sensor_log), **pre_stats,
             "v4_pi": round(res["pi"], 4) if res.get("pi") is not None else None, "v4_onset_root": res.get("onset_root"),
             "v4_n_onsets": res.get("n_onsets"), "v5_mm_k": res.get("mm_k"), "v5_mm_beta": res.get("mm_beta"),
             "v5_mm_n_corroborated": res.get("mm_n_corroborated"), "v5_load_explained": "|".join(res.get("load_explained") or []),
             **({"v5_err_traces": ts_.get("n_err_traces"),
                 "v5_err_origin": "|".join(f"{k}:{v}" for k, v in sorted((ts_.get("err_origin") or {}).items(), key=lambda kv: -kv[1])[:3]),
                 "v5_relayed": "|".join(f"{k}:{v}" for k, v in (ts_.get("relayed_to_callee") or {}).items()),
                 "v5_origin_msgs": "|".join(sorted((ts_.get("origin_messages") or {}).keys())),
                 "v5_slow_origin": "|".join(f"{k}:{v}" for k, v in sorted((ts_.get("slow_origin") or {}).items(), key=lambda kv: -kv[1])[:3])}
                if ts_ else {}),
             **stats_llmtype}
    ev_lines = [f"- {e['entity']} {e['signal']} {'holds' if e['holds'] else 'absent'} (log Bayes factor {e['log_bf']}): {e['ev']}"
                for e in res["evidence"]]
    prop = _safe(compute_propagation_evidence, case, at, entry)
    infra = _safe(compute_infra_evidence, case, at, entry)
    trace = _safe(compute_trace_evidence, case, at, entry)
    tables = "\n\n".join(t for t in (prop.get("text"), infra.get("text"), trace.get("text")) if t)
    op_name = _node_name(case.topology, entry) if (entry and entry in case.topology) else None
    path = call_path(callees, db.get("hosts", {}), alert, res["entity"])
    step_metrics = None
    if m.chain:                                          # M6: each chain step cites the window values of its target
        step_metrics = chain_step_metrics(case, db["w0"], db["w1"], [res["entity"]] + list(path or []), op_name)
        if step_metrics:
            tables = (tables + "\n\nWindow values of the entities on the chain (cite them as metric=value):\n" + "\n".join(
                f"- {e}: " + "; ".join(f"{mt}={v}" for mt, v in vals) for e, vals in step_metrics.items())).strip()
    if llm is not None:                                  # M7 narrator: kept for the reader, scored separately (rp_*_llm)
        prompt = gllm.narrator_prompt(case, alert, op_name, path, res, ev_lines, tables)
        out = llm.generate_json(prompt, system=gllm.NARRATOR_SYSTEM)
        narrative = [str(x) for x in (out.get("reasoning_chain") or [])] if isinstance(out, dict) else []
        if narrative:
            from agents.grounding import _evidence_pool, _grounding
            pool, _ = _evidence_pool({"propagation_text": prop.get("text") or "", "infra_text": (infra.get("text") or "") + "\n"
                                      + "\n".join(ev_lines), "trace_text": trace.get("text") or ""})
            g, n, _bad = _grounding(narrative, pool)
            stats.update(v3_grounded=round(g, 3), v3_numbers=n)
        stats["v3_narrator_prompt_chars"] = len(prompt)
        if narrative:
            stats["llm_chain"] = " || ".join(narrative)
    chain = rule_chain(res, path, op_name, step_metrics)  # the scored chain: computed along the topology path
    stats["reasoning_chain"] = " || ".join(chain)
    stats["_trace"] = {                                   # decision trail (agents/trace.py); popped by main.py, never a column
        "alert_text": case.alert.alert_text, "alert_service": alert, "alert_operation": op_name, "parts": res.get("parts"),
        "n_observations": res.get("n_obs"), "n_hypotheses": res.get("n_hyp"), "holding_observations": res.get("holding"),
        "pi": res.get("pi"), "onsets": res.get("onsets"), "trace_structure": res.get("trace_structure"), "top_hypotheses": res.get("top"),
        "type_rank": res.get("type_rank"),
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


def gama(case_id, llm, cases_dir=config.CASES_DIR):
    """main.py system 'gama' (modules from GAMA_MODULES)."""
    return run_case(Case(case_id, cases_dir=cases_dir), llm, Modules.from_env())
