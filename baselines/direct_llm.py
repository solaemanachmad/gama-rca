"""Baseline Direct LLM: the alert text and the 28 fault types, nothing else (one call). Moved unchanged from pipeline/baselines.py
(archived; prompt and output handling byte-identical)."""
import time

import config
from data.loader import Case, normalize_entity_ids
from data.taxonomy import taxonomy_prompt_block
from schema import RCAResult

FINAL_SCHEMA_HINT = (
    '{"predicted_entity_ids": ["entity1", "entity2"], '
    '"predicted_fault_type": "<one of the 28 RCA100 fault types or best guess>", '
    '"reasoning_chain": ["cause step", "propagation step", "impact step"], '
    '"confidence": <float 0-1>}'
)
DIRECT_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis from an alert alone, with "
    "no observability data provided. Respond ONLY with valid JSON: "
    f"{FINAL_SCHEMA_HINT}"
)


def _to_rca_result(case_id: str, raw: dict, stats: dict, evidence_items=None,
                   topology=None, name_index=None) -> RCAResult:
    predicted_entity_ids = raw.get("predicted_entity_ids", []) or []
    if topology is not None and name_index is not None:
        predicted_entity_ids = normalize_entity_ids(predicted_entity_ids, topology, name_index)
    return RCAResult(
        case_id=case_id,
        predicted_entity_ids=predicted_entity_ids,
        predicted_fault_type=raw.get("predicted_fault_type", "unknown"),
        reasoning_chain=raw.get("reasoning_chain", []) or [],
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        agent_findings=[],
        retrieval_stats=stats,
        evidence_items=evidence_items or [],
    )


def direct_llm(case_id: str, llm, cases_dir: str = config.CASES_DIR) -> RCAResult:
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    prompt = f"Alert: {case.alert.alert_text}\n\n{taxonomy_prompt_block()}\n\nDiagnose the root cause."
    result = llm.generate_json(prompt, system=DIRECT_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0, **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats, topology=case.topology, name_index=case.name_index)
