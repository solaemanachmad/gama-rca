"""
baselines.py
=============
Comparison systems for the ablation study (RQ1-RQ5):

  1. direct_llm           - alert text straight to the LLM, zero retrieval
  2. standard_rag         - vector-only retrieval, single LLM call
  3. graphrag_only        - graph-only retrieval (no vector), single LLM call
  4. multi_agent_only     - full multi-agent system but fed UNFILTERED
                            evidence (no hybrid ranking) to isolate the
                            retrieval contribution from the agent contribution
  5. proposed_hybrid      - full framework -> see pipeline.GraphRAGPipeline

Each baseline returns an RCAResult with the SAME schema as the proposed
framework so evaluation.py can score all five identically.
"""

import time
from typing import Dict, List

import config
from data_loader import Case
from graph_retrieval import GraphRetriever
from vector_retrieval import build_case_index, build_index_from_observations
from hybrid_retrieval import fuse_scores
from evidence_summarizer import summarize_evidence, render_summary_text
from agents import build_agent_graph, build_agent_findings_list
from llm_client import LLMClient
from schema import RCAResult
from pipeline import parse_alert
from taxonomy import fault_shortlist_prompt_block, taxonomy_prompt_block

DIRECT_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis from an alert alone, with "
    "no observability data provided. Respond ONLY with valid JSON: "
    '{"predicted_entity_ids": [], "predicted_fault_type": "", '
    '"reasoning_chain": [], "confidence": 0.0}'
)


def direct_llm(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    prompt = f"Alert: {case.alert.alert_text}\n\n{taxonomy_prompt_block()}\n\nDiagnose the root cause."
    result = llm.generate_json(prompt, system=DIRECT_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0, **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats)


RAG_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis using retrieved log/metric/"
    "trace snippets (no topology information). Respond ONLY with valid JSON: "
    '{"predicted_entity_ids": [], "predicted_fault_type": "", '
    '"reasoning_chain": [], "confidence": 0.0}'
)


def standard_rag(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    """Vector retrieval only: no graph score, alpha=0 equivalent."""
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)

    # DEV_QUICK_TEST cap: standard_rag is deliberately topology-blind (no
    # graph filtering by design -- that's the whole point of this baseline),
    # so during fast iteration we cap raw volume per modality instead. This
    # is a SPEED-ONLY concession for dev testing; disable DEV_QUICK_TEST for
    # the real experiment so this baseline searches the full case, as intended.
    observations = case.observations
    if config.DEV_QUICK_TEST:
        observations = {m: obs[:config.DEV_MAX_UNRESOLVED_PER_MODALITY * 5]
                         for m, obs in observations.items()}

    vector_index = build_index_from_observations(observations)
    hits = vector_index.search(parsed["alert_text"], top_k=config.VECTOR_TOP_K)
    evidence_items = fuse_scores(graph_scores={}, vector_hits=hits, alpha=0.0, beta=1.0)
    summary = summarize_evidence(evidence_items)
    summary_text = render_summary_text(summary)

    prompt = f"Alert: {parsed['alert_text']}\n\nRetrieved evidence:\n{summary_text}\n\n{fault_shortlist_prompt_block(summary_text)}\n\nDiagnose the root cause."
    result = llm.generate_json(prompt, system=RAG_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0, "evidence_items_retrieved": len(evidence_items),
              **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats, evidence_items=evidence_items)


GRAPHRAG_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis using topology-derived "
    "candidate entities (no log/metric/trace text). Respond ONLY with valid "
    'JSON: {"predicted_entity_ids": [], "predicted_fault_type": "", '
    '"reasoning_chain": [], "confidence": 0.0}'
)


def graphrag_only(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    """Graph retrieval only: ranked candidate entities, no vector evidence text."""
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)

    graph_retriever = GraphRetriever(case.topology)
    graph_result = graph_retriever.retrieve(parsed["entry_entity_id"])
    ranked = graph_result["ranked_entities"]

    prompt = (
        f"Alert: {parsed['alert_text']}\n"
        f"Entry entity: {parsed['entry_entity_id']}\n"
        f"Topology-ranked candidate entities (entity_id, propagation_score): {ranked}\n\n"
        f"{taxonomy_prompt_block()}\n\n"
        f"Diagnose the root cause using only this structural information."
    )
    result = llm.generate_json(prompt, system=GRAPHRAG_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0,
              "candidate_subgraph_size": graph_result["subgraph"].number_of_nodes(),
              **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats)


def multi_agent_only(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR,
                      max_raw_observations: int = 200) -> RCAResult:
    """Full multi-agent + coordinator pipeline, but evidence is an
    UNFILTERED (truncated) dump rather than hybrid-ranked — isolates the
    agent-collaboration contribution from the retrieval contribution.

    NOTE ON SAMPLING: naively slicing case.all_observations()[:N] is biased
    -- modalities are concatenated in dict order (metrics first), and the
    earliest metrics rows are dominated by generic k8s-node-level readings
    with no entity_id (e.g. "node_ready_status"), giving the LLM mostly
    non-specific evidence regardless of N. This interleaves across
    modalities and prefers entity-resolved observations, so the "no smart
    retrieval" baseline still gets a representative sample instead of an
    accidentally-degenerate one."""
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)

    per_modality_budget = max(1, max_raw_observations // 5)
    sampled = []
    for modality, obs_list in case.observations.items():
        resolved = [o for o in obs_list if o.entity_id]
        unresolved = [o for o in obs_list if not o.entity_id]
        # prefer entity-resolved observations first, pad with unresolved if short
        chosen = (resolved + unresolved)[:per_modality_budget]
        sampled.extend(chosen)

    fake_items = [type("Item", (), {
        "observation": o, "graph_score": 0.0, "vector_score": 0.0, "hybrid_score": 0.0
    })() for o in sampled]
    summary = summarize_evidence(fake_items)

    agent_graph = build_agent_graph(llm)
    state = {
        "case_id": case_id,
        "alert_text": parsed["alert_text"],
        "evidence_summary": summary,
        "graph_neighbors": {},
        "candidate_entities": [],
    }
    final_state = agent_graph.invoke(state)
    final = final_state.get("final_result") or {}
    findings = build_agent_findings_list(final_state)

    stats = {"total_pipeline_time_s": time.time() - t0, **llm.usage_stats()}
    return RCAResult(
        case_id=case_id,
        predicted_entity_ids=final.get("predicted_entity_ids", []),
        predicted_fault_type=final.get("predicted_fault_type", "unknown"),
        reasoning_chain=final.get("reasoning_chain", []),
        confidence=float(final.get("confidence", 0.0) or 0.0),
        agent_findings=findings,
        retrieval_stats=stats,
        evidence_items=fake_items,
    )


def _to_rca_result(case_id: str, raw: dict, stats: dict, evidence_items=None) -> RCAResult:
    return RCAResult(
        case_id=case_id,
        predicted_entity_ids=raw.get("predicted_entity_ids", []) or [],
        predicted_fault_type=raw.get("predicted_fault_type", "unknown"),
        reasoning_chain=raw.get("reasoning_chain", []) or [],
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        agent_findings=[],
        retrieval_stats=stats,
        evidence_items=evidence_items or [],
    )


BASELINE_REGISTRY = {
    "direct_llm": direct_llm,
    "standard_rag": standard_rag,
    "graphrag_only": graphrag_only,
    "multi_agent_only": multi_agent_only,
    # "proposed_hybrid" is run via pipeline.GraphRAGPipeline, not this registry
}