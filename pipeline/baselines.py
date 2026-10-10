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
  5. sea_rca              - Single-Shot Evidence-Augmented RCA: graphrag_only's
                            topology prompt + a SMALL hybrid-ranked evidence
                            slice, one LLM call. Tests whether proposed_hybrid's
                            reasoning_process/explainability edge can be had
                            without its 6-call multi-agent cost (see sea_rca()
                            docstring for the 8-case finding that motivated it).
  6. proposed_hybrid      - full framework -> see pipeline.GraphRAGPipeline

Each baseline returns an RCAResult with the SAME schema as the proposed
framework so evaluation.py can score all five identically.

FINAL_SCHEMA_HINT (shared with the Coordinator, agents/multi_agent.py) is used
for every baseline's JSON schema instruction, not a locally-hardcoded one.
Found via a sea_rca ablation (2026-10-02): the Coordinator's version includes
a worked example -- '"reasoning_chain": ["cause step", "propagation step",
"impact step"]' -- whose wording directly echoes gt.reasoning_chain's own
format (evaluation/scoring.py's load_ground_truth builds it as "{step_type}:
{target}", e.g. "cause: payment"). Every baseline here previously used a bare
'"reasoning_chain": []' placeholder with none of that vocabulary, which meant
cross-system reasoning_process comparisons were confounded by an inconsistent
prompt, not just by each system's retrieval/evidence design -- proposed_hybrid
had a built-in head start on this metric that had nothing to do with its
multi-agent architecture. Standardizing on one shared constant removes that
confound for every baseline at once.
"""

import time
from typing import Dict, List

import config
from data.loader import Case, normalize_entity_ids
from retrieval.graph import GraphRetriever
from retrieval.vector import build_case_index, build_index_from_observations, VectorIndex
from retrieval.hybrid import fuse_scores, HybridRetriever, graph_direct_evidence, merge_evidence
from pipeline.evidence_summarizer import summarize_evidence, render_summary_text
from agents.multi_agent import build_agent_graph, build_agent_findings_list, FINAL_SCHEMA_HINT
from agents.llm_client import LLMClient
from schema import RCAResult
from pipeline.pipeline import parse_alert
from data.taxonomy import taxonomy_prompt_block

DIRECT_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis from an alert alone, with "
    "no observability data provided. Respond ONLY with valid JSON: "
    f"{FINAL_SCHEMA_HINT}"
)


def direct_llm(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    prompt = f"Alert: {case.alert.alert_text}\n\n{taxonomy_prompt_block()}\n\nDiagnose the root cause."
    result = llm.generate_json(prompt, system=DIRECT_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0, **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats, topology=case.topology, name_index=case.name_index)


RAG_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis using retrieved log/metric/"
    "trace snippets (no topology information). Respond ONLY with valid JSON: "
    f"{FINAL_SCHEMA_HINT}"
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
    summary = summarize_evidence(evidence_items, alert_timestamp=case.alert.alert_timestamp)
    summary_text = render_summary_text(summary)

    prompt = f"Alert: {parsed['alert_text']}\n\nRetrieved evidence:\n{summary_text}\n\n{taxonomy_prompt_block()}\n\nDiagnose the root cause."
    result = llm.generate_json(prompt, system=RAG_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0, "evidence_items_retrieved": len(evidence_items),
              **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats, evidence_items=evidence_items, topology=case.topology, name_index=case.name_index)


GRAPHRAG_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis using topology-derived "
    "candidate entities (no log/metric/trace text). Respond ONLY with valid "
    f"JSON: {FINAL_SCHEMA_HINT}"
)


def graphrag_only(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    """Graph retrieval only: ranked candidate entities, no vector evidence text."""
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)

    graph_retriever = GraphRetriever(case.topology)
    graph_result = graph_retriever.retrieve(parsed["entry_entity_id"])
    # Cap at top-20: ranked_entities now returns the FULL scored list (can be
    # 65-200+ entities after graph_retrieval.py stopped truncating to
    # PPR_TOP_K=15 -- that fix was for the real pipeline's evidence-selection
    # stage, not for dumping raw into an LLM prompt here). A 7B local model
    # fed a 100+-tuple wall of text reliably breaks JSON output entirely
    # (empty reasoning_chain, "unknown" fault type) -- this baseline is
    # deliberately meant to give a SHORT structural signal, not the whole
    # scored graph.
    ranked = graph_result["ranked_entities"][:20]
    ranked_text = "\n".join(f"  {eid}: score={score:.4f}" for eid, score in ranked)

    prompt = (
        f"Alert: {parsed['alert_text']}\n"
        f"Entry entity: {parsed['entry_entity_id']}\n"
        f"Top-20 topology-ranked candidate entities (entity_id: propagation_score):\n{ranked_text}\n\n"
        f"{taxonomy_prompt_block()}\n\n"
        f"Diagnose the root cause using only this structural information."
    )
    result = llm.generate_json(prompt, system=GRAPHRAG_SYSTEM_PROMPT)
    stats = {"total_pipeline_time_s": time.time() - t0,
              "candidate_subgraph_size": graph_result["subgraph"].number_of_nodes(),
              **llm.usage_stats()}
    return _to_rca_result(case_id, result, stats, topology=case.topology, name_index=case.name_index)


SEA_RCA_SYSTEM_PROMPT = (
    "You are an SRE performing root cause analysis using BOTH topology-derived "
    "candidate entities AND a small set of retrieved log/metric/event evidence "
    "snippets, in a single pass. Respond ONLY with valid JSON: "
    f"{FINAL_SCHEMA_HINT}"
)


def sea_rca(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR,
            max_observations_per_index: int = None, top_k_evidence: int = None) -> RCAResult:
    """Single-Shot Evidence-Augmented RCA (SEA-RCA).

    Motivated by an 8-case head-to-head (t001/t013/t026/t039/t052/t064/t077/
    t090, 2026-10-02): graphrag_only (1 LLM call, topology only) BEATS
    proposed_hybrid on entity_localization (0.3203 vs 0.2383 mean) and TIES
    on fault_identification (0.25 vs 0.25), but scores near-zero on
    reasoning_process (0.018 vs 0.381 mean) because it never sees any
    evidence text -- reasoning_process/explainability specifically reward an
    evidence-grounded cause -> propagation -> impact chain, not just a
    correct final entity/type. Separately, pipeline.py's predicted_entity_ids
    assignment already overrides the Coordinator's own entity pick with
    graph_anchor's whenever graph_anchor produces one -- so proposed_hybrid's
    4 specialist-agent calls + Coordinator call (5 of its 6 total_calls)
    contribute ZERO measured entity_localization value in that same sample.

    SEA_RCA tests whether folding a SMALL amount of hybrid-ranked evidence
    (top-`top_k_evidence` items, from an index capped at
    `max_observations_per_index` -- both deliberately independent of
    config.MAX_OBSERVATIONS_PER_INDEX, see config/__init__.py) into
    graphrag_only's single prompt recovers most of proposed_hybrid's
    reasoning_process/explainability value, at a cost close to graphrag_only's
    (1 LLM call, a small fast index) rather than proposed_hybrid's (6 calls,
    a 10k-capped index). If it does, that's a direct speed+cost contribution:
    comparable accuracy, a fraction of the wall-clock time and token spend."""
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)

    graph_retriever = GraphRetriever(case.topology)
    graph_result = graph_retriever.retrieve(parsed["entry_entity_id"])
    ranked = graph_result["ranked_entities"][:20]
    ranked_text = "\n".join(f"  {eid}: score={score:.4f}" for eid, score in ranked)

    cap = max_observations_per_index or config.SEA_RCA_MAX_OBSERVATIONS_PER_INDEX
    top_k = top_k_evidence or config.SEA_RCA_TOP_K_EVIDENCE

    observations = case.observations
    if config.DEV_QUICK_TEST:
        observations = {m: obs[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
                         for m, obs in observations.items()}

    # Small, independent vector index -- NOT build_index_from_observations()
    # (that reads config.MAX_OBSERVATIONS_PER_INDEX, tuned for
    # proposed_hybrid's own needs). Still only logs/metrics/events, matching
    # the rest of the pipeline's vector-modality scope.
    vector_index = VectorIndex(max_observations=cap)
    for modality in ("logs", "metrics", "events"):
        vector_index.add(observations.get(modality, []))

    hybrid = HybridRetriever(graph_result, vector_index)
    queries = [parsed["alert_text"]] + parsed["keywords"][:5]
    vector_based_items = hybrid.retrieve_multi(queries, top_k=top_k)

    direct_items = graph_direct_evidence(observations, graph_result["graph_scores"],
                                          top_n_entities=10, max_per_entity=2,
                                          force_include=graph_result.get("boosted_entities"))
    evidence_items = merge_evidence(vector_based_items, direct_items)[:top_k]

    summary = summarize_evidence(evidence_items, alert_timestamp=case.alert.alert_timestamp)
    summary_text = render_summary_text(summary)

    prompt = (
        f"Alert: {parsed['alert_text']}\n"
        f"Entry entity: {parsed['entry_entity_id']}\n"
        f"Top-20 topology-ranked candidate entities (entity_id: propagation_score):\n{ranked_text}\n\n"
        f"Top-{len(evidence_items)} retrieved evidence snippets (hybrid graph+vector ranked):\n{summary_text}\n\n"
        f"{taxonomy_prompt_block()}\n\n"
        f"Diagnose the root cause. Build an explicit cause -> propagation -> impact "
        f"reasoning chain, citing the specific evidence and/or entities above that support it."
    )
    result = llm.generate_json(prompt, system=SEA_RCA_SYSTEM_PROMPT)
    stats = {
        "total_pipeline_time_s": time.time() - t0,
        "candidate_subgraph_size": graph_result["subgraph"].number_of_nodes(),
        "evidence_items_retrieved": len(evidence_items),
        "indexed_observations": vector_index.index.ntotal,
        **llm.usage_stats(),
    }
    return _to_rca_result(case_id, result, stats, evidence_items=evidence_items,
                           topology=case.topology, name_index=case.name_index)


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
    summary = summarize_evidence(fake_items, alert_timestamp=case.alert.alert_timestamp)

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
    predicted_entity_ids = normalize_entity_ids(final.get("predicted_entity_ids", []), case.topology, case.name_index)
    return RCAResult(
        case_id=case_id,
        predicted_entity_ids=predicted_entity_ids,
        predicted_fault_type=final.get("predicted_fault_type", "unknown"),
        reasoning_chain=final.get("reasoning_chain", []),
        confidence=float(final.get("confidence", 0.0) or 0.0),
        agent_findings=findings,
        retrieval_stats=stats,
        evidence_items=fake_items,
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


BASELINE_REGISTRY = {
    "direct_llm": direct_llm,
    "standard_rag": standard_rag,
    "graphrag_only": graphrag_only,
    "multi_agent_only": multi_agent_only,
    "sea_rca": sea_rca,
    "gala_lite": lambda case_id, llm, cases_dir=config.CASES_DIR: __import__(
        "pipeline.gala_lite", fromlist=["gala_lite"]).gala_lite(case_id, llm, cases_dir),
    # "proposed_hybrid" is run via pipeline.GraphRAGPipeline, not this registry
}