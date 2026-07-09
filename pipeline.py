"""
pipeline.py
============
Wires Modules 1-6 into the full GraphRAG-RCA pipeline described in the
project architecture diagram:

  Alert -> Alert Parser -> Graph Retrieval -> Hybrid Retrieval
        -> Evidence Summarizer -> Multi-Agent -> Coordinator -> RCAResult

This is the ONLY module that touches every other module — data_loader,
graph_retrieval, vector_retrieval, hybrid_retrieval, evidence_summarizer,
agents, llm_client are all composed here and nowhere else, so each stays
independently testable/replaceable per the "keep modular" design principle.
"""

import time
from typing import Dict, Optional

import config
from data_loader import Case, build_service_membership_index
from graph_retrieval import GraphRetriever
from vector_retrieval import build_index_from_observations
from hybrid_retrieval import HybridRetriever, graph_direct_evidence, merge_evidence
from evidence_summarizer import summarize_evidence
from agents import build_agent_graph, build_agent_findings_list
from llm_client import LLMClient
from schema import RCAResult


# ---------------------------------------------------------------------------
# Module 1 — Alert Parser
# ---------------------------------------------------------------------------
def parse_alert(case: Case) -> Dict:
    """Extract candidate entities / keywords from the alert text + entry
    entity. Kept intentionally simple (regex/keyword split); swap in an
    NER model here if alert text is richer than the RCA100 structured form."""
    alert = case.alert
    keywords = [w.strip(".,:;") for w in alert.alert_text.split() if len(w) > 3]
    return {
        "entry_entity_id": alert.entry_entity_id,
        "keywords": list(dict.fromkeys(keywords))[:20],   # dedup, cap
        "alert_text": alert.alert_text,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class GraphRAGPipeline:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        self.agent_graph = build_agent_graph(self.llm)

    def run(self, case_id: str, cases_dir: str = config.CASES_DIR) -> RCAResult:
        self.llm.reset_usage()
        t0 = time.time()
        stats = {"case_id": case_id}

        # --- Stage 0: ingestion -------------------------------------------------
        case = Case(case_id, cases_dir=cases_dir)
        stats["load_time_s"] = time.time() - t0

        # --- Module 1: Alert Parser ----------------------------------------------
        t1 = time.time()
        parsed_alert = parse_alert(case)
        stats["alert_parse_time_s"] = time.time() - t1

        # --- Module 2: Graph Retrieval --------------------------------------------
        t2 = time.time()
        graph_retriever = GraphRetriever(case.topology)
        graph_result = graph_retriever.retrieve(parsed_alert["entry_entity_id"])
        stats["graph_retrieval_time_s"] = time.time() - t2
        stats["candidate_subgraph_size"] = graph_result["subgraph"].number_of_nodes()

        # --- Module 3: Vector Retrieval (index build) -----------------------------
        t3 = time.time()
        subgraph_node_ids = set(graph_result["subgraph"].nodes)

        def _filter_modality(obs_list):
            resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
            unresolved = [o for o in obs_list if o.entity_id is None]
            if config.DEV_QUICK_TEST:
                unresolved = unresolved[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
            return resolved + unresolved

        filtered_observations = {
            modality: _filter_modality(obs_list)
            for modality, obs_list in case.observations.items()
        }
        vector_index = build_index_from_observations(filtered_observations)
        stats["vector_index_build_time_s"] = time.time() - t3
        stats["indexed_observations"] = vector_index.index.ntotal
        stats["observations_before_graph_filter"] = sum(len(v) for v in case.observations.values())

        # --- Module 4: Hybrid Retrieval --------------------------------------------
        t4 = time.time()
        hybrid = HybridRetriever(graph_result, vector_index)
        queries = [parsed_alert["alert_text"]] + parsed_alert["keywords"][:5]
        vector_based_items = hybrid.retrieve_multi(queries, top_k=config.VECTOR_TOP_K)

        # Guarantee representation for structurally-central entities (e.g. a
        # root cause 2-3 hops upstream) regardless of whether their text
        # semantically resembles the alert wording -- see graph_direct_evidence
        # docstring for why this is necessary on top of vector_based_items alone.
        service_membership = build_service_membership_index(case.topology)
        direct_items = graph_direct_evidence(filtered_observations, graph_result["graph_scores"],
                                              top_n_entities=10, max_per_entity=6,
                                              service_membership=service_membership)
        evidence_items = merge_evidence(vector_based_items, direct_items)

        stats["hybrid_retrieval_time_s"] = time.time() - t4
        stats["evidence_items_retrieved"] = len(evidence_items)
        stats["evidence_items_from_vector"] = len(vector_based_items)
        stats["evidence_items_from_graph_direct"] = len(direct_items)

        # --- Module 5: Evidence Summarizer -----------------------------------------
        t5 = time.time()
        evidence_summary = summarize_evidence(evidence_items)
        stats["summarization_time_s"] = time.time() - t5

        # --- Module 6: Multi-Agent + Coordinator -----------------------------------
        t6 = time.time()
        neighbors = graph_result.get("neighbors", {})
        candidate_entities = [eid for eid, _ in graph_result["ranked_entities"]]

        agent_state = {
            "case_id": case_id,
            "alert_text": parsed_alert["alert_text"],
            "evidence_summary": evidence_summary,
            "graph_neighbors": neighbors,
            "candidate_entities": candidate_entities,
        }
        final_state = self.agent_graph.invoke(agent_state)
        stats["multi_agent_time_s"] = time.time() - t6
        stats.update(self.llm.usage_stats())
        stats["total_pipeline_time_s"] = time.time() - t0

        final = final_state.get("final_result") or {}
        agent_findings = build_agent_findings_list(final_state)

        llm_predicted_entities = final.get("predicted_entity_ids", []) or []
        used_fallback = not llm_predicted_entities
        stats["used_entity_fallback"] = used_fallback

        return RCAResult(
            case_id=case_id,
            predicted_entity_ids=llm_predicted_entities or
                                 ([parsed_alert["entry_entity_id"]] if parsed_alert["entry_entity_id"] else []),
            predicted_fault_type=final.get("predicted_fault_type", "unknown"),
            reasoning_chain=final.get("reasoning_chain", []),
            confidence=float(final.get("confidence", 0.0) or 0.0),
            agent_findings=agent_findings,
            retrieval_stats=stats,
            evidence_items=evidence_items,
        )
