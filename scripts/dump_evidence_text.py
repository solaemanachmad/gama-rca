"""
dump_evidence_text.py
=======================
Prints the EXACT evidence_summary text handed to the multi-agent Coordinator
for a given case -- i.e. what the LLM actually sees, no more no less. Use
this to check whether the evidence carries any signal that would let even a
human distinguish "Redis unavailable" from "traffic surge", or whether it's
just generic request_count/workload numbers that look the same regardless
of the real fault type.

Usage:
    python scripts/dump_evidence_text.py t002   # gt=redisUnavailable
    python scripts/dump_evidence_text.py t003   # gt=nodeMemoryOOM
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.loader import Case, build_service_membership_index
from retrieval.graph import GraphRetriever
from retrieval.vector import build_index_from_observations
from retrieval.hybrid import HybridRetriever, graph_direct_evidence, merge_evidence
from pipeline.evidence_summarizer import summarize_evidence, render_summary_text
from pipeline.pipeline import parse_alert
from evaluation.scoring import load_ground_truth

case_id = sys.argv[1] if len(sys.argv) > 1 else "t002"
case = Case(case_id)
gt = load_ground_truth(case_id, name_index=case.name_index)

print(f"=== {case_id} ===")
print(f"gt.fault_type: {gt.fault_type}")
print(f"gt.target_entity_ids: {gt.target_entity_ids}\n")

parsed_alert = parse_alert(case)
graph_retriever = GraphRetriever(case.topology)
graph_result = graph_retriever.retrieve(parsed_alert["entry_entity_id"])
subgraph_node_ids = set(graph_result["subgraph"].nodes)

def _filter_modality(obs_list, subgraph_node_ids):
    """Exact copy of pipeline.py's _filter_modality -- sorts resolved
    observations by recency before capping, separately caps unresolved.
    (An earlier version of this script used a naive unsorted slice here,
    which gave a misleading picture of what evidence survives filtering --
    always replicate the real pipeline's logic exactly, don't approximate.)"""
    resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
    unresolved = [o for o in obs_list if o.entity_id is None]
    if config.DEV_QUICK_TEST:
        unresolved = unresolved[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
    if len(resolved) > config.MAX_RESOLVED_PER_MODALITY:
        resolved = sorted(resolved, key=lambda o: o.timestamp or 0, reverse=True)[:config.MAX_RESOLVED_PER_MODALITY]
    return resolved + unresolved


filtered_observations = {
    modality: _filter_modality(obs_list, subgraph_node_ids)
    for modality, obs_list in case.observations.items()
}
vector_index = build_index_from_observations(filtered_observations)
hybrid = HybridRetriever(graph_result, vector_index)
queries = [parsed_alert["alert_text"]] + parsed_alert["keywords"][:5]
vector_based_items = hybrid.retrieve_multi(queries, top_k=config.VECTOR_TOP_K)

service_membership = build_service_membership_index(case.topology)
direct_items = graph_direct_evidence(filtered_observations, graph_result["graph_scores"],
                                      top_n_entities=25, max_per_entity=6,
                                      service_membership=service_membership)
evidence_items = merge_evidence(vector_based_items, direct_items)

print(f"total evidence_items: {len(evidence_items)}")
by_modality = {}
for it in evidence_items:
    by_modality.setdefault(it.observation.modality, 0)
    by_modality[it.observation.modality] += 1
print(f"by modality: {by_modality}\n")

evidence_summary = summarize_evidence(evidence_items)
text = render_summary_text(evidence_summary)
print("=== EXACT TEXT HANDED TO THE COORDINATOR ===")
print(text)

print("\n=== raw evidence items (first 40, for cross-check) ===")
for it in evidence_items[:40]:
    print(f"  [{it.observation.modality}] entity={it.observation.entity_id} "
          f"score={it.hybrid_score:.3f} | {it.observation.text[:120]}")
