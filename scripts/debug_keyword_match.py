"""
debug_keyword_match.py
=========================
Dumps the EXACT evidence text snippet(s) that trigger a keyword_fault_type
match for a given case -- run this when keyword_fault_candidates shows
something suspicious (e.g. "httpError5xx" on a case whose ground truth is
totally unrelated), to see whether it's a genuine signal (real "500" in
real evidence, possibly from a cascading effect) or a matching bug.

Usage:
    python scripts/debug_keyword_match.py t002
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import config
from data.loader import Case, build_service_membership_index
from retrieval.graph import GraphRetriever, apply_temporal_boost
from retrieval.vector import build_index_from_observations
from retrieval.hybrid import HybridRetriever, graph_direct_evidence, merge_evidence
from pipeline.pipeline import parse_alert
from data.taxonomy import FAULT_TYPE_KEYWORDS

case_id = sys.argv[1] if len(sys.argv) > 1 else "t002"
case = Case(case_id)

parsed_alert = parse_alert(case)
graph_retriever = GraphRetriever(case.topology)
graph_result = graph_retriever.retrieve(parsed_alert["entry_entity_id"])
graph_result["graph_scores"] = apply_temporal_boost(
    graph_result["graph_scores"], case.observations, case.topology, case.alert.alert_timestamp)
graph_result["ranked_entities"] = sorted(
    graph_result["graph_scores"].items(), key=lambda kv: kv[1], reverse=True)

subgraph_node_ids = set(graph_result["subgraph"].nodes)


def _filter_modality(obs_list):
    resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
    unresolved = [o for o in obs_list if o.entity_id is None]
    if config.DEV_QUICK_TEST:
        unresolved = unresolved[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
    if len(resolved) > config.MAX_RESOLVED_PER_MODALITY:
        resolved = sorted(resolved, key=lambda o: o.timestamp or 0, reverse=True)[:config.MAX_RESOLVED_PER_MODALITY]
    return resolved + unresolved


filtered_observations = {m: _filter_modality(o) for m, o in case.observations.items()}
vector_index = build_index_from_observations(filtered_observations)
hybrid = HybridRetriever(graph_result, vector_index)
queries = [parsed_alert["alert_text"]] + parsed_alert["keywords"][:5]
vector_based_items = hybrid.retrieve_multi(queries, top_k=config.VECTOR_TOP_K)
service_membership = build_service_membership_index(case.topology)
direct_items = graph_direct_evidence(filtered_observations, graph_result["graph_scores"],
                                      top_n_entities=25, max_per_entity=6,
                                      service_membership=service_membership)
evidence_items = merge_evidence(vector_based_items, direct_items)

top_entity_ids = set()
for it in sorted(evidence_items, key=lambda x: x.hybrid_score, reverse=True):
    if it.observation.entity_id:
        top_entity_ids.add(it.observation.entity_id)
    if len(top_entity_ids) >= 3:
        break

print(f"=== {case_id}: top 3 entities used for keyword scoping ===")
for eid in top_entity_ids:
    entity = case.topology.nodes[eid].get("entity") if eid in case.topology else None
    name = f"{entity.name} ({entity.entity_type})" if entity else "?"
    print(f"  {eid}  ->  {name}")

scoped_items = [it for it in evidence_items if it.observation.entity_id in top_entity_ids]
print(f"\n=== {len(scoped_items)} evidence items in scope ===\n")

for slug, keywords in FAULT_TYPE_KEYWORDS.items():
    for kw in keywords:
        pattern = (r"(?<![\d.])" + re.escape(kw) + r"(?![\d.])") if kw.isdigit() else (r"\b" + re.escape(kw) + r"\b")
        for it in scoped_items:
            matches = re.findall(pattern, it.observation.text.lower())
            if matches:
                print(f"[{slug}] keyword={kw!r} matched {len(matches)}x in:")
                print(f"    entity={it.observation.entity_id} modality={it.observation.modality}")
                print(f"    text={it.observation.text[:200]!r}")
                print()
