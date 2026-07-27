"""
debug_retrieval.py
====================
Diagnostic for the "retrieval_precision / retrieval_recall keep coming back
0.0" issue. Runs stages 0-4 of the pipeline (ingestion -> alert parse ->
graph retrieval -> vector retrieval -> hybrid retrieval) WITHOUT the LLM, so
it's fast and doesn't need `ollama serve` running.

For each case it prints:
  1. Ground-truth target entity IDs (raw, as stored in answer_key)
  2. Their entity_type in the topology (are they already apm.service, or do
     they need find_service_ancestor() roll-up?)
  3. The set of entity IDs that actually got retrieved as evidence
     (both raw and after service-ancestor roll-up)
  4. The overlap between (2) and (3) -- this is exactly what
     retrieval_precision_recall() computes internally, but broken out
     step-by-step so you can see WHERE it breaks: no overlap at all
     (wrong entities retrieved), or overlap only before/only after roll-up
     (a topology/roll-up problem), etc.
  5. How many observations failed entity resolution entirely (entity_id is
     None) -- these never get graph_score and never roll up, so a case with
     a very high fraction of unresolved rows is a data-loading problem, not
     a retrieval-logic problem.

Usage:
    export RCA100_ROOT=/path/to/RCA100
    python debug_retrieval.py                          # default case list below
    python debug_retrieval.py t003 t005 t007 t008       # specific cases
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sys
from typing import Dict, List, Set

import config
from data.loader import Case, build_service_membership_index, find_service_ancestor
from retrieval.graph import GraphRetriever
from retrieval.vector import build_index_from_observations
from retrieval.hybrid import HybridRetriever, graph_direct_evidence, merge_evidence
from evaluation.scoring import load_ground_truth
from pipeline.pipeline import parse_alert

DEFAULT_CASES = ["t003", "t005", "t006", "t007", "t008"]


def rollup(entity_id: str, topology) -> str:
    return find_service_ancestor(entity_id, topology) or entity_id


def debug_case(case_id: str):
    print(f"\n{'=' * 70}\nCASE {case_id}\n{'=' * 70}")

    case = Case(case_id)
    gt = load_ground_truth(case_id, name_index=case.name_index)

    print(f"entry_entity_id (alert):        {case.alert.entry_entity_id}")
    print(f"  in topology?                  {case.alert.entry_entity_id in case.topology}")
    print(f"gt.fault_type:                  {gt.fault_type}")
    print(f"gt.target_entity_ids (raw):     {gt.target_entity_ids}")

    for eid in gt.target_entity_ids:
        in_topo = eid in case.topology
        etype = case.topology.nodes[eid].get("entity").entity_type if in_topo else "NOT IN TOPOLOGY"
        rolled = rollup(eid, case.topology) if in_topo else None
        print(f"  - {eid}: in_topology={in_topo}, entity_type={etype}, rolls_up_to={rolled}")

    # --- unresolved-entity rate, per modality --------------------------------
    print("\nEntity resolution rate per modality (entity_id is None = unresolved):")
    for modality, obs_list in case.observations.items():
        total = len(obs_list)
        unresolved = sum(1 for o in obs_list if o.entity_id is None)
        pct = (unresolved / total * 100) if total else 0.0
        print(f"  {modality:8s}: {total:6d} rows, {unresolved:6d} unresolved ({pct:5.1f}%)")

    # --- replicate pipeline.py stages 1-4 (no LLM) ----------------------------
    parsed_alert = parse_alert(case)
    graph_retriever = GraphRetriever(case.topology)
    graph_result = graph_retriever.retrieve(parsed_alert["entry_entity_id"])
    subgraph_node_ids = set(graph_result["subgraph"].nodes)

    def _filter_modality(obs_list):
        resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
        unresolved = [o for o in obs_list if o.entity_id is None]
        if config.DEV_QUICK_TEST:
            unresolved = unresolved[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
        if len(resolved) > config.MAX_RESOLVED_PER_MODALITY:
            resolved = sorted(resolved, key=lambda o: o.timestamp or 0,
                               reverse=True)[:config.MAX_RESOLVED_PER_MODALITY]
        return resolved + unresolved

    filtered_observations = {
        modality: _filter_modality(obs_list)
        for modality, obs_list in case.observations.items()
    }
    vector_index = build_index_from_observations(filtered_observations)
    hybrid = HybridRetriever(graph_result, vector_index)
    queries = [parsed_alert["alert_text"]] + parsed_alert["keywords"][:5]
    vector_based_items = hybrid.retrieve_multi(queries, top_k=config.VECTOR_TOP_K)

    service_membership = build_service_membership_index(case.topology)
    direct_items = graph_direct_evidence(filtered_observations, graph_result["graph_scores"],
                                          top_n_entities=25, max_per_entity=6,
                                          service_membership=service_membership,
                                          force_include=graph_result.get("boosted_entities"))
    evidence_items = merge_evidence(vector_based_items, direct_items)

    print(f"\ncandidate_subgraph_size (BFS+PPR):  {len(subgraph_node_ids)} nodes")
    print(f"evidence_items retrieved (total):   {len(evidence_items)}"
          f"  (vector={len(vector_based_items)}, graph_direct={len(direct_items)})")

    retrieved_raw: Set[str] = {it.observation.entity_id for it in evidence_items if it.observation.entity_id}
    retrieved_rolled: Set[str] = {rollup(eid, case.topology) for eid in retrieved_raw}

    gt_set = set(gt.target_entity_ids)
    # BUG FIX: roll up the GT side too before comparing -- previously this
    # compared raw GT ids against rolled-up retrieved ids, which under-counts
    # matches whenever the GT entity is not already an apm.service.
    gt_set_rolled = {rollup(eid, case.topology) for eid in gt_set}

    print(f"\nRetrieved entity IDs (raw, n={len(retrieved_raw)}):      {sorted(retrieved_raw)[:10]}"
          f"{' ...' if len(retrieved_raw) > 10 else ''}")
    print(f"Retrieved entity IDs (after roll-up, n={len(retrieved_rolled)}): "
          f"{sorted(retrieved_rolled)[:10]}{' ...' if len(retrieved_rolled) > 10 else ''}")
    print(f"GT entity IDs (after roll-up):                     {sorted(gt_set_rolled)}")

    overlap_raw = retrieved_raw & gt_set
    overlap_rolled = retrieved_rolled & gt_set_rolled

    # --- NEW: was the GT entity even in the candidate subgraph at all? -------
    print("\n--- Subgraph membership & graph-score rank for GT entity ---")
    graph_scores: Dict[str, float] = graph_result["graph_scores"]
    ranked = sorted(graph_scores.items(), key=lambda kv: kv[1], reverse=True)
    rank_by_id = {eid: i + 1 for i, (eid, _) in enumerate(ranked)}
    for eid in gt_set:
        in_sub = eid in subgraph_node_ids
        gscore = graph_scores.get(eid)
        rank = rank_by_id.get(eid)
        print(f"  raw GT {eid}: in_subgraph={in_sub}, graph_score={gscore}, "
              f"rank={rank} / {len(ranked)} scored nodes (top_n_entities cutoff=25)")
        rolled_eid = rollup(eid, case.topology)
        if rolled_eid != eid:
            in_sub_r = rolled_eid in subgraph_node_ids
            gscore_r = graph_scores.get(rolled_eid)
            rank_r = rank_by_id.get(rolled_eid)
            print(f"  rolled GT {rolled_eid}: in_subgraph={in_sub_r}, graph_score={gscore_r}, "
                  f"rank={rank_r} / {len(ranked)} scored nodes (top_n_entities cutoff=25)")

    print(f"\nOverlap with GT (raw, no roll-up):     {overlap_raw or '(none)'}")
    print(f"Overlap with GT (after roll-up):       {overlap_rolled or '(none)'}")

    if not overlap_raw and not overlap_rolled:
        print("  -> DIAGNOSIS: retrieval never surfaced the GT entity at all "
              "(check whether the GT entity's observations exist in this case's "
              "parquet files, and whether they resolved to a non-None entity_id).")
    elif overlap_raw and not overlap_rolled:
        print("  -> DIAGNOSIS: raw match existed but roll-up broke it "
              "(find_service_ancestor bug, or GT entity_id isn't reachable "
              "from the retrieved node within max_hops=3).")
    elif overlap_rolled and not overlap_raw:
        print("  -> Roll-up IS working as intended (fine-grained hit -> correct service).")
    else:
        print("  -> Retrieval + roll-up both look correct for this case; if "
              "retrieval_precision/recall still show 0.0 in results.csv, the "
              "bug is likely in how full_case_report() calls this function, "
              "not in retrieval itself.")


if __name__ == "__main__":
    case_ids: List[str] = sys.argv[1:] or DEFAULT_CASES
    for cid in case_ids:
        try:
            debug_case(cid)
        except Exception as e:
            print(f"\n[ERROR] case {cid} failed: {type(e).__name__}: {e}")