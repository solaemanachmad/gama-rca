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
from collections import Counter
from typing import Dict, Optional

import config
from data.loader import Case, build_service_membership_index, normalize_entity_ids
from data.taxonomy import taxonomy_prompt_block, detect_fault_keywords, fault_group
from retrieval.graph import GraphRetriever, apply_temporal_boost, compute_propagation_path
from retrieval.vector import build_index_from_observations
from retrieval.hybrid import HybridRetriever, graph_direct_evidence, merge_evidence
from pipeline.evidence_summarizer import summarize_evidence, compute_metric_trends, compute_baseline_stats, ERROR_PATTERN
from pipeline.fault_group_classifier import predict_fault_group
from pipeline.fault_type_classifier import predict_fault_type
from pipeline.case_based_reasoning import find_similar_case
from pipeline.zero_shot_matching import zero_shot_type_match
from agents.multi_agent import build_agent_graph, build_agent_findings_list
from agents.llm_client import LLMClient
from schema import RCAResult


ANCHOR_SYSTEM_PROMPT = (
    "You are an SRE performing a FAST, FIRST-PASS root cause guess using "
    "only topology structure -- no detailed evidence text yet. Pick the single "
    "most topologically central candidate and the single most likely fault "
    "type. This is a quick prior, not a final answer -- a second pass with "
    "full evidence will confirm or correct it."
)


def _compute_graph_anchor(case: Case, parsed_alert: Dict, graph_result: Dict, llm, use_llm: bool = True) -> Dict:
    """Stage 0.5 -- cheap, single-call structural prior computed BEFORE the
    multi-agent stage, mirroring the graphrag_only baseline's approach
    (topology-ranked candidates + full taxonomy, nothing else). Empirically,
    on a 4-case spot check this simple approach scored ~2x higher than the
    full multi-agent Coordinator on both entity_localization (0.504 vs 0.160)
    and fault_identification (0.50 vs 0.0) -- the hypothesis is that a small
    local LLM (7B) reasons more reliably over a short, structured signal than
    over a large multi-agent-findings-plus-evidence-text prompt. Rather than
    replace the multi-agent stage (which wins clearly on reasoning_process),
    this anchor is handed to the Coordinator as a prior to confirm or
    override -- see coordinator_node's prompt in agents/multi_agent.py.

    use_llm=False: skip the LLM call entirely, just take the top-ranked
    entity directly from graph_result["ranked_entities"] -- already a pure
    numerical ranking (BFS + temporal boost), no LLM needed. Exploratory
    ablation motivated by a session-long pattern: every prior LLM-decision
    replaced with a structural/statistical one (this anchor's entity guess,
    fault-group/type classification, propagation path, keyword detection,
    evidence-priority sorting) measurably improved results. This tests
    whether the anchor's own LLM call is similarly replaceable."""
    ranked = graph_result["ranked_entities"][:20]

    if not use_llm:
        if not ranked:
            return {"anchor_entity_ids": [], "anchor_fault_type": "unknown", "anchor_confidence": 0.0}
        top_entity_id, top_score = ranked[0]
        # Normalize confidence as this entity's share of the top-20 total
        # score -- a simple, defensible proxy for "how much more likely is
        # this entity than the alternatives", without needing an LLM to
        # articulate it.
        total = sum(s for _, s in ranked) or 1.0
        return {
            "anchor_entity_ids": [top_entity_id],
            "anchor_fault_type": "unknown",  # not used downstream (classifier handles fault type)
            "anchor_confidence": round(top_score / total, 4),
        }

    ranked_text = "\n".join(f"  {eid}: score={score:.4f}" for eid, score in ranked)
    prompt = (
        f"Alert: {parsed_alert['alert_text']}\n"
        f"Entry entity: {parsed_alert['entry_entity_id']}\n"
        f"Top-20 topology-ranked candidate entities (entity_id: propagation_score):\n{ranked_text}\n\n"
        f"{taxonomy_prompt_block()}\n\n"
        f"Give your fast first-pass guess using only this structural information.\n"
        f'Respond as JSON: {{"predicted_entity_ids": ["entity1"], '
        f'"predicted_fault_type": "<one of the 28 RCA100 fault types>", "confidence": <float 0-1>}}'
    )
    raw = llm.generate_json(prompt, system=ANCHOR_SYSTEM_PROMPT)
    entity_ids = normalize_entity_ids(raw.get("predicted_entity_ids", []) or [], case.topology, case.name_index)
    return {
        "anchor_entity_ids": entity_ids,
        "anchor_fault_type": raw.get("predicted_fault_type", "unknown"),
        "anchor_confidence": float(raw.get("confidence", 0.0) or 0.0),
    }


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
    def __init__(self, llm=None):
        from agents.factory import get_llm_client, get_coordinator_llm_client
        self.llm = llm or get_llm_client()
        self.coordinator_llm = get_coordinator_llm_client()  # None unless explicitly configured
        self.agent_graph = build_agent_graph(self.llm, coordinator_llm=self.coordinator_llm,
                                              use_llm_agents=config.USE_LLM_SPECIALIST_AGENTS)

    def run(self, case_id: str, cases_dir: str = config.CASES_DIR) -> RCAResult:
        self.llm.reset_usage()
        if self.coordinator_llm is not None:
            self.coordinator_llm.reset_usage()
        t0 = time.time()
        stats = {"case_id": case_id}

        # --- Stage 0: ingestion -------------------------------------------------
        case = Case(case_id, cases_dir=cases_dir)
        stats["load_time_s"] = time.time() - t0
        stats.update(case.load_times)  # per-modality breakdown: load_metrics_s, load_logs_s, etc.

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

        # Temporal pre-processing (not just a display label): boosts scores
        # for candidates whose earliest evidence precedes the alert
        # (candidate causes) and discounts entities whose evidence only
        # appears after (likely downstream effects). Applied BEFORE the
        # anchor is computed, so it directly influences entity choice, not
        # just what the LLM happens to notice in a text block.
        if config.TEMPORAL_BOOST_ENABLED:
            graph_result["graph_scores"] = apply_temporal_boost(
                graph_result["graph_scores"], case.observations, case.topology, case.alert.alert_timestamp)
            graph_result["ranked_entities"] = sorted(
                graph_result["graph_scores"].items(), key=lambda kv: kv[1], reverse=True)

        # --- Stage 0.5: Graph Anchor (cheap structural prior) ---------------------
        t2b = time.time()
        graph_anchor = _compute_graph_anchor(case, parsed_alert, graph_result, self.llm,
                                              use_llm=config.USE_LLM_GRAPH_ANCHOR)
        stats["graph_anchor_time_s"] = time.time() - t2b
        stats["graph_anchor_entity_ids"] = "|".join(graph_anchor["anchor_entity_ids"])
        stats["graph_anchor_fault_type"] = graph_anchor["anchor_fault_type"]

        # Factually-grounded propagation path (free -- pure graph algorithm,
        # no LLM call) from the alert's entry entity (the symptom/"impact")
        # to the anchor's top candidate (the "cause"). Maps directly onto
        # RCA100's cause -> propagation -> impact reasoning structure: this
        # path IS the "propagation" stage, computed from real topology
        # edges rather than left for an LLM to invent from scratch.
        propagation_path = None
        if graph_anchor["anchor_entity_ids"] and parsed_alert["entry_entity_id"]:
            propagation_path = compute_propagation_path(
                case.topology, parsed_alert["entry_entity_id"], graph_anchor["anchor_entity_ids"][0])
        stats["propagation_path_hops"] = len(propagation_path) if propagation_path else 0
        stats["propagation_path"] = " -> ".join(propagation_path) if propagation_path else ""

        # --- Module 3: Vector Retrieval (index build) -----------------------------
        t3 = time.time()
        subgraph_node_ids = set(graph_result["subgraph"].nodes)

        # Compute BEFORE the recency-cap in _filter_modality below discards
        # early-window data -- a trend needs both ends of the window, and
        # "keep only the most recent N" would frequently throw away exactly
        # the earliest points a trend comparison needs.
        metrics_in_subgraph = [o for o in case.observations.get("metrics", []) if o.entity_id in subgraph_node_ids]
        # Genuine baseline preprocessing: per-(entity, metric) mean from ALL
        # pre-alert observations (not just whatever survives retrieval), so
        # trend bullets can compare "current value" against a real
        # historical baseline -- mirrors the ground truth's own "基线均值"
        # (baseline mean) methodology (see compute_baseline_stats() for the
        # full rationale). Uses the same subgraph-filtered-but-not-yet-
        # truncated metrics list as compute_metric_trends() below: relevant
        # entities only, but not restricted by the later recency cap.
        baseline_stats = compute_baseline_stats(metrics_in_subgraph, case.alert.alert_timestamp, case.topology)
        metric_trends = compute_metric_trends(metrics_in_subgraph, case.topology, baseline_stats=baseline_stats)

        truncated_modalities = []
        alert_ts = case.alert.alert_timestamp
        if alert_ts is not None and alert_ts.tzinfo is not None:
            alert_ts = alert_ts.replace(tzinfo=None)

        def _filter_modality(obs_list, modality_name):
            resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
            unresolved = [o for o in obs_list if o.entity_id is None]
            if config.DEV_QUICK_TEST:
                unresolved = unresolved[:config.DEV_MAX_UNRESOLVED_PER_MODALITY]
            if len(resolved) > config.MAX_RESOLVED_PER_MODALITY:
                truncated_modalities.append(modality_name)
                budget = config.MAX_RESOLVED_PER_MODALITY
                if alert_ts is not None:
                    # Temporally-stratified truncation around the alert time,
                    # instead of naive "keep the temporally-latest N" -- the
                    # latter can disproportionately discard early-window
                    # evidence (candidate CAUSES, per the causal-precedence
                    # principle behind TEMPORAL_BOOST_ENABLED) in favor of
                    # late-stage noise, whenever a modality genuinely exceeds
                    # the cap. Bias 60/40 toward before-alert evidence
                    # (closest to the alert first, since a precursor right
                    # before the alert is more diagnostic than one from the
                    # very start of a long window), backfilling from the
                    # other side if either side has fewer than its share.
                    def _naive_ts(o):
                        ts = o.timestamp
                        return ts.replace(tzinfo=None) if ts and ts.tzinfo else ts
                    before = sorted([o for o in resolved if _naive_ts(o) and _naive_ts(o) < alert_ts],
                                     key=_naive_ts, reverse=True)  # closest-to-alert first
                    after = sorted([o for o in resolved if _naive_ts(o) and _naive_ts(o) >= alert_ts],
                                    key=_naive_ts)  # closest-to-alert first
                    before_budget = int(budget * 0.6)
                    after_budget = budget - before_budget
                    kept_before = before[:before_budget]
                    kept_after = after[:after_budget]
                    # Backfill unused budget from whichever side has more
                    # available, so the full cap is still used even when
                    # evidence is lopsided to one side of the alert.
                    leftover = budget - len(kept_before) - len(kept_after)
                    if leftover > 0:
                        kept_before += before[len(kept_before):len(kept_before) + leftover]
                        leftover = budget - len(kept_before) - len(kept_after)
                    if leftover > 0:
                        kept_after += after[len(kept_after):len(kept_after) + leftover]
                    resolved = kept_before + kept_after
                else:
                    # No alert timestamp available for this case -- fall
                    # back to the original recency-based truncation.
                    resolved = sorted(resolved, key=lambda o: o.timestamp or 0,
                                       reverse=True)[:budget]
            return resolved + unresolved

        filtered_observations = {
            modality: _filter_modality(obs_list, modality)
            for modality, obs_list in case.observations.items()
        }
        stats["truncated_modalities"] = "|".join(truncated_modalities)
        vector_index = build_index_from_observations(filtered_observations)
        stats["vector_index_build_time_s"] = time.time() - t3
        stats["indexed_observations"] = vector_index.index.ntotal
        stats["observations_before_graph_filter"] = sum(len(v) for v in case.observations.values())
        # Whether MAX_OBSERVATIONS_PER_INDEX actually capped the vector
        # index for this case (indexed_observations hitting the ceiling
        # exactly, vs the natural/uncapped total being smaller).
        total_filtered = sum(len(v) for v in filtered_observations.values())
        stats["vector_index_was_capped"] = total_filtered > config.MAX_OBSERVATIONS_PER_INDEX

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
                                              top_n_entities=25, max_per_entity=6,
                                              service_membership=service_membership,
                                              force_include=graph_result.get("boosted_entities"))
        evidence_items = merge_evidence(vector_based_items, direct_items)

        stats["hybrid_retrieval_time_s"] = time.time() - t4
        stats["evidence_items_retrieved"] = len(evidence_items)
        stats["evidence_items_from_vector"] = len(vector_based_items)
        stats["evidence_items_from_graph_direct"] = len(direct_items)

        # Semantic features for the fault-group classifier (see
        # scripts/train_fault_group_classifier.py) -- per-modality evidence
        # counts specifically, since the 30-case modality audit
        # (scripts/audit_fault_groups.py) found Events correlates strongly
        # with K8s lifecycle faults (5/5 usable there vs near-zero
        # elsewhere) while being a near-useless raw signal for other
        # groups -- exactly the kind of category-discriminating feature a
        # classifier can exploit that pure volume/timing features can't.
        modality_counts = Counter(it.observation.modality for it in evidence_items)
        for m in ("metrics", "logs", "traces", "events", "alerts"):
            stats[f"evidence_count_{m}"] = modality_counts.get(m, 0)
        stats["log_error_pattern_count"] = sum(
            1 for it in evidence_items
            if it.observation.modality == "logs" and ERROR_PATTERN.search(it.observation.text))

        # Metric-trend direction counts (from compute_metric_trends above) --
        # a fault whose evidence is dominated by "increase" trends looks
        # structurally different (as a feature vector) from one dominated
        # by "dropped to zero" or "new nonzero" signals, even before any
        # LLM reads the text.
        trend_bullets = [b for bullets in metric_trends.values() for b in bullets]
        stats["trend_increase_count"] = sum(1 for b in trend_bullets if "increase" in b)
        stats["trend_decrease_count"] = sum(1 for b in trend_bullets if "decrease" in b)
        stats["trend_new_nonzero_count"] = sum(1 for b in trend_bullets if "new nonzero" in b)
        stats["trend_dropped_zero_count"] = sum(1 for b in trend_bullets if "dropped to zero" in b)

        # --- Module 5: Evidence Summarizer -----------------------------------------
        t5 = time.time()
        evidence_summary = summarize_evidence(evidence_items, alert_timestamp=case.alert.alert_timestamp,
                                               metric_trends=metric_trends)
        stats["summarization_time_s"] = time.time() - t5

        # Keyword-based fault-type detection: scans the RAW evidence text
        # (before summarization -- the summarizer's own bullet compression
        # can drop the exact phrase a keyword needs) for literal terms tied
        # to each of the 28 fault types (see data.taxonomy.FAULT_TYPE_KEYWORDS).
        # Deterministic, not embedding-similarity-based -- targets exactly
        # the failure mode found earlier (the removed shortlist mechanism
        # missing domain terms like "valkey" that generic embeddings don't
        # associate with "redisUnavailable").
        #
        # SCOPED to the top-ranked candidate entities only (by hybrid_score),
        # not the entire evidence pool. evidence_items routinely spans 100+
        # items across dozens of different services (top_n_entities=25 in
        # graph_direct_evidence, plus vector hits) -- if even one item from
        # an unrelated service has a genuine but irrelevant "500" (normal
        # background noise in any distributed system), it was flooding
        # httpError5xx as a false "detection" on nearly every case
        # regardless of the actual injected fault. This is a scope bug, not
        # a regex bug: a keyword only means something if it's near the
        # SUSPECTED root cause, not anywhere in the whole retrieved pool.
        top_entity_ids = set()
        for it in sorted(evidence_items, key=lambda x: x.hybrid_score, reverse=True):
            if it.observation.entity_id:
                top_entity_ids.add(it.observation.entity_id)
            if len(top_entity_ids) >= 3:
                break
        scoped_evidence_text = " ".join(
            it.observation.text for it in evidence_items
            if it.observation.entity_id in top_entity_ids)
        keyword_fault_candidates = detect_fault_keywords(scoped_evidence_text)[:5]
        stats["keyword_fault_candidates"] = "|".join(keyword_fault_candidates)

        # Two-stage fault classification: predict the coarse group (0.66-0.68
        # LOOCV accuracy on 103 cases, stable across runs where the LLM's own
        # end-to-end fault_group accuracy swung 0.39-0.44 -- see
        # findings_session_20260728.md Section 4) BEFORE the Coordinator
        # runs, so its taxonomy prompt can be narrowed to that group's 3-7
        # types instead of all 28. Soft narrowing, not a hard restriction --
        # see coordinator_node in agents/multi_agent.py for why: the
        # earlier graph_anchor hard-override design (forcing its own
        # fault_type guess with no escape hatch) was found to collapse to a
        # single generic answer, since the anchor has no evidence text to
        # ground a fault-type decision in. A classifier with real accuracy
        # deserves more trust than that, but still not unconditional
        # override, given ~1/3 of predictions are wrong.
        classifier_prediction = predict_fault_group(stats, case_id=case_id)
        if classifier_prediction:
            stats["classifier_predicted_group"] = classifier_prediction[0]
            stats["classifier_confidence"] = classifier_prediction[1]
        else:
            stats["classifier_predicted_group"] = None
            stats["classifier_confidence"] = None

        # Tier-2 classifier: fine-grained type WITHIN the group just
        # predicted above. Masked to only the ~3-7 types belonging to that
        # group (see pipeline/fault_type_classifier.py) -- this is what
        # makes the sparse per-type training data (several types have only
        # 1 example in the full 103-case corpus) usable at all.
        type_prediction = predict_fault_type(stats, known_group=stats["classifier_predicted_group"], case_id=case_id)
        if type_prediction:
            stats["classifier_predicted_type"] = type_prediction[0]
            stats["classifier_type_confidence"] = type_prediction[1]
        else:
            stats["classifier_predicted_type"] = None
            stats["classifier_type_confidence"] = None

        # Case-Based Reasoning fallback (see pipeline/case_based_reasoning.py):
        # nearest-neighbor lookup that only surfaces a hint when the closest
        # historical case is one of the singleton (n=1) fault types the
        # RandomForest tier-2 classifier structurally cannot learn from.
        # Motivated by MicroCBR (Liu et al. 2022) -- complements, doesn't
        # replace, the classifier above.
        cbr_result = find_similar_case(stats, case_id=case_id)
        if cbr_result:
            stats["cbr_suggested_type"] = cbr_result[0]
            stats["cbr_similarity"] = cbr_result[1]
            stats["cbr_matched_case_id"] = cbr_result[2]
        else:
            stats["cbr_suggested_type"] = None
            stats["cbr_similarity"] = None
            stats["cbr_matched_case_id"] = None

        # Zero-shot semantic matching (see pipeline/zero_shot_matching.py):
        # compares the same scoped evidence text used for keyword detection
        # against fault-type DEFINITION text (not examples) -- works even
        # for types with zero labeled examples. Scoped to a candidate
        # group -- prefer the tier-1 classifier's prediction when
        # available, but fall back to graph_anchor's own fault_type guess
        # (converted to its group) when the classifier .pkl isn't present.
        # This fallback matters: without it, zero-shot silently depended on
        # the classifier for group scoping, defeating its whole point of
        # working without any trained/labeled-example model at all.
        zeroshot_group = stats["classifier_predicted_group"]
        if not zeroshot_group and graph_anchor.get("anchor_fault_type") not in (None, "unknown"):
            zeroshot_group = fault_group(graph_anchor["anchor_fault_type"])
        zeroshot_result = zero_shot_type_match(scoped_evidence_text, zeroshot_group)
        if zeroshot_result:
            stats["zeroshot_suggested_type"] = zeroshot_result[0]
            stats["zeroshot_similarity"] = zeroshot_result[1]
        else:
            stats["zeroshot_suggested_type"] = None
            stats["zeroshot_similarity"] = None

        # --- Module 6: Multi-Agent + Coordinator -----------------------------------
        t6 = time.time()
        neighbors = graph_result.get("neighbors", {})
        candidate_entities = [eid for eid, _ in graph_result["ranked_entities"][:20]]

        agent_state = {
            "case_id": case_id,
            "alert_text": parsed_alert["alert_text"],
            "evidence_summary": evidence_summary,
            "graph_neighbors": neighbors,
            "candidate_entities": candidate_entities,
            "graph_anchor": graph_anchor,
            "propagation_path": propagation_path,
            "keyword_fault_candidates": keyword_fault_candidates,
            "classifier_predicted_group": stats["classifier_predicted_group"],
            "classifier_confidence": stats["classifier_confidence"],
            "classifier_predicted_type": stats["classifier_predicted_type"],
            "classifier_type_confidence": stats["classifier_type_confidence"],
            "cbr_suggested_type": stats["cbr_suggested_type"],
            "cbr_similarity": stats["cbr_similarity"],
            "zeroshot_suggested_type": stats["zeroshot_suggested_type"],
            "zeroshot_similarity": stats["zeroshot_similarity"],
        }
        final_state = self.agent_graph.invoke(agent_state)
        stats["multi_agent_time_s"] = time.time() - t6
        combined_usage = self.llm.usage_stats()
        if self.coordinator_llm is not None:
            coord_usage = self.coordinator_llm.usage_stats()
            combined_usage = {
                "total_calls": combined_usage.get("total_calls", 0) + coord_usage.get("total_calls", 0),
                "total_tokens": combined_usage.get("total_tokens", 0) + coord_usage.get("total_tokens", 0),
            }
        stats.update(combined_usage)
        stats["total_pipeline_time_s"] = time.time() - t0

        final = final_state.get("final_result") or {}
        agent_findings = build_agent_findings_list(final_state)

        # Split override by task, NOT a uniform anchor-wins-everything rule.
        # entity_id: anchor is demonstrably strong here (pure topology
        # ranking is a good WHERE signal, confirmed by entity_localization
        # jumping to exact matches like t002=1.0).
        # fault_type: anchor is structurally blind here -- its prompt has
        # NO evidence text at all (only entity_id:score pairs), so it has
        # zero signal to distinguish "Redis unavailable" from "node OOM"
        # from "traffic surge" and collapsed to the same generic
        # "Application logic"-group guess (often literally
        # "F006-trafficSurge") across every case regardless of the real
        # fault. The Coordinator DOES see real evidence bullets, so
        # fault_type stays its responsibility, not the anchor's.
        anchor_entities = graph_anchor.get("anchor_entity_ids") or []
        if anchor_entities:
            predicted_entity_ids = anchor_entities
            used_fallback = False
        else:
            llm_predicted_entities = final.get("predicted_entity_ids", []) or []
            predicted_entity_ids = normalize_entity_ids(llm_predicted_entities, case.topology, case.name_index) or \
                ([parsed_alert["entry_entity_id"]] if parsed_alert["entry_entity_id"] else [])
            used_fallback = not llm_predicted_entities
        predicted_fault_type = final.get("predicted_fault_type", "unknown")
        stats["used_entity_fallback"] = used_fallback

        return RCAResult(
            case_id=case_id,
            predicted_entity_ids=predicted_entity_ids,
            predicted_fault_type=predicted_fault_type,
            reasoning_chain=final.get("reasoning_chain", []),
            confidence=float(final.get("confidence", 0.0) or 0.0),
            agent_findings=agent_findings,
            retrieval_stats=stats,
            evidence_items=evidence_items,
        )