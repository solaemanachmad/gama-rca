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

import json
import random
import re
import time
from collections import Counter
from typing import Dict, List, Optional
from pipeline.twist_scoring import (compute_twist_scores, twist_scores_to_summary,
                                     twist_top_entity, twist_scores_to_observations)

import os
import sys
import config
from data.loader import Case, build_service_membership_index, normalize_entity_ids
from data.taxonomy import taxonomy_prompt_block, detect_fault_keywords
from retrieval.graph import GraphRetriever, apply_temporal_boost, compute_propagation_path
from retrieval.vector import build_index_from_observations
from retrieval.hybrid import HybridRetriever, graph_direct_evidence, merge_evidence
from pipeline.evidence_summarizer import summarize_evidence, compute_metric_trends, compute_baseline_stats, ERROR_PATTERN
from pipeline.zero_shot_matching import zero_shot_type_match_topk
from agents.multi_agent import build_agent_graph, build_agent_findings_list
from agents.llm_client import LLMClient
from schema import EvidenceItem, RCAResult
from pipeline.propagation_evidence import compute_propagation_evidence
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.colocation_evidence import compute_colocation_evidence
from pipeline.log_templates import compute_log_templates


ANCHOR_SYSTEM_PROMPT = (
    "You are an SRE performing a FAST, FIRST-PASS root cause guess using "
    "only topology structure -- no detailed evidence text yet. Pick the single "
    "most topologically central candidate and the single most likely fault "
    "type. This is a quick prior, not a final answer -- a second pass with "
    "full evidence will confirm or correct it."
)


ANCHOR_SYSTEM_PROMPT_PROPAGATION = (
    "You are an SRE doing a first-pass root cause guess. An alert fires on the "
    "service that NOTICES a failure, which is often only a VICTIM: the origin "
    "is frequently a downstream dependency (callee) whose error is relayed "
    "upstream, sometimes the alert service itself, sometimes infrastructure. "
    "Use the call-graph table to decide whether the alert service is the "
    "origin or a victim, then pick the ORIGIN: the dependency whose anomaly is "
    "not explained by one of its own callees. Do not choose a service merely "
    "because it is close to the alert or central in the graph."
)


def _compute_graph_anchor(case: Case, parsed_alert: Dict, graph_result: Dict, llm, use_llm: bool = True,
                          propagation_text: str = "") -> Dict:
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
        total = sum(s for _, s in ranked) or 1.0
        return {
            "anchor_entity_ids": [top_entity_id],
            "anchor_fault_type": "unknown",
            "anchor_confidence": round(top_score / total, 4),
        }

    ranked_text = "\n".join(f"  {eid}: score={score:.4f}" for eid, score in ranked)
    prompt = (
        f"Alert: {parsed_alert['alert_text']}\n"
        f"Entry entity: {parsed_alert['entry_entity_id']}\n"
        f"Top-20 topology-ranked candidate entities (entity_id: propagation_score):\n{ranked_text}\n\n"
        + (f"{propagation_text}\n\n" if propagation_text else "")
        + f"{taxonomy_prompt_block()}\n\n"
        f"Give your fast first-pass guess using only this structural information.\n"
        f'Respond as JSON: {{"predicted_entity_ids": ["entity1"], '
        f'"predicted_fault_type": "<one of the 28 RCA100 fault types>", "confidence": <float 0-1>}}'
    )
    # Force greedy decoding for this one call. llm.temperature defaults to
    # config.LLM_TEMPERATURE=0.1 (nonzero), and kaggle_client.py's generate()
    # does `do_sample = self.temperature > 0` -- so at the shared client's
    # normal temperature this "cheap structural prior" call is NOT
    # deterministic: identical input (same case, same code) can produce a
    # different anchor_entity_ids/anchor_fault_type on separate runs, purely
    # from sampling noise. Caught 2026-10-04 by cross-referencing two
    # separate 8-case Kaggle runs of nominally-identical code: t052's
    # graph_anchor_fault_type was F006-trafficSurge in one run and
    # F004-trafficHotspot in the other. That confounds every anchor-related
    # ablation in this project (self-consistency, two-stage anchor, anchor
    # wording) -- a change in predicted_fault_type between two runs could be
    # the mechanism under test, or just anchor noise, and there was no way to
    # tell them apart. Setting temperature=0.0 here makes do_sample=False
    # (greedy argmax) for this call only -- the anchor becomes reproducible
    # across runs holding the rest of the pipeline fixed, without touching
    # the Coordinator's own temperature/self-consistency sampling elsewhere.
    old_temperature = llm.temperature
    llm.temperature = 0.0
    try:
        raw = llm.generate_json(
            prompt, system=ANCHOR_SYSTEM_PROMPT_PROPAGATION if propagation_text else ANCHOR_SYSTEM_PROMPT)
    finally:
        llm.temperature = old_temperature
    entity_ids = normalize_entity_ids(raw.get("predicted_entity_ids", []) or [], case.topology, case.name_index)
    return {
        "anchor_entity_ids": entity_ids,
        "anchor_fault_type": raw.get("predicted_fault_type", "unknown"),
        "anchor_confidence": float(raw.get("confidence", 0.0) or 0.0),
    }


_CJK_RE = re.compile(r"[一-鿿]")
_JIEBA_WARNED = False


def _tokenize_alert_text(text: str) -> List[str]:
    """Word/phrase-level tokens from alert text, handling both Latin
    (whitespace-delimited) and Chinese (RCA100's alert_title is frequently
    Chinese, e.g. "checkout响应时间突增告警" -- no whitespace between
    Chinese words at all, so a plain .split() treats the whole run as ONE
    meaningless blob token, and even "checkout" ends up fused into it).
    Uses jieba (Chinese word segmentation) to split CJK runs into their
    actual words (e.g. "响应时间"/"突增"/"告警" -- "response time" /
    "surge" / "alert"), which then work as real vector-search query terms
    (see parse_alert's caller in run()) instead of being silently lost.

    Falls back to the previous naive whitespace split (Chinese portions
    stay unsegmented as one blob, but nothing crashes) if jieba isn't
    installed -- `pip install jieba`.

    Length filters differ by script: Latin tokens keep the original >3-char
    threshold (filters short filler words like "the"/"for"); meaningful
    Chinese words are frequently just 2 characters, so those only need
    length > 1."""
    global _JIEBA_WARNED
    try:
        import jieba
    except ImportError:
        if not _JIEBA_WARNED:
            print("[parse_alert] jieba not installed -- Chinese alert text "
                  "(common in RCA100's alert_title) will not be segmented "
                  "into keywords, only kept as one unsplit blob. "
                  "`pip install jieba` to fix.")
            _JIEBA_WARNED = True
        return [w.strip(".,:;") for w in text.split() if len(w) > 3]

    tokens = []
    for t in jieba.cut(text, cut_all=False):
        t = t.strip(".,:;、，。 ")
        if not t:
            continue
        min_len = 1 if _CJK_RE.search(t) else 3
        if len(t) > min_len:
            tokens.append(t)
    return tokens


# ---------------------------------------------------------------------------
# Module 1 — Alert Parser
# ---------------------------------------------------------------------------
def parse_alert(case: Case) -> Dict:
    """Extract candidate entities / keywords from the alert text + entry
    entity. Kept intentionally simple (regex/keyword split, now with jieba
    segmentation for Chinese runs -- see _tokenize_alert_text); swap in an
    NER model here if alert text is richer than the RCA100 structured form."""
    alert = case.alert
    keywords = _tokenize_alert_text(alert.alert_text)
    return {
        "entry_entity_id": alert.entry_entity_id,
        "keywords": list(dict.fromkeys(keywords))[:20],
        "alert_text": alert.alert_text,
    }


# ---------------------------------------------------------------------------
# Evidence-volume reduction for UNRESOLVED observations (entity_id is None,
# so they can't be bounded by subgraph membership the way `resolved`
# observations are). Adapted from GALA's (Tian et al. 2025, arXiv:2508.12472,
# Section 4.2) "Error-Centric Log Abstraction" and "Temporal Performance
# Profiling": reduce volume by filtering to RELEVANT signal first (error/
# exception severity, de-duplicated by message template), and only fall back
# to sampling -- a representative random sample, not a positional cut -- once
# that already-filtered set is still oversized. This replaces the previous
# DEV_QUICK_TEST-only unresolved[:300] positional truncation, which (a) had
# no effect at all outside dev mode (the real 103-case run saw the FULL
# unresolved set, causing multi-thousand-second per-case runtimes) and
# (b) was biased toward whatever order the data happens to arrive in, not
# toward the more informative entries.
#
# The budget itself reuses config.MAX_RESOLVED_PER_MODALITY rather than
# introducing a second, separately-tuned magic number -- the same evidence
# budget already accepted for `resolved` observations applies here too, so
# there is exactly one volume-control knob in config to reason about.
_LOG_TEMPLATE_DIGITS_RE = re.compile(r"\d+")


def _log_template(text: str) -> str:
    """Canonicalizes a log line to its "shape" by blanking out digit runs
    (timestamps, ports, latencies, IDs, status codes, ...), so that many
    near-identical lines differing only in those values collapse to one
    template -- the de-duplication step GALA's Error-Centric Log Abstraction
    performs before any threshold/sampling is applied."""
    return _LOG_TEMPLATE_DIGITS_RE.sub("#", text.lower())


def _reduce_unresolved_logs(unresolved: List, budget: int, rng: random.Random) -> List:
    """GALA-style log abstraction: keep error/exception-severity lines,
    de-duplicate by template (one representative per distinct shape), then
    -- only if that still exceeds budget -- take a representative RANDOM
    sample (not the first N) so the kept subset isn't skewed toward
    whichever timestamp happens to sort first. If the error-only set is
    under budget, supplement with a random sample of non-error lines up to
    budget, mirroring GALA's rationale: a case with genuinely few errors
    should still show the LLM that logging was happening (to distinguish
    "no failures logged" from "no logs collected at all"), not be silently
    padded with duplicate error noise instead."""
    if len(unresolved) <= budget:
        return unresolved

    error_obs = [o for o in unresolved if ERROR_PATTERN.search(o.text)]
    non_error_obs = [o for o in unresolved if not ERROR_PATTERN.search(o.text)]

    seen_templates = set()
    deduped_errors = []
    for o in error_obs:
        tmpl = _log_template(o.text)
        if tmpl not in seen_templates:
            seen_templates.add(tmpl)
            deduped_errors.append(o)

    if len(deduped_errors) >= budget:
        return rng.sample(deduped_errors, budget)

    remaining = budget - len(deduped_errors)
    supplement = rng.sample(non_error_obs, min(remaining, len(non_error_obs)))
    return deduped_errors + supplement


def _reduce_unresolved_generic(unresolved: List, budget: int, rng: random.Random) -> List:
    """Non-log modalities (metrics/traces/events/alerts) don't have an
    error-pattern equivalent to filter by, so the volume-control fallback is
    an unbiased random sample rather than a positional truncation -- still a
    real improvement over unresolved[:N], which always kept the same
    (arbitrary, arrival-order) subset every run."""
    if len(unresolved) <= budget:
        return unresolved
    return rng.sample(unresolved, budget)


def reduce_unresolved_observations(unresolved: List, modality_name: str, budget: int,
                                    seed: int = config.RANDOM_SEED) -> List:
    """Dispatches to the modality-appropriate reduction. A fresh Random(seed)
    per call (rather than one shared/global RNG) keeps this deterministic
    and reproducible across runs regardless of call order or how many other
    modalities/cases were processed first."""
    rng = random.Random(seed)
    if modality_name == "logs":
        return _reduce_unresolved_logs(unresolved, budget, rng)
    return _reduce_unresolved_generic(unresolved, budget, rng)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class GraphRAGPipeline:
    def __init__(self, llm=None):
        from agents.factory import get_llm_client, get_coordinator_llm_client
        self.llm = llm or get_llm_client()
        self.coordinator_llm = get_coordinator_llm_client()
        self.agent_graph = build_agent_graph(self.llm, coordinator_llm=self.coordinator_llm,
                                              use_llm_agents=config.USE_LLM_SPECIALIST_AGENTS)

    def run(self, case_id: str, cases_dir: str = config.CASES_DIR) -> RCAResult:
        self.llm.reset_usage()
        if self.coordinator_llm is not None:
            self.coordinator_llm.reset_usage()
        t0 = time.time()
        stats = {"case_id": case_id}

        # --- Stage 0: ingestion -------------------------------------------
        case = Case(case_id, cases_dir=cases_dir)
        stats["load_time_s"] = time.time() - t0
        stats.update(case.load_times)

        # --- Module 1: Alert Parser ----------------------------------------
        t1 = time.time()
        parsed_alert = parse_alert(case)
        stats["alert_parse_time_s"] = time.time() - t1

        # --- Module 2: Graph Retrieval -------------------------------------
        t2 = time.time()
        graph_retriever = GraphRetriever(case.topology)
        graph_result = graph_retriever.retrieve(parsed_alert["entry_entity_id"])
        stats["graph_retrieval_time_s"] = time.time() - t2
        stats["candidate_subgraph_size"] = graph_result["subgraph"].number_of_nodes()

        if config.TEMPORAL_BOOST_ENABLED:
            graph_result["graph_scores"] = apply_temporal_boost(
                graph_result["graph_scores"], case.observations, case.topology, case.alert.alert_timestamp)
            graph_result["ranked_entities"] = sorted(
                graph_result["graph_scores"].items(), key=lambda kv: kv[1], reverse=True)

        # --- Stage 0.5: Graph Anchor --------------------------------------
        t2b = time.time()
        propagation_ev = {"rows": [], "text": ""}
        if config.USE_PROPAGATION_EVIDENCE:
            t_prop = time.time()
            propagation_ev = compute_propagation_evidence(
                case, case.alert.alert_timestamp, parsed_alert["entry_entity_id"])
            stats["propagation_evidence_time_s"] = round(time.time() - t_prop, 3)
        stats["use_propagation_evidence"] = bool(config.USE_PROPAGATION_EVIDENCE)
        stats["propagation_evidence_rows"] = len(propagation_ev["rows"])
        infra_ev = {"rows": [], "text": "", "observations": []}
        if config.USE_INFRA_EVIDENCE:
            t_inf = time.time()
            infra_ev = compute_infra_evidence(
                case, case.alert.alert_timestamp, parsed_alert["entry_entity_id"])
            stats["infra_evidence_time_s"] = round(time.time() - t_inf, 3)
        stats["use_infra_evidence"] = bool(config.USE_INFRA_EVIDENCE)
        stats["infra_evidence_rows"] = len(infra_ev["rows"])
        stats["infra_evidence_chars"] = len(infra_ev["text"])
        stats["chain_first"] = bool(config.CHAIN_FIRST)
        coloc_ev = {"text": "", "rows": 0, "stats": {}}
        if config.USE_COLOCATION_EVIDENCE:
            t_col = time.time()
            coloc_ev = compute_colocation_evidence(case, case.alert.alert_timestamp)
            stats["colocation_time_s"] = round(time.time() - t_col, 3)
            stats["colocation_error"] = coloc_ev.get("error")
        stats["use_colocation_evidence"] = bool(config.USE_COLOCATION_EVIDENCE)
        stats["colocation_nodes_shown"] = coloc_ev["rows"]
        stats["colocation_chars"] = len(coloc_ev["text"])
        stats["layer_agent"] = bool(config.LAYER_AGENT)
        stats["chain_first_entity"] = config.CHAIN_FIRST_ENTITY
        graph_anchor = _compute_graph_anchor(case, parsed_alert, graph_result, self.llm,
                                              use_llm=config.USE_LLM_GRAPH_ANCHOR,
                                              propagation_text=propagation_ev["text"])
        stats["graph_anchor_time_s"] = time.time() - t2b
        stats["graph_anchor_entity_ids"] = "|".join(graph_anchor["anchor_entity_ids"])
        stats["graph_anchor_fault_type"] = graph_anchor["anchor_fault_type"]

        propagation_path = None
        if graph_anchor["anchor_entity_ids"] and parsed_alert["entry_entity_id"]:
            propagation_path = compute_propagation_path(
                case.topology, parsed_alert["entry_entity_id"], graph_anchor["anchor_entity_ids"][0])
        stats["propagation_path_hops"] = len(propagation_path) if propagation_path else 0
        stats["propagation_path"] = " -> ".join(propagation_path) if propagation_path else ""

        # --- TWIST: Trace-based anomaly scoring (GALA adaptation) ---------
        # Computes 4 complementary service-level scores from distributed
        # traces BEFORE any filtering/truncation, using the full raw
        # observation list for maximum signal coverage:
        #   c1 = self-anomaly (service's own spans anomalous)
        #   c2 = trace impact (fraction of anomalous traces it appears in)
        #   c3 = blast radius (downstream fan-out, propagation risk)
        #   c4 = delay severity (magnitude of latency deviation)
        # Adapted from GALA (Tian et al. 2025, arXiv:2508.12472). GALA uses
        # TWIST for entity ranking only; we additionally expose the 4 scores
        # as evidence for the Coordinator (via twist_summary in
        # evidence_summary) so the LLM has quantitative anomaly profiles to
        # reason over, not just text bullets.
        t_twist = time.time()
        twist_scores = compute_twist_scores(
            case.observations.get("traces", []),
            name_index=case.name_index,
        )
        twist_summary = twist_scores_to_summary(twist_scores, top_k=5)
        stats["twist_time_s"] = round(time.time() - t_twist, 4)
        # Save top TWIST entity and score for results.csv inspection
        twist_top = twist_top_entity(twist_scores, name_index=case.name_index)
        stats["twist_top_entity"] = twist_top[0] if twist_top else None
        stats["twist_top_score"] = twist_top[1] if twist_top else None
        if twist_scores:
            top3_svcs = sorted(twist_scores, key=lambda s: twist_scores[s]["twist_score"], reverse=True)[:3]
            stats["twist_c1_max"] = max(twist_scores[s]["c1_self_anomaly"] for s in top3_svcs)
            stats["twist_c2_max"] = max(twist_scores[s]["c2_trace_impact"] for s in top3_svcs)
            stats["twist_c3_max"] = max(twist_scores[s]["c3_blast_radius"] for s in top3_svcs)
            stats["twist_c4_max"] = max(twist_scores[s]["c4_delay_severity"] for s in top3_svcs)
        else:
            stats["twist_c1_max"] = stats["twist_c2_max"] = stats["twist_c3_max"] = stats["twist_c4_max"] = None

        # --- Proposal 2: TWIST-to-Text Evidence Synthesis ------------------
        # Turns each service's TWIST c1..c4 into one synthesized sentence so
        # the trace-derived anomaly signal becomes retrievable by the vector
        # index's semantic search (see config.USE_TWIST_TEXT_EVIDENCE and
        # twist_scoring.twist_scores_to_observations for the full rationale).
        twist_text_observations = []
        if config.USE_TWIST_TEXT_EVIDENCE:
            twist_text_observations = twist_scores_to_observations(
                twist_scores, name_index=case.name_index,
                top_n=config.TWIST_TEXT_TOP_N_SERVICES,
            )
        stats["twist_text_observations"] = len(twist_text_observations)

        # --- Module 3: Vector Retrieval ------------------------------------
        t3 = time.time()
        subgraph_node_ids = set(graph_result["subgraph"].nodes)

        metrics_in_subgraph = [o for o in case.observations.get("metrics", []) if o.entity_id in subgraph_node_ids]
        baseline_stats = compute_baseline_stats(metrics_in_subgraph, case.alert.alert_timestamp, case.topology)
        metric_trends = compute_metric_trends(metrics_in_subgraph, case.topology, baseline_stats=baseline_stats)

        truncated_modalities = []
        alert_ts = case.alert.alert_timestamp
        if alert_ts is not None and alert_ts.tzinfo is not None:
            alert_ts = alert_ts.replace(tzinfo=None)

        def _filter_modality(obs_list, modality_name):
            resolved = [o for o in obs_list if o.entity_id in subgraph_node_ids]
            unresolved = [o for o in obs_list if o.entity_id is None]
            # Relevance-based reduction (GALA-style), always applied -- not
            # gated behind DEV_QUICK_TEST. See reduce_unresolved_observations
            # docstring: previously this was unresolved[:300] and ONLY in dev
            # mode, so a real (--no-dev-quick-test) run saw the entire
            # unfiltered unresolved set, which is what caused ~3000s/case
            # runtimes once DEV_QUICK_TEST was turned off. DEV_QUICK_TEST now
            # only shrinks the budget for faster dev iteration; it no longer
            # changes WHETHER reduction happens.
            unresolved_budget = (config.DEV_MAX_UNRESOLVED_PER_MODALITY if config.DEV_QUICK_TEST
                                  else config.MAX_RESOLVED_PER_MODALITY)
            unresolved = reduce_unresolved_observations(unresolved, modality_name, unresolved_budget)
            if len(resolved) > config.MAX_RESOLVED_PER_MODALITY:
                truncated_modalities.append(modality_name)
                budget = config.MAX_RESOLVED_PER_MODALITY
                if alert_ts is not None:
                    def _naive_ts(o):
                        ts = o.timestamp
                        return ts.replace(tzinfo=None) if ts and ts.tzinfo else ts
                    before = sorted([o for o in resolved if _naive_ts(o) and _naive_ts(o) < alert_ts],
                                     key=_naive_ts, reverse=True)
                    after = sorted([o for o in resolved if _naive_ts(o) and _naive_ts(o) >= alert_ts],
                                    key=_naive_ts)
                    before_budget = int(budget * 0.6)
                    after_budget = budget - before_budget
                    kept_before = before[:before_budget]
                    kept_after = after[:after_budget]
                    leftover = budget - len(kept_before) - len(kept_after)
                    if leftover > 0:
                        kept_before += before[len(kept_before):len(kept_before) + leftover]
                        leftover = budget - len(kept_before) - len(kept_after)
                    if leftover > 0:
                        kept_after += after[len(kept_after):len(kept_after) + leftover]
                    resolved = kept_before + kept_after
                else:
                    resolved = sorted(resolved, key=lambda o: o.timestamp or 0,
                                       reverse=True)[:budget]
            return resolved + unresolved

        filtered_observations = {
            modality: _filter_modality(obs_list, modality)
            for modality, obs_list in case.observations.items()
        }
        # twist_synth is NOT entity-filtered via _filter_modality() -- it's
        # already small (<= TWIST_TEXT_TOP_N_SERVICES rows, one per service)
        # and was synthesized directly from the full-case trace scan, so the
        # same resolved/unresolved subgraph-membership filtering that
        # protects against raw observation volume doesn't apply here.
        if twist_text_observations:
            filtered_observations["twist_synth"] = twist_text_observations
        stats["truncated_modalities"] = "|".join(truncated_modalities)
        vector_modalities = ("logs", "metrics", "events")
        stats["use_log_templates"] = bool(config.USE_LOG_TEMPLATES)
        if config.USE_LOG_TEMPLATES:
            # Whole log modality as templates (hundreds of entries, no cap needed);
            # metrics leave the vector index (numeric operators serve them).
            t_tpl = time.time()
            log_tpl = compute_log_templates(case, case.alert.alert_timestamp)
            filtered_observations["log_template"] = log_tpl["observations"]
            vector_modalities = ("log_template", "events")
            stats["log_template_time_s"] = round(time.time() - t_tpl, 2)
            stats["log_rows_raw"] = log_tpl["stats"].get("log_rows", 0)
            stats["log_templates"] = log_tpl["stats"].get("log_templates", 0)
            stats["log_template_engine"] = os.environ.get("LOG_TEMPLATE_ENGINE", "regex")
            if log_tpl.get("error"):
                stats["log_template_error"] = log_tpl["error"]
        if config.USE_TWIST_TEXT_EVIDENCE and twist_text_observations:
            vector_modalities = vector_modalities + ("twist_synth",)
        if config.USE_VECTOR_RETRIEVAL:
            # Template path: no hand-picked size cap (index size = #distinct templates + events).
            vector_index = build_index_from_observations(
                filtered_observations, modalities=vector_modalities,
                max_observations=(sys.maxsize if config.USE_LOG_TEMPLATES else None))
        else:
            vector_index = build_index_from_observations({}, modalities=vector_modalities)   # empty index (ablation)
        stats["vector_index_build_time_s"] = time.time() - t3
        stats["indexed_observations"] = vector_index.index.ntotal
        # How much of retrieval/vector.py's text-dedup optimization paid off
        # for this case -- 1.0 means every text was unique (no savings),
        # lower means many repeated log/metric/span strings were collapsed
        # to a single encode() call. Helps explain why vector_index_build_time_s
        # varies a lot case-to-case even at the same observation count.
        stats["vector_texts_submitted"] = vector_index.texts_submitted
        stats["vector_texts_encoded_raw"] = vector_index.texts_encoded_raw
        stats["vector_embed_dedup_ratio"] = (
            round(vector_index.texts_encoded_raw / vector_index.texts_submitted, 4)
            if vector_index.texts_submitted else None
        )
        stats["observations_before_graph_filter"] = sum(len(v) for v in case.observations.values())
        # Count only the modalities that actually go into the vector index
        # (with log templates the raw logs/metrics are not indexed, so counting
        # them would report "capped" although the index is far below the cap).
        total_filtered = sum(len(filtered_observations.get(m, ())) for m in vector_modalities)
        stats["vector_index_was_capped"] = (False if config.USE_LOG_TEMPLATES
                                            else total_filtered > config.MAX_OBSERVATIONS_PER_INDEX)

        # --- Module 4: Hybrid Retrieval -----------------------------------
        t4 = time.time()
        hybrid = HybridRetriever(graph_result, vector_index)
        queries = [parsed_alert["alert_text"]] + parsed_alert["keywords"][:5]
        vector_based_items = (hybrid.retrieve_multi(queries, top_k=config.VECTOR_TOP_K)
                              if config.USE_VECTOR_RETRIEVAL else [])
        stats["use_vector_retrieval"] = bool(config.USE_VECTOR_RETRIEVAL)
        stats["embedding_model"] = config.EMBEDDING_MODEL

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

        modality_counts = Counter(it.observation.modality for it in evidence_items)
        for m in ("metrics", "logs", "traces", "events", "alerts", "twist_synth", "log_template"):
            stats[f"evidence_count_{m}"] = modality_counts.get(m, 0)
        stats["log_error_pattern_count"] = sum(
            1 for it in evidence_items
            if it.observation.modality == "logs" and ERROR_PATTERN.search(it.observation.text))

        trend_bullets = [b for bullets in metric_trends.values() for b in bullets]
        stats["trend_increase_count"] = sum(1 for b in trend_bullets if "increase" in b)
        stats["trend_decrease_count"] = sum(1 for b in trend_bullets if "decrease" in b)
        stats["trend_new_nonzero_count"] = sum(1 for b in trend_bullets if "new nonzero" in b)
        stats["trend_dropped_zero_count"] = sum(1 for b in trend_bullets if "dropped to zero" in b)

        # --- Module 5: Evidence Summarizer --------------------------------
        t5 = time.time()
        evidence_summary = summarize_evidence(evidence_items, alert_timestamp=case.alert.alert_timestamp,
                                               metric_trends=metric_trends)

        # Inject TWIST summary as an additional evidence block -- placed
        # AFTER the per-entity metric/log/trace bullets so the Coordinator
        # sees both raw evidence AND the quantitative anomaly profile.
        # Keyed as "_twist" (underscore prefix = not a real entity_id) so
        # the Coordinator prompt renders it as a separate section.
        if twist_summary:
            evidence_summary["_twist_scores"] = [twist_summary]
        if propagation_ev["text"]:
            evidence_summary["_propagation"] = [propagation_ev["text"]]

        stats["summarization_time_s"] = time.time() - t5

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

        # Zero-shot semantic matching -- the ONLY structured fault-type
        # signal reaching the Coordinator now. The two-stage RandomForest
        # classifier, its CBR fallback, and TWIST-guided narrowing were all
        # removed: each was trained or hand-tuned on RCA100's OWN
        # ground-truth labels (even the classifier's leave-one-out
        # cross-validation only avoids literal same-case leakage, not
        # overfitting to this specific 103-case benchmark's label
        # distribution), which the project treats as invalid for evaluating
        # genuine agentic reasoning. zero_shot_matching.py has no such
        # dependency -- it only compares evidence text against each fault
        # type's public taxonomy definition, so it works standalone with no
        # group hint (searches all 28 types directly) and needs no prior
        # pipeline run or trained model.
        zeroshot_topk = zero_shot_type_match_topk(scoped_evidence_text, candidate_group=None, k=3)
        if zeroshot_topk:
            stats["zeroshot_suggested_type"] = zeroshot_topk[0][0]
            stats["zeroshot_similarity"] = zeroshot_topk[0][1]
            stats["zeroshot_topk"] = "|".join(f"{t}:{s}" for t, s in zeroshot_topk)
        else:
            stats["zeroshot_suggested_type"] = None
            stats["zeroshot_similarity"] = None
            stats["zeroshot_topk"] = None

        # --- Module 6: Multi-Agent + Coordinator --------------------------
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
            "zeroshot_suggested_type": stats["zeroshot_suggested_type"],
            "zeroshot_similarity": stats["zeroshot_similarity"],
            "zeroshot_topk": zeroshot_topk,
            # TWIST top entity for optional Coordinator re-ranking hint
            "twist_top_entity": stats.get("twist_top_entity"),
            "twist_top_score": stats.get("twist_top_score"),
            "propagation_text": propagation_ev["text"] or None,
            "infra_text": infra_ev["text"] or None,
            "colocation_text": coloc_ev["text"] or None,
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
        layer = final_state.get("layer_result") or {}
        if config.LAYER_AGENT:
            stats["layer_level"] = layer.get("level")
            stats["layer_suspect"] = layer.get("suspect")
            stats["layer_confidence"] = layer.get("confidence")
            stats["layer_parse_error"] = bool(layer.get("_parse_error"))

        sc_stats = final_state.get("self_consistency_stats") or {}
        if sc_stats:
            stats["self_consistency_n_samples"] = sc_stats.get("n_samples")
            stats["self_consistency_n_valid"] = sc_stats.get("n_valid")
            stats["self_consistency_agreement"] = sc_stats.get("agreement")
            stats["self_consistency_votes"] = json.dumps(sc_stats.get("votes", {}))

        ts_stats = final_state.get("two_stage_stats") or {}
        if ts_stats:
            s1_type = ts_stats.get("stage1_fault_type")
            stats["stage1_fault_type"] = s1_type
            stats["stage1_entity_ids"] = "|".join(
                str(e) for e in (ts_stats.get("stage1_entity_ids") or []))
            stats["stage1_confidence"] = ts_stats.get("stage1_confidence")
            stats["stage1_parse_error"] = ts_stats.get("stage1_parse_error")
            # Faithfulness probes (no GT involved -- prediction vs prediction):
            # did exposure to the anchor change the evidence-only answer, and
            # does the evidence-only answer already equal the anchor's guess.
            stats["stage1_vs_final_flipped"] = (
                s1_type != final.get("predicted_fault_type"))
            stats["stage1_matches_anchor"] = (
                s1_type == graph_anchor.get("anchor_fault_type"))

        # Record which Coordinator-ablation config actually ran for THIS case,
        # read straight from config/env at call time. Added 2026-10-04 after a
        # mix-up where a results.csv was re-sent and assumed to be from the
        # COORDINATOR_TWO_STAGE_ANCHOR=1 run when it wasn't -- with these
        # columns saved to results.csv, which config produced a given file is
        # verifiable from the file itself instead of relying on remembering
        # which shell command was run.
        stats["coordinator_two_stage_anchor"] = bool(config.COORDINATOR_TWO_STAGE_ANCHOR)
        stats["coordinator_anchor_confirm_bias"] = bool(config.COORDINATOR_ANCHOR_CONFIRM_BIAS)
        stats["self_consistency_enabled"] = bool(config.USE_SELF_CONSISTENCY)

        anchor_entities = graph_anchor.get("anchor_entity_ids") or []
        coord_entities = normalize_entity_ids(
            final.get("predicted_entity_ids", []) or [], case.topology, case.name_index)
        stats["entity_source"] = config.ENTITY_SOURCE
        stats["entity_source_effective"] = (
            "coordinator" if (config.ENTITY_SOURCE == "coordinator"
                              or (config.CHAIN_FIRST and config.CHAIN_FIRST_ENTITY == "coordinator"))
            else "anchor")
        if (config.ENTITY_SOURCE == "coordinator"
                or (config.CHAIN_FIRST and config.CHAIN_FIRST_ENTITY == "coordinator")) and coord_entities:
            predicted_entity_ids = coord_entities
            used_fallback = False
        elif anchor_entities:
            predicted_entity_ids = anchor_entities
            used_fallback = False
        else:
            llm_predicted_entities = final.get("predicted_entity_ids", []) or []
            predicted_entity_ids = normalize_entity_ids(llm_predicted_entities, case.topology, case.name_index) or \
                ([parsed_alert["entry_entity_id"]] if parsed_alert["entry_entity_id"] else [])
            used_fallback = not llm_predicted_entities
        predicted_fault_type = final.get("predicted_fault_type", "unknown")
        stats["fault_type_equals_anchor"] = bool(
            predicted_fault_type and predicted_fault_type == graph_anchor.get("anchor_fault_type"))
        stats["used_entity_fallback"] = used_fallback
        stats["coordinator_entity_empty"] = not bool(coord_entities)

        # EVALUATION-ONLY diagnostics (no effect on predictions or on the
        # official score): record the entity each alternative source WOULD
        # have predicted, so scoring.full_case_report can report a
        # counterfactual entity_localization per source. No GT is touched here.
        _coord_ents = normalize_entity_ids(
            final.get("predicted_entity_ids", []) or [], case.topology, case.name_index)
        stats["alt_entity_coordinator"] = "|".join(_coord_ents)
        stats["alt_entity_anchor"] = "|".join(anchor_entities)
        stats["alt_entity_twist_top"] = stats.get("twist_top_entity") or ""
        stats["alt_entity_entry"] = parsed_alert.get("entry_entity_id") or ""

        return RCAResult(
            case_id=case_id,
            predicted_entity_ids=predicted_entity_ids,
            predicted_fault_type=predicted_fault_type,
            reasoning_chain=final.get("reasoning_chain", []),
            confidence=float(final.get("confidence", 0.0) or 0.0),
            agent_findings=agent_findings,
            retrieval_stats=stats,
            # infra tables were shown to the Coordinator: include so the checkpoint
            # proxy sees the same text the LLM saw
            evidence_items=list(evidence_items) + [EvidenceItem(observation=o) for o in infra_ev["observations"]],
        )