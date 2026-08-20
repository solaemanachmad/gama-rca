"""
agents.py
==========
Module 6 — Multi-Agent Collaboration, built on LangGraph.

Four specialist agents run in parallel over the SAME evidence summary
(scoped to their modality of interest), each producing an AgentFinding.
A Coordinator agent then fuses findings into the final RCAResult via the LLM.

Install: pip install langgraph
"""

from typing import Dict, List, Optional, TypedDict
import json

from langgraph.graph import StateGraph, END

from data.taxonomy import taxonomy_prompt_block, taxonomy_prompt_block_for_group

import config
from schema import AgentFinding, RCAResult
from agents.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    case_id: str
    alert_text: str
    evidence_summary: Dict[str, List[str]]     # {entity_id: [bullets]}
    graph_neighbors: Dict[str, List[str]]       # {"upstream": [...], "downstream": [...]}
    candidate_entities: List[str]               # ranked by graph score
    graph_anchor: Optional[dict]                 # cheap structural prior (Stage 0.5,
                                                   # see pipeline.py's _compute_graph_anchor)
                                                   # {"anchor_entity_ids", "anchor_fault_type",
                                                   #  "anchor_confidence"}
    propagation_path: Optional[List[str]]          # graph-derived hop-by-hop path from the
                                                   # alert entry entity to graph_anchor's top
                                                   # candidate (see pipeline.py's
                                                   # compute_propagation_path) -- factually
                                                   # grounded, not LLM-invented
    keyword_fault_candidates: Optional[List[str]]   # fault-type slugs detected via literal
                                                   # keyword match against raw evidence text
                                                   # (see data.taxonomy.detect_fault_keywords)
                                                   # -- a textual-evidence-grounded companion
                                                   # to graph_anchor's purely structural prior
    classifier_predicted_group: Optional[str]       # fault_group predicted by the trained
                                                   # structured-feature classifier (see
                                                   # pipeline/fault_group_classifier.py),
                                                   # None if no classifier has been trained yet
    classifier_confidence: Optional[float]          # the classifier's own confidence (max
                                                   # predict_proba) for classifier_predicted_group
    classifier_predicted_type: Optional[str]        # fine-grained (28-type) prediction from the
                                                   # tier-2 classifier, masked to only the types
                                                   # within classifier_predicted_group (see
                                                   # pipeline/fault_type_classifier.py)
    classifier_type_confidence: Optional[float]     # renormalized confidence among just the
                                                   # in-group candidates (not the diluted raw
                                                   # 28-way predict_proba)
    cbr_suggested_type: Optional[str]               # nearest-neighbor case-based-reasoning
                                                   # suggestion, only present for singleton
                                                   # (n=1) fault types the tier-2 classifier
                                                   # cannot learn (see
                                                   # pipeline/case_based_reasoning.py)
    cbr_similarity: Optional[float]                 # cosine similarity to the matched case
    zeroshot_suggested_type: Optional[str]          # zero-shot match against fault-type
                                                   # DEFINITIONS (not examples) -- see
                                                   # pipeline/zero_shot_matching.py, works
                                                   # even for types with zero examples
    zeroshot_similarity: Optional[float]
    metrics_finding: Optional[dict]
    logs_finding: Optional[dict]
    trace_finding: Optional[dict]
    topology_finding: Optional[dict]
    final_result: Optional[dict]


AGENT_SYSTEM_PROMPT = (
    "You are a specialist Site Reliability Engineering agent performing root "
    "cause analysis on microservice observability evidence. Only reason from "
    "the evidence given to you. Respond ONLY with valid JSON matching the "
    "requested schema — no prose outside the JSON object."
)

FINDING_SCHEMA_HINT = (
    '{"entity_id": "<most-suspect entity id or null>", '
    '"summary": "<2-3 sentence finding>", '
    '"supporting_evidence": ["bullet1", "bullet2"], '
    '"confidence": <float 0-1>}'
)


def _make_agent_node(agent_name: str, modality_filter: Optional[str],
                      llm: LLMClient):
    """Factory producing a LangGraph node function for one specialist agent."""

    def node(state: AgentState) -> AgentState:
        summary_text = _filter_summary_text(state["evidence_summary"], modality_filter)
        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Evidence relevant to {agent_name} (entity: bullet list):\n{summary_text}\n\n"
            f"Task: identify which entity is most likely implicated and why, "
            f"from a {agent_name.replace('_', ' ')} perspective only.\n"
            f"Respond as JSON: {FINDING_SCHEMA_HINT}"
        )
        result = llm.generate_json(prompt, system=AGENT_SYSTEM_PROMPT)
        state[f"{agent_name}_finding"] = result
        return state

    return node


def _filter_summary_dict(evidence_summary: Dict[str, List[str]],
                          modality_filter: Optional[str]) -> Dict[str, List[str]]:
    """Metrics/Logs/Trace agents only see bullets relevant to their modality
    (cheap heuristic keyword filter on the bullet text); Topology agent sees
    everything since its job is cross-entity structure, not signal content."""
    if modality_filter is None:
        return evidence_summary
    keep = {}
    for eid, bullets in evidence_summary.items():
        filtered = [b for b in bullets if modality_filter.lower() in b.lower()]
        if filtered:
            keep[eid] = filtered
    if not keep:  # fall back to full summary if the filter emptied everything
        keep = evidence_summary
    return keep


def _filter_summary_text(evidence_summary: Dict[str, List[str]],
                          modality_filter: Optional[str]) -> str:
    keep = _filter_summary_dict(evidence_summary, modality_filter)
    lines = []
    for eid, bullets in keep.items():
        lines.append(eid)
        lines.extend(f"  • {b}" for b in bullets)
    return "\n".join(lines)


def _rule_based_agent_node(agent_name: str, modality_filter: Optional[str]):
    """Non-LLM alternative to _make_agent_node(): picks the entity with the
    most bullets for this modality as "most implicated" (a simple proxy for
    evidence density -- more anomalous/notable observations for an entity
    means more bullets survived the Evidence Summarizer's filtering), and
    builds the finding directly from those bullets rather than asking an
    LLM to restate them. Exploratory ablation -- see _compute_graph_anchor's
    use_llm=False docstring for the full motivation. Same output schema
    (FINDING_SCHEMA_HINT) as the LLM version, so downstream code (the
    Coordinator) needs no changes to consume either."""
    def node(state: AgentState) -> AgentState:
        filtered = _filter_summary_dict(state["evidence_summary"], modality_filter)
        if not filtered:
            state[f"{agent_name}_finding"] = {
                "entity_id": None, "summary": f"No {agent_name} evidence found.",
                "supporting_evidence": [], "confidence": 0.0,
            }
            return state
        best_entity = max(filtered, key=lambda e: len(filtered[e]))
        bullets = filtered[best_entity][:3]
        summary = (f"{agent_name.replace('_', ' ').title()} evidence points to {best_entity}: "
                   + "; ".join(bullets))
        confidence = round(min(len(filtered[best_entity]) / 5.0, 1.0), 4)
        state[f"{agent_name}_finding"] = {
            "entity_id": best_entity, "summary": summary,
            "supporting_evidence": bullets, "confidence": confidence,
        }
        return state
    return node


def _rule_based_topology_node():
    """Non-LLM alternative to topology_agent_node(): the propagation_path
    (already a real graph-computed path, not a guess) directly names the
    likely origin entity -- its FIRST hop. No LLM reasoning needed to
    restate what the graph already computed."""
    def node(state: AgentState) -> AgentState:
        path = state.get("propagation_path")
        if not path:
            state["topology_finding"] = {
                "entity_id": None, "summary": "No propagation path available.",
                "supporting_evidence": [], "confidence": 0.0,
            }
            return state
        origin = path[0]
        summary = f"Graph-computed propagation path identifies {origin} as the likely origin: " + " -> ".join(path)
        state["topology_finding"] = {
            "entity_id": origin, "summary": summary,
            "supporting_evidence": [summary], "confidence": 0.7,
        }
        return state
    return node


def topology_agent_node(llm: LLMClient):
    def node(state: AgentState) -> AgentState:
        neighbors = state.get("graph_neighbors", {})
        path = state.get("propagation_path")
        path_block = ""
        if path:
            path_block = (
                f"\nComputed propagation path in the topology graph, from the alerted "
                f"entity to the top structurally-ranked candidate root cause "
                f"(this is a REAL path from the graph, not a guess):\n"
                f"  {' -> '.join(path)}\n"
                f"Use this path as your primary reasoning basis -- confirm it fits the "
                f"evidence, or explain specifically why it doesn't.\n"
            )
        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Upstream (callers) of the alerted entity: {neighbors.get('upstream', [])}\n"
            f"Downstream (dependencies): {neighbors.get('downstream', [])}\n"
            f"Ranked candidate entities by graph propagation score: "
            f"{state.get('candidate_entities', [])}\n"
            f"{path_block}\n"
            f"Task: reason about the most plausible fault-propagation path "
            f"(which entity is the likely origin vs. which are downstream victims).\n"
            f"Respond as JSON: {FINDING_SCHEMA_HINT}"
        )
        result = llm.generate_json(prompt, system=AGENT_SYSTEM_PROMPT)
        state["topology_finding"] = result
        return state
    return node


COORDINATOR_SYSTEM_PROMPT = (
    "You are the Coordinator agent for a microservice root-cause-analysis "
    "system. You receive independent findings from Metrics, Logs, Trace, and "
    "Topology specialist agents and must produce ONE final diagnosis. Weigh "
    "agreement across agents heavily. Respond ONLY with valid JSON."
)

FINAL_SCHEMA_HINT = (
    '{"predicted_entity_ids": ["entity1", "entity2"], '
    '"predicted_fault_type": "<one of the 28 RCA100 fault types or best guess>", '
    '"reasoning_chain": ["cause step", "propagation step", "impact step"], '
    '"confidence": <float 0-1>}'
)


def coordinator_node(llm: LLMClient):
    def node(state: AgentState) -> AgentState:
        findings_block = json.dumps({
            "metrics_agent": state.get("metrics_finding"),
            "logs_agent": state.get("logs_finding"),
            "trace_agent": state.get("trace_finding"),
            "topology_agent": state.get("topology_finding"),
        }, indent=2)

        # Two-stage classification: if the trained structured-feature
        # classifier made a confident prediction (see
        # pipeline/fault_group_classifier.py), narrow the taxonomy shown
        # here to just that group's 3-7 types instead of all 28. This is
        # SOFT narrowing, not a hard restriction: the LLM can still name a
        # type outside the list if evidence clearly contradicts the
        # classifier, with justification in reasoning_chain. This is
        # deliberately different from the earlier graph_anchor
        # hard-override design (forcing its own fault_type guess with no
        # escape hatch), which collapsed to one generic answer every time
        # because the anchor has no evidence text to ground a fault-type
        # decision in -- the classifier's LOOCV-validated 0.66-0.68
        # accuracy earns more trust, but not unconditional override, since
        # roughly 1/3 of its predictions are still wrong.
        #
        # Falls back to the full list (same as before) when no classifier
        # has been trained yet, or its confidence is low (<0.3 -- barely
        # above the 1/6 random-guess floor for 6 classes, not worth
        # narrowing on).
        classifier_group = state.get("classifier_predicted_group")
        classifier_conf = state.get("classifier_confidence") or 0.0
        if classifier_group and classifier_conf >= 0.3:
            # Escape-hatch strength now scales with confidence. Found via a
            # 10-case spot check: at HIGH confidence (e.g. 84.7% for
            # 'Cloud resource' in one case), the Coordinator still deviated
            # to its default Application-logic/trafficSurge bias under the
            # earlier flat "prefer, but you MAY deviate" wording -- the
            # instruction wasn't forceful enough to overcome the model's own
            # prior at high confidence. Threshold lowered from 0.6 to 0.45
            # after a follow-up 10-case check: cases with confidence
            # 0.40-0.56 (t001/t002/t006/t009) still fell back to the LLM's
            # default favorite-type-per-group bias (trafficSurge,
            # cacheBreakdown, fullGC) under the softer wording, while a
            # 0.58-confidence case succeeded anyway -- moderate-confidence
            # predictions appear to deserve the same firm treatment as
            # high-confidence ones, not just >=0.6. Below 0.45, keep the
            # framing (classifier is only modestly more likely to be right
            # than not, so genuine deviation should stay easy).
            if classifier_conf >= 0.45:
                deviation_clause = (
                    f"This is a HIGH-confidence prediction ({classifier_conf:.0%}) from a "
                    f"classifier validated at 0.66-0.68 accuracy via cross-validation -- "
                    f"noticeably more reliable than your own unaided guess tends to be on "
                    f"this benchmark. Only pick a fault type outside this list if the "
                    f"specialist findings contain a SPECIFIC, named piece of evidence that "
                    f"directly contradicts it (cite it explicitly in reasoning_chain). "
                    f"'the evidence looks like a generic traffic/error pattern' is NOT "
                    f"sufficient justification to override this prediction."
                )
            else:
                deviation_clause = (
                    f"Prefer a type from this list, but you MAY pick any of the 28 RCA100 "
                    f"fault types instead if the specialist findings clearly point "
                    f"elsewhere; explain why in your reasoning_chain if so."
                )
            fault_hint = (
                taxonomy_prompt_block_for_group(classifier_group)
                + f"\n(Structured-feature classifier prediction: '{classifier_group}' "
                  f"at {classifier_conf:.0%} confidence. {deviation_clause})"
            )
        else:
            fault_hint = taxonomy_prompt_block()

        # Tier-2: fine-grained TYPE within the already-narrowed group.
        # Targets a specific failure pattern found empirically across
        # several spot checks: even once the group is correctly narrowed,
        # the Coordinator tends to collapse to one "favorite" type within
        # it regardless of case-specific evidence (e.g. F009-cacheBreakdown
        # guessed for BOTH t002 and t010 when the real answer was
        # F029-redisUnavailable in both; F006-trafficSurge as the default
        # for nearly every Application-logic case). Same confidence-scaled
        # escape-hatch pattern as the group-level hint above -- soft
        # narrowing, not a hard override.
        classifier_type = state.get("classifier_predicted_type")
        classifier_type_conf = state.get("classifier_type_confidence") or 0.0
        if classifier_type and classifier_type_conf >= 0.2:
            # Threshold lowered from 0.4 to 0.2, and confidence-scaled
            # language added -- same pattern that fixed tier-1's escape-
            # hatch (t008 case). Real-run evidence: several cases had
            # tier-2 confidence 0.24-0.29 (just below the old 0.4 cutoff),
            # giving the Coordinator NO hint at all and falling back to its
            # default bias; the one case with high confidence (0.91)
            # correctly followed the hint (F026-nodeCpuHigh, exact match).
            # Tier-2 confidence is inherently lower on average than tier-1
            # given sparser per-type training data, so both the activation
            # threshold and the "firm" tier are set lower than tier-1's.
            if classifier_type_conf >= 0.45:
                deviation_clause = (
                    "This is a HIGH-confidence prediction, comparable to cases where "
                    "this classifier has been exactly correct. Only pick a different type "
                    "within the group if a specialist finding cites SPECIFIC evidence "
                    "contradicting it (name it in reasoning_chain)."
                )
            else:
                deviation_clause = (
                    "Prefer this type, but you may pick a different one within the same "
                    "group if specialist findings clearly point elsewhere."
                )
            fault_hint += (
                f"\n\nTIER-2 PREDICTION: among the group above, the classifier's specific "
                f"guess is '{classifier_type}' ({classifier_type_conf:.0%} confidence among "
                f"the in-group candidates). {deviation_clause}"
            )

        # Case-Based Reasoning hint (see pipeline/case_based_reasoning.py):
        # only present for fault types with just 1 historical example --
        # the tier-2 classifier above has no statistical basis to learn
        # these, so this is a DIFFERENT mechanism (nearest-neighbor
        # similarity on structured features, not a learned classifier).
        cbr_type = state.get("cbr_suggested_type")
        cbr_sim = state.get("cbr_similarity") or 0.0
        if cbr_type:
            fault_hint += (
                f"\n\nSIMILAR HISTORICAL CASE: this case's structured features closely "
                f"resemble ({cbr_sim:.0%} similarity) exactly one prior labeled case, of type "
                f"'{cbr_type}' -- a fault type too rare (1 example) for the classifier above to "
                f"learn, so this is a nearest-neighbor match rather than a trained prediction. "
                f"Consider it alongside the specialist findings, especially if no other strong "
                f"signal points elsewhere."
            )

        # Zero-shot semantic hint (see pipeline/zero_shot_matching.py):
        # compares evidence TEXT against fault-type DEFINITIONS, not
        # examples -- works even for types with zero training cases.
        zeroshot_type = state.get("zeroshot_suggested_type")
        zeroshot_sim = state.get("zeroshot_similarity") or 0.0
        if zeroshot_type and zeroshot_type not in (cbr_type, classifier_type):
            fault_hint += (
                f"\n\nDEFINITION-BASED MATCH: the evidence text semantically resembles "
                f"({zeroshot_sim:.0%} similarity) the OFFICIAL DEFINITION of fault type "
                f"'{zeroshot_type}' -- this comes from comparing evidence wording to each "
                f"candidate type's own description, not from any labeled example, so it can "
                f"surface types no other signal above covers. Weigh this alongside, not above, "
                f"the specialist findings and other hints."
            )

        # Structural prior from Stage 0.5 (see pipeline.py's
        # _compute_graph_anchor). A 4-case spot check showed a topology-only
        # single-call guess (graphrag_only-style) scoring ~2x higher on both
        # entity_localization and fault_identification than the full
        # multi-agent Coordinator -- likely because a small local LLM
        # reasons more reliably over a short, structured signal than a large
        # multi-agent-findings-plus-evidence prompt. Handing that prior to
        # the Coordinator as something to confirm-or-override (rather than
        # re-deriving from scratch) is meant to recover that accuracy while
        # keeping the Coordinator's richer reasoning_chain. Absent for
        # multi_agent_only, which doesn't run graph retrieval at all.
        anchor = state.get("graph_anchor") or {}
        if anchor.get("anchor_entity_ids") or anchor.get("anchor_fault_type"):
            anchor_block = (
                f"\nSTRUCTURAL PRIOR (fast topology-only first pass, before detailed "
                f"evidence was considered):\n"
                f"  candidate entity: {anchor.get('anchor_entity_ids')}\n"
                f"  candidate fault type: {anchor.get('anchor_fault_type')}\n"
                f"This prior is often right (topology structure alone is a strong signal for "
                f"this benchmark) -- CONFIRM it unless the specialist findings below clearly "
                f"contradict it. If you override it, say why in your reasoning_chain.\n"
            )
        else:
            anchor_block = ""

        path = state.get("propagation_path")
        path_block = ""
        if path:
            path_block = (
                f"\nGRAPH-COMPUTED PROPAGATION PATH (real path in the topology, from the "
                f"alerted/impacted entity to the structural candidate cause -- use this as "
                f"the basis for your 'propagation' reasoning_chain step instead of "
                f"inventing one):\n  {' -> '.join(path)}\n"
            )

        keyword_candidates = state.get("keyword_fault_candidates")
        keyword_block = ""
        if keyword_candidates:
            keyword_block = (
                f"\nKEYWORD-DETECTED FAULT-TYPE CANDIDATES (literal terms found in the raw "
                f"evidence text, ordered by match strength -- these are the fault types with "
                f"actual textual support in this case's evidence, not a guess):\n"
                f"  {', '.join(keyword_candidates)}\n"
                f"Prefer one of these if it's consistent with the specialist findings above; "
                f"only pick outside this list if the evidence clearly points elsewhere.\n"
            )

        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Specialist agent findings:\n{findings_block}\n"
            f"{anchor_block}"
            f"{path_block}"
            f"{keyword_block}\n"
            f"{fault_hint}\n\n"
            f"Task: synthesize a final root-cause diagnosis with an explicit "
            f"cause -> propagation -> impact reasoning chain.\n"
            f"Respond as JSON: {FINAL_SCHEMA_HINT}"
        )
        result = llm.generate_json(prompt, system=COORDINATOR_SYSTEM_PROMPT)
        state["final_result"] = result
        return state
    return node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_agent_graph(llm: Optional[LLMClient] = None, coordinator_llm: Optional[LLMClient] = None,
                       use_llm_agents: bool = True) -> StateGraph:
    llm = llm or LLMClient()
    coordinator_llm = coordinator_llm or llm  # falls back to the same client for everything

    graph = StateGraph(AgentState)
    if use_llm_agents:
        graph.add_node("metrics_agent", _make_agent_node("metrics", "metric", llm))
        graph.add_node("logs_agent", _make_agent_node("logs", "log", llm))
        graph.add_node("trace_agent", _make_agent_node("trace", "span", llm))
        graph.add_node("topology_agent", topology_agent_node(llm))
    else:
        # Rule-based specialist agents -- see _rule_based_agent_node's
        # docstring for the motivation. The Coordinator ALWAYS stays
        # LLM-based (coordinator_llm above) -- this only removes LLM calls
        # from the 4 specialists, which mostly restate structured evidence
        # bullets rather than performing genuine open-ended reasoning.
        graph.add_node("metrics_agent", _rule_based_agent_node("metrics", "metric"))
        graph.add_node("logs_agent", _rule_based_agent_node("logs", "log"))
        graph.add_node("trace_agent", _rule_based_agent_node("trace", "span"))
        graph.add_node("topology_agent", _rule_based_topology_node())
    graph.add_node("coordinator", coordinator_node(coordinator_llm))

    graph.set_entry_point("metrics_agent")
    # Fan out: entry triggers all four specialists (LangGraph runs nodes with
    # satisfied dependencies in the same superstep when reachable from START
    # via parallel edges). Simpler/robust alternative used here: chain then
    # join, since all four only depend on the shared input state, not on
    # each other's output — order does not affect correctness.
    graph.add_edge("metrics_agent", "logs_agent")
    graph.add_edge("logs_agent", "trace_agent")
    graph.add_edge("trace_agent", "topology_agent")
    graph.add_edge("topology_agent", "coordinator")
    graph.add_edge("coordinator", END)

    return graph.compile()


def build_agent_findings_list(state: AgentState) -> List[AgentFinding]:
    findings = []
    for name in ("metrics_finding", "logs_finding", "trace_finding", "topology_finding"):
        f = state.get(name) or {}
        if f.get("_parse_error"):
            continue
        findings.append(AgentFinding(
            agent_name=name.replace("_finding", ""),
            entity_id=f.get("entity_id"),
            summary=f.get("summary", ""),
            supporting_evidence=f.get("supporting_evidence", []),
            confidence=float(f.get("confidence", 0.0) or 0.0),
        ))
    return findings