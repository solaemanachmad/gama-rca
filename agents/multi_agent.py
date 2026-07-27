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

from data.taxonomy import taxonomy_prompt_block

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


def _filter_summary_text(evidence_summary: Dict[str, List[str]],
                          modality_filter: Optional[str]) -> str:
    """Metrics/Logs/Trace agents only see bullets relevant to their modality
    (cheap heuristic keyword filter on the bullet text); Topology agent sees
    everything since its job is cross-entity structure, not signal content."""
    if modality_filter is None:
        keep = evidence_summary
    else:
        keep = {}
        for eid, bullets in evidence_summary.items():
            filtered = [b for b in bullets if modality_filter.lower() in b.lower()]
            if filtered:
                keep[eid] = filtered
        if not keep:  # fall back to full summary if the filter emptied everything
            keep = evidence_summary

    lines = []
    for eid, bullets in keep.items():
        lines.append(eid)
        lines.extend(f"  • {b}" for b in bullets)
    return "\n".join(lines)


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

        # Full taxonomy list, not a narrowed embedding-similarity shortlist --
        # 15-case evaluation showed the shortlist mechanism (raw evidence
        # text vs abstract fault definitions via generic MiniLM similarity)
        # correlated with fault_identification scoring ~5x worse than
        # direct_llm/graphrag_only, which both use this same full list.
        # The likely mechanism: the correct fault type gets excluded from
        # the top-5 shortlist before the LLM ever gets to reason about it.
        fault_hint = taxonomy_prompt_block()

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

        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Specialist agent findings:\n{findings_block}\n"
            f"{anchor_block}"
            f"{path_block}\n"
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
def build_agent_graph(llm: Optional[LLMClient] = None) -> StateGraph:
    llm = llm or LLMClient()

    graph = StateGraph(AgentState)
    graph.add_node("metrics_agent", _make_agent_node("metrics", "metric", llm))
    graph.add_node("logs_agent", _make_agent_node("logs", "log", llm))
    graph.add_node("trace_agent", _make_agent_node("trace", "span", llm))
    graph.add_node("topology_agent", topology_agent_node(llm))
    graph.add_node("coordinator", coordinator_node(llm))

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