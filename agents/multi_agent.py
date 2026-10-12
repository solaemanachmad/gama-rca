"""
agents.py
==========
Module 6 — Multi-Agent Collaboration, built on LangGraph.

Four specialist agents run in parallel over the SAME evidence summary
(scoped to their modality of interest), each producing an AgentFinding.
A Coordinator agent then fuses findings into the final RCAResult via the LLM.

Install: pip install langgraph
"""

from typing import Dict, List, Optional, Tuple, TypedDict
import json

from langgraph.graph import StateGraph, END

from data.taxonomy import taxonomy_prompt_block, knowledge_prompt_block, knowledge_prompt_block_pos, FAULT_DEFINITIONS

import re
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
    topology_view: Optional[dict]               # TOPOLOGY_NAMES: names + alphabetical neighbourhood (pipeline/topology_view.py)
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
    zeroshot_suggested_type: Optional[str]          # zero-shot match against fault-type
                                                   # DEFINITIONS (not examples) -- see
                                                   # pipeline/zero_shot_matching.py, works
                                                   # even for types with zero examples and no
                                                   # trained model. Since the GT-trained
                                                   # classifier, its CBR fallback, and the
                                                   # GT-tuned TWIST group narrowing were all
                                                   # removed (each depended on RCA100's own
                                                   # ground-truth labels), this is now the
                                                   # ONLY structured fault-type hint reaching
                                                   # the Coordinator.
    zeroshot_similarity: Optional[float]
    zeroshot_topk: Optional[List[tuple]]            # up to top-3 (fault_type, similarity)
                                                   # pairs from zero_shot_type_match_topk(),
                                                   # a ranked shortlist rather than a single
                                                   # forced guess
    twist_top_entity: Optional[str]                 # highest-TWIST-score entity (pure trace
                                                   # statistics, no GT dependency -- see
                                                   # pipeline/twist_scoring.py)
    twist_top_score: Optional[float]
    infra_text: Optional[str]                       # numeric APM/k8s tables (pipeline/infra_evidence.py)
    colocation_text: Optional[str]                  # per-node hosted-service table (pipeline/colocation_evidence.py)
    coordinator_prompt_chars: Optional[int]         # diagnostic: chars of the chain-first Coordinator prompt
    onset_text: Optional[str]                       # per-minute robust-z onset table (pipeline/onset_evidence.py)
    trace_text: Optional[str]                       # HTTP code / error message table (pipeline/trace_evidence.py)
    log_evidence_text: Optional[str]                # per-burst log composition (pipeline/log_evidence.py)
    trace_edges_text: Optional[str]                 # per call-edge trace evidence per burst (pipeline/trace_edges.py)
    interval_strip_text: Optional[str]              # arm X: time-aligned strip (pipeline/interval_strip.py)
    metric_structure_text: Optional[str]           # identity-free K8s aggregates + call-edge flow ratios (pipeline/metric_structure.py)
    alert_lifecycle_text: Optional[str]             # condensed alert stream (pipeline/alert_lifecycle.py)
    layer_result: Optional[dict]                    # LAYER_AGENT output (level/suspect/evidence/runner_up)
    type_check_stats: Optional[dict]                # VERIFY_FAULT_TYPE diagnostics (checked, satisfied, quote grounded, revised)
    verify_stats: Optional[dict]                    # VERIFY_CHAIN diagnostics (issues, revised, ungrounded numbers)
    debug_trace: Optional[list]                     # prompts + raw outputs per node (only when DEBUG_DUMP_DIR is set)
    propagation_text: Optional[str]                 # call-graph/trace table (evidence only; see
                                                   # pipeline/propagation_evidence.py)
    metrics_finding: Optional[dict]
    logs_finding: Optional[dict]
    trace_finding: Optional[dict]
    topology_finding: Optional[dict]
    final_result: Optional[dict]
    two_stage_stats: Optional[dict]                 # Stage-1 (anchor-free) answer from
                                                   # _two_stage_coordinate, kept so the
                                                   # Stage1-vs-final flip rate (an
                                                   # explanation-faithfulness probe) can
                                                   # be computed post hoc
    self_consistency_stats: Optional[dict]          # diagnostics from coordinator_node's
                                                   # self-consistency ensemble (see
                                                   # _self_consistency_sample below) --
                                                   # {n_samples, n_valid, votes, agreement}


AGENT_SYSTEM_PROMPT = (
    "You are a specialist Site Reliability Engineering agent performing root "
    "cause analysis on microservice observability evidence. Only reason from "
    "the evidence given to you. Respond ONLY with valid JSON matching the "
    "requested schema — no prose outside the JSON object."
)

_ENGLISH_NOTE = (" Write the summary and supporting_evidence in English (keep entity names, metric names and "
                 "log text exactly as given).")


def _agent_system() -> str:
    return AGENT_SYSTEM_PROMPT + (_ENGLISH_NOTE if config.SPECIALIST_ENGLISH else "")


FINDING_SCHEMA_HINT = (
    '{"entity_id": "<most-suspect entity id or null>", '
    '"summary": "<2-3 sentence finding>", '
    '"supporting_evidence": ["bullet1", "bullet2"], '
    '"confidence": <float 0-1>}'
)


def _trace(state, name, prompt, output):
    """Debug only (config.DEBUG_DUMP_DIR): keep the exact prompt and raw output of one LLM call.
    Contains no ground truth -- only what the model saw and said."""
    if getattr(config, "DEBUG_DUMP_DIR", ""):
        state.setdefault("debug_trace", []).append({"node": name, "prompt": prompt, "output": output})


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
        result = llm.generate_json(prompt, system=_agent_system())
        _trace(state, f"{agent_name}_agent", prompt, result)
        state[f"{agent_name}_finding"] = result
        return state

    return node


_KIND_PREFIX = {
    "metric": ("Metric observed:",),
    "log": ("Log observed:", "Log pattern:", "Error/exception patterns in"),
}
_SPAN_COUNT = re.compile(r"^\d+ span\(s\) with error/timeout status")


def _bullet_for(entity_key: str, bullet: str, modality: str) -> bool:
    """STRICT_MODALITY_ROUTING: explicit kind of a bullet instead of a substring match (see config)."""
    if entity_key == "_propagation":          # call-graph table computed from spans
        return modality == "span"
    if entity_key.startswith("_"):            # _twist_scores: cross-modality, not for a single-modality specialist
        return False
    if modality == "span":
        return bool(_SPAN_COUNT.match(bullet))
    return bullet.startswith(_KIND_PREFIX.get(modality, ()))


def _filter_summary_dict(evidence_summary: Dict[str, List[str]],
                          modality_filter: Optional[str]) -> Dict[str, List[str]]:
    """Metrics/Logs/Trace agents only see bullets relevant to their modality
    (cheap heuristic keyword filter on the bullet text); Topology agent sees
    everything since its job is cross-entity structure, not signal content."""
    if modality_filter is None:
        return evidence_summary
    keep = {}
    for eid, bullets in evidence_summary.items():
        if config.STRICT_MODALITY_ROUTING:
            filtered = [b for b in bullets if _bullet_for(eid, b, modality_filter)]
        else:
            filtered = [b for b in bullets if modality_filter.lower() in b.lower()]
        if filtered:
            keep[eid] = filtered
    if not keep and not config.STRICT_MODALITY_ROUTING:  # legacy: fall back to the full summary if the filter emptied everything
        keep = evidence_summary                           # (strict routing: an empty modality stays empty, no leakage)
    return keep


def _filter_summary_text(evidence_summary: Dict[str, List[str]],
                          modality_filter: Optional[str]) -> str:
    keep = _filter_summary_dict(evidence_summary, modality_filter)
    lines = []
    for eid, bullets in keep.items():
        lines.append(eid)
        lines.extend(f"  • {b}" for b in bullets)
    if not lines:
        return "(no evidence of this modality was retrieved for this case)"
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
        tv = state.get("topology_view") if config.TOPOLOGY_NAMES else None
        if tv:
            path_txt = (f"\nCall-graph path from the alerted entity (a structural fact about nearness, not evidence about the cause):\n"
                        f"  {' -> '.join(tv['path'])}\n") if tv.get("path") else ""
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"Alerted service: {tv.get('alert_service')}\n"
                f"Upstream (callers) of the alerted entity: {tv['upstream']}\n"
                f"Downstream (dependencies): {tv['downstream']}\n"
                f"Services in the call-graph neighbourhood of the alert (alphabetical; the order carries no information): "
                f"{tv['services']}\n"
                f"{path_txt}\n"
                f"Task: reason about the most plausible fault-propagation path "
                f"(which entity is the likely origin vs. which are downstream victims).\n"
                f"Respond as JSON: {FINDING_SCHEMA_HINT}"
            )
            result = llm.generate_json(prompt, system=_agent_system())
            _trace(state, "topology_agent", prompt, result)
            state["topology_finding"] = result
            return state
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
        result = llm.generate_json(prompt, system=_agent_system())
        _trace(state, "topology_agent", prompt, result)
        state["topology_finding"] = result
        return state
    return node


COORDINATOR_SYSTEM_PROMPT = (
    "You are the Coordinator agent for a microservice root-cause-analysis "
    "system. You receive independent findings from Metrics, Logs, Trace, and "
    "Topology specialist agents and must produce ONE final diagnosis. Weigh "
    "agreement across agents heavily. Respond ONLY with valid JSON."
)

CHAIN_FIRST_SCHEMA_HINT = (
    '{"reasoning_chain": ["cause: <target> <signal>=<value> ...", '
    '"propagation: <target> <signal>=<value> ...", "impact: <target> <signal>=<value> ..."], '
    '"predicted_entity_ids": ["<target of the cause step>"], '
    '"predicted_fault_type": "<one taxonomy slug that best explains the cause step>", '
    '"confidence": <float 0-1>}'
)

# SYMPTOM_FIRST=1 (arm S, claude/design-arm-P-grounded-fault-choice-2026-10-12.md): the 7B model quotes numbers correctly but maps them to a
# familiar fault name (cpu x670 -> "memoryPressure"; error words -> httpError5xx). The JSON asks for a per-signal-family symptom reading of
# the cause target (direction + value, from the tables) BEFORE predicted_fault_type, so the type is derived from the model's own symptom list.
# Nothing is filtered; same keys for every case; no ground truth.
CHAIN_FIRST_SCHEMA_HINT_S = (
    '{"reasoning_chain": ["cause: <target> <signal>=<value> ...", '
    '"propagation: <target> <signal>=<value> ...", "impact: <target> <signal>=<value> ..."], '
    '"predicted_entity_ids": ["<target of the cause step>"], '
    '"symptoms": {"cpu": "<up|down|normal|no data> <value>", "memory": "<up|down|normal|no data> <value>", '
    '"request_count": "<up|down|normal|no data> <value>", "error_count": "<up|down|normal|no data> <value>", '
    '"latency": "<up|down|normal|no data> <value>", "pods_replicas_restarts_events": "<what the tables show or no data>", '
    '"logs_traces": "<what the tables show or no data>"}, '
    '"predicted_fault_type": "<one taxonomy slug whose defining observable appears in symptoms>", '
    '"confidence": <float 0-1>}'
)
# GROUNDED_FAULT=1 (arm P1, claude/design-arm-P-grounded-fault-choice-2026-10-12.md): one uniform rule + a 'fault_evidence' field BEFORE
# predicted_fault_type: the type must follow from the CHAIN (2026-10-12 audit: chain said cpu=670, label said httpError5xx); fault_evidence copies the
# signal=value from the cause step that the chosen type's definition requires. Probe 2026-10-12:
# Qwen2.5-7B/14B both know 403 is 4xx and reject it against a 5xx definition when asked directly; the long Coordinator prompt never asks that
# question. No call added, nothing filtered, same rule for every case, no ground truth.
CHAIN_FIRST_SCHEMA_HINT_P = (
    '{"reasoning_chain": ["cause: <target> <signal>=<value> ...", '
    '"propagation: <target> <signal>=<value> ...", "impact: <target> <signal>=<value> ..."], '
    '"predicted_entity_ids": ["<target of the cause step>"], '
    '"fault_evidence": "<signal=value copied from your cause step that the chosen type\'s definition requires>", '
    '"predicted_fault_type": "<one taxonomy slug whose definition is satisfied by fault_evidence>", '
    '"confidence": <float 0-1>}'
)
_GROUNDED_TASK = (
    " The fault type must follow from the chain: the cause step must contain the signals that identify the kind of fault (for example the "
    "status-code class or error message, which resource is saturated and its value, replica/restart/event reasons), not only an error rate. "
    "In fault_evidence copy the signal=value from your own cause step that the definition of the chosen type requires. A definition is "
    "satisfied only by what that signal shows: a different status-code class, a different resource or an absent signal does not satisfy it. "
    "If the cause step does not contain the observable a type requires, do not choose that type; choose the type whose definition matches "
    "the signals in the cause step."
)
# CHAIN_STEPS_FLEX=1: RCA100 chains have 3 steps (91 cases) or 4 (12 cases: one extra propagation hop). The default prompt asks for exactly 3.
_FLEX_TASK = (" The chain has 3 steps (cause, propagation, impact); add one more propagation step only when the tables show one more "
              "hop between the cause and the impact.")
_SYMPTOM_TASK = (
    " Before the fault type, fill 'symptoms' for the cause target only from the tables above: the direction against its own baseline and "
    "the value, or 'no data'. Then choose predicted_fault_type from those symptoms: a type whose defining observable is not among the "
    "symptoms must not be chosen."
)

LAYER_SCHEMA_HINT = (
    '{"evidence": ["<name> <signal>=<value> ..."], '
    '"level": "<service | runtime | dependency | pod_deployment | node | cloud_resource>", '
    '"suspect": "<service, pod, deployment or node name>", '
    '"runner_up": {"level": "<other level>", "why_not": "<one sentence citing a value>"}, '
    '"confidence": <float 0-1>}'
)

LAYER_SYSTEM_PROMPT = (
    "You are an on-call SRE doing the first triage step of a microservice incident: decide at "
    "which LEVEL the fault originates before naming a fault type. Reason only from the tables "
    "given. Respond ONLY with valid JSON."
)


def layer_node(llm: LLMClient):
    """Operator-style layer attribution (flag LAYER_AGENT). One LLM call; output is a soft
    hypothesis passed to the Coordinator -- nothing is filtered and the Coordinator still sees
    the full taxonomy."""
    def node(state: AgentState) -> AgentState:
        parts = [f"Alert: {state['alert_text']}"]
        if state.get("propagation_text"):
            parts.append(state["propagation_text"])
        if state.get("infra_text"):
            parts.append(state["infra_text"])
        if state.get("colocation_text"):
            parts.append(state["colocation_text"])
        if state.get("trace_text"):
            parts.append(state["trace_text"])
        if state.get("onset_text"):
            parts.append(state["onset_text"])
        parts.append(
            "Levels: service = application code/config/traffic of one service; runtime = JVM/GC/"
            "thread pool of a service; dependency = database, cache, queue or external call; "
            "pod_deployment = pod restarts, replicas, scheduling, limits of a deployment; node = "
            "host CPU/memory/disk/network or node readiness; cloud_resource = cloud-managed "
            "resource, quota or network.")
        parts.append(
            "Task: (1) list the numeric evidence you rely on (write it first), (2) say whether the "
            "degradation follows a SERVICE (the same service is abnormal on every node it runs on) "
            "or a HOST (several services on one node are abnormal together while the same services "
            "on other nodes are not), (3) pick the level of the ORIGIN (not of the victim that "
            "raised the alert) and name the suspect, (4) name the runner-up level and one value "
            "that argues against it.\nRespond as JSON: " + LAYER_SCHEMA_HINT)
        _layer_prompt = "\n\n".join(parts)
        state["layer_result"] = llm.generate_json(_layer_prompt, system=LAYER_SYSTEM_PROMPT)
        _trace(state, "layer_agent", _layer_prompt, state["layer_result"])
        return state
    return node


VERIFY_SCHEMA_HINT = (
    '{"issues": [{"claim": "<value or statement from the answer>", "problem": "<what is unsupported or '
    'contradicted>", "table_quote": "<verbatim text copied from the tables>", "suggestion": "<how to correct>"}]}'
)

VERIFY_SYSTEM_PROMPT = (
    "You audit one root-cause answer against the evidence tables it was derived from. Report an issue only "
    "when a cited value or conclusion is contradicted by, or cannot be found in, the tables. Absence of "
    "evidence in a partial table is not a contradiction. Copy table text verbatim in table_quote. Report at "
    "most 3 issues; return an empty list when the answer is consistent. Respond ONLY with valid JSON."
)


def ungrounded_numbers(chain, evidence_text: str, limit: int = 8):
    """Deterministic grounding aid (VERIFY_CHAIN): numbers written in the chain that appear in no evidence
    table. Evidence checking only -- it decides nothing."""
    out, seen = [], set()
    for step in chain or []:
        for tok in re.findall(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", str(step)):
            if tok in seen or ("." not in tok and len(tok) < 3):
                continue
            seen.add(tok)
            variants = {tok, tok.rstrip("0").rstrip(".") if "." in tok else tok}
            if not any(v and v in evidence_text for v in variants):
                out.append(tok)
    return out[:limit]


def _verify_and_revise(llm, state, first_prompt: str, result: dict, evidence_blocks: str) -> dict:
    """VERIFY_CHAIN: one audit call; if it reports issues, one revision call. Returns the final result."""
    stats = {"issues": 0, "revised": False, "parse_error": False, "ungrounded": 0}
    state["verify_stats"] = stats
    if not isinstance(result, dict) or result.get("_parse_error"):
        return result
    chain = result.get("reasoning_chain") or []
    ung = ungrounded_numbers(chain, evidence_blocks)
    stats["ungrounded"] = len(ung)
    answer = {k: result.get(k) for k in ("reasoning_chain", "predicted_entity_ids", "predicted_fault_type")}
    vprompt = (
        f"{evidence_blocks}\n\nAnswer to audit:\n{json.dumps(answer, ensure_ascii=False)}\n\n"
        + (f"Automatic check: these numbers in the chain appear in none of the tables above: "
           f"{', '.join(ung)}.\n" if ung else "Automatic check: every number in the chain appears in the tables.\n")
        + f"Respond as JSON: {VERIFY_SCHEMA_HINT}"
    )
    v = llm.generate_json(vprompt, system=VERIFY_SYSTEM_PROMPT)
    _trace(state, "verifier", vprompt, v)
    if not isinstance(v, dict) or v.get("_parse_error"):
        stats["parse_error"] = True
        return result
    issues = [i for i in (v.get("issues") or []) if isinstance(i, dict)][:3]
    stats["issues"] = len(issues)
    if not issues:
        return result
    rprompt = (
        first_prompt + "\n\nYour first answer was:\n" + json.dumps(answer, ensure_ascii=False)
        + "\n\nAn auditor checked it against the tables and reported:\n"
        + json.dumps(issues, ensure_ascii=False)
        + "\nRe-check each point against the tables. Correct only what is actually wrong, keep what is "
          "supported, and answer with the same JSON schema."
    )
    revised = llm.generate_json(rprompt, system=COORDINATOR_SYSTEM_PROMPT)
    _trace(state, "coordinator_revision", rprompt, revised)
    if (isinstance(revised, dict) and not revised.get("_parse_error")
            and revised.get("predicted_fault_type") and revised.get("predicted_entity_ids")):
        stats["revised"] = True
        return revised
    return result


def _check_fault_type(llm, state, first_prompt: str, result: dict, blocks: list) -> dict:
    """VERIFY_FAULT_TYPE (arm C): one short check of the chosen type's definition against the rows of the chosen entity; if not satisfied,
    one re-choice of predicted_fault_type only (entity and chain of the first answer are kept)."""
    from agents.type_check import entity_rows, check_type, rechoice_prompt
    st = {"checked": False, "satisfied": None, "grounded": None, "revised": False, "type_before": None, "type_after": None, "rows_chars": 0}
    state["type_check_stats"] = st
    if not isinstance(result, dict) or result.get("_parse_error"):
        return result
    ids = result.get("predicted_entity_ids") or []
    ftype = str(result.get("predicted_fault_type") or "")
    st["type_before"] = st["type_after"] = ftype
    defs = FAULT_DEFINITIONS
    if not ids or ftype not in defs:
        return result
    entity = str(ids[0])
    rows = entity_rows([entity], blocks)
    st["rows_chars"] = len(rows)
    if not rows.strip():
        return result
    chk = check_type(llm, state["alert_text"], entity, rows, defs[ftype])
    _trace(state, "type_check", chk.pop("prompt"), chk.get("raw") or {"_parse_error": True})
    st.update(checked=True, satisfied=chk["satisfied"], grounded=chk["grounded"], quote=chk.get("quote"))
    if chk["satisfied"] is not False:
        return result
    answer = {k: result.get(k) for k in ("reasoning_chain", "predicted_entity_ids", "predicted_fault_type")}
    rp = rechoice_prompt(first_prompt, answer, entity, chk)
    revised = llm.generate_json(rp, system=COORDINATOR_SYSTEM_PROMPT)
    _trace(state, "coordinator_type_rechoice", rp, revised)
    new_t = str((revised or {}).get("predicted_fault_type") or "") if isinstance(revised, dict) and not revised.get("_parse_error") else ""
    if new_t in defs and new_t != ftype:
        out = dict(result); out["predicted_fault_type"] = new_t; out["fault_type_rechosen_from"] = ftype
        st.update(revised=True, type_after=new_t)
        return out
    return result


FINAL_SCHEMA_HINT = (
    '{"predicted_entity_ids": ["entity1", "entity2"], '
    '"predicted_fault_type": "<one of the 28 RCA100 fault types or best guess>", '
    '"reasoning_chain": ["cause step", "propagation step", "impact step"], '
    '"confidence": <float 0-1>}'
)


def _findings_json(findings: dict, raw_cap: int = 300) -> str:
    """Specialist findings as prompt text. ensure_ascii=False (escaped CJK was ~3x more tokens) and an unparsable
    specialist output is cut to raw_cap chars (a runaway raw output was ~2k chars of junk in the Coordinator prompt)."""
    clean = {}
    for k, v in findings.items():
        if isinstance(v, dict) and v.get("_parse_error"):
            v = {"_parse_error": True, "_raw": str(v.get("_raw", ""))[:raw_cap]}
        clean[k] = v
    return json.dumps(clean, indent=1, ensure_ascii=False)


def coordinator_node(llm: LLMClient):
    def node(state: AgentState) -> AgentState:
        findings_block = _findings_json({
            "metrics_agent": state.get("metrics_finding"),
            "logs_agent": state.get("logs_finding"),
            "trace_agent": state.get("trace_finding"),
            "topology_agent": state.get("topology_finding"),
        })

        # Fault-type hint construction. Previously this narrowed the shown
        # taxonomy using a two-stage RandomForest classifier (group, then
        # type-within-group) plus a case-based-reasoning fallback -- both
        # removed. Even with leave-one-out cross-validation, training or
        # tuning on RCA100's own ground-truth labels (even for OTHER cases
        # within this same closed 103-case benchmark) is overfitting to the
        # benchmark rather than a genuine transferable signal, which
        # undermines RCA100's validity as an agentic-reasoning benchmark.
        # See pipeline/zero_shot_matching.py and pipeline/twist_scoring.py's
        # module docstrings for why those two remain: both are grounded
        # only in public, per-type-invariant information (the taxonomy's
        # own definitions; each case's own trace statistics), never in
        # RCA100's labels.
        #
        # The full 28-type taxonomy is always shown -- nothing is narrowed
        # away before the Coordinator reasons over it. The zero-shot
        # shortlist below is a soft, ranked hint on top of that full list.
        fault_hint = taxonomy_prompt_block()

        # Zero-shot semantic hint (see pipeline/zero_shot_matching.py):
        # compares evidence TEXT against fault-type DEFINITIONS, not
        # examples or per-case labels -- works even for types with zero
        # training cases, and needs no upstream group hint (searches all 28
        # types directly). This is now the only structured fault-type
        # signal reaching the Coordinator.
        zeroshot_topk = state.get("zeroshot_topk") or []
        if zeroshot_topk and config.USE_KEYWORD_HINT:
            shortlist = "; ".join(f"'{t}' ({sim:.0%})" for t, sim in zeroshot_topk)
            fault_hint += (
                f"\n\nDEFINITION-BASED SHORTLIST: ranked by how closely the evidence text's "
                f"wording resembles each candidate type's OFFICIAL DEFINITION (not any labeled "
                f"example or per-case ground truth): {shortlist}. This can surface types no "
                f"other signal covers, but it is a soft ranking, not an answer -- weigh it "
                f"alongside the specialist findings above, and feel free to pick a type outside "
                f"this shortlist if the evidence clearly points elsewhere."
            )

        keyword_candidates = state.get("keyword_fault_candidates")
        keyword_block = ""
        if keyword_candidates and config.USE_KEYWORD_HINT:
            keyword_block = (
                f"\nKEYWORD-DETECTED FAULT-TYPE CANDIDATES (literal terms found in the raw "
                f"evidence text, ordered by match strength -- these are the fault types with "
                f"actual textual support in this case's evidence, not a guess):\n"
                f"  {', '.join(keyword_candidates)}\n"
                f"Prefer one of these if it's consistent with the specialist findings above; "
                f"only pick outside this list if the evidence clearly points elsewhere.\n"
            )

        prop_text = state.get("propagation_text")
        if prop_text:
            keyword_block = keyword_block + (
                f"\n{prop_text}\n"
                f"For predicted_entity_ids, name the ORIGIN of the fault (the dependency whose "
                f"anomaly is not explained by one of its own callees), not merely the service "
                f"that raised the alert -- the alert service may only be relaying an error.\n"
            )

        # path_block is built here (needed by both the two-stage and
        # single-call paths below) but IMPORTANT: it is DERIVED FROM the
        # Stage-0.5 anchor's own top candidate entity (pipeline.py's
        # compute_propagation_path call passes
        # graph_anchor["anchor_entity_ids"][0] as the path's destination) --
        # so it is NOT evidence-independent. _two_stage_coordinate() below
        # deliberately withholds it from Stage 1 for this reason and only
        # reveals it in Stage 2, alongside the anchor itself.
        path = state.get("propagation_path")
        path_block = ""
        if path:
            path_block = (
                f"\nGRAPH-COMPUTED PROPAGATION PATH (real path in the topology, from the "
                f"alerted/impacted entity to the structural candidate cause -- use this as "
                f"the basis for your 'propagation' reasoning_chain step instead of "
                f"inventing one):\n  {' -> '.join(path)}\n"
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
        #
        # Found 2026-10-04 on the 8-case sample: predicted_fault_type matched
        # graph_anchor_fault_type in 7/8 proposed_hybrid cases, and in 6 of
        # those 7 the Coordinator ignored a DIFFERING zero-shot shortlist
        # suggestion to do so -- not just "the anchor usually happens to be
        # right," an active preference for the anchor over the one other
        # structured fault-type signal available. Also found: specialist
        # agents (FINDING_SCHEMA_HINT) have NO fault_type field at all --
        # entity_id/summary/supporting_evidence/confidence only -- so the
        # anchor is the ONLY ready-made structured fault-type candidate in
        # the whole prompt; the Coordinator has to synthesize anything else
        # from loose prose. Two independent, compounding causes, not one:
        # (a) the "CONFIRM unless contradicted" wording (COORDINATOR_ANCHOR_
        # CONFIRM_BIAS toggles this, but only changes wording AFTER the
        # anchor is already visible), and (b) no competing evidence-based
        # fault-type judgment ever gets formed independently of the anchor.
        # COORDINATOR_TWO_STAGE_ANCHOR (below) addresses (b) directly via
        # order-of-exposure: elicit an independent evidence-only diagnosis
        # with the anchor NOT YET in context, then reveal the anchor and
        # ask for an explicit, cited reconciliation. Standard anchoring-bias
        # mitigation (Tversky & Kahneman) -- costs ONE extra Coordinator
        # call, not N like self-consistency.
        if config.CHAIN_FIRST:
            # Chain-first: the structured chain is written BEFORE the entity and
            # fault type (JSON key order = generation order), and the Stage-0.5
            # anchor's fault type is deliberately NOT shown, so the fault type
            # follows the cited evidence instead of copying the anchor.
            infra_text = state.get("infra_text") or ""
            coloc_text = state.get("colocation_text") or ""
            trace_text = state.get("trace_text") or ""
            log_ev_text = state.get("log_evidence_text") or ""
            trace_edges_text = state.get("trace_edges_text") or ""
            metric_struct_text = state.get("metric_structure_text") or ""
            strip_text = state.get("interval_strip_text") or ""
            onset_text = state.get("onset_text") or ""
            alert_lc_text = state.get("alert_lifecycle_text") or ""
            layer_res = state.get("layer_result")
            knowledge_block = ((knowledge_prompt_block_pos() if config.USE_KNOWLEDGE_POS else knowledge_prompt_block()) + "\n\n") if (config.USE_KNOWLEDGE_PROMPT or config.USE_KNOWLEDGE_POS) else ""
            layer_block = ""
            if layer_res and not layer_res.get("_parse_error"):
                layer_block = ("\nLayer analysis by a preceding on-call analyst (a hypothesis to verify "
                               "against the tables, not a conclusion):\n"
                               + json.dumps({k: layer_res.get(k) for k in
                                             ("evidence", "level", "suspect", "runner_up", "confidence")},
                                            ensure_ascii=False) + "\n")
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"{('' if not alert_lc_text else alert_lc_text + chr(10) + chr(10))}"
                f"Specialist agent findings:\n{findings_block}\n"
                f"{keyword_block}\n"
                f"{('' if not infra_text else chr(10) + infra_text + chr(10))}"
                f"{('' if not metric_struct_text else chr(10) + metric_struct_text + chr(10))}"
                f"{('' if not strip_text else chr(10) + strip_text + chr(10))}"
                f"{('' if not coloc_text else chr(10) + coloc_text + chr(10))}"
                f"{('' if not trace_text else chr(10) + trace_text + chr(10))}"
                f"{('' if not trace_edges_text else chr(10) + trace_edges_text + chr(10))}"
                f"{('' if not log_ev_text else chr(10) + log_ev_text + chr(10))}"
                f"{('' if not onset_text else chr(10) + onset_text + chr(10))}"
                f"{layer_block}"
                f"{fault_hint}\n\n"
                f"{knowledge_block}"
                f"Task: write the root-cause reasoning chain FIRST, then derive the answer from it. "
                f"Each chain step is 'step_type: target signal=value ...' with step_type in "
                f"cause / propagation / impact. 'target' is a service, operation, node, pod or "
                f"deployment name taken from the evidence; cite the concrete numeric signal values "
                f"(request count, error count, latency, cpu/memory usage, replicas, event reasons "
                f"and counts) from the tables above. The cause step names the ORIGIN of the fault, "
                f"not merely the entity that raised the alert. Then set predicted_entity_ids to the "
                f"target of the cause step, and predicted_fault_type to the single taxonomy type "
                f"that best explains the cause step's evidence (node/pod/deployment faults are "
                f"visible in the node, deployment and event tables)."
                f"{_SYMPTOM_TASK if config.SYMPTOM_FIRST else ''}"
                f"{_GROUNDED_TASK if config.GROUNDED_FAULT else ''}"
                f"{_FLEX_TASK if config.CHAIN_STEPS_FLEX else ''}\n"
                f"Respond as JSON: {CHAIN_FIRST_SCHEMA_HINT_P if config.GROUNDED_FAULT else (CHAIN_FIRST_SCHEMA_HINT_S if config.SYMPTOM_FIRST else CHAIN_FIRST_SCHEMA_HINT)}"
            )
            state["coordinator_prompt_chars"] = len(prompt)
            state["final_result"] = llm.generate_json(prompt, system=COORDINATOR_SYSTEM_PROMPT)
            _trace(state, "coordinator_chain_first", prompt, state["final_result"])
            if config.VERIFY_CHAIN:
                _ev = "\n\n".join(t for t in (state.get("propagation_text"), infra_text, metric_struct_text, strip_text, coloc_text, trace_text, trace_edges_text, log_ev_text, onset_text) if t) \
                    or findings_block
                state["final_result"] = _verify_and_revise(llm, state, prompt, state["final_result"], _ev)
            if config.VERIFY_FAULT_TYPE:
                state["final_result"] = _check_fault_type(llm, state, prompt, state["final_result"], [
                    state.get("propagation_text"), infra_text, metric_struct_text, strip_text, coloc_text, trace_text, trace_edges_text,
                    log_ev_text, onset_text])
            return state

        anchor = state.get("graph_anchor") or {}
        has_anchor = bool(anchor.get("anchor_entity_ids") or anchor.get("anchor_fault_type"))

        if has_anchor and config.COORDINATOR_TWO_STAGE_ANCHOR:
            result, stage1_result = _two_stage_coordinate(
                llm, state, findings_block, fault_hint, keyword_block, path_block, anchor)
            state["final_result"] = result
            state["two_stage_stats"] = {
                "stage1_fault_type": stage1_result.get("predicted_fault_type"),
                "stage1_entity_ids": stage1_result.get("predicted_entity_ids"),
                "stage1_confidence": stage1_result.get("confidence"),
                "stage1_parse_error": bool(stage1_result.get("_parse_error")),
            }
            return state

        if has_anchor:
            if config.COORDINATOR_ANCHOR_CONFIRM_BIAS:
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
                anchor_block = (
                    f"\nSTRUCTURAL PRIOR (fast topology-only first pass, computed BEFORE any "
                    f"evidence was examined):\n"
                    f"  candidate entity: {anchor.get('anchor_entity_ids')}\n"
                    f"  candidate fault type: {anchor.get('anchor_fault_type')}\n"
                    f"Treat this as only ONE input among several, on equal footing with the "
                    f"specialist findings below -- it has not seen any evidence, so it is not "
                    f"more trustworthy by default. Independently judge which entity and fault "
                    f"type the EVIDENCE actually supports, and explain in your reasoning_chain "
                    f"whether you agree or disagree with this prior and why.\n"
                )
        else:
            anchor_block = ""

        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Specialist agent findings:\n{findings_block}\n"
            f"{anchor_block}"
            f"{path_block}"
            f"{keyword_block}\n"
            f"{fault_hint}\n\n"
            f"Task: synthesize a final root-cause diagnosis with an explicit "
            f"cause -> propagation -> impact reasoning chain. Each reasoning_chain "
            f"step MUST name the specific entity_id or evidence detail (from the "
            f"specialist findings, structural prior, or propagation path above) "
            f"that supports it -- do not write a generic step with nothing cited.\n"
            f"Respond as JSON: {FINAL_SCHEMA_HINT}"
        )

        if config.USE_SELF_CONSISTENCY and config.SELF_CONSISTENCY_SAMPLES > 1:
            result, sc_stats = _self_consistency_sample(llm, prompt, COORDINATOR_SYSTEM_PROMPT)
            state["self_consistency_stats"] = sc_stats
        else:
            result = llm.generate_json(prompt, system=COORDINATOR_SYSTEM_PROMPT)
        _trace(state, "coordinator", prompt, result)
        state["final_result"] = result
        return state
    return node


def _two_stage_coordinate(llm: LLMClient, state: AgentState, findings_block: str,
                           fault_hint: str, keyword_block: str, path_block: str,
                           anchor: dict) -> Tuple[dict, dict]:
    """Returns (final_result, stage1_result) -- stage1_result is returned only
    for diagnostics (see AgentState.two_stage_stats); it never changes the
    final answer.

    Anchoring-bias mitigation via order-of-exposure (Tversky & Kahneman:
    an independent judgment formed before exposure to an anchor resists it;
    one formed after exposure can be pulled toward it regardless of how the
    anchor is worded). Stage 1 elicits a diagnosis from the specialist
    findings + keyword hints + taxonomy/zero-shot hint ONLY -- the anchor is
    not in context at all. Stage 2 then reveals the anchor (and the
    propagation path, which is withheld from Stage 1 because it is computed
    FROM the anchor's own top entity -- see the caller's comment -- so
    showing it earlier would leak the anchor's choice) and asks the
    Coordinator to explicitly compare its own Stage-1 answer against it and
    justify whichever it keeps.

    The anchor is this pipeline's OWN cheap topology-only guess (Stage 0.5),
    never RCA100's ground truth -- nothing here sees or references ground
    truth at any point. "Two stages" means only the order in which two of
    this pipeline's own intermediate guesses are shown to the Coordinator.

    Costs ONE extra Coordinator call total (two calls here vs. one in the
    single-call path), not N like self-consistency. Mutually exclusive with
    USE_SELF_CONSISTENCY in this implementation -- combining 2-stage
    reconciliation with N-way voting would be 2N calls and a confounded
    experiment; test one mechanism at a time."""
    stage1_prompt = (
        f"Alert: {state['alert_text']}\n\n"
        f"Specialist agent findings:\n{findings_block}\n"
        f"{keyword_block}\n"
        f"{fault_hint}\n\n"
        f"Task: based ONLY on the evidence above, give your independent first-pass "
        f"diagnosis with an explicit cause -> propagation -> impact reasoning chain. "
        f"Each reasoning_chain step MUST name the specific entity_id or evidence "
        f"detail that supports it.\n"
        f"Respond as JSON: {FINAL_SCHEMA_HINT}"
    )
    stage1_result = llm.generate_json(stage1_prompt, system=COORDINATOR_SYSTEM_PROMPT)

    stage2_prompt = (
        f"Alert: {state['alert_text']}\n\n"
        f"Your own independent, evidence-only diagnosis from a moment ago (before you'd "
        f"seen anything else):\n{json.dumps(stage1_result)}\n\n"
        f"A SEPARATE fast first-pass guess -- topology structure only, computed BEFORE "
        f"any evidence was examined, NOT a verified answer -- produced:\n"
        f"  candidate entity: {anchor.get('anchor_entity_ids')}\n"
        f"  candidate fault type: {anchor.get('anchor_fault_type')}\n"
        f"{path_block}\n"
        f"Task: decide your FINAL diagnosis. Explicitly compare your own evidence-based "
        f"answer above against this structural guess in your reasoning_chain -- state "
        f"whether you are keeping your own answer or switching to the structural guess, "
        f"and cite the specific evidence or structural reasoning that justifies your "
        f"choice. Do not default to the structural guess just because it's shown second.\n"
        f"Respond as JSON: {FINAL_SCHEMA_HINT}"
    )
    return llm.generate_json(stage2_prompt, system=COORDINATOR_SYSTEM_PROMPT), stage1_result


def _self_consistency_sample(llm: LLMClient, prompt: str, system: str) -> Tuple[dict, dict]:
    """Self-Consistency ensemble (Wang et al., ICLR 2023, "Self-Consistency
    Improves Chain of Thought Reasoning in Language Models") applied ONLY to
    the Coordinator's final-diagnosis call: sample config.SELF_CONSISTENCY_SAMPLES
    independent completions at a temporarily raised temperature, take a
    majority vote on predicted_fault_type, and return the highest-confidence
    sample among those agreeing with the majority. Training-free, no GT
    dependency -- pure test-time compute, the same family as the project's
    other inference-time-only improvements (TWIST, zero-shot taxonomy
    matching). Motivated by a direct observation on this project's 8-case
    sample runs: this benchmark's 7B local model's single-shot
    predicted_fault_type is noisy from run to run on the SAME evidence (no
    GT leakage involved -- just ordinary LLM sampling variance), so voting
    across a few samples should be a strictly-cheaper, training-free way to
    reduce that noise than adding yet more upstream evidence.

    The temperature bump matters: config.LLM_TEMPERATURE defaults to 0.1,
    deliberately low so the Graph Anchor / Specialist Agents stay
    near-deterministic single-pass priors (by design -- see
    _compute_graph_anchor's docstring). At that temperature, repeated calls
    on the same prompt are too similar for a majority vote to mean anything;
    SELF_CONSISTENCY_TEMPERATURE restores real sampling diversity for just
    these N extra calls, then the client's original temperature is restored
    (try/finally -- even if a sample's generate_json raises).

    Deliberately scoped to ONLY the Coordinator, not Graph Anchor or
    Specialist Agents -- those are meant to stay cheap, minimal single-pass
    signals; re-sampling them would multiply pipeline cost for stages that
    were never the bottleneck. The Coordinator's fault-type call is the one
    stage this project has repeatedly found to be noise-sensitive."""
    original_temperature = getattr(llm, "temperature", None)
    if original_temperature is not None:
        llm.temperature = config.SELF_CONSISTENCY_TEMPERATURE
    samples = []
    try:
        for _ in range(config.SELF_CONSISTENCY_SAMPLES):
            samples.append(llm.generate_json(prompt, system=system))
    finally:
        if original_temperature is not None:
            llm.temperature = original_temperature

    valid = [s for s in samples
             if isinstance(s, dict) and s.get("predicted_fault_type") and not s.get("_parse_error")]
    if not valid:
        # Every sample failed to parse -- fall back to the first raw sample,
        # the same failure mode a single-shot call would have had anyway.
        fallback = samples[0] if samples else {}
        return fallback, {"n_samples": len(samples), "n_valid": 0, "votes": {}, "agreement": 0.0}

    votes: Dict[str, int] = {}
    for s in valid:
        fault_type = str(s.get("predicted_fault_type"))
        votes[fault_type] = votes.get(fault_type, 0) + 1
    majority_type, majority_count = max(votes.items(), key=lambda kv: kv[1])

    agreeing = [s for s in valid if str(s.get("predicted_fault_type")) == majority_type]
    best = max(agreeing, key=lambda s: float(s.get("confidence", 0.0) or 0.0))

    sc_stats = {
        "n_samples": len(samples),
        "n_valid": len(valid),
        "votes": votes,
        "agreement": round(majority_count / len(valid), 4),
    }
    return best, sc_stats


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
    if config.LAYER_AGENT:
        graph.add_node("layer_agent", layer_node(coordinator_llm))

    graph.set_entry_point("metrics_agent")
    # Fan out: entry triggers all four specialists (LangGraph runs nodes with
    # satisfied dependencies in the same superstep when reachable from START
    # via parallel edges). Simpler/robust alternative used here: chain then
    # join, since all four only depend on the shared input state, not on
    # each other's output — order does not affect correctness.
    graph.add_edge("metrics_agent", "logs_agent")
    graph.add_edge("logs_agent", "trace_agent")
    graph.add_edge("trace_agent", "topology_agent")
    if config.LAYER_AGENT:
        graph.add_edge("topology_agent", "layer_agent")
        graph.add_edge("layer_agent", "coordinator")
    else:
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