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

from data.taxonomy import taxonomy_prompt_block, knowledge_prompt_block_pos, FAULT_DEFINITIONS, fault_group

import re
import config
from schema import AgentFinding
from agents.llm_protocol import LLMClient


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    case_id: str
    alert_text: str
    evidence_summary: Dict[str, List[str]]     # {entity_id: [bullets]}
    graph_neighbors: Dict[str, List[str]]       # {"upstream": [...], "downstream": [...]}
    candidate_entities: List[str]               # ranked by graph score
    graph_anchor: Optional[dict]                 # cheap structural prior (Stage 0.5, pipeline.py's _compute_graph_anchor):
                                                 # {"anchor_entity_ids", "anchor_fault_type", "anchor_confidence"}
    propagation_path: Optional[List[str]]        # graph-derived hop-by-hop path from the alert entry entity to the
                                                 # anchor's top candidate (compute_propagation_path), not LLM-invented
    twist_top_entity: Optional[str]              # highest-TWIST-score entity (pure trace statistics, no GT dependency)
    twist_top_score: Optional[float]
    propagation_text: Optional[str]              # call-graph/trace table (pipeline/propagation_evidence.py)
    infra_text: Optional[str]                    # numeric APM/k8s tables (pipeline/infra_evidence.py)
    trace_text: Optional[str]                    # HTTP code / error message table (pipeline/trace_evidence.py)
    coordinator_prompt_chars: Optional[int]      # diagnostic: chars of the chain-first Coordinator prompt
    debug_trace: Optional[list]                  # prompts + raw outputs per node (only when DEBUG_DUMP_DIR is set)
    metrics_finding: Optional[dict]
    logs_finding: Optional[dict]
    trace_finding: Optional[dict]
    topology_finding: Optional[dict]
    final_result: Optional[dict]
    fp_db: Optional[dict]                        # arm FP: per-service/node signal database (pipeline/fault_fingerprint.py)


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


# SPEC_CHAIN_SEGMENTS=1 (arm SEG, claude/audit-fault-choice-and-design-seg-2026-10-13.md): a specialist no longer names one "most suspect
# entity"; it writes the chain steps it can SEE in its own modality. The Coordinator assembles ONE chain from the segments and the tables.
# Nothing is filtered by significance; same schema for every case; no ground truth.
SEGMENT_SCHEMA_HINT = '{"segment": ["<step_type>: <target> <signal>=<value> ...", "..."], "confidence": <float 0-1>}'
_SEG_RULES = (" Each step is 'step_type: target signal=value ...' with step_type in cause / propagation / impact. 'target' is the entity NAME "
              "exactly as written in the evidence (a service, service::operation, node, pod or deployment), not an id hash; cite absolute values "
              "as given. Write only steps this evidence supports. An empty list is the right answer when this modality shows nothing abnormal. "
              "Do not name a final fault type.")
COORDINATOR_SYSTEM_PROMPT_SEG = (
    "You are the Coordinator agent for a microservice root-cause-analysis system. You receive chain segments written by Metrics, Logs, Trace and "
    "Topology specialists, each of whom saw only its own modality, plus numeric tables. Assemble ONE causal chain from the segments and the "
    "tables, and produce ONE final diagnosis. Respond ONLY with valid JSON."
)
_SEG_TASK = (" The specialist findings are chain segments, each written from one modality only; they can overlap, differ or be empty. "
             "Merge them into one chain and check every step against the tables.")


def _seg_finding(result):
    """Normalises a segment answer and adds the legacy keys (entity_id, summary, supporting_evidence) used by AgentFinding / scoring."""
    if not isinstance(result, dict) or result.get("_parse_error"):
        return result
    seg = result.get("segment")
    seg = [str(x) for x in seg] if isinstance(seg, list) else ([str(seg)] if isinstance(seg, str) and seg else [])
    result["segment"] = seg
    first = re.match(r"\s*\w+\s*:\s*(\S+)", seg[0]) if seg else None
    result["entity_id"] = first.group(1).strip(",;:").split("::")[0] if first else None
    result["summary"] = " | ".join(seg)[:300]
    result["supporting_evidence"] = list(seg)
    return result


def _seg_block(state) -> str:
    """Coordinator view of the specialists in SEG mode: segments and confidence only."""
    out = {}
    for key, name in (("metrics_finding", "metrics_agent"), ("logs_finding", "logs_agent"), ("trace_finding", "trace_agent"),
                      ("topology_finding", "topology_agent")):
        f = state.get(key)
        if isinstance(f, dict) and "segment" in f:
            out[name] = {"segment": f["segment"], "confidence": f.get("confidence")}
        else:
            out[name] = f
    return _findings_json(out)


# SPEC_EVIDENCE_ONLY=1 (arm EVOBS): a specialist copies observations of its own modality; it does not name a suspect entity, summary or confidence.
OBS_SCHEMA_HINT = '{"observations": ["<entity name> <signal>=<value>  or a quoted message", "..."]}'
_OBS_TASK = (" Copy, from the evidence above only, the observations that show abnormal behaviour or errors: the entity NAME exactly as written (not an id "
             "hash) with the signal=value or the message text as given. Do not name a cause, a suspect entity or a fault type; the Coordinator decides. "
             "At most 6 items; an empty list is the right answer when this modality shows nothing abnormal.")
COORDINATOR_SYSTEM_PROMPT_OBS = (
    "You are the Coordinator agent for a microservice root-cause-analysis system. You receive observations copied from Metrics, Logs and Trace "
    "evidence (they are not conclusions) plus numeric tables, and must produce ONE final diagnosis. Respond ONLY with valid JSON."
)
COORDINATOR_SYSTEM_PROMPT_NOSPEC = (
    "You are the Coordinator agent for a microservice root-cause-analysis system. You receive numeric evidence tables and must produce ONE final "
    "diagnosis. Respond ONLY with valid JSON."
)
TYPING_SYSTEM_PROMPT = (
    "You are a fault-typing agent for a microservice root-cause-analysis system. Given the origin entity, a causal chain and the evidence rows that "
    "mention that entity, you choose the fault type whose definition the evidence satisfies. Respond ONLY with valid JSON."
)


def _obs_finding(result):
    """Normalises an observations answer and adds the legacy keys used by AgentFinding."""
    if not isinstance(result, dict) or result.get("_parse_error"):
        return result
    obs = result.get("observations")
    obs = [str(x) for x in obs] if isinstance(obs, list) else ([str(obs)] if isinstance(obs, str) and obs else [])
    result["observations"] = obs
    result["entity_id"] = None
    result["summary"] = " | ".join(obs)[:300]
    result["supporting_evidence"] = list(obs)
    result["confidence"] = 0.0
    return result


def _obs_block(state) -> str:
    out = {}
    for key, name in (("metrics_finding", "metrics_agent"), ("logs_finding", "logs_agent"), ("trace_finding", "trace_agent")):
        f = state.get(key)
        out[name] = {"observations": f["observations"]} if isinstance(f, dict) and "observations" in f else f
    return _findings_json(out)


_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _origin_name(final):
    """Name of the origin entity: the target of the Coordinator's cause step, else its predicted entity (when that is a name, not an id hash)."""
    chain = final.get("reasoning_chain") or []
    m = re.match(r"\s*cause\s*:\s*(\S+)", str(chain[0])) if chain else None
    tgt = m.group(1).strip(",;:").split("::")[0] if m else None
    if tgt and not _HEX32.match(tgt.lower()):
        return tgt
    ids = final.get("predicted_entity_ids") or []
    if ids:
        e = str(ids[0]).strip().split("::")[0]
        if e and not _HEX32.match(e.lower()):
            return e
    return None


def _lines_with(text, pat, cap_lines, cap_chars):
    out = []
    for ln in (text or "").split("\n"):
        s = ln.strip()
        if s and pat.search(s):
            out.append(s[:cap_chars])
            if len(out) >= cap_lines:
                break
    return out


def _bullets_with(summary, pat, cap_lines, cap_chars):
    out = []
    for k, bl in (summary or {}).items():
        if str(k).startswith("_"):
            continue
        for b in bl:
            if pat.search(b):
                out.append(b[:cap_chars])
                if len(out) >= cap_lines:
                    return out
    return out


def _scoped_evidence(state, name):
    """Evidence rows that mention the entity NAME (word match, case-insensitive). Selection by name only, never by significance; the caps bound the prompt."""
    pat = re.compile(r"(?<![\w-])" + re.escape(name) + r"(?!\w)", re.I)
    parts = []
    for title, lines in (
        ("Call-graph rows", _lines_with(state.get("propagation_text"), pat, 6, 600)),
        ("Numeric tables (APM, nodes, deployments, events)", _lines_with(state.get("infra_text"), pat, 16, 900)),
        ("Trace rows (HTTP codes, error messages)", _lines_with(state.get("trace_text"), pat, 6, 900)),
        ("Metric and log observations", _bullets_with(state.get("evidence_summary"), pat, 14, 500)),
    ):
        if lines:
            parts.append(title + ":\n" + "\n".join(lines))
    return "\n\n".join(parts)


_TYPING_TASK = (
    "Task: choose the fault type of the origin entity. First copy into fault_evidence the signal=value or the message text from the evidence rows above "
    "that identifies the KIND of fault (status-code class or error message, which resource is saturated and its value, replica/restart/event reasons). "
    "Then choose the one taxonomy type whose definition is satisfied by fault_evidence. A definition is satisfied only by what that signal shows: a "
    "different status-code class, a different resource or an absent signal does not satisfy it.\n"
    "Respond as JSON: " + '{"fault_evidence": "<signal=value or message copied from the rows above>", "predicted_fault_type": "<one taxonomy slug whose definition is satisfied by fault_evidence>", "confidence": <float 0-1>}'
)
# Arm TYPECMP (TYPING_COMPARE=1): the model itself proposes up to three candidate types from the taxonomy and, for each, copies the row that supports it and
# the row that contradicts it (or "none"), then chooses. Nothing is excluded by code; the LLM does the comparison.
_TYPING_TASK_CMP = (
    "Task: name up to three taxonomy types whose definition could fit this origin entity. For each candidate copy, from the evidence rows above, the row "
    "that supports its definition and the row that contradicts it (or write \"none\" when no row does). A definition is supported only by what the row "
    "shows: a different status-code class, a different resource, or an absent signal does not support it. Then choose the candidate with supporting "
    "evidence and no contradicting evidence; if two remain, choose the one whose definition names the signal you copied.\n"
    "Respond as JSON: " + '{"candidates": [{"type": "<taxonomy slug>", "supports": "<row copied or none>", "contradicts": "<row copied or none>"}], '
    '"fault_evidence": "<row copied for the chosen type>", "predicted_fault_type": "<the chosen slug>", "confidence": <float 0-1>}'
)


# Arm FP (FAULT_FINGERPRINT=show|shortlist|rule): the typing step sees a code-checked signal profile of the origin entity and the fault types whose key
# signal is present (pipeline/fault_fingerprint.py). shortlist constrains the answer to those candidates; rule takes the top one without an LLM call.
FP_TYPING_SYSTEM_PROMPT = (
    "You are a fault-typing agent for a microservice root-cause-analysis system. Given the origin entity, a causal chain, the evidence rows that "
    "mention that entity and a code-checked signal profile, you choose the fault type whose definition the evidence satisfies. Respond ONLY with valid JSON."
)
_FP_SHORTLIST_TASK = (
    "Task: choose the fault type of the origin entity from the numbered candidates above (a code shortlist: only types whose key signal is present "
    "are listed). First copy into fault_evidence the signal=value or the message text from the rows above that identifies the KIND of fault, then "
    "choose the candidate whose definition that evidence satisfies. When two candidates fit, prefer the one with supporting and without "
    "contradicting signals.\n"
    "Respond as JSON: " + '{"fault_evidence": "<signal=value or message copied from the rows above>", "predicted_fault_type": "<one candidate slug, verbatim>", "confidence": <float 0-1>}'
)
_FP_SHOW_NOTE = ("The signal profile and the candidate list above were computed by code with fixed thresholds; treat them as evidence, not as the "
                 "answer: a candidate is only a type whose key signal is present.\n")


def _fp_origin(state, fr):
    name = _origin_name(fr)
    if name:
        return name
    ids = fr.get("predicted_entity_ids") or []
    return str(ids[0]).split("::")[0] if ids else None


def _apply_fingerprint_typing(llm, state, fault_hint):
    """Arm FP. Records fp_* diagnostics on final_result; keeps the Coordinator's type on any failure."""
    if config.FP_SCORING == "fixed":
        from pipeline.fault_fingerprint import origin_signals, rank_candidates, profile_text, NODE_TYPES
        rank = lambda o: rank_candidates(o, top_k=config.FP_TOP_K)
    else:
        from pipeline.fault_calibrated import origin_signals, rank_candidates, profile_text
        NODE_TYPES = set()
        rank = lambda o: rank_candidates(o, top_k=config.FP_TOP_K, alpha=config.FP_ALPHA)
    fr = state["final_result"]
    fr["coordinator_fault_type"] = fr.get("predicted_fault_type")
    mode = config.FAULT_FINGERPRINT
    name = _fp_origin(state, fr)
    osig = origin_signals(state.get("fp_db") or {}, name)
    cands = rank(osig)
    fr["fp_mode"] = mode
    fr["fp_scoring"] = config.FP_SCORING
    fr["fp_origin"] = name
    fr["fp_origin_kind"] = osig.get("kind")
    fr["fp_signals"] = "|".join(sorted(osig.get("signals", {})))
    fr["fp_candidates"] = "|".join(c["type"] for c in cands)
    fr["typing_applied"] = False
    chosen = None
    if mode == "rule":
        chosen = cands[0]["type"] if cands else None
    elif osig.get("kind") != "unknown" and (cands or mode == "show"):
        scoped = _scoped_evidence(state, name) if name else ""
        chain = "\n".join(f"- {s}" for s in (fr.get("reasoning_chain") or []))
        knowledge_block = (knowledge_prompt_block_pos() + "\n\n") if config.USE_KNOWLEDGE_POS else ""
        if mode == "shortlist":
            defs = "\n".join(f"- {c['type']} [{fault_group(c['type'])}]" + (f" on {c['entity']}" if c.get("entity") else "")
                              + f": {FAULT_DEFINITIONS.get(c['type'], '')}" for c in cands)
            label_block, task = f"Definitions of the candidate types:\n{defs}", _FP_SHORTLIST_TASK
        else:
            label_block, task = fault_hint, _FP_SHOW_NOTE + _TYPING_TASK
        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Origin entity chosen by the Coordinator: {name}\n"
            f"Causal chain written by the Coordinator:\n{chain}\n\n"
            f"Evidence rows that mention {name} (selected by name only; nothing is filtered by significance):\n"
            f"{scoped or '(no row mentions this name)'}\n\n"
            f"{profile_text(name, osig, cands)}\n\n"
            f"{label_block}\n\n"
            f"{knowledge_block}"
            f"{task}"
        )
        res = llm.generate_json(prompt, system=FP_TYPING_SYSTEM_PROMPT)
        _trace(state, "fp_typing_call", prompt, res)
        fr["typing_prompt_chars"] = len(prompt)
        ft = res.get("predicted_fault_type") if isinstance(res, dict) else None
        ft = ft.strip() if isinstance(ft, str) else ""
        code = ft[:4]
        if mode == "shortlist":
            match = [c["type"] for c in cands if c["type"] == ft or (re.match(r"^F\d{3}$", code) and c["type"].startswith(code))]
            fr["fp_choice_in_shortlist"] = bool(match)
            chosen = match[0] if match else cands[0]["type"]
        elif re.match(r"^F\d{3}", ft):
            chosen = ft
        if isinstance(res, dict) and res.get("fault_evidence"):
            fr["fault_evidence"] = str(res.get("fault_evidence"))[:600]
    if chosen:
        fr["predicted_fault_type"] = chosen
        fr["typing_applied"] = True
        hyp = next((c for c in cands if c["type"] == chosen), None)
        if config.FP_LEVEL_SWITCH and hyp and hyp.get("entity") and hyp["entity"] != osig.get("origin", name):
            fr["fp_entity_switched_from"] = name                      # calibrated: the hypothesis (entity, type) names its own entity
            fr["predicted_entity_ids"] = [hyp["entity"]]
        elif (config.FP_LEVEL_SWITCH and chosen in NODE_TYPES and osig.get("kind") == "service"):
            for c in cands:
                if c["type"] == chosen:
                    node = next((osig.get("signal_node", {}).get(k) for k in c["key"] if osig.get("signal_node", {}).get(k)), None)
                    if node:
                        fr["fp_entity_switched_from"] = name
                        fr["predicted_entity_ids"] = [node]
                    break
    elif config.ENTITY_TYPING and mode != "rule":
        _apply_entity_typing(llm, state, fault_hint, _from_fp=True)
    elif config.COORD_LEAN and not fr.get("predicted_fault_type"):
        fr["predicted_fault_type"] = "unknown"


# Arm VER (VERIFY_CHAIN=1): deterministic soundness check of the chain-first answer, then at most ONE repair call.
#   (1) the answered entity must be the target of the cause step (otherwise the answer is set to that target: the chain is the reasoning);
#   (2) every number quoted as signal=value in the chain must occur in the evidence text the Coordinator saw (1% tolerance, s/ms and
#       fraction/percent conversions). Below VERIFY_MIN_GROUNDED, one call rewrites the chain with values copied from the rows; the rewrite
#       is kept only if it is better grounded. No ground truth, no threshold tuned on RCA100.
VERIFY_SYSTEM_PROMPT = (
    "You are the verifier of a microservice root-cause-analysis system. You correct a causal chain so that every quoted value is copied from the "
    "evidence rows. Respond ONLY with valid JSON."
)
_QUOTED_NUM = re.compile(r"=\s*[-+]?(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
_ANY_NUM = re.compile(r"(?<![\w.])[-+]?(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)


def _evidence_pool(state):
    texts = [state.get("propagation_text") or "", state.get("infra_text") or "", state.get("trace_text") or ""]
    for bl in (state.get("evidence_summary") or {}).values():
        texts.extend(str(b) for b in bl)
    pool = set()
    for t in texts:
        for m in _ANY_NUM.finditer(t):
            try:
                pool.add(float(m.group(1)))
            except ValueError:
                pass
    return sorted(pool), "\n".join(texts)


def _is_grounded(x, pool):
    import bisect
    for v in (x, x / 1000.0, x * 1000.0, x / 100.0, x * 100.0):
        tol = max(1e-9, 0.01 * abs(v))
        i = bisect.bisect_left(pool, v - tol)
        if i < len(pool) and pool[i] <= v + tol:
            return True
    return False


def _grounding(chain, pool):
    nums, bad = 0, []
    for step in chain or []:
        for m in _QUOTED_NUM.finditer(str(step)):
            try:
                x = float(m.group(1))
            except ValueError:
                continue
            nums += 1
            if not _is_grounded(x, pool):
                bad.append(m.group(0).lstrip("= "))
    return (1.0 if nums == 0 else (nums - len(bad)) / nums), nums, bad


def _cause_target(fr):
    chain = fr.get("reasoning_chain") or []
    m = re.match(r"\s*cause\s*:\s*(\S+)", str(chain[0])) if chain else None
    return m.group(1).strip(",;:") if m else None


def _verify_chain(llm, state):
    fr = state.get("final_result")
    if not isinstance(fr, dict) or fr.get("_parse_error") or not fr.get("reasoning_chain"):
        return
    pool, _ = _evidence_pool(state)
    g0, n0, bad = _grounding(fr.get("reasoning_chain"), pool)
    fr["verify_grounded_before"] = round(g0, 3)
    fr["verify_numbers"] = n0
    fr["verify_repaired"] = False
    if n0 and g0 < config.VERIFY_MIN_GROUNDED:
        chain = "\n".join(f"- {s}" for s in fr.get("reasoning_chain") or [])
        infra_text, trace_text, prop_text = state.get("infra_text") or "", state.get("trace_text") or "", state.get("propagation_text") or ""
        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"{prop_text}\n\n{infra_text}\n\n{trace_text}\n\n"
            f"Causal chain to correct:\n{chain}\n\n"
            f"These quoted values do not occur in the evidence rows above: {', '.join(bad[:12])}.\n"
            "Task: rewrite the chain with the same step types (cause / propagation / impact) and the same targets, unless a row above shows that a "
            "target is wrong. Each step is 'step_type: target signal=value ...'; copy every value exactly as a row above shows it, and drop a "
            "signal that no row shows.\n"
            'Respond as JSON: {"reasoning_chain": ["cause: <target> <signal>=<value> ...", "propagation: ...", "impact: ..."]}'
        )
        res = llm.generate_json(prompt, system=VERIFY_SYSTEM_PROMPT)
        _trace(state, "verify_repair", prompt, res)
        new_chain = res.get("reasoning_chain") if isinstance(res, dict) else None
        if isinstance(new_chain, list) and new_chain and re.match(r"\s*cause\s*:", str(new_chain[0])):
            g1, n1, _ = _grounding(new_chain, pool)
            if g1 > g0:
                fr["verify_chain_before"] = list(fr.get("reasoning_chain") or [])
                fr["reasoning_chain"] = [str(x) for x in new_chain]
                fr["verify_repaired"] = True
                g0 = g1
    fr["verify_grounded_after"] = round(g0, 3)
    # (1) the answer follows the chain
    tgt = _cause_target(fr)
    ids = fr.get("predicted_entity_ids") or []
    cur = str(ids[0]).split("::")[0].lower() if ids else ""
    if tgt and not _HEX32.match(tgt.lower()) and tgt.split("::")[0].lower() != cur:
        fr["verify_entity_from"] = ids[0] if ids else None
        fr["predicted_entity_ids"] = [tgt.split("::")[0]]


def _apply_entity_typing(llm, state, fault_hint, _from_fp=False):
    """Arm TYPE: second, short call that chooses the fault type from the chain and the entity-scoped evidence. Keeps the Coordinator's type on any failure."""
    fr = state.get("final_result")
    if not isinstance(fr, dict) or fr.get("_parse_error"):
        return
    if config.FAULT_FINGERPRINT and not _from_fp:
        return _apply_fingerprint_typing(llm, state, fault_hint)
    if not _from_fp:
        fr["coordinator_fault_type"] = fr.get("predicted_fault_type")
    name = _origin_name(fr)
    scoped = _scoped_evidence(state, name) if name else ""
    chain = "\n".join(f"- {s}" for s in (fr.get("reasoning_chain") or []))
    knowledge_block = (knowledge_prompt_block_pos() + "\n\n") if config.USE_KNOWLEDGE_POS else ""
    prompt = (
        f"Alert: {state['alert_text']}\n\n"
        f"Origin entity chosen by the Coordinator: {name or '(unknown)'}\n"
        f"Causal chain written by the Coordinator:\n{chain}\n\n"
        f"Evidence rows that mention {name or 'the origin entity'} (selected by name only; nothing is filtered by significance):\n"
        f"{scoped or '(no row mentions this name)'}\n\n"
        f"{fault_hint}\n\n"
        f"{knowledge_block}"
        f"{_TYPING_TASK_CMP if config.TYPING_COMPARE else _TYPING_TASK}"
    )
    res = llm.generate_json(prompt, system=TYPING_SYSTEM_PROMPT)
    _trace(state, "typing_call", prompt, res)
    fr["typing_scoped_chars"] = len(scoped)
    fr["typing_prompt_chars"] = len(prompt)
    ft = res.get("predicted_fault_type") if isinstance(res, dict) else None
    if isinstance(ft, str) and re.match(r"^F\d{3}", ft.strip()):
        fr["predicted_fault_type"] = ft.strip()
        fr["fault_evidence"] = str(res.get("fault_evidence", ""))[:600]
        fr["typing_applied"] = True
    else:
        fr["typing_applied"] = False
        if config.COORD_LEAN:
            fr["predicted_fault_type"] = "unknown"


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
        if config.SPEC_EVIDENCE_ONLY:
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"Evidence relevant to {agent_name} (entity: bullet list):\n{summary_text}\n\n"
                f"Task: from the {agent_name.replace('_', ' ')} evidence above only.{_OBS_TASK}\n"
                f"Respond as JSON: {OBS_SCHEMA_HINT}"
            )
            result = _obs_finding(llm.generate_json(prompt, system=_agent_system()))
            _trace(state, f"{agent_name}_agent", prompt, result)
            state[f"{agent_name}_finding"] = result
            return state
        if config.SPEC_CHAIN_SEGMENTS:
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"Evidence relevant to {agent_name} (entity: bullet list):\n{summary_text}\n\n"
                f"Task: from the {agent_name.replace('_', ' ')} evidence above only, write the causal-chain steps you can see, in order."
                f"{_SEG_RULES}\n"
                f"Respond as JSON: {SEGMENT_SCHEMA_HINT}"
            )
            result = _seg_finding(llm.generate_json(prompt, system=_agent_system()))
            _trace(state, f"{agent_name}_agent", prompt, result)
            state[f"{agent_name}_finding"] = result
            return state
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
        if config.SPEC_EVIDENCE_ONLY:      # structure is already in the call-graph table; no verdict wanted from this specialist
            state["topology_finding"] = {"observations": [], "entity_id": None, "summary": "", "supporting_evidence": [], "confidence": 0.0}
            return state
        if config.SPEC_CHAIN_SEGMENTS:
            _up = neighbors.get("upstream", [])
            _down = neighbors.get("downstream", [])
            _path = path
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"Upstream (callers) of the alerted entity: {_up}\n"
                f"Downstream (dependencies): {_down}\n"
                f"Computed call-graph path from the alerted entity (structure only, not evidence about the cause): "
                f"{' -> '.join(_path) if _path else 'none'}\n\n"
                f"Task: from the structure only, write the propagation steps that the call graph allows (who calls whom). "
                f"Use step_type propagation or impact; there is no signal to cite, so write 'target' and, if useful, the relation "
                f"(for example 'propagation: cart calls inventory').{_SEG_RULES}\n"
                f"Respond as JSON: {SEGMENT_SCHEMA_HINT}"
            )
            result = _seg_finding(llm.generate_json(prompt, system=_agent_system()))
            _trace(state, "topology_agent", prompt, result)
            state["topology_finding"] = result
            return state
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
# CHAIN_ABSOLUTE_SIGNALS=1: RCA100 checkpoints are numeric constraints on request count (204 of 661), average request latency in seconds (178) and
# error count (97) (paper, section 5.3), compared with >= against absolute values, and the GT example names operations as service::operation.
# One uniform instruction, same for every case; derived from the public paper, not from any label of ours.
CHAIN_LEAN_SCHEMA_HINT = (
    '{"reasoning_chain": ["cause: <target> <signal>=<value> ...", '
    '"propagation: <target> <signal>=<value> ...", "impact: <target> <signal>=<value> ..."], '
    '"predicted_entity_ids": ["<target of the cause step>"], '
    '"confidence": <float 0-1>}'
)
_LEAN_TASK = (
    "Task: write the root-cause reasoning chain. Each chain step is 'step_type: target signal=value ...' with step_type in cause / propagation / impact. "
    "'target' is a service, operation, node, pod or deployment name taken from the evidence; cite the concrete numeric signal values (request count, error "
    "count, latency, cpu/memory usage, replicas, event reasons and counts) from the tables above. The cause step names the ORIGIN of the fault, not merely "
    "the entity that raised the alert (node/pod/deployment faults are visible in the node, deployment and event tables). Then set predicted_entity_ids to the "
    "target of the cause step. The kind of fault is decided in a later step from your chain, so the cause step should contain the signals that identify it "
    "(the status-code class or error message, which resource is saturated and its value, replica/restart/event reasons), not only an error rate."
)
_ABS_TASK = (" Cite signals as absolute values in their own units, as the tables show them: request count, average request latency in seconds, "
             "error count and rate, cpu and memory usage, replicas, event reasons and counts. Do not write change factors or ratios as if they "
             "were values (not 'x670', not p95_latency_ratio). Name each step's target at the finest level the tables show: a service, "
             "service::operation, node, pod or deployment (whichever the evidence points to).")
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
        if config.SPEC_EVIDENCE_ONLY:
            findings_block = _obs_block(state)
        elif config.SPEC_CHAIN_SEGMENTS:
            findings_block = _seg_block(state)
        else:
            findings_block = _findings_json({
                "metrics_agent": state.get("metrics_finding"),
                "logs_agent": state.get("logs_finding"),
                "trace_agent": state.get("trace_finding"),
                "topology_agent": state.get("topology_finding"),
            })
        if config.NO_SPECIALISTS:
            findings_head = ""
        elif config.SPEC_EVIDENCE_ONLY:
            findings_head = f"Observations copied by the specialists (one modality each; they are not conclusions):\n{findings_block}\n"
        else:
            findings_head = f"Specialist agent findings:\n{findings_block}\n"

        # The full 28-type taxonomy is always shown: nothing is narrowed away before the Coordinator
        # reasons over it. No classifier or shortlist trained on RCA100 labels is used.
        fault_hint = taxonomy_prompt_block()

        # Call-graph table (evidence only) and the origin-versus-victim instruction.
        prop_block = ""
        prop_text = state.get("propagation_text")
        if prop_text:
            prop_block = (
                f"\n{prop_text}\n"
                f"For predicted_entity_ids, name the ORIGIN of the fault (the dependency whose "
                f"anomaly is not explained by one of its own callees), not merely the service "
                f"that raised the alert -- the alert service may only be relaying an error.\n"
            )

        if config.CHAIN_FIRST:
            # Chain-first: the structured chain is written BEFORE the entity and fault type
            # (JSON key order = generation order), and the Stage-0.5 anchor's fault type is
            # deliberately NOT shown, so the fault type follows the cited evidence instead of
            # copying the anchor.
            infra_text = state.get("infra_text") or ""
            trace_text = state.get("trace_text") or ""
            knowledge_block = (knowledge_prompt_block_pos() + "\n\n") if (config.USE_KNOWLEDGE_POS and not config.COORD_LEAN) else ""
            if config.COORD_LEAN:      # arm TYPELEAN: taxonomy, KP and the type field move to the typing call
                prompt = (
                    f"Alert: {state['alert_text']}\n\n"
                    f"{findings_head}"
                    f"{prop_block}\n"
                    f"{('' if not infra_text else chr(10) + infra_text + chr(10))}"
                    f"{('' if not trace_text else chr(10) + trace_text + chr(10))}\n"
                    f"{_LEAN_TASK}"
                    f"{_ABS_TASK if config.CHAIN_ABSOLUTE_SIGNALS else ''}\n"
                    f"Respond as JSON: {CHAIN_LEAN_SCHEMA_HINT}"
                )
                state["coordinator_prompt_chars"] = len(prompt)
                _sys = (COORDINATOR_SYSTEM_PROMPT_NOSPEC if config.NO_SPECIALISTS else COORDINATOR_SYSTEM_PROMPT_OBS if config.SPEC_EVIDENCE_ONLY
                        else COORDINATOR_SYSTEM_PROMPT)
                state["final_result"] = llm.generate_json(prompt, system=_sys)
                _trace(state, "coordinator_chain_first", prompt, state["final_result"])
                if config.VERIFY_CHAIN:
                    _verify_chain(llm, state)
                if config.ENTITY_TYPING or config.FAULT_FINGERPRINT:
                    _apply_entity_typing(llm, state, fault_hint)
                return state
            prompt = (
                f"Alert: {state['alert_text']}\n\n"
                f"{findings_head}"
                f"{prop_block}\n"
                f"{('' if not infra_text else chr(10) + infra_text + chr(10))}"
                f"{('' if not trace_text else chr(10) + trace_text + chr(10))}"
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
                f"{_GROUNDED_TASK if config.GROUNDED_FAULT else ''}"
                f"{_ABS_TASK if config.CHAIN_ABSOLUTE_SIGNALS else ''}"
                f"{_SEG_TASK if config.SPEC_CHAIN_SEGMENTS else ''}\n"
                f"Respond as JSON: {CHAIN_FIRST_SCHEMA_HINT_P if config.GROUNDED_FAULT else CHAIN_FIRST_SCHEMA_HINT}"
            )
            state["coordinator_prompt_chars"] = len(prompt)
            _sys = (COORDINATOR_SYSTEM_PROMPT_NOSPEC if config.NO_SPECIALISTS else COORDINATOR_SYSTEM_PROMPT_OBS if config.SPEC_EVIDENCE_ONLY
                    else COORDINATOR_SYSTEM_PROMPT_SEG if config.SPEC_CHAIN_SEGMENTS else COORDINATOR_SYSTEM_PROMPT)
            state["final_result"] = llm.generate_json(prompt, system=_sys)
            _trace(state, "coordinator_chain_first", prompt, state["final_result"])
            if config.VERIFY_CHAIN:
                _verify_chain(llm, state)
            if config.ENTITY_TYPING or config.FAULT_FINGERPRINT:
                _apply_entity_typing(llm, state, fault_hint)
            return state

        # Single-call path (used by the multi_agent_only baseline): the Stage-0.5 anchor is shown as a prior.
        path = state.get("propagation_path")
        path_block = ""
        if path:
            path_block = (
                f"\nGRAPH-COMPUTED PROPAGATION PATH (real path in the topology, from the "
                f"alerted/impacted entity to the structural candidate cause -- use this as "
                f"the basis for your 'propagation' reasoning_chain step instead of "
                f"inventing one):\n  {' -> '.join(path)}\n"
            )
        anchor = state.get("graph_anchor") or {}
        has_anchor = bool(anchor.get("anchor_entity_ids") or anchor.get("anchor_fault_type"))
        if has_anchor:
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

        prompt = (
            f"Alert: {state['alert_text']}\n\n"
            f"Specialist agent findings:\n{findings_block}\n"
            f"{anchor_block}"
            f"{path_block}"
            f"{prop_block}\n"
            f"{fault_hint}\n\n"
            f"Task: synthesize a final root-cause diagnosis with an explicit "
            f"cause -> propagation -> impact reasoning chain. Each reasoning_chain "
            f"step MUST name the specific entity_id or evidence detail (from the "
            f"specialist findings, structural prior, or propagation path above) "
            f"that supports it -- do not write a generic step with nothing cited.\n"
            f"Respond as JSON: {FINAL_SCHEMA_HINT}"
        )
        result = llm.generate_json(prompt, system=COORDINATOR_SYSTEM_PROMPT)
        _trace(state, "coordinator", prompt, result)
        state["final_result"] = result
        return state
    return node


def build_agent_graph(llm: Optional[LLMClient] = None, coordinator_llm: Optional[LLMClient] = None,
                       use_llm_agents: bool = True) -> StateGraph:
    if llm is None:
        from agents.factory import get_llm_client
        llm = get_llm_client()
    coordinator_llm = coordinator_llm or llm  # falls back to the same client for everything

    graph = StateGraph(AgentState)
    if config.NO_SPECIALISTS:
        graph.add_node("coordinator", coordinator_node(coordinator_llm))
        graph.set_entry_point("coordinator")
        graph.add_edge("coordinator", END)
        return graph.compile()
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