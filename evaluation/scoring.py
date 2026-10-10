"""
evaluation.py
==============
Ground truth is loaded ONLY here — never inside data_loader/retrieval/agents —
to guarantee the framework cannot leak answers into its own reasoning path.

GROUND TRUTH FILE LAYOUT (confirmed real answer_key/t001.gt.json):
--------------------------------------------------------------------------
answer_key/t001.gt.json (top level):
{
  "incident_id": "...", "case_id": "...", "alert_title": "...",
  "root_cause_entities": ["payment"],          # convenient shortcut, service names only
  "raw_ground_truth": "{...}"                  # <-- JSON-ENCODED STRING, must be json.loads()'d again
}

json.loads(raw_ground_truth) gives:
{
  "outcome": {
    "expected_fault_id": "F014-httpError5xx",
    "target_entity_ids": ["06e538f4a2950039a09fd3bba1d3b7b2"],      # direct UModel entity IDs
    "target_entities": [{"entity_id": "...", "entity_name": "payment",
                          "entity_domain": "apm", "entity_type": "apm.service"}],
    "expected_conclusion": "<free-text final diagnosis, Chinese>"
  },
  "reasoning": {
    "steps": [
      {"step": 1, "title": "...", "step_type": "cause", "target": "payment",
       "fault_id": "F014-httpError5xx", "description": "<free-text, Chinese>",
       "required": true, "queryable": true,
       "observability": [{"source_type": "metric", "source": "apm", "signal": "error_count",
                           "required": true,
                           "expected": {"comparator": ">=", "value": 8829, "unit": "count/min"}}],
       "conclusion_constraints": null, "time_range_hint": "..."},
      {"step": 2, "step_type": "propagation", "target": "checkout", ...},
      {"step": 3, "step_type": "impact", "target": "checkout::/oteldemo.CheckoutService/PlaceOrder", ...}
    ]
  }
}
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

import config
from schema import RCAResult, EvidenceItem
from data.loader import resolve_entity_by_name, find_service_ancestor
from data.taxonomy import fault_group

_MAPPING_CACHE: Optional[Dict[str, Any]] = None


def _load_mapping() -> Dict[str, Any]:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        path = os.path.join(config.ANSWER_KEY_DIR, "mapping.json")
        with open(path, "r", encoding="utf-8") as f:
            _MAPPING_CACHE = json.load(f)
    return _MAPPING_CACHE


_TAXONOMY_CACHE: Optional[Dict[str, Any]] = None


def _load_taxonomy() -> Dict[str, Any]:
    """Official taxonomy.json (answer_key/). EVALUATION ONLY: never read by the
    agent path or put in a prompt (dataset README: no answer_key/ content in the
    agent's context). Returns {} if the file is absent."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        path = os.path.join(config.ANSWER_KEY_DIR, "taxonomy.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _TAXONOMY_CACHE = json.load(f)
        except (OSError, ValueError):
            _TAXONOMY_CACHE = {}
    return _TAXONOMY_CACHE


def _slug_levels(fault_type) -> Optional[tuple]:
    """(slug, L1, L2) for a type string like 'F014-httpError5xx' / 'httpError5xx'
    / free text containing a slug; None if it cannot be resolved."""
    tax = _load_taxonomy()
    defs = tax.get("fault_definitions") or {}
    if not defs or not isinstance(fault_type, str):
        return None
    t = fault_type.strip().lower()
    for key, d in defs.items():
        slug = key.split("-", 1)[-1].lower()
        if t == key.lower() or t == slug or slug in t:
            return (slug, d.get("L1"), d.get("L2"))
    return None


def fault_identification_tiered(predicted_fault_type, gt: "GroundTruth") -> Optional[float]:
    """Tiered rubric from taxonomy.json (exact 1.0 / same L2 0.5 / same L1 0.25 /
    else 0). Reported alongside, not instead of, the official binary FI."""
    if not _load_taxonomy().get("scoring_rubric"):
        return None
    rub = _load_taxonomy()["scoring_rubric"]
    g = _slug_levels(gt.fault_type)
    if g is None:
        return None
    p = _slug_levels(predicted_fault_type)
    if p is None:
        return rub.get("T0_different", 0.0)
    if p[0] == g[0]:
        return rub.get("T3_exact_L3", 1.0)
    if p[2] and p[2] == g[2]:
        return rub.get("T2_same_L2", 0.5)
    if p[1] and p[1] == g[1]:
        return rub.get("T1_same_L1", 0.25)
    return rub.get("T0_different", 0.0)


def get_real_case_id(task_id: str) -> Optional[str]:
    return _load_mapping().get("task_to_case_id", {}).get(task_id)


@dataclass
class GroundTruth:
    task_id: str
    case_id: str                                # real case_id, e.g. "F014-httpError5xx.tbdh9alum..."
    fault_type: str                             # e.g. "F014-httpError5xx"
    root_cause_entities: List[str]              # shortcut name list, e.g. ["payment"]
    target_entity_ids: List[str]                # direct UModel entity IDs from target_entities
    target_entity_names: List[Dict[str, str]]   # [{"entity_id":..,"entity_name":..,"entity_type":..}, ...]
    reasoning_chain: List[str]                  # ["cause: payment", "propagation: checkout", ...]
    reasoning_descriptions: List[str]           # the free-text `description` per step (richer chain matching)
    expected_conclusion: str                    # free-text final diagnosis
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


def load_ground_truth(task_id: str, name_index: Optional[Dict[str, str]] = None,
                       answer_key_dir: str = config.ANSWER_KEY_DIR) -> GroundTruth:
    path = os.path.join(answer_key_dir, f"{task_id}.gt.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        top = json.load(f)

    inner = json.loads(top["raw_ground_truth"])  # nested JSON-encoded string
    outcome = inner.get("outcome", {})
    steps = inner.get("reasoning", {}).get("steps", [])

    target_entities = outcome.get("target_entities", [])
    target_entity_ids = outcome.get("target_entity_ids", []) or [
        te.get("entity_id") for te in target_entities if te.get("entity_id")
    ]

    # Fallback: if target_entity_ids is somehow empty, resolve root_cause_entities
    # (plain service names) against topology by name.
    if not target_entity_ids and name_index:
        for name in top.get("root_cause_entities", []):
            resolved = resolve_entity_by_name(name, name_index)
            if resolved:
                target_entity_ids.append(resolved)

    chain = [f"{s.get('step_type')}: {s.get('target')}" for s in steps]
    descriptions = [s.get("description", "") for s in steps]

    checkpoints = []
    for s in steps:
        for obs in s.get("observability", []):
            checkpoints.append({
                "step": s.get("step"),
                "step_type": s.get("step_type"),
                "target": s.get("target"),
                "source_type": obs.get("source_type"),
                "signal": obs.get("signal"),
                "comparator": obs.get("expected", {}).get("comparator"),
                "value": obs.get("expected", {}).get("value"),
                "unit": obs.get("expected", {}).get("unit"),
            })

    return GroundTruth(
        task_id=task_id,
        case_id=top.get("case_id") or get_real_case_id(task_id) or task_id,
        fault_type=str(outcome.get("expected_fault_id", "unknown")),
        root_cause_entities=top.get("root_cause_entities", []),
        target_entity_ids=target_entity_ids,
        target_entity_names=target_entities,
        reasoning_chain=chain,
        reasoning_descriptions=descriptions,
        expected_conclusion=outcome.get("expected_conclusion", ""),
        checkpoints=checkpoints,
        raw=top,
    )


# ---------------------------------------------------------------------------
# RCA100 official protocol
# ---------------------------------------------------------------------------
def entity_localization_score(predicted_ids: List[str], gt: GroundTruth,
                               topology: nx.DiGraph) -> float:
    """Exact match = 1.0; partial credit for topologically adjacent entities;
    0 otherwise. Falls back to name-based comparison if target_entity_ids
    couldn't be resolved (no name_index was passed to load_ground_truth).

    Rolls up both predicted and target entity IDs to their apm.service
    ancestor before comparing (in addition to keeping the raw/unrolled
    comparison too, since exact-match at fine granularity is still the best
    possible signal when it happens). Without this, a prediction at
    instance/operation granularity that is structurally correct (e.g. the
    LLM names checkout's specific apm.operation node while ground truth
    names the checkout *service*) scores a hard 0.0 purely from ID-format
    mismatch, identical to the retrieval_precision_recall bug found
    earlier -- this function needed the same fix, separately."""
    targets = gt.target_entity_ids or [te.get("entity_name") for te in gt.target_entity_names]
    if not predicted_ids or not targets:
        return 0.0

    def _rollup(eid: str) -> str:
        return find_service_ancestor(eid, topology) or eid

    undirected = topology.to_undirected(as_view=True)
    best = 0.0
    for pred in predicted_ids:
        pred_rolled = _rollup(pred)
        for target in targets:
            target_rolled = _rollup(target)
            if pred == target or pred_rolled == target_rolled:
                best = max(best, 1.0)
                continue
            if pred in undirected and target in undirected:
                try:
                    dist = nx.shortest_path_length(undirected, pred, target)
                    best = max(best, 0.5 ** dist)
                except nx.NetworkXNoPath:
                    continue
    return round(best, 4)


def fault_identification_score(predicted_fault_type, gt: GroundTruth) -> float:
    """Loose containment match, since predicted_fault_type is free-form LLM
    output and gt.fault_type is a slug like 'httpError5xx' possibly prefixed
    with a group id like 'F014-'.

    Accepts non-str input defensively: a denser prompt (see sea_rca) can lead
    the LLM to emit a structured object here (e.g. {"type": "...",
    "confidence": ...}) instead of a plain string -- same failure class
    documented on _stringify_chain_step above. Falls back to "unknown" for a
    dict without an obvious type-like field, so it still scores 0.0 cleanly
    instead of raising."""
    if not isinstance(predicted_fault_type, str):
        predicted_fault_type = _stringify_chain_step(predicted_fault_type) or "unknown"
    pred = predicted_fault_type.strip().lower()
    truth = gt.fault_type.strip().lower()
    truth_slug = truth.split("-")[-1] if "-" in truth else truth
    return 1.0 if (pred == truth or pred == truth_slug or truth_slug in pred) else 0.0


def fault_group_score(predicted_fault_type: str, gt: GroundTruth) -> float:
    """Coarser companion to fault_identification_score: 1.0 if the predicted
    and ground-truth fault types fall in the same one of 6 groups (see
    data/taxonomy.FAULT_GROUPS), even if the specific 28-way type is wrong.
    Useful when a model can reliably tell 'this is a resource problem' but
    not reliably distinguish nodeCpuHigh from cpuFullLoad -- a real,
    reportable capability distinct from exact-type accuracy."""
    pred_group = fault_group(predicted_fault_type)
    gt_group = fault_group(gt.fault_type)
    return 1.0 if (pred_group != "unknown" and pred_group == gt_group) else 0.0


def _stringify_chain_step(step) -> str:
    """Coerces one reasoning_chain entry to plain text. Most systems' prompts
    (graphrag_only, standard_rag) elicit a flat List[str] from the LLM, but a
    richer prompt (sea_rca's "cite the specific evidence/entities" instruction)
    can lead a 7B model to emit structured objects instead -- e.g.
    {"step": "cause", "target": "payment", "evidence": "error_count=8829"}
    rather than "cause: payment (error_count=8829)". Observed in practice:
    sea_rca's first 8-case run crashed on 7/8 cases with 'dict object has no
    attribute lower' inside the old version of this function, which assumed
    every pred_chain entry was already a string. Join a dict's own values
    (covers str/number/nested-list values generically) rather than discarding
    the step entirely -- a partially-recovered text step still contributes to
    the word-overlap score instead of silently zeroing out that case's
    reasoning_process. Any other non-string type falls back to str()."""
    if isinstance(step, str):
        return step
    if isinstance(step, dict):
        parts = []
        for v in step.values():
            if isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v)
            elif v is not None:
                parts.append(str(v))
        return " ".join(parts)
    return str(step)


def _chain_overlap_score(pred_chain: List[str], gt_chain: List[str]) -> float:
    if not pred_chain or not gt_chain:
        return 0.0
    scores = []
    for gt_step in gt_chain:
        gt_words = set(gt_step.lower().replace(":", " ").split())
        best = 0.0
        for pred_step in pred_chain:
            pred_words = set(_stringify_chain_step(pred_step).lower().split())
            if not gt_words or not pred_words:
                continue
            jaccard = len(gt_words & pred_words) / len(gt_words | pred_words)
            best = max(best, jaccard)
        scores.append(best)
    return sum(scores) / len(scores)


_COMPARATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _extract_signal_value(signal: str, text: str) -> Optional[float]:
    """Extracts a numeric value for `signal` from raw evidence text -- e.g.
    signal='error_count' matches 'error_count=8829.0' in
    '[metric] payment::.../Charge error_count=8829.0' -> 8829.0. Returns
    None if the signal name appears in the text but isn't immediately
    followed by a parseable number (e.g. it's mentioned in a log sentence,
    not a key=value metric reading)."""
    pattern = re.escape(signal) + r"[=:]\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _checkpoint_satisfied(cp: Dict, retrieved_texts: List[str]) -> bool:
    """Checks whether any retrieved evidence text satisfies this
    checkpoint's numeric comparator/value constraint (e.g. error_count >=
    8829) -- matching the official protocol's checkpoint definition, where
    99.5% of RCA100's 661 checkpoints carry an explicit
    <comparator, value, unit> constraint the agent's evidence must satisfy,
    not merely mention. Falls back to signal-name-only presence (the
    previous, weaker behavior) only when the comparator/value aren't
    available, or when the signal is mentioned but no evidence text carries
    a machine-parseable number for it -- this avoids scoring a hard miss
    purely due to text-extraction failure on an otherwise-correct citation."""
    signal = (cp.get("signal") or "").lower()
    if not signal:
        return False
    comparator = cp.get("comparator")
    expected_value = cp.get("value")

    found_signal_mention = False
    for t in retrieved_texts:
        if signal not in t.lower():
            continue
        found_signal_mention = True
        if comparator in _COMPARATORS and expected_value is not None:
            actual = _extract_signal_value(signal, t)
            if actual is not None:
                try:
                    if _COMPARATORS[comparator](actual, float(expected_value)):
                        return True
                except (TypeError, ValueError):
                    pass

    # Fallback: comparator/value missing from this checkpoint, or the
    # signal was mentioned but no text had an extractable number for it --
    # credit name-only presence rather than a hard miss.
    if found_signal_mention and (comparator not in _COMPARATORS or expected_value is None):
        return True
    return False


def reasoning_process_score(predicted_chain: List[str], gt: GroundTruth,
                             retrieved_texts: Optional[List[str]] = None) -> float:
    """0.5 * chain-node overlap + 0.5 * checkpoint hit rate. A checkpoint
    counts as hit only if its numeric comparator/value constraint (e.g.
    error_count >= 8829) is actually satisfied by a parseable value in
    retrieved evidence -- see _checkpoint_satisfied() -- not merely whether
    the signal name is textually present, matching the official protocol's
    checkpoint semantics (Section 5.3/5.4 of the RCA100 paper)."""
    chain_score = _chain_overlap_score(predicted_chain, gt.reasoning_chain)

    if gt.checkpoints and retrieved_texts:
        hits = sum(1 for cp in gt.checkpoints if _checkpoint_satisfied(cp, retrieved_texts))
        checkpoint_score = hits / len(gt.checkpoints)
    else:
        checkpoint_score = 0.0

    return round(0.5 * chain_score + 0.5 * checkpoint_score, 4)


def rca100_final_score(result: RCAResult, gt: GroundTruth, topology: nx.DiGraph,
                        retrieved_texts: Optional[List[str]] = None) -> Dict[str, float]:
    el = entity_localization_score(result.predicted_entity_ids, gt, topology)
    fi = fault_identification_score(result.predicted_fault_type, gt)
    rp = reasoning_process_score(result.reasoning_chain, gt, retrieved_texts)

    final = (config.WEIGHT_ENTITY_LOCALIZATION * el +
             config.WEIGHT_FAULT_IDENTIFICATION * fi +
             config.WEIGHT_REASONING_PROCESS * rp)

    return {
        "entity_localization": el,
        "fault_identification": fi,
        "reasoning_process": rp,
        "final_score": round(final, 4),
    }


# ---------------------------------------------------------------------------
# Additional proposed research metrics
# ---------------------------------------------------------------------------
def retrieval_precision_recall(evidence_items: List[EvidenceItem], gt: GroundTruth,
                                topology: Optional[nx.DiGraph] = None) -> Dict[str, float]:
    """Rolls up retrieved entity IDs to their apm.service ancestor (if a
    topology is given) before matching, since telemetry is often tagged at
    instance/pod/operation granularity while ground truth target_entity_ids
    are service-level. Without this rollup, correctly-retrieved evidence for
    the right service can silently score as a miss.

    Ground-truth entities are rolled up the same way before comparison --
    a k8s.node-type target with a valid apm.service ancestor was previously
    compared in its raw (unrolled) form against rolled-up retrieved
    entities, which under-counted matches asymmetrically."""
    retrieved_entities = set()
    for it in evidence_items:
        eid = it.observation.entity_id
        if not eid:
            continue
        if topology is not None:
            eid = find_service_ancestor(eid, topology) or eid
        retrieved_entities.add(eid)

    gt_entities = set(gt.target_entity_ids) or {te.get("entity_name") for te in gt.target_entity_names}
    if topology is not None:
        gt_entities = {find_service_ancestor(e, topology) or e for e in gt_entities}

    if not retrieved_entities:
        return {"retrieval_precision": 0.0, "retrieval_recall": 0.0}

    tp = len(retrieved_entities & gt_entities)
    precision = tp / len(retrieved_entities)
    recall = tp / len(gt_entities) if gt_entities else 0.0
    return {"retrieval_precision": round(precision, 4), "retrieval_recall": round(recall, 4)}


def explainability_proxy(result: RCAResult) -> float:
    if not result.reasoning_chain:
        return 0.0
    chain_len_score = min(len(result.reasoning_chain) / 3.0, 1.0)
    agreeing_agents = sum(
        1 for f in result.agent_findings
        if f.entity_id in result.predicted_entity_ids and f.entity_id is not None
    )
    agreement_score = min(agreeing_agents / 2.0, 1.0)
    evidence_score = min(
        sum(len(f.supporting_evidence) for f in result.agent_findings) / 8.0, 1.0
    )
    return round((chain_len_score + agreement_score + evidence_score) / 3.0, 4)


def full_case_report(result: RCAResult, gt: GroundTruth, topology: nx.DiGraph,
                      evidence_items: Optional[List[EvidenceItem]] = None) -> Dict[str, Any]:
    retrieved_texts = [it.observation.text for it in evidence_items] if evidence_items else None
    report = {"case_id": result.case_id}
    report.update(rca100_final_score(result, gt, topology, retrieved_texts))
    if evidence_items is not None:
        report.update(retrieval_precision_recall(evidence_items, gt, topology))
    report["explainability"] = explainability_proxy(result)
    # Reasoning-process components, reported separately. The official
    # combination (0.5 chain + 0.5 checkpoint) and final_score are unchanged;
    # this only exposes which half drives the score.
    report["rp_chain_overlap"] = round(_chain_overlap_score(result.reasoning_chain, gt.reasoning_chain), 4)
    if gt.checkpoints and retrieved_texts:
        _hits = sum(1 for cp in gt.checkpoints if _checkpoint_satisfied(cp, retrieved_texts))
        report["rp_checkpoint_hit_rate"] = round(_hits / len(gt.checkpoints), 4)
    else:
        report["rp_checkpoint_hit_rate"] = 0.0
    # Stricter evaluation-only variant: checkpoint satisfied by a value the agent
    # itself wrote in its reasoning chain (not merely present in retrieved evidence).
    _chain_texts = [str(x) for x in (result.reasoning_chain or [])]
    if gt.checkpoints and _chain_texts:
        _ch = sum(1 for cp in gt.checkpoints if _checkpoint_satisfied(cp, _chain_texts))
        report["rp_checkpoint_hit_chain"] = round(_ch / len(gt.checkpoints), 4)
    else:
        report["rp_checkpoint_hit_chain"] = 0.0
    # Strict RP / final (evaluation-only, added 2026-10-10 audit): the local
    # rp_checkpoint_hit_rate counts any checkpoint value that merely appears in
    # RETRIEVED evidence, including the synthetic infra tables, so arms that add
    # evidence tables gain RP by construction. The *_chain variants credit only values
    # the agent wrote in its own chain; compare arms on these.
    report["reasoning_process_chain"] = round(
        0.5 * report["rp_chain_overlap"] + 0.5 * report["rp_checkpoint_hit_chain"], 4)
    report["final_score_chain"] = round(
        config.WEIGHT_ENTITY_LOCALIZATION * report["entity_localization"]
        + config.WEIGHT_FAULT_IDENTIFICATION * report["fault_identification"]
        + config.WEIGHT_REASONING_PROCESS * report["reasoning_process_chain"], 4)
    # Counterfactual entity_localization per entity source (diagnostic only;
    # the official entity_localization above still uses result.predicted_entity_ids).
    for _key, _col in (("alt_entity_coordinator", "el_if_coordinator"),
                       ("alt_entity_anchor", "el_if_anchor"),
                       ("alt_entity_twist_top", "el_if_twist_top"),
                       ("alt_entity_entry", "el_if_entry")):
        _ids = [e for e in str(result.retrieval_stats.get(_key) or "").split("|") if e]
        report[_col] = entity_localization_score(_ids, gt, topology) if _ids else None
    report.update(result.retrieval_stats)
    # Raw predictions alongside the scores -- without this, diagnosing WHY a
    # score is low/zero means re-running the case with a separate debug
    # script just to see what the LLM actually said. Kept as compact
    # strings (not lists) so they stay single CSV cells.
    report["predicted_fault_type"] = result.predicted_fault_type
    report["gt_fault_type"] = gt.fault_type
    report["fi_tiered"] = fault_identification_tiered(result.predicted_fault_type, gt)
    report["fault_group_identification"] = fault_group_score(result.predicted_fault_type, gt)
    report["predicted_fault_group"] = fault_group(result.predicted_fault_type)
    report["gt_fault_group"] = fault_group(gt.fault_type)
    report["predicted_entity_ids"] = "|".join(result.predicted_entity_ids)
    report["gt_target_entity_ids"] = "|".join(gt.target_entity_ids)
    return report