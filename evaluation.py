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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

import config
from schema import RCAResult, EvidenceItem
from data_loader import resolve_entity_by_name, find_service_ancestor

_MAPPING_CACHE: Optional[Dict[str, Any]] = None


def _load_mapping() -> Dict[str, Any]:
    global _MAPPING_CACHE
    if _MAPPING_CACHE is None:
        path = os.path.join(config.ANSWER_KEY_DIR, "mapping.json")
        with open(path, "r", encoding="utf-8") as f:
            _MAPPING_CACHE = json.load(f)
    return _MAPPING_CACHE


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
    couldn't be resolved (no name_index was passed to load_ground_truth)."""
    targets = gt.target_entity_ids or [te.get("entity_name") for te in gt.target_entity_names]
    if not predicted_ids or not targets:
        return 0.0

    undirected = topology.to_undirected(as_view=True)
    best = 0.0
    for pred in predicted_ids:
        for target in targets:
            if pred == target:
                best = max(best, 1.0)
                continue
            if pred in undirected and target in undirected:
                try:
                    dist = nx.shortest_path_length(undirected, pred, target)
                    best = max(best, 0.5 ** dist)
                except nx.NetworkXNoPath:
                    continue
    return round(best, 4)


def fault_identification_score(predicted_fault_type: str, gt: GroundTruth) -> float:
    """Loose containment match, since predicted_fault_type is free-form LLM
    output and gt.fault_type is a slug like 'httpError5xx' possibly prefixed
    with a group id like 'F014-'."""
    pred = predicted_fault_type.strip().lower()
    truth = gt.fault_type.strip().lower()
    truth_slug = truth.split("-")[-1] if "-" in truth else truth
    return 1.0 if (pred == truth or pred == truth_slug or truth_slug in pred) else 0.0


def _chain_overlap_score(pred_chain: List[str], gt_chain: List[str]) -> float:
    if not pred_chain or not gt_chain:
        return 0.0
    scores = []
    for gt_step in gt_chain:
        gt_words = set(gt_step.lower().replace(":", " ").split())
        best = 0.0
        for pred_step in pred_chain:
            pred_words = set(pred_step.lower().split())
            if not gt_words or not pred_words:
                continue
            jaccard = len(gt_words & pred_words) / len(gt_words | pred_words)
            best = max(best, jaccard)
        scores.append(best)
    return sum(scores) / len(scores)


def reasoning_process_score(predicted_chain: List[str], gt: GroundTruth,
                             retrieved_texts: Optional[List[str]] = None) -> float:
    """0.5 * chain-node overlap + 0.5 * checkpoint hit rate. A checkpoint
    counts as hit if its signal name (e.g. "error_count") appears in any
    retrieved evidence text — this checks SIGNAL coverage, not the numeric
    comparator/value match (that would need the raw numeric payload, which
    the Evidence Summarizer already compresses away by design; extend this
    if you want strict numeric-checkpoint scoring)."""
    chain_score = _chain_overlap_score(predicted_chain, gt.reasoning_chain)

    if gt.checkpoints and retrieved_texts:
        hits = 0
        for cp in gt.checkpoints:
            signal = (cp.get("signal") or "").lower()
            if signal and any(signal in t.lower() for t in retrieved_texts):
                hits += 1
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
    the right service can silently score as a miss."""
    retrieved_entities = set()
    for it in evidence_items:
        eid = it.observation.entity_id
        if not eid:
            continue
        if topology is not None:
            eid = find_service_ancestor(eid, topology) or eid
        retrieved_entities.add(eid)

    gt_entities = set(gt.target_entity_ids) or {te.get("entity_name") for te in gt.target_entity_names}

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
    report.update({k: v for k, v in result.retrieval_stats.items()
                   if k in ("total_pipeline_time_s", "total_calls", "total_tokens", "used_entity_fallback")})
    return report
