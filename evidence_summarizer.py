"""
evidence_summarizer.py
========================
Module 5 — Evidence Summarizer.

Compresses a ranked EvidenceItem list into a short, per-entity bullet
summary (e.g. "Payment Service: • Error rate increased • Timeout observed
• CPU normal • Downstream checkout failed") BEFORE anything reaches the
LLM. Two-stage design:

  1. Rule-based aggregation (cheap, deterministic, no LLM call): group by
     entity + modality, compute simple signals (error/exception counts,
     latency direction, status-code mix).
  2. Optional LLM compression pass for entities with too much heterogeneous
     text to summarize with rules alone (kept optional to control token
     cost — see llm_client.py).
"""

from collections import defaultdict
from typing import Dict, List, Optional
import re

from schema import EvidenceItem

ERROR_PATTERN = re.compile(r"(error|exception|fail|timeout|5\d\d)", re.IGNORECASE)


def _bucket_by_entity(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    buckets = defaultdict(list)
    for it in items:
        key = it.observation.entity_id or "unresolved"
        buckets[key].append(it)
    return buckets


def _rule_based_bullets(entity_items: List[EvidenceItem]) -> List[str]:
    bullets = []
    by_modality = defaultdict(list)
    for it in entity_items:
        by_modality[it.observation.modality].append(it)

    if "logs" in by_modality:
        error_count = sum(1 for it in by_modality["logs"] if ERROR_PATTERN.search(it.observation.text))
        total = len(by_modality["logs"])
        if error_count > 0:
            bullets.append(f"Error/exception patterns in {error_count}/{total} retrieved log lines")

    if "metrics" in by_modality:
        # surface each distinct metric name mentioned, most-relevant first
        seen_metrics = []
        for it in by_modality["metrics"]:
            name_part = it.observation.text.split("]", 1)[-1].strip()
            if name_part not in seen_metrics:
                seen_metrics.append(name_part)
            if len(seen_metrics) >= 4:
                break
        bullets.extend(f"Metric observed: {m}" for m in seen_metrics)

    if "traces" in by_modality:
        error_spans = sum(1 for it in by_modality["traces"]
                           if ERROR_PATTERN.search(it.observation.text))
        if error_spans > 0:
            bullets.append(f"{error_spans} span(s) with error/timeout status among retrieved traces")

    if "events" in by_modality:
        for it in by_modality["events"][:3]:
            bullets.append(it.observation.text.replace("[event] ", "Lifecycle event: "))

    if "alerts" in by_modality:
        for it in by_modality["alerts"][:2]:
            bullets.append(it.observation.text.replace("[alert:", "Alert stage ["))

    return bullets or ["No strong signal in retrieved evidence for this entity"]


def summarize_evidence(items: List[EvidenceItem], max_entities: int = 8,
                        max_bullets_per_entity: int = 5) -> Dict[str, List[str]]:
    """Returns {entity_id: [bullet, bullet, ...]}, ordered by the entities'
    best hybrid_score, ready to hand to the multi-agent layer."""
    buckets = _bucket_by_entity(items)
    entity_order = sorted(
        buckets.keys(),
        key=lambda eid: max(it.hybrid_score for it in buckets[eid]),
        reverse=True,
    )[:max_entities]

    summary = {}
    for eid in entity_order:
        bullets = _rule_based_bullets(buckets[eid])[:max_bullets_per_entity]
        summary[eid] = bullets
    return summary


def render_summary_text(summary: Dict[str, List[str]], entity_names: Optional[Dict[str, str]] = None) -> str:
    """Flatten the per-entity bullet dict into the LLM-ready text block."""
    entity_names = entity_names or {}
    lines = []
    for eid, bullets in summary.items():
        label = entity_names.get(eid, eid)
        lines.append(f"{label}")
        lines.extend(f"  • {b}" for b in bullets)
    return "\n".join(lines)
