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

Also surfaces TEMPORAL context relative to the alert trigger time (see
_relative_time_label): timestamps were loaded into every Observation from
the start, but were never actually shown to the LLM anywhere in the
pipeline. RCA has a standard temporal-causality principle -- the root
cause's anomaly precedes and propagates to the symptom the alert fired
on -- so knowing "this entity's evidence started 47s BEFORE the alert" vs
"12s AFTER" is a real, previously-unused signal. We surface it as text (an
extra fact for the LLM to reason over) rather than silently re-ranking
entities by it, so retrieval ranking behavior everything else depends on
is untouched -- this is additive, not a replacement for hybrid_score
ordering.
"""

from collections import defaultdict
from typing import Dict, List, Optional
import datetime as dt
import re

from schema import EvidenceItem, Observation

ERROR_PATTERN = re.compile(r"(error|exception|fail|timeout|5\d\d)", re.IGNORECASE)


def compute_metric_trends(metrics_observations: List[Observation], topology, min_ratio: float = 1.5) -> Dict[str, List[str]]:
    """Groups metric observations by (rolled-up entity, metric name) and
    compares the earliest vs. latest value within the retrieved window,
    producing bullets like:
        "workload: 2767.0 -> 18016.0 (6.5x increase within window)"
        "error_rate: 0.0 -> 0.02 (new, was zero)"

    Rationale: raw point values ("workload=17736.0") carry no sense of
    whether that number is normal or abnormal for this entity -- a large
    absolute value looks like "traffic surge" regardless of the real fault
    (Redis timeout, node OOM, thread exhaustion all can also show elevated
    request/latency numbers as a side effect). Showing the WITHIN-WINDOW
    change instead gives the LLM a comparison point it never had before,
    without needing any external "normal" baseline outside the alert
    window's own data (nothing here comes from outside evidence already
    available, and nothing is derived from ground truth).

    Call this on the subgraph-filtered metrics list BEFORE any
    recency-based truncation (see pipeline.py) -- truncating to "most
    recent N" first would frequently discard the early-window baseline
    this function needs to compute a trend at all.

    min_ratio: only report a trend if last/first (or first/last) exceeds
    this ratio, to avoid flooding bullets with noise-level fluctuations."""
    from data.loader import find_service_ancestor  # local import: avoid a
    # data.loader <-> pipeline.evidence_summarizer circular import at
    # module load time

    ancestor_cache: Dict[str, str] = {}

    def _rolled(entity_id: str) -> str:
        if entity_id not in ancestor_cache:
            ancestor_cache[entity_id] = find_service_ancestor(entity_id, topology) or entity_id
        return ancestor_cache[entity_id]

    # {(entity, metric_name): [(timestamp, value), ...]}
    series: Dict[tuple, list] = defaultdict(list)
    for o in metrics_observations:
        if not o.entity_id or not o.timestamp:
            continue
        metric_name = o.payload.get("metric")
        value = o.payload.get("value")
        if metric_name is None or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        series[(_rolled(o.entity_id), metric_name)].append((o.timestamp, value))

    trends: Dict[str, List[str]] = defaultdict(list)
    for (entity_id, metric_name), points in series.items():
        if len(points) < 2:
            continue
        points.sort(key=lambda p: p[0])
        first_val, last_val = points[0][1], points[-1][1]

        if first_val == 0 and last_val != 0:
            trends[entity_id].append(f"{metric_name}: 0 -> {last_val:.4g} (new nonzero signal within window)")
            continue
        if first_val == 0:
            continue  # both zero, nothing to report
        if last_val == 0:
            # Symmetric case to the "new nonzero" branch above -- a metric
            # dropping to exactly zero (e.g. request_count going silent)
            # would otherwise make ratio=0.0 and crash the 1/ratio call
            # below with ZeroDivisionError (measured: happened on 3/103
            # cases in a full run before this fix).
            trends[entity_id].append(f"{metric_name}: {first_val:.4g} -> 0 (dropped to zero within window)")
            continue

        ratio = last_val / first_val
        if ratio >= min_ratio:
            trends[entity_id].append(f"{metric_name}: {first_val:.4g} -> {last_val:.4g} ({ratio:.1f}x increase within window)")
        elif ratio <= 1 / min_ratio:
            trends[entity_id].append(f"{metric_name}: {first_val:.4g} -> {last_val:.4g} ({1/ratio:.1f}x decrease within window)")
        # else: within min_ratio of stable -- not worth a bullet, avoids
        # flooding the LLM with noise-level fluctuations on every metric

    return dict(trends)


def _bucket_by_entity(items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    buckets = defaultdict(list)
    for it in items:
        key = it.observation.entity_id or "unresolved"
        buckets[key].append(it)
    return buckets


def _relative_time_label(entity_items: List[EvidenceItem], alert_timestamp: Optional[dt.datetime]) -> Optional[str]:
    """'first evidence: T-47s (before alert)' or 'T+12s (after alert)'.
    Returns None if timestamps aren't available for this entity or no
    alert_timestamp was given -- callers should skip the line entirely
    rather than print a misleading default."""
    if alert_timestamp is None:
        return None
    # Observation timestamps come from two different parsers with
    # inconsistent tz-awareness: _parse_ts_iso (alerts/events) uses
    # pd.to_datetime(utc=True) -> tz-AWARE, while _parse_ts_epoch
    # (metrics/traces) uses datetime.utcfromtimestamp() -> tz-NAIVE, even
    # though both represent UTC. A single entity's evidence can mix BOTH
    # (e.g. metrics + alerts for the same entity), so min() itself can
    # crash comparing naive vs aware -- normalize every timestamp to naive
    # BEFORE taking min(), not just the final result.
    timestamps = [it.observation.timestamp.replace(tzinfo=None) if it.observation.timestamp.tzinfo else it.observation.timestamp
                  for it in entity_items if it.observation.timestamp]
    if not timestamps:
        return None
    earliest = min(timestamps)
    if alert_timestamp.tzinfo is not None:
        alert_timestamp = alert_timestamp.replace(tzinfo=None)
    delta = (earliest - alert_timestamp).total_seconds()
    direction = "before" if delta < 0 else "after"
    return f"first evidence: T{delta:+.0f}s ({direction} alert trigger)"


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
        # ALWAYS surface actual log content too, not just a count -- a log
        # line can carry critical diagnostic context (backend/dependency
        # names, connection details) even without literally containing
        # "error"/"exception"/"timeout". Discarding raw content whenever
        # error_count==0 was hiding exactly this kind of clue: a "cart
        # ...ValkeyCartStore..." log line is the single strongest hint that
        # a case is Redis-related, but it doesn't match ERROR_PATTERN and
        # was previously never shown to the LLM at all.
        for it in by_modality["logs"][:2]:
            snippet = it.observation.text[:200]
            bullets.append(f"Log observed: {snippet}")

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
                        max_bullets_per_entity: int = 5,
                        alert_timestamp: Optional[dt.datetime] = None,
                        metric_trends: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """Returns {entity_id: [bullet, bullet, ...]}, ordered by the entities'
    best hybrid_score, ready to hand to the multi-agent layer.

    alert_timestamp: if given, each entity's bullet list is prefixed with a
    relative-timing line (see _relative_time_label) -- surfaces temporal
    ordering to the LLM without changing entity ranking, which stays purely
    hybrid_score-based as before.

    metric_trends: if given (see compute_metric_trends), prepends
    within-window change bullets ("workload: 2767 -> 18016 (6.5x increase)")
    ahead of the raw point-value bullets -- gives the LLM an "is this
    abnormal" signal that isolated readings never carried."""
    buckets = _bucket_by_entity(items)
    entity_order = sorted(
        buckets.keys(),
        key=lambda eid: max(it.hybrid_score for it in buckets[eid]),
        reverse=True,
    )[:max_entities]

    metric_trends = metric_trends or {}
    summary = {}
    for eid in entity_order:
        bullets = _rule_based_bullets(buckets[eid])[:max_bullets_per_entity]
        trend_bullets = metric_trends.get(eid, [])[:3]  # cap to top 3, avoid crowding out other signals
        time_label = _relative_time_label(buckets[eid], alert_timestamp)
        prefix = ([time_label] if time_label else []) + trend_bullets
        summary[eid] = prefix + bullets
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