"""
twist_scoring.py
==================
TWIST (Trace-based Weighted Impact Scoring & Thresholding) adapted from
GALA (Tian, Y. et al. "Can Graph-Augmented Large Language Model Agentic
Workflows Elevate Root Cause Analysis?" arXiv:2508.12472, 2025).

GALA's TWIST computes 4 complementary service-level scores from
distributed traces, producing a quantitative, LLM-free characterization
of where anomalies originate and how they propagate:

  c1 (Self-Anomaly Score):  how often the service's OWN spans are anomalous
  c2 (Trace Impact Score):  how often the service appears in anomalous traces
  c3 (Blast Radius Score):  how many downstream services it calls (propagation risk)
  c4 (Delay Severity Score): how severe its latency deviations are

Our adaptation differs from GALA's in one important way:
  - GALA uses TWIST for SERVICE ENTITY RANKING only (AC@k metric)
  - We additionally use TWIST scores as INPUT FEATURES for the fault-type
    classifier (tier-1 and tier-2 RandomForest), since RCA100 evaluates
    both entity localization AND fault_identification -- a service with
    high c1 (self-anomalous) + high c3 (blast radius) + high c4 (severe
    delay) has a different failure profile than one with low c1 + high c2
    (it's affected by upstream, not the origin) -- this profile difference
    is a signal for fault type even if the raw metric values aren't.

Dynamic thresholding (for c1):
  GALA uses "dynamic thresholding" for span-level anomaly detection.
  We implement this as: a span is anomalous if its duration exceeds
  mean + 2*std of ALL spans for the same (serviceName, spanName) pair
  within the case's time window. This is computed from the traces
  themselves without any external reference baseline -- genuinely
  data-driven, not a hardcoded threshold.
"""

from typing import Dict, List, Optional, Tuple
import math
from collections import defaultdict

from schema import Observation


def compute_twist_scores(
    observations: List,
    candidate_entities: Optional[List[str]] = None,
    name_index: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes TWIST scores for each service observed in the trace
    observations list. Returns a dict:
      {service_name: {c1, c2, c3, c4, twist_score}}

    observations: the pipeline's Observation list for the 'traces' modality.
    candidate_entities: if given, only score these entity IDs (matched via
      name_index back to service names) for efficiency.
    name_index: entity_id -> entity_name mapping (from Case.name_index).

    All scores are in [0, 1]. twist_score is the weighted composite
    following GALA's equal weighting (w1=w2=w3=w4=0.25) as a starting
    point -- can be tuned via the TWIST_WEIGHTS constant below.
    """
    TWIST_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # w1, w2, w3, w4

    # --- Parse spans from observation payloads ----------------------------
    # Each trace Observation's .payload contains the raw span dict from
    # data/loader.py's load_traces() (the full parquet row as _asdict()).
    spans = []
    for o in observations:
        if o.modality != "traces" or not o.payload:
            continue
        p = o.payload
        spans.append({
            "traceId": p.get("traceId", ""),
            "spanId": p.get("spanId", ""),
            "parentSpanId": p.get("parentSpanId", ""),
            "serviceName": p.get("serviceName", ""),
            "spanName": p.get("spanName", ""),
            "duration": _safe_float(p.get("duration", 0)),  # nanoseconds
            "statusCode": str(p.get("statusCode", "0")),
        })

    if not spans:
        return {}

    # --- Step 1: Dynamic span-level anomaly detection (c1 numerator) ------
    # Anomalous = duration > mean + 2*std for (serviceName, spanName) pair.
    # GALA calls this "dynamic thresholding" -- computed from observed spans,
    # no external reference needed.
    duration_by_op: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for s in spans:
        key = (s["serviceName"], s["spanName"])
        duration_by_op[key].append(s["duration"])

    op_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (mean, std)
    for key, durations in duration_by_op.items():
        mean = sum(durations) / len(durations)
        variance = sum((d - mean) ** 2 for d in durations) / max(len(durations), 1)
        std = math.sqrt(variance)
        op_stats[key] = (mean, std)

    def _is_anomalous_span(s: dict) -> bool:
        key = (s["serviceName"], s["spanName"])
        mean, std = op_stats.get(key, (0.0, 0.0))
        threshold = mean + 2.0 * std
        return s["duration"] > threshold or s["statusCode"] == "2"  # ERROR always anomalous

    span_anomalous = {s["spanId"]: _is_anomalous_span(s) for s in spans}

    # --- Step 2: Trace-level anomaly classification -----------------------
    # A trace is anomalous if ANY of its spans is anomalous.
    trace_has_anomaly: Dict[str, bool] = defaultdict(bool)
    for s in spans:
        if span_anomalous.get(s["spanId"], False):
            trace_has_anomaly[s["traceId"]] = True

    # Services appearing in each trace
    trace_to_services: Dict[str, set] = defaultdict(set)
    for s in spans:
        trace_to_services[s["traceId"]].add(s["serviceName"])

    anomalous_traces = {tid for tid, anom in trace_has_anomaly.items() if anom}
    all_traces = set(trace_to_services.keys())

    # --- Step 3: Build trace DAG for blast radius -------------------------
    # Parent-child: spanId -> list of child spanIds within same trace
    span_children: Dict[str, List[str]] = defaultdict(list)
    span_to_service: Dict[str, str] = {}
    for s in spans:
        span_to_service[s["spanId"]] = s["serviceName"]
        if s["parentSpanId"]:
            span_children[s["parentSpanId"]].append(s["spanId"])

    # Per-span: unique downstream services called (immediate children)
    def _immediate_child_services(span_id: str) -> set:
        children = span_children.get(span_id, [])
        return {span_to_service[cid] for cid in children if cid in span_to_service}

    # --- Step 4: Compute per-service scores --------------------------------
    services = set(s["serviceName"] for s in spans)
    service_spans: Dict[str, List[dict]] = defaultdict(list)
    for s in spans:
        service_spans[s["serviceName"]].append(s)

    scores: Dict[str, Dict[str, float]] = {}
    max_delay_excess = 1.0  # will be set below for normalization

    # First pass: gather raw delay excesses for normalization
    delay_excesses: Dict[str, float] = {}
    for svc, svc_spans in service_spans.items():
        anomalous_svc_spans = [s for s in svc_spans if span_anomalous.get(s["spanId"], False)]
        if not anomalous_svc_spans:
            delay_excesses[svc] = 0.0
            continue
        max_excess = 0.0
        for s in anomalous_svc_spans:
            key = (s["serviceName"], s["spanName"])
            mean, std = op_stats.get(key, (0.0, 0.0))
            excess = (s["duration"] - mean) / max(std, 1.0)
            max_excess = max(max_excess, excess)
        delay_excesses[svc] = max_excess
        max_delay_excess = max(max_delay_excess, max_excess)

    for svc in services:
        svc_spans = service_spans[svc]
        n_spans = len(svc_spans)

        # c1: Self-Anomaly Score
        n_anomalous = sum(1 for s in svc_spans if span_anomalous.get(s["spanId"], False))
        c1 = n_anomalous / max(n_spans, 1)

        # c2: Trace Impact Score -- fraction of ALL anomalous traces containing this service
        traces_with_svc = {s["traceId"] for s in svc_spans}
        anomalous_traces_with_svc = traces_with_svc & anomalous_traces
        c2 = len(anomalous_traces_with_svc) / max(len(anomalous_traces), 1)

        # c3: Blast Radius Score -- mean immediate downstream services per span
        downstream_counts = [len(_immediate_child_services(s["spanId"])) for s in svc_spans]
        mean_downstream = sum(downstream_counts) / max(len(downstream_counts), 1)
        # Normalize: cap at a reasonable maximum (e.g. 10 downstream services)
        c3 = min(mean_downstream / 10.0, 1.0)

        # c4: Delay Severity Score
        c4 = delay_excesses.get(svc, 0.0) / max_delay_excess

        # Composite TWIST score
        w1, w2, w3, w4 = TWIST_WEIGHTS
        twist = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4

        scores[svc] = {
            "c1_self_anomaly": round(c1, 4),
            "c2_trace_impact": round(c2, 4),
            "c3_blast_radius": round(c3, 4),
            "c4_delay_severity": round(c4, 4),
            "twist_score": round(twist, 4),
            "n_spans": n_spans,
            "n_anomalous_spans": n_anomalous,
        }

    return scores


def twist_scores_to_summary(
    scores: Dict[str, Dict[str, float]],
    top_k: int = 5,
) -> str:
    """Converts TWIST scores to a human-readable summary string suitable
    for inclusion in the evidence summary passed to the Coordinator LLM.
    Sorted by twist_score descending (most suspect service first)."""
    if not scores:
        return ""
    sorted_svcs = sorted(scores.items(), key=lambda x: x[1]["twist_score"], reverse=True)
    lines = ["=== TWIST Trace Anomaly Scores (higher = more suspect) ==="]
    for svc, sc in sorted_svcs[:top_k]:
        lines.append(
            f"  {svc}: twist={sc['twist_score']:.3f} "
            f"(self_anomaly={sc['c1_self_anomaly']:.2f}, "
            f"trace_impact={sc['c2_trace_impact']:.2f}, "
            f"blast_radius={sc['c3_blast_radius']:.2f}, "
            f"delay_severity={sc['c4_delay_severity']:.2f}) "
            f"[{sc['n_anomalous_spans']}/{sc['n_spans']} anomalous spans]"
        )
    return "\n".join(lines)


def twist_top_entity(
    scores: Dict[str, Dict[str, float]],
    name_index: Optional[Dict] = None,
) -> Optional[Tuple[str, float]]:
    """Returns (entity_id_or_name, twist_score) for the highest-scoring
    service. Used to contribute to entity re-ranking alongside the
    graph-based score."""
    if not scores:
        return None
    best_svc = max(scores, key=lambda s: scores[s]["twist_score"])
    best_score = scores[best_svc]["twist_score"]
    # Try to resolve to entity_id via name_index for consistency
    if name_index:
        from data.loader import resolve_entity_by_name
        eid = resolve_entity_by_name(best_svc, name_index)
        if eid:
            return eid, best_score
    return best_svc, best_score


def twist_scores_to_observations(
    scores: Dict[str, Dict[str, float]],
    name_index: Optional[Dict] = None,
    top_n: Optional[int] = None,
) -> List[Observation]:
    """Proposal 2 (TWIST-to-Text Evidence Synthesis): converts each service's
    already-computed TWIST c1..c4 scores into one short natural-language
    Observation, so this quantitative-only signal becomes retrievable by the
    SAME semantic (vector) search the pipeline already runs over
    logs/metrics/events -- instead of being visible to the LLM only as a
    fixed top-5 text block (twist_scores_to_summary) or not at all.

    Why this matters: build_index_from_observations() only embeds
    ("logs", "metrics", "events") by default -- raw trace spans never enter
    the vector index, so a query like the alert text or its keywords can
    never semantically retrieve "service X looks like the origin" UNLESS
    that exact wording happens to appear in a log/metric/event observation.
    These synthesized sentences give the trace-derived anomaly profile a
    textual form that CAN be matched by query embeddings, closing that gap
    without training anything and without touching fault-type reasoning
    (this only adds evidence text; it does not rank or exclude candidates).

    Deliberately a NEW modality tag ("twist_synth") rather than "traces" --
    keeps these synthesized rows distinguishable from raw span observations
    everywhere downstream (evidence_count_* breakdowns, retrieval stats),
    and makes the with/without ablation a one-line change to which
    modalities build_index_from_observations() is given.

    top_n: if given, only synthesize for the top_n highest twist_score
    services (by default all scored services -- RCA100 cases have at most a
    few dozen services, so this is rarely a real cap; it exists mainly so a
    pathological case can't blow up the embedding budget with 1:1 sentences
    for every service).
    """
    if not scores:
        return []
    sorted_svcs = sorted(scores.items(), key=lambda x: x[1]["twist_score"], reverse=True)
    if top_n:
        sorted_svcs = sorted_svcs[:top_n]

    resolve_entity_by_name = None
    if name_index:
        from data.loader import resolve_entity_by_name  # local import: avoids circular import,
                                                          # matches twist_top_entity()'s own pattern

    observations: List[Observation] = []
    for svc, sc in sorted_svcs:
        eid = None
        if name_index:
            eid = resolve_entity_by_name(svc, name_index)
        text = (
            f"[TWIST trace anomaly profile] {svc}: composite twist_score={sc['twist_score']:.3f}. "
            f"Self-anomaly (own spans anomalous): {sc['c1_self_anomaly']:.2f}. "
            f"Trace impact (share of anomalous traces it appears in): {sc['c2_trace_impact']:.2f}. "
            f"Blast radius (mean downstream services called): {sc['c3_blast_radius']:.2f}. "
            f"Delay severity (latency deviation magnitude): {sc['c4_delay_severity']:.2f}. "
            f"Based on {sc['n_anomalous_spans']} anomalous spans out of {sc['n_spans']} observed."
        )
        observations.append(Observation(
            entity_id=eid or svc,
            timestamp=None,
            modality="twist_synth",
            text=text,
            payload={"service": svc, "synthesized": True, **sc},
            source_file="twist_scoring.py",
        ))
    return observations


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0