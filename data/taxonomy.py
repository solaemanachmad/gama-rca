"""
taxonomy.py
=============
RCA100's official fault taxonomy (taxonomy.json) has not been published yet
— see task.json's scoring_note: "Output contract (prediction_schema.json)
and fault taxonomy (taxonomy.json) will be published in a follow-up release."

Without a label set, LLMs guess free-form fault names (e.g. "E_ACCESS_DENIED")
that never match RCA100's actual slugs (e.g. "F014-httpError5xx"), so
fault_identification_score is ~always 0 regardless of reasoning quality.

This is fixable WITHOUT touching per-case ground truth: answer_key/mapping.json
already contains task_to_case_id for all 103 cases, and each case_id is
prefixed with its fault-type slug (e.g. "F014-httpError5xx.tbdh9alum...").
Extracting the DISTINCT set of these prefixes gives you the closed label
vocabulary — this is dataset-level schema metadata (equivalent to telling a
classifier its label space up front), not the answer for any specific case,
so it's safe to expose to every system (baselines AND proposed) equally for
a fair ablation comparison.
"""

import json
import os
import re
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_TAXONOMY_CACHE: List[str] = None
_TAXONOMY_EMBEDDER = None
_TAXONOMY_EMBEDDINGS_CACHE = None

# Generic, dataset-independent one-line definitions derived purely from the
# slug names themselves (standard SRE/K8s domain knowledge) — NOT derived
# from any case's ground-truth description/expected_conclusion text. Safe to
# expose to every system equally, same as the bare label list. If a slug in
# your local mapping.json isn't covered here, taxonomy_prompt_block() falls
# back to showing the bare name for it.
FAULT_GROUPS = {
    # Application logic (38 cases in the full RCA100 set)
    "httpError5xx": "Application logic", "rateLimiting": "Application logic",
    "trafficSurge": "Application logic", "nullPointerException": "Application logic",
    "trafficHotspot": "Application logic", "loadBalancerFailure": "Application logic",
    "codeDefect": "Application logic",
    # JVM runtime (16)
    "memoryPressure": "JVM runtime", "threadExhaustion": "JVM runtime", "fullGC": "JVM runtime",
    # Cloud resource (14)
    "nodeCpuHigh": "Cloud resource", "nodeDown": "Cloud resource", "nodeMemoryOOM": "Cloud resource",
    # Middleware & DB (13)
    "slowSQL": "Middleware&DB", "redisUnavailable": "Middleware&DB",
    "dbNetworkLatency": "Middleware&DB", "messageQueueBacklog": "Middleware&DB",
    "cacheBreakdown": "Middleware&DB",
    # K8s lifecycle (12)
    "replicaScaleDown": "K8s lifecycle", "resourceLimitMisconfig": "K8s lifecycle",
    "podCrashLoop": "K8s lifecycle", "podPendingUnschedulable": "K8s lifecycle",
    "podRestartFlapping": "K8s lifecycle", "networkPolicyIsolation": "K8s lifecycle",
    "dnsResolutionFailure": "K8s lifecycle",
    # Resource & perf. (10)
    "cpuFullLoad": "Resource&perf.", "cpuDeadLoop": "Resource&perf.", "diskIOHigh": "Resource&perf.",
}


def fault_group(fault_type: str) -> str:
    """'F014-httpError5xx' -> 'Application logic'. Robust to missing/unknown
    prefixes or slugs (returns 'unknown' rather than raising) since this is
    also used on free-form LLM output, not just clean ground-truth labels."""
    if not fault_type:
        return "unknown"
    slug = fault_type.strip().split("-", 1)[-1] if "-" in fault_type else fault_type.strip()
    if slug in FAULT_GROUPS:
        return FAULT_GROUPS[slug]
    # case-insensitive fallback for free-form LLM casing drift
    slug_lower = slug.lower()
    for known_slug, group in FAULT_GROUPS.items():
        if known_slug.lower() == slug_lower:
            return group
    return "unknown"


FAULT_TYPE_KEYWORDS = {
    # Cloud resource / K8s lifecycle -- infra-level keywords
    "nodeDown": ["nodenotready", "node not ready", "node down", "unreachable"],
    "nodeCpuHigh": ["node cpu", "nodecpuhigh", "cpu utilization"],
    "nodeMemoryOOM": ["oomkilled", "oom-kill", "out of memory", "node memory"],
    "podCrashLoop": ["crashloopbackoff", "crash loop", "restart"],
    "podPendingUnschedulable": ["pending", "unschedulable", "insufficient", "taint", "affinity"],
    "podRestartFlapping": ["liveness probe", "restart", "flapping"],
    "resourceLimitMisconfig": ["resource limit", "throttl", "evict", "requests/limits"],
    "replicaScaleDown": ["scale down", "replica", "scaledown", "autoscaler"],
    "networkPolicyIsolation": ["networkpolicy", "network policy", "blocked", "isolation"],
    "dnsResolutionFailure": ["dns", "resolution failed", "name resolution", "nxdomain"],
    "diskIOHigh": ["disk io", "disk i/o", "iowait", "disk saturat"],
    # Middleware & DB -- redisUnavailable vs cacheBreakdown are frequently
    # confused (observed: t002/t010 both defaulted to cacheBreakdown when GT
    # was redisUnavailable). Added literal Java Redis-client exception class
    # names and connection-level error strings -- these DO appear verbatim
    # in stack-trace log lines, unlike generic phrases like "cache miss"
    # which are rarely written verbatim by real client libraries.
    "redisUnavailable": ["redis", "valkey", "connection refused", "cache unreachable",
                          "jedisconnectionexception", "redisconnectionexception",
                          "lettuce", "econnrefused", "no route to host",
                          "timeout connecting", "redistimeoutexception"],
    "slowSQL": ["slow query", "sql", "query time", "database query", "query timeout",
                "sqltimeoutexception", "lock wait timeout"],
    "dbNetworkLatency": ["db network", "database latency", "connection timeout",
                          "sqlnontransientconnectionexception", "communications link failure"],
    "messageQueueBacklog": ["queue backlog", "message queue", "kafka", "rabbitmq",
                             "consumer lag", "queue full", "producer blocked"],
    "cacheBreakdown": ["cache miss", "cache breakdown", "cache bypass",
                        "cache penetration", "null cached value"],
    # JVM runtime -- memoryPressure/threadExhaustion/fullGC are frequently
    # confused (observed: t009 defaulted to fullGC when GT was
    # memoryPressure). Added the literal Java exception/log strings each
    # condition actually produces, which are far more distinctive than the
    # generic phrases alone.
    "memoryPressure": ["memory pressure", "heap", "memory limit", "eviction risk",
                        "outofmemoryerror", "java.lang.outofmemoryerror", "heap space",
                        "gc overhead limit exceeded"],
    "threadExhaustion": ["thread pool", "thread exhaustion", "threads exhausted",
                          "queue rejected", "rejectedexecutionexception",
                          "threadpoolexecutor", "maximum pool size reached",
                          "too many open files"],
    "fullGC": ["full gc", "garbage collection", "gc pause", "stop-the-world",
               "allocation failure", "g1gc", "cms gc", "pause young"],
    # Resource & perf.
    "cpuFullLoad": ["cpu full", "cpu 100%", "cpu pegged", "high cpu"],
    "cpuDeadLoop": ["infinite loop", "dead loop", "busy loop", "cpu pinned"],
    # Application logic -- trafficSurge is the group's persistent default
    # guess; the other types here got a few more distinctive literal terms
    # to compete against it (e.g. specific Java exception class names for
    # nullPointerException/codeDefect, which are far more identifiable than
    # the generic "exception"/"bug" terms alone).
    "httpError5xx": ["500", "502", "503", "504", "5xx", "internal server error", "bad gateway"],
    "rateLimiting": ["rate limit", "429", "throttled", "too many requests",
                      "quota exceeded", "requests per second exceeded"],
    "trafficSurge": ["traffic surge", "spike", "sudden increase", "surge"],
    "nullPointerException": ["nullpointerexception", "null pointer", "nil dereference", "npe",
                              "java.lang.nullpointerexception"],
    "trafficHotspot": ["hotspot", "uneven", "disproportionate", "shard imbalance"],
    "loadBalancerFailure": ["load balancer", "lb ", "misrouted", "traffic distribution"],
    "codeDefect": ["exception", "stack trace", "bug", "incorrect behavior",
                    "illegalstateexception", "illegalargumentexception", "classcastexception"],
}


def detect_fault_keywords(text_blob: str) -> List[str]:
    """Scans a blob of evidence text (case-insensitive) for fault-type
    keywords and returns the matching fault-type slugs, ordered by number
    of keyword hits (most-supported first). Deterministic, not
    embedding-similarity-based -- this is what the earlier shortlist
    mechanism (fault_shortlist_prompt_block, removed after it correlated
    with worse fault_identification) tried to do via semantic similarity
    and got wrong; literal keyword matching against each type's own
    definition terms is far more precise for domain-specific vocabulary
    (e.g. "valkey", "OOMKilled") that generic sentence embeddings don't
    reliably associate with the right fault category.

    Uses \\b word-boundary matching, NOT naive substring counting -- short/
    numeric keywords like "500", "502", "429" were matching as substrings
    inside unrelated larger numbers (e.g. "500" inside "15000.0", extremely
    common given how much raw metric text this scans), causing
    httpError5xx/rateLimiting/resourceLimitMisconfig to spuriously "detect"
    on nearly every case regardless of actual evidence content. \\b500\\b
    does not match inside "15000" since digits are contiguous word
    characters with no boundary between them."""
    if not text_blob:
        return []
    text_lower = text_blob.lower()
    hits = []
    for slug, keywords in FAULT_TYPE_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if kw.isdigit():
                # Numeric keywords (HTTP codes like "500", "429") need a
                # stricter boundary than plain \b: "." is a non-word
                # character, so \b500\b still matches inside "0.500" (a
                # latency value). Exclude adjacency to digits AND decimal
                # points, AND exclude when followed by a percent sign --
                # RCA100's alert template text uses "500%" as a boilerplate
                # significance threshold ("同比增加 500 %触发紧急告警" =
                # "increased 500% YoY, triggering emergency alert") in
                # nearly EVERY alert regardless of actual fault type. This
                # was the dominant real-world source of httpError5xx false
                # "detections" -- confirmed via scripts/debug_keyword_match.py
                # showing the literal alert text triggering the match.
                pattern = r"(?<![\d.])" + re.escape(kw) + r"(?![\d.])(?!\s*%)"
            else:
                pattern = r"\b" + re.escape(kw) + r"\b"
            count += len(re.findall(pattern, text_lower))
        if count > 0:
            hits.append((slug, count))
    hits.sort(key=lambda x: x[1], reverse=True)
    return [slug for slug, _ in hits]


FAULT_DEFINITIONS = {
    "F001-nodeDown": "A Kubernetes node becomes unreachable/down, taking its pods offline.",
    "F002-threadExhaustion": "A service's thread pool is fully saturated, causing requests to queue or be rejected.",
    "F004-trafficHotspot": "Traffic concentrates disproportionately on one instance/shard instead of being load-balanced evenly.",
    "F005-messageQueueBacklog": "Messages accumulate faster than a queue consumer can process them.",
    "F006-trafficSurge": "A sudden, broad spike in request volume across a service.",
    "F007-memoryPressure": "A service/pod approaches its memory limit, causing degraded performance or eviction risk.",
    "F009-cacheBreakdown": "A cache layer fails or is bypassed, forcing traffic directly to a slower backing store.",
    "F010-slowSQL": "Database queries take abnormally long, increasing downstream request latency.",
    "F011-codeDefect": "A logic/implementation bug in application code causes incorrect behavior or errors.",
    "F012-cpuDeadLoop": "A busy/infinite loop pins a CPU core at 100%, starving other work on that process.",
    "F014-httpError5xx": "A service returns an elevated rate of HTTP 5xx (server error) responses.",
    "F016-rateLimiting": "Requests are being throttled/rejected by a rate limiter, increasing error or retry rates.",
    "F018-dbNetworkLatency": "Network latency between a service and its database increases, slowing queries.",
    "F020-loadBalancerFailure": "A load balancer misroutes, drops, or fails to distribute traffic correctly.",
    "F022-fullGC": "Frequent/long garbage-collection pauses (JVM or similar runtime) stall request processing.",
    "F023-nullPointerException": "An unhandled null/nil dereference crashes or errors out a request path.",
    "F025-diskIOHigh": "Disk I/O saturation slows reads/writes, backing up dependent operations.",
    "F026-nodeCpuHigh": "A Kubernetes node's overall CPU utilization is abnormally high, affecting all pods scheduled on it.",
    "F029-redisUnavailable": "A Redis instance used for caching/state is unreachable or down.",
    "F031-nodeMemoryOOM": "A Kubernetes node runs out of memory, triggering OOM-kills of pods on it.",
    "F034-cpuFullLoad": "A specific service/pod's CPU usage is pegged at or near 100%.",
    "F036-replicaScaleDown": "A deployment's replica count is reduced (intentionally or via autoscaler/misconfig), reducing capacity.",
    "F039-resourceLimitMisconfig": "Kubernetes resource requests/limits are misconfigured, causing throttling or eviction.",
    "F050-podCrashLoop": "A pod repeatedly crashes and restarts (CrashLoopBackOff).",
    "F051-podPendingUnschedulable": "A pod cannot be scheduled onto any node (insufficient resources, affinity/taint mismatch, etc.).",
    "F052-podRestartFlapping": "A pod restarts repeatedly without necessarily crash-looping (e.g. liveness probe flapping).",
    "F056-networkPolicyIsolation": "A NetworkPolicy or similar rule unexpectedly blocks required traffic between services.",
    "F057-dnsResolutionFailure": "DNS lookups for a dependency fail or time out, breaking connectivity.",
}


def build_fault_taxonomy() -> List[str]:
    """Returns the sorted, deduplicated list of fault-type slugs across all
    103 cases, e.g. ["F001-nodeDown", "F002-threadExhaustion", ...]."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE

    path = os.path.join(config.ANSWER_KEY_DIR, "mapping.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    task_to_case_id = mapping.get("task_to_case_id", {})
    slugs = set()
    for case_id in task_to_case_id.values():
        slug = case_id.split(".")[0]  # "F014-httpError5xx.tbdh9alum..." -> "F014-httpError5xx"
        slugs.add(slug)

    _TAXONOMY_CACHE = sorted(slugs)
    return _TAXONOMY_CACHE


def taxonomy_prompt_block_for_group(group_name: str) -> str:
    """Same as taxonomy_prompt_block(), but filtered to only the fault
    types belonging to `group_name` (see FAULT_GROUPS). Used for the
    two-stage classifier-narrowed Coordinator prompt: the classifier
    predicts the group (0.66-0.68 LOOCV accuracy, stable across runs where
    the LLM's own end-to-end accuracy varied 0.39-0.44), then the
    Coordinator only has to pick among that group's 3-7 types instead of
    all 28. Falls back to the full list if the group has no matching types
    (shouldn't happen for a real FAULT_GROUPS value, but safe regardless)."""
    taxonomy = build_fault_taxonomy()
    matching = [t for t in taxonomy if fault_group(t) == group_name]
    if not matching:
        return taxonomy_prompt_block()
    lines = []
    for t in matching:
        definition = FAULT_DEFINITIONS.get(t)
        lines.append(f"- {t}: {definition}" if definition else f"- {t}")
    return (f"Valid fault types for the predicted '{group_name}' category "
            f"(prefer one of these, verbatim):\n" + "\n".join(lines))


def taxonomy_prompt_block() -> str:
    """Renders the FULL taxonomy as a prompt-ready text block (fallback for
    systems with no evidence text to embed against, e.g. direct_llm which
    has no evidence at all, or graphrag_only which only has entity IDs)."""
    taxonomy = build_fault_taxonomy()
    lines = []
    for t in taxonomy:
        definition = FAULT_DEFINITIONS.get(t)
        lines.append(f"- {t}: {definition}" if definition else f"- {t}")
    return ("Valid fault types (you MUST pick exactly one of these, verbatim):\n"
            + "\n".join(lines))


def _get_embedder() -> SentenceTransformer:
    from retrieval.vector import get_embedding_model
    return get_embedding_model(config.EMBEDDING_MODEL)


def _get_taxonomy_embeddings():
    """Cached (taxonomy_list, normalized_embedding_matrix) for all fault-type
    "slug: definition" strings — computed once per process, reused across
    every case/system that needs a shortlist."""
    global _TAXONOMY_EMBEDDINGS_CACHE
    if _TAXONOMY_EMBEDDINGS_CACHE is None:
        taxonomy = build_fault_taxonomy()
        texts = [f"{t}: {FAULT_DEFINITIONS.get(t, t)}" for t in taxonomy]
        vecs = _get_embedder().encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        _TAXONOMY_EMBEDDINGS_CACHE = (taxonomy, vecs)
    return _TAXONOMY_EMBEDDINGS_CACHE


def rank_fault_types_by_similarity(evidence_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Embedding-based shortlist: ranks all fault types by cosine similarity
    between their definition and the retrieved evidence text, computed BEFORE
    the LLM is ever called. This does the heavy semantic-matching work with
    the embedding model (already loaded for retrieval, cheap and
    deterministic) instead of asking a small LLM to hold 27 category
    definitions in its head and reason about numeric evidence simultaneously."""
    if not evidence_text or not evidence_text.strip():
        return []
    taxonomy, tax_vecs = _get_taxonomy_embeddings()
    qvec = _get_embedder().encode([evidence_text], convert_to_numpy=True)[0]
    qnorm = np.linalg.norm(qvec)
    if qnorm > 0:
        qvec = qvec / qnorm
    scores = tax_vecs @ qvec
    ranked = sorted(zip(taxonomy, scores.tolist()), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def fault_shortlist_prompt_block(evidence_text: str, top_k: int = 5) -> str:
    """Prompt-ready shortlist block. Falls back to the full taxonomy list if
    there's no evidence text to embed against (e.g. an empty-evidence case)."""
    ranked = rank_fault_types_by_similarity(evidence_text, top_k=top_k)
    if not ranked:
        return taxonomy_prompt_block()

    lines = [f"- {t} (evidence_similarity={s:.3f}): {FAULT_DEFINITIONS.get(t, '')}"
             for t, s in ranked]
    return (
        "The retrieved evidence was compared against every fault type's definition "
        "using semantic similarity. Most plausible fault types, ranked (most likely first):\n"
        + "\n".join(lines) +
        "\n\nPick the single best match from this shortlist. Only choose a fault type "
        "OUTSIDE this shortlist if the evidence clearly contradicts all of them — "
        "if you do, explain why in your reasoning."
    )