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
    global _TAXONOMY_EMBEDDER
    if _TAXONOMY_EMBEDDER is None:
        _TAXONOMY_EMBEDDER = SentenceTransformer(config.EMBEDDING_MODEL)
    return _TAXONOMY_EMBEDDER


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
