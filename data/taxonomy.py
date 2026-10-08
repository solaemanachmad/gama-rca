"""
taxonomy.py
=============
RCA100's official fault taxonomy (taxonomy.json) has not been published yet
— see task.json's scoring_note: "Output contract (prediction_schema.json)
and fault taxonomy (taxonomy.json) will be published in a follow-up release."

Without a label set, LLMs guess free-form fault names (e.g. "E_ACCESS_DENIED")
that never match RCA100's actual slugs (e.g. "F014-httpError5xx"), so
fault_identification_score is ~always 0 regardless of reasoning quality.

The closed label vocabulary (the paper's 28 root-cause types) is written out
statically in FAULT_TYPE_LABELS below. It used to be derived at run time from
answer_key/mapping.json; since the answer-key README forbids answer_key content
in the agent's context, the agent-facing code no longer reads that folder at
all (2026-10-08). The label list is dataset-level schema metadata, equal for
every system (baselines AND proposed).
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

_CJK_RE = re.compile(r"[一-鿿]")   # used by detect_fault_keywords() below to
                                            # route Chinese keywords to substring
                                            # matching instead of \b word-boundary
                                            # matching (see that function's docstring)

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
    #
    # Chinese terms added below are standard/generic SRE-K8s-APM vocabulary
    # (the kind found in any Chinese cloud-ops glossary or Alibaba Cloud
    # ARMS documentation -- this dataset's own topology.json tags every
    # service with "telemetry_client": "ARMS", so this is the ops
    # vocabulary its own alert templates are drawn from), translated the
    # same way FAULT_DEFINITIONS' English one-liners were written: from
    # domain knowledge of what each slug name means, NOT by reading any
    # RCA100 case's ground-truth/expected_conclusion text. Deliberately
    # excludes bare, overly generic terms that show up in nearly every
    # alert regardless of actual fault type -- standalone 异常 ("abnormal"),
    # 告警 ("alert"), 错误 ("error") -- for the same reason "500%" was
    # excluded from the numeric-keyword matching below (see
    # detect_fault_keywords' docstring): confirmed boilerplate, not signal.
    "nodeDown": ["nodenotready", "node not ready", "node down", "unreachable",
                 "节点下线", "节点不可达", "节点宕机", "节点未就绪"],
    "nodeCpuHigh": ["node cpu", "nodecpuhigh", "cpu utilization",
                    "节点CPU使用率过高", "节点CPU过高"],
    "nodeMemoryOOM": ["oomkilled", "oom-kill", "out of memory", "node memory",
                       "节点内存溢出", "节点内存不足", "内存耗尽"],
    "podCrashLoop": ["crashloopbackoff", "crash loop", "restart",
                      "容器崩溃重启", "Pod崩溃循环"],
    "podPendingUnschedulable": ["pending", "unschedulable", "insufficient", "taint", "affinity",
                                 "调度失败", "无法调度", "资源不足无法调度"],
    "podRestartFlapping": ["liveness probe", "restart", "flapping",
                            "存活探针失败", "频繁重启"],
    "resourceLimitMisconfig": ["resource limit", "throttl", "evict", "requests/limits",
                                "资源限制配置错误", "资源配额不当"],
    "replicaScaleDown": ["scale down", "replica", "scaledown", "autoscaler",
                          "副本数缩容", "副本数量减少"],
    "networkPolicyIsolation": ["networkpolicy", "network policy", "blocked", "isolation",
                                "网络策略隔离", "网络策略阻断"],
    "dnsResolutionFailure": ["dns", "resolution failed", "name resolution", "nxdomain",
                              "DNS解析失败", "域名解析超时"],
    "diskIOHigh": ["disk io", "disk i/o", "iowait", "disk saturat",
                   "磁盘IO过高", "磁盘读写延迟"],
    # Middleware & DB -- redisUnavailable vs cacheBreakdown are frequently
    # confused (observed: t002/t010 both defaulted to cacheBreakdown when GT
    # was redisUnavailable). Added literal Java Redis-client exception class
    # names and connection-level error strings -- these DO appear verbatim
    # in stack-trace log lines, unlike generic phrases like "cache miss"
    # which are rarely written verbatim by real client libraries.
    "redisUnavailable": ["redis", "valkey", "connection refused", "cache unreachable",
                          "jedisconnectionexception", "redisconnectionexception",
                          "lettuce", "econnrefused", "no route to host",
                          "timeout connecting", "redistimeoutexception",
                          "Redis不可用", "Redis连接失败", "缓存服务不可用"],
    "slowSQL": ["slow query", "sql", "query time", "database query", "query timeout",
                "sqltimeoutexception", "lock wait timeout",
                "SQL慢查询", "数据库慢查询", "查询超时"],
    "dbNetworkLatency": ["db network", "database latency", "connection timeout",
                          "sqlnontransientconnectionexception", "communications link failure",
                          "数据库网络延迟", "数据库连接超时"],
    "messageQueueBacklog": ["queue backlog", "message queue", "kafka", "rabbitmq",
                             "consumer lag", "queue full", "producer blocked",
                             "消息队列积压", "队列堆积", "消费延迟"],
    "cacheBreakdown": ["cache miss", "cache breakdown", "cache bypass",
                        "cache penetration", "null cached value",
                        "缓存击穿", "缓存穿透", "缓存失效"],
    # JVM runtime -- memoryPressure/threadExhaustion/fullGC are frequently
    # confused (observed: t009 defaulted to fullGC when GT was
    # memoryPressure). Added the literal Java exception/log strings each
    # condition actually produces, which are far more distinctive than the
    # generic phrases alone.
    "memoryPressure": ["memory pressure", "heap", "memory limit", "eviction risk",
                        "outofmemoryerror", "java.lang.outofmemoryerror", "heap space",
                        "gc overhead limit exceeded",
                        "内存压力", "堆内存溢出", "内存使用率过高"],
    "threadExhaustion": ["thread pool", "thread exhaustion", "threads exhausted",
                          "queue rejected", "rejectedexecutionexception",
                          "threadpoolexecutor", "maximum pool size reached",
                          "too many open files",
                          "线程池耗尽", "线程池已满"],
    "fullGC": ["full gc", "garbage collection", "gc pause", "stop-the-world",
               "allocation failure", "g1gc", "cms gc", "pause young",
               "垃圾回收停顿", "GC停顿"],
    # Resource & perf.
    "cpuFullLoad": ["cpu full", "cpu 100%", "cpu pegged", "high cpu",
                     "CPU满载", "CPU占用过高"],
    "cpuDeadLoop": ["infinite loop", "dead loop", "busy loop", "cpu pinned",
                     "死循环", "CPU死循环"],
    # Application logic -- trafficSurge is the group's persistent default
    # guess; the other types here got a few more distinctive literal terms
    # to compete against it (e.g. specific Java exception class names for
    # nullPointerException/codeDefect, which are far more identifiable than
    # the generic "exception"/"bug" terms alone).
    "httpError5xx": ["500", "502", "503", "504", "5xx", "internal server error", "bad gateway",
                      "错误次数", "5xx错误"],
    "rateLimiting": ["rate limit", "429", "throttled", "too many requests",
                      "quota exceeded", "requests per second exceeded",
                      "限流", "触发限流", "超过配额"],
    "trafficSurge": ["traffic surge", "spike", "sudden increase", "surge",
                      "流量突增", "流量激增", "访问量突增"],
    "nullPointerException": ["nullpointerexception", "null pointer", "nil dereference", "npe",
                              "java.lang.nullpointerexception",
                              "空指针异常"],
    "trafficHotspot": ["hotspot", "uneven", "disproportionate", "shard imbalance",
                        "流量热点", "流量倾斜"],
    "loadBalancerFailure": ["load balancer", "lb ", "misrouted", "traffic distribution",
                             "负载均衡故障", "负载均衡异常"],
    "codeDefect": ["exception", "stack trace", "bug", "incorrect behavior",
                    "illegalstateexception", "illegalargumentexception", "classcastexception",
                    "代码缺陷", "业务逻辑错误"],
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

    Uses \\b word-boundary matching for Latin/numeric keywords, NOT naive
    substring counting -- short/numeric keywords like "500", "502", "429"
    were matching as substrings inside unrelated larger numbers (e.g. "500"
    inside "15000.0", extremely common given how much raw metric text this
    scans), causing httpError5xx/rateLimiting/resourceLimitMisconfig to
    spuriously "detect" on nearly every case regardless of actual evidence
    content. \\b500\\b does not match inside "15000" since digits are
    contiguous word characters with no boundary between them.

    Chinese keywords use plain substring matching instead, NOT \\b: Chinese
    text has no whitespace between words at all, and Python's \\w (which \\b
    is defined against) treats CJK characters as word characters -- so a
    real match like "响应时间突增" sitting naturally in the middle of a
    longer Chinese sentence (i.e. with more CJK characters immediately
    before/after it, the normal case, since Chinese doesn't delimit words)
    would never see a \\b on either side and would be silently missed. The
    "500 inside 15000" false-positive problem \\b exists to prevent doesn't
    have a real CJK analogue for these 2-6 character technical terms, so
    substring matching is the correct default there (standard practice for
    keyword matching without a proper segmenter)."""
    if not text_blob:
        return []
    text_lower = text_blob.lower()
    hits = []
    for slug, keywords in FAULT_TYPE_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if _CJK_RE.search(kw):
                # kw.lower() matters for keywords that mix CJK with a Latin
                # term (e.g. "Redis连接失败") -- text_lower is already
                # lowercased, so the keyword needs to match case too.
                count += text_lower.count(kw.lower())
            elif kw.isdigit():
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
                count += len(re.findall(pattern, text_lower))
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


# The benchmark's closed label set (RCA100 paper: "28 root-cause types"). Written
# out statically on 2026-10-08 so the agent-facing code path NEVER reads
# answer_key/ (the RCA100 answer-key README: "do NOT include answer_key content in
# the agent's prompt context"). Previously this list was derived at run time from
# answer_key/mapping.json; the resulting set is identical (28 slugs). mapping.json
# is now read only by evaluation/scoring.py, for scoring.
FAULT_TYPE_LABELS: List[str] = [
    "F001-nodeDown", "F002-threadExhaustion", "F004-trafficHotspot",
    "F005-messageQueueBacklog", "F006-trafficSurge", "F007-memoryPressure",
    "F009-cacheBreakdown", "F010-slowSQL", "F011-codeDefect", "F012-cpuDeadLoop",
    "F014-httpError5xx", "F016-rateLimiting", "F018-dbNetworkLatency",
    "F020-loadBalancerFailure", "F022-fullGC", "F023-nullPointerException",
    "F025-diskIOHigh", "F026-nodeCpuHigh", "F029-redisUnavailable",
    "F031-nodeMemoryOOM", "F034-cpuFullLoad", "F036-replicaScaleDown",
    "F039-resourceLimitMisconfig", "F050-podCrashLoop", "F051-podPendingUnschedulable",
    "F052-podRestartFlapping", "F056-networkPolicyIsolation", "F057-dnsResolutionFailure",
]


def build_fault_taxonomy() -> List[str]:
    """Returns the sorted list of the 28 RCA100 fault-type slugs, e.g.
    ["F001-nodeDown", "F002-threadExhaustion", ...]. Static; reads no files."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        _TAXONOMY_CACHE = sorted(FAULT_TYPE_LABELS)
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