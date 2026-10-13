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

from typing import List


_TAXONOMY_CACHE: List[str] = None
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


def knowledge_prompt_block_pos() -> str:
    """Arm KP. 'How to tell similar fault types apart', paraphrased from the agent-facing fault catalog and diagnostic methodology of the
    PoS project (github.com/luoyu100/PoS, benchmarks/RCA100/info.py, MIT licence, (c) 2026 PoS Contributors; paper arXiv:2610.01415), whose
    authors overlap with the RCA100 authors. Public text; not derived from any RCA100 label of ours. Slugs are ours (28 types); PoS's
    extra type slowResource does not occur in RCA100 v1.1 (audit 2026-10-12: all 103 accepted types are in our 28).
    NOTE for the paper: the rateLimiting line keeps PoS's remark that the status can be 403 with throttling wording; disclose it."""
    return (
        "How to tell similar fault types apart. Several types share a symptom and differ in one checkable signal; take the types consistent "
        "with the symptom, verify the discriminating signal of each in the tables, and rule out the nearest alternative before answering.\n"
        "- Application CPU high: F012-cpuDeadLoop = one hot method pins a single core at 100%; F034-cpuFullLoad = CPU high evenly across "
        "cores; F026-nodeCpuHigh = the host NODE cpu is high and several co-located pods degrade together (a node-level fault); "
        "F006-trafficSurge / F004-trafficHotspot = request count or workload rises first (hotspot: one key, product or endpoint).\n"
        "- JVM / memory: F022-fullGC = GC fires continuously at high frequency with long stop-the-world pauses; F007-memoryPressure = memory "
        "near its limit with bursty or occasional GC, the service keeps answering and is not OOM-killed; F031-nodeMemoryOOM = the host's "
        "physical memory is exhausted and the kernel OOM killer acts; F039-resourceLimitMisconfig = the container limit itself is clearly too "
        "low, the pod is OOMKilled or throttled again soon after restart.\n"
        "- Errors returned by a service: F016-rateLimiting = throttling, quota or protective rejection, typically HTTP 429, sometimes a 403 "
        "with throttling wording; F023-nullPointerException = an explicit null-dereference stack; F011-codeDefect = other code bugs "
        "(validation, logic, configuration); F014-httpError5xx = the service clearly returns 5xx and the evidence does not converge on a "
        "more specific mechanism (a fallback type).\n"
        "- Slow dependency calls: F010-slowSQL = long database query spans; F018-dbNetworkLatency = added latency, jitter or loss on the path "
        "to the dependency while the query itself is not slow; F029-redisUnavailable = cache calls fail outright (unreachable, connection "
        "refused); F009-cacheBreakdown = cache misses push a flood of requests to the backend; F005-messageQueueBacklog = consumption falls "
        "behind production, queue grows; F002-threadExhaustion = the thread pool is full, requests queue or time out.\n"
        "- Pods not serving: F036-replicaScaleDown = desired replicas at 0, endpoints empty; F050-podCrashLoop = containers exit right after "
        "start, CrashLoopBackOff, fast-growing restarts; F052-podRestartFlapping = periodic restarts, intermittent availability; "
        "F051-podPendingUnschedulable = pods stuck Pending with a scheduling reason; F001-nodeDown = the whole node is unreachable.\n"
        "- Calls fail while both services run: F056-networkPolicyIsolation = specific service pairs rejected or timing out; "
        "F057-dnsResolutionFailure = name-resolution errors or a wrong address; F020-loadBalancerFailure = Service endpoints broken or the "
        "load balancer misconfigured.\n"
        "- F025-diskIOHigh = abnormal disk IOPS or latency slows reads and writes."
    )
