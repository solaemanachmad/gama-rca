"""AIOps2025 fault taxonomy as fingerprints (PREREGISTRATION_AIOPS2025.md A5). Active only with DATASET_PROFILE=aiops2025.

Written from the published type definitions (Cai et al. 2026, Table 2; the release README's injection list) and the Chaos-Mesh
documentation, over the signal vocabulary of the frozen core (pipeline/fault_calibrated.py) plus one family, `net` (network bytes /
packets), used only by this profile. Roles as on RCA100: key (one must hold; "a&b" both), sup (support), may (possible consequence,
absence uninformative), con (contradiction). Written without AIOps2025 labels; fault-type counts are not used (uniform priors).

Deviations from the A5 table, all forced by the frozen vocabulary (no signal 'msg_error' exists in the core: the log family 'error' is
not an observation of entity_signals): jvm-exception, erroneous-code and io-fault drop 'msg_error'. Pairs the telemetry cannot separate
are resolved by the order of this table (general before special), as fixed in A5.
"""
from collections import defaultdict

LEVEL = defaultdict(lambda: "service", {"node_cpu": "node", "node_memory": "node", "node_disk": "node",
                                        "node_network_delay": "node", "node_network_loss": "node"})
RESIDUAL = ()                                    # no fallback type in this taxonomy: uniform priors per entity
FALLBACK = {"node": "node_cpu", "service": "erroneous-code"}   # entity decided but no typed hypothesis testable (decide._fallback_type)

FINGERPRINTS = {
    # resource stress (StressChaos / JVMChaos)
    "cpu_stress":            dict(key=["svc_cpu_rise"], sup=["svc_latency_rise"], con=["svc_request_rise"]),
    "memory_stress":         dict(key=["svc_mem_rise"], sup=["svc_latency_rise"], con=[]),
    "jvm-cpu-stress":        dict(key=["svc_cpu_rise"], sup=["svc_latency_rise", "msg_gc"], con=["svc_request_rise"]),
    "jvm-gc":                dict(key=["msg_gc", "svc_gc_rise"], sup=["svc_latency_rise", "svc_cpu_rise"], con=[]),
    "jvm-latency":           dict(key=["svc_latency_rise"], sup=["trace_slow_origin"], con=["svc_cpu_rise"]),
    "jvm-exception":         dict(key=["svc_error_rise", "http5xx"], sup=["error_spans", "trace_err_origin"], con=[]),
    # pod lifecycle (PodChaos)
    "pod_failure":           dict(key=["svc_request_drop", "msg_refused"], sup=["trace_err_origin", "http503"], may=["svc_net_drop"],
                                  con=["svc_cpu_rise"]),
    "pod_kill":              dict(key=["svc_request_drop", "msg_refused"], sup=["trace_err_origin", "svc_restart_rise"], con=[]),
    # network (NetworkChaos / DNSChaos / misconfiguration)
    "network_delay":         dict(key=["svc_latency_rise"], sup=["trace_slow_origin", "msg_timeout"], con=["svc_cpu_rise"]),
    "network_loss":          dict(key=["msg_timeout", "svc_error_rise"], sup=["trace_err_origin", "svc_latency_rise"], con=["svc_cpu_rise"]),
    "network_corrupt":       dict(key=["svc_error_rise", "msg_timeout"], sup=["trace_err_origin", "svc_latency_rise"], con=["svc_cpu_rise"]),
    "dns-error":             dict(key=["msg_dns"], sup=["svc_error_rise", "trace_err_origin"], con=[]),
    "target_port_misconfig": dict(key=["msg_refused", "trace_err_origin"], sup=["svc_request_drop", "http503"], con=[]),
    # code / IO (HTTPChaos, JVMChaos, IOChaos)
    "erroneous-code":        dict(key=["svc_error_rise", "http5xx"], sup=["error_spans", "trace_err_origin"], con=[]),
    "io-fault":              dict(key=["svc_disk_rise"], sup=["svc_latency_rise"], con=[]),
    # node level
    "node_cpu":              dict(key=["node_cpu_rise"], sup=[], con=[]),
    "node_memory":           dict(key=["node_mem_rise"], sup=[], con=[]),
    "node_disk":             dict(key=["node_disk_rise"], sup=[], con=[]),
    "node_network_delay":    dict(key=["node_net_drop", "node_net_rise"], sup=[], con=["node_cpu_rise"]),
    "node_network_loss":     dict(key=["node_net_drop", "node_net_rise"], sup=[], con=["node_cpu_rise"]),
}

# One-line definitions shown to the LLM roles (typing shortlist, narrator, evidence-only reference arm), written like the RCA100 ones
# (data/taxonomy.py FAULT_DEFINITIONS): from the type names and the Chaos-Mesh fault kinds only, not from any AIOps2025 case or label.
DEFINITIONS = {
    "cpu_stress": "A stress process pins the CPU of a service's pod (StressChaos), slowing its request handling.",
    "memory_stress": "A stress process allocates memory in a service's pod (StressChaos), raising its memory use towards the limit.",
    "jvm-cpu-stress": "CPU stress injected inside the JVM of a Java service (JVMChaos), raising its CPU use and latency.",
    "jvm-gc": "Garbage collection is triggered repeatedly inside the JVM of a Java service (JVMChaos), stalling requests.",
    "jvm-latency": "Latency injected into a method of a Java service (JVMChaos): its calls become slow without a resource rise.",
    "jvm-exception": "A method of a Java service throws an injected exception (JVMChaos), so its requests fail with errors.",
    "pod_failure": "A service's pod is made unavailable for a period (PodChaos pod-failure): requests to it are refused or drop.",
    "pod_kill": "A service's pod is killed and restarted (PodChaos pod-kill): a short outage followed by a restart.",
    "network_delay": "Added latency on a service's network (NetworkChaos delay): its calls and responses become slow.",
    "network_loss": "Packet loss on a service's network (NetworkChaos loss): timeouts, retries and errors on its calls.",
    "network_corrupt": "Packet corruption on a service's network (NetworkChaos corrupt): failed or retried calls and errors.",
    "dns-error": "DNS resolution fails for a service (DNSChaos): it cannot resolve the names of its dependencies.",
    "target_port_misconfig": "A Service's target port is misconfigured: connections to the service are refused although its pods run.",
    "erroneous-code": "Faulty application code (or injected HTTP faults) makes a service return errors, e.g. HTTP 5xx.",
    "io-fault": "Disk I/O of a service's pod is delayed or fails (IOChaos), slowing reads and writes.",
    "node_cpu": "CPU stress on a host node: all pods on the node compete for CPU.",
    "node_memory": "Memory stress on a host node: the node's memory is exhausted for the pods on it.",
    "node_disk": "Disk stress on a host node: high disk I/O on the node slows the pods on it.",
    "node_network_delay": "Added network latency on a host node: traffic of all pods on the node is slowed.",
    "node_network_loss": "Packet loss on a host node: traffic of all pods on the node suffers timeouts and errors.",
}
assert set(DEFINITIONS) == set(FINGERPRINTS)

# the `net` family: signals of this profile only (appended to fault_calibrated.NODE_SIGNALS / SVC_SIGNALS at import)
NODE_EXTRA = [("node_net_rise", ("net",), "up"), ("node_net_drop", ("net",), "down")]
SVC_EXTRA = [("svc_net_rise", ("net",), "up"), ("svc_net_drop", ("net",), "down")]


def phrase(t: str) -> str:
    """The taxonomy phrase written first in the answer's reason (Type Accuracy is keyword-matched): 'jvm-cpu-stress' -> 'jvm cpu stress'."""
    return t.replace("_", " ").replace("-", " ")
