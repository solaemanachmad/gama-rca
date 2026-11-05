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

# the `net` family: signals of this profile only (appended to fault_calibrated.NODE_SIGNALS / SVC_SIGNALS at import)
NODE_EXTRA = [("node_net_rise", ("net",), "up"), ("node_net_drop", ("net",), "down")]
SVC_EXTRA = [("svc_net_rise", ("net",), "up"), ("svc_net_drop", ("net",), "down")]


def phrase(t: str) -> str:
    """The taxonomy phrase written first in the answer's reason (Type Accuracy is keyword-matched): 'jvm-cpu-stress' -> 'jvm cpu stress'."""
    return t.replace("_", " ").replace("-", " ")
