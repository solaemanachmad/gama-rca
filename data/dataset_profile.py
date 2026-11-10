"""Dataset profile: the ONLY dataset-specific knowledge of the statistical core (v5, PREREGISTRATION_V5.md amendment 4).

GAMA-RCA's inference (calibrated evidence, Bayesian model, EM propagation, onset, trace structure, LLM sensor) is dataset-agnostic. To
run it on another benchmark, adapt the items below and the data loader (data/loader.py: file names and column names of metrics, logs,
traces, events, alerts and topology), and map that benchmark's fault taxonomy to fingerprints (pipeline/fault_calibrated.FINGERPRINTS).
Nothing here is learned from labels.

RCA100 (OpenTelemetry Demo on Alibaba Cloud ACK, UModel entities) values:"""
import re

# metric name -> signal family (first match wins, lower-case metric name). RCA100 APM exports the request count as 'workload'.
METRIC_PATTERNS = [
    ("throttle", r"throttl"), ("gc", r"gc"), ("thread", r"thread"), ("error", r"error|exception|fail"),
    ("latency", r"latency|(^|_)rt($|_)|duration|response_time|elapsed"), ("replica", r"replica"), ("restart", r"restart"),
    ("ready", r"ready"), ("pods", r"pod.*(running|count)|running.*pod"), ("disk", r"disk|(^|_)io($|_)|iops|(^|_)fs_"),
    ("cpu", r"cpu"), ("mem", r"mem|heap"), ("request", r"request|qps|workload|(^|_)count$"),
]
METRIC_FAMILY_PATTERNS = METRIC_PATTERNS   # backward compatibility alias
# workloads of the fault-injection tooling (present in the benchmark, absent in production): excluded from the evidence
EXCLUDED_WORKLOADS = re.compile(r"^(l3reg-|chaosblade|chaos-)", re.I)
# configuration constants exported as metrics (not behaviour): excluded
CONSTANT_METRICS = re.compile(r"_(init|max|committed)_bytes$|core_pool_size|max_pool_size|max_thread_count|desired_replicas", re.I)
# OpenTelemetry span status code of an error (OTLP: 0 unset, 1 ok, 2 error); traces use traceId / spanId / parentSpanId / serviceName /
# duration (ns) / statusCode / attributes (http.status_code)
TRACE_ERROR_STATUS = "2"
# entity types of the topology (UModel). Other benchmarks: derive services from trace serviceName and nodes from host metrics.
SERVICE_ENTITY_TYPES = ("apm.service", "apm.svc")
NODE_ENTITY_TYPES = ("k8s.node",)
SERVICE_TYPES = SERVICE_ENTITY_TYPES
NODE_TYPES = NODE_ENTITY_TYPES

# ------------------------------------------------------------------------------------------------------------------------------
# Profile selection (PREREGISTRATION_AIOPS2025.md A2). DATASET_PROFILE is read once at import; unset = RCA100 (values above,
# unchanged). The AIOps2025 values below are dataset knowledge only (metric names of the release, injection tooling), written
# without AIOps2025 labels. Fingerprints of the AIOps2025 taxonomy: data/aiops2025/taxonomy.py.
# ------------------------------------------------------------------------------------------------------------------------------
import os

PROFILE = os.environ.get("DATASET_PROFILE", "rca100").strip().lower() or "rca100"
assert PROFILE in ("rca100", "aiops2025"), f"unknown DATASET_PROFILE {PROFILE!r}"

if PROFILE == "aiops2025":
    # AIOps2025 metric names (APM: request, response, rrt [ms], rrt_max, error, client_error, server_error, timeout, *_ratio;
    # infra_pod / infra_node / TiDB / TiKV / PD kpi_key names). Added to the RCA100 patterns: rrt -> latency, timeout -> error,
    # network bytes / packets -> net (new family, this profile only), TiKV MB/s and filesystem usage -> disk, *_is_up -> ready.
    METRIC_PATTERNS = [
        ("net", r"network_(?:receive|transmit)"),
        ("throttle", r"throttl"), ("gc", r"(^|_)gc"), ("thread", r"thread"), ("error", r"error|exception|fail|timeout"),
        ("latency", r"latency|(^|_)rr?t($|_)|duration|response_time|elapsed|_wait$|write_stall"), ("replica", r"replica"),
        ("restart", r"restart"), ("ready", r"ready|is_up$"), ("pods", r"pod.*(running|count)|running.*pod"),
        ("disk", r"disk|(^|_)io($|_)|iops|(^|_)fs_|_mbps$|filesystem_usage"),
        ("cpu", r"cpu"), ("mem", r"mem|heap"), ("request", r"request|qps|workload|(^|_)count$"),
    ]
    METRIC_FAMILY_PATTERNS = METRIC_PATTERNS
    # Chaos-Mesh / injection tooling and the k6 load generator are not entities of the system
    EXCLUDED_WORKLOADS = re.compile(r"^(chaos-|chaos-mesh|chaos-daemon|k6)", re.I)
    # capacities, totals, sizes, topology counts and uptime describe configuration or bookkeeping, not behaviour
    CONSTANT_METRICS = re.compile(
        r"memtotal|memavailable|filesystem_(?:free|size)_bytes|capacity|available_size|storage_size|store_size|uptime|"
        r"block_cache_size|leader_primary|(?:region|leader|learner|witness)_count|region_health|abnormal_region|"
        r"store_(?:up|down|low_space|slow|unhealth)_count|sockstat|snapshot_apply|region_pending|_ratio$", re.I)
    # the adapter writes the canonical OTLP status: "2" = error (Jaeger status.code != 0), "1" = ok
    TRACE_ERROR_STATUS = "2"
