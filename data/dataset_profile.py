"""Dataset profile: the ONLY dataset-specific knowledge of the statistical core (v5, PREREGISTRATION_V5.md amendment 4).

GAMA-RCA's inference (calibrated evidence, Bayesian model, EM propagation, onset, trace structure, LLM sensor) is dataset-agnostic. To
run it on another benchmark, adapt the items below and the data loader (data/loader.py: file names and column names of metrics, logs,
traces, events, alerts and topology), and map that benchmark's fault taxonomy to fingerprints (pipeline/fault_calibrated.FINGERPRINTS).
Nothing here is learned from labels.

RCA100 (OpenTelemetry Demo on Alibaba Cloud ACK, UModel entities) values:"""
import re

# metric name -> signal family (first match wins, lower-case metric name). RCA100 APM exports the request count as 'workload'.
METRIC_FAMILY_PATTERNS = [
    ("throttle", r"throttl"), ("gc", r"gc"), ("thread", r"thread"), ("error", r"error|exception|fail"),
    ("latency", r"latency|(^|_)rt($|_)|duration|response_time|elapsed"), ("replica", r"replica"), ("restart", r"restart"),
    ("ready", r"ready"), ("pods", r"pod.*(running|count)|running.*pod"), ("disk", r"disk|(^|_)io($|_)|iops|(^|_)fs_"),
    ("cpu", r"cpu"), ("mem", r"mem|heap"), ("request", r"request|qps|workload|(^|_)count$"),
]
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
