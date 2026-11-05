# Pre-registration: zero-shot generalisation of GAMA-RCA to AIOps2025 (DRAFT, 2026-11-03)

Purpose: test whether the method designed on RCA100 transfers to an unseen system (HipsterShop + TiDB + Redis on Kubernetes, Chaos-Mesh
faults, 400 cases; Cai et al. 2026). This file is finalised (dated, committed) BEFORE the first scored run; changes after that are
dated amendments.

## What stays fixed (the method)

The `gama/` package as frozen for RCA100: likelihood model, fingerprint-role semantics (K/U/M/C), constants (THETA_*, ALPHA*, priors,
EM bounds, onset tolerance), two-stage decision, modules M1-M7 and their switches, LLM prompts and settings. No constant is re-tuned.

## What may change (dataset knowledge only; written without AIOps2025 labels)

A1 Adapter: AIOps2025 files -> the canonical case used by `gama/` (topology graph; metric / log / trace observations in the canonical
   fields; case window from the input description; no alert entity, as in the 13 RCA100 cases without one).
A2 Dataset profile (`data/dataset_profile.py` equivalent): metric-name -> signal-family patterns, entity kinds, trace error status,
   workloads of the load generator / injection tooling excluded.
A3 Topology from the data: call edges from parent-child spans across services; membership pod -> service and hosting pod -> node from
   metric labels. (RCA100 ships a topology; AIOps2025 does not.)
A4 Pod entities (the only structural extension): hypotheses also at pod level. Same likelihood functions; a pod belongs to its service
   and runs on its node. Tier rule written before the run, by analogy with the RCA100 node / service rule: a service-level fault shows
   in all its pods, a pod-level fault in one pod while its sibling pods stay normal.
A5 Fingerprints for the 18 AIOps2025 fault types, from their published definitions (Cai et al. Table 2) and the injection tool's
   documentation, on the canonical signals (e.g. network delay: latency rise at the callee edge without CPU rise; packet loss /
   corrupt: connection errors / retries; pod failure / kill: availability drop, restarts; jvm gc: GC rise; dns error: DNS messages).
A6 Output mapping to the official scorer: component (service / pod / node name; for network attacks the source service), reason = the
   type's phrase of the AIOps2025 taxonomy (Type Accuracy is keyword-matched on the first 20 words), reasoning trace = the decided
   evidence list (metric / log / trace items), kept short (Efficiency).

## Checks allowed before the run (no labels)

Structure audit (`scripts/aiops2025_audit.py`: never opens ground-truth files); adapter coverage (entities found, signals computed per
entity and modality); false-alarm rate on normal windows (pseudo-windows); run time. Ground truth is opened only by the scorer.

## Runs and analysis

- GAMA (CPU, modules core) on all 400 cases, once; optionally GAMA + LLM (Qwen2.5-7B, same settings as RCA100).
- Metrics: the benchmark's own (LA, TA, Explainability, Efficiency; Final = 0.4 LA + 0.4 TA + 0.1 Exp + 0.1 Eff) with its official
  scoring code; 95% bootstrap CIs; breakdown by fault category and injection level (service / pod / node).
- Context, not a controlled comparison (different protocol and backbones): published Acc@1 on AIOps2025 of ThinkDepth.ai with and
  without DiagGuard (Lu et al. 2026).
- Data: check the AIOps2025 licence before use; keep the data and per-case outputs private.

## Data facts (from the label-free audit and probe, 2026-11-05; groundtruth.jsonl never opened)

- input.json: 400 x {uuid, "Anomaly Description"}: free text with the window start / end in UTC (ISO, "Z").
- 18 daily folders `<date>/<date>/` (CST date; data timestamps UTC; a CST day = UTC 16:00 previous day .. 16:00). macOS `._*` files
  are upload artefacts and are ignored.
- Metrics (per minute): APM per pod and per service (request, response, rrt [ms], rrt_max, error, client_error, server_error,
  timeout, *_ratio); namespace total (pod_ns_hipstershop); infra_pod (pod_cpu_usage, pod_memory_working_set_bytes, pod_fs_reads /
  writes_bytes, pod_network_receive / transmit_bytes / packets, pod_processes; `instance` = the hosting node); infra_node (11 nodes:
  aiops-k8s-01..08, k8s-master1..3: cpu, memory, disk read / write bytes and time, filesystem usage, network bytes / packets, TCP
  sockets); TiDB SQL layer (infra_tidb), PD and TiKV (other/): cpu, memory, qps, duration, failed queries, io_util, server_is_up, ...
- Logs (Filebeat): k8_pod, k8_node_name, message, @timestamp.
- Traces (Jaeger): traceID, spanID, references (CHILD_OF parent), startTimeMillis, duration [us], operationName; tags span.kind,
  status.code (gRPC code: 0 ok, 1 cancelled, 13 internal, 14 unavailable), status.message, http.status_code, grpc.status_code,
  net.peer.name / ip, db.system (redis), rpc.service / method, error; process: serviceName, tags name / podName (pod), node_name /
  nodeName (node).
- Services seen: frontend, adservice, cartservice, checkoutservice, currencyservice, emailservice, paymentservice,
  productcatalogservice, recommendationservice, shippingservice, redis-cart; storage: TiDB (tidb, tikv, pd). Three pods per
  service (`<service>-0..2`), some renamed "(deleted)" after restarts.

## Adapter specification v1 (fixed before any run; dataset knowledge only)

A1 Case: case_id = uuid; window [start, end] parsed from the description (both ISO timestamps); no alert entity; alert text = the
   description. Data loaded from the hourly files covering [start - 3 h, end] (baseline for the calibrated tests), converted
   to the canonical observation fields used by `gama/` (metrics: entity_set / entity_name / metric / value; logs:
   _container_name_ / content; traces: traceId / spanId / parentSpanId / serviceName / duration [ns] / statusCode / statusMessage /
   attributes / spanName).
A2 Profile `aiops2025` (selected by the adapter; RCA100 keeps its profile): metric families add `rrt` -> latency and `timeout` ->
   error; a span is failing when status.code is not 0 (canonical statusCode "2"); HTTP / gRPC codes carried in attributes; the load
   generator (k6, `http.user_agent`) is not an entity.
A3 Topology from the data: service -> service call edges from parent / child spans of different services; cartservice -> redis-cart
   from `net.peer.name`; productcatalogservice -> TiDB from the published system description (storage tier behind the product
   catalog; Cai et al. Fig. 3) because these calls carry no spans. Hosting pod -> node from infra_pod `instance` (and log /
   trace node names).
A4 Pods: every pod is an entity of the service kind, named as in the data (`cartservice-1`); its service is an entity too, with
   the service-level APM series and the mean of its pods' infra series. Membership edges service -> pod are added to the call
   map, so a pod fault explains its service's and callers' symptoms (existing explain-away). One new likelihood rule, active only
   when the case defines memberships (RCA100 unaffected): a hypothesis at a SERVICE predicts the mechanism signals (fingerprint
   roles K / U) at each of its pods (THETA_P when present, 1 - THETA_P when absent: a service-level fault shows in every pod); a
   hypothesis at a POD explains the same mechanism signals in its service's aggregate (absence uninformative). Trace origins at a
   pod count as own evidence of its service. Constants unchanged (THETA_P, EM pi). TiDB components are entities `tidb-tidb`,
   `tidb-tikv`, `tidb-pd` (pods `-0`).
A5 Fingerprints (roles over the existing signal vocabulary, plus one family `net` = network bytes / packets, used only by this
   profile; LEVEL service = also pod):

   | Type (taxonomy name) | level | key | support | consequence (may) | contradiction |
   |---|---|---|---|---|---|
   | cpu_stress | service | svc_cpu_rise | svc_latency_rise | | svc_request_rise |
   | memory_stress | service | svc_mem_rise | svc_latency_rise | | |
   | jvm-cpu-stress | service | svc_cpu_rise & msg_gc? -> svc_cpu_rise | svc_latency_rise, msg_gc | | svc_request_rise |
   | jvm-gc | service | msg_gc / svc_gc_rise | svc_latency_rise, svc_cpu_rise | | |
   | jvm-latency | service | svc_latency_rise | trace_slow_origin | | svc_cpu_rise |
   | jvm-exception | service | msg_error / svc_error_rise | error_spans, http5xx, trace_err_origin | | |
   | pod_failure | service | svc_request_drop / msg_refused | trace_err_origin, http503 | svc_net_drop | svc_cpu_rise |
   | pod_kill | service | svc_request_drop / msg_refused | trace_err_origin, svc_restart_rise | | |
   | network_delay | service | svc_latency_rise | trace_slow_origin, msg_timeout | | svc_cpu_rise |
   | network_loss | service | msg_timeout / svc_error_rise | trace_err_origin, svc_latency_rise | | svc_cpu_rise |
   | network_corrupt | service | svc_error_rise / msg_timeout | trace_err_origin, svc_latency_rise | | svc_cpu_rise |
   | dns-error | service | msg_dns | svc_error_rise, trace_err_origin | | |
   | target_port_misconfig | service | msg_refused / trace_err_origin | svc_request_drop, http503 | | |
   | erroneous-code | service | svc_error_rise / http5xx | error_spans, trace_err_origin, msg_error | | |
   | io-fault | service | svc_disk_rise | svc_latency_rise, msg_error | | |
   | node_cpu | node | node_cpu_rise | | | |
   | node_memory (paper Table 2) | node | node_mem_rise | | | |
   | node_disk | node | node_disk_rise | | | |
   | node_network_delay | node | node_net_drop / node_net_rise | latency of hosted services (propagated) | | node_cpu_rise |
   | node_network_loss | node | node_net_drop / node_net_rise | errors of hosted services (propagated) | | node_cpu_rise |

   ("/" = alternatives, as in RCA100.) Pairs that the telemetry cannot separate (network_loss vs network_corrupt; pod_failure vs
   pod_kill; jvm-cpu-stress vs cpu_stress outside the JVM service) are resolved by the general-before-special order of this table,
   fixed now.
A6 Output: component = decided entity (pod / service / node name; TiDB pod name); reason = the taxonomy phrase of the decided type
   (e.g. "cpu stress"), first in the text; reasoning trace = up to 5 decided evidence items (metric name, log text, trace span).
   Score: Cai et al. eq. (1)-(4) re-implemented (no official scorer in the release): LA exact match (network faults: source or
   destination; pods: exact pod), TA keyword part only (embedding fallback unspecified -> TA reported as a lower bound),
   Explainability, Efficiency.
A7 Volume: per case all spans of traces that contain a failing span, plus the spans of a fixed 20% of the other traces (selected
   by a hash of the traceID, independent of content); metrics and logs complete.
