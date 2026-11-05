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
   description. Data loaded from the hourly files covering [start - 5 h, end] (the calibrated tests use B_SPAN = 4 window lengths of history and
   K_PSEUDO = 5 pseudo-windows, i.e. about 10 window lengths before the window), converted
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
   | jvm-cpu-stress | service | svc_cpu_rise | svc_latency_rise, msg_gc | | svc_request_rise |
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

## Self-audit of the specification (2026-11-05, before implementation)

| # | Finding | Severity | Resolution (fixed now) |
|---|---|---|---|
| 1 | Faults are injected about every hour (input windows at hh:10 for ~20-25 min), so the 4-5 h of history used for calibration contains earlier fault windows. The calibrated tests assume a mostly normal history (on RCA100 the export starts at the window). | High | A8 below: samples inside the windows of the OTHER cases of input.json (label-free: input.json is the agent input) are removed from every history series. |
| 2 | input.json windows of other cases are used (A8). | Note | They are inputs, not labels; disclosed as a preprocessing step. |
| 3 | The README's per-type "manifestation" column (metric / trace / log) and the published type list informed the fingerprints. | Note | Public documentation, not per-case labels; disclosed. Fault-type counts (Cai et al. Table 2) are NOT used: priors stay uniform per entity. |
| 4 | Entity naming of TiDB pods (`tidb-tikv-0`) follows the one published example (Lu et al. Fig. 6b) and the TiDB operator convention. | Low | Disclosed; if the scorer's names differ, LA for TiDB faults is reported both raw and with a documented name normalisation. |
| 5 | The probe showed real log lines (input data, not labels), e.g. product catalog "Region is unavailable". A3's product catalog -> TiDB edge is justified by the published architecture, not by that line. | Low | Disclosed. No further design from test-set inputs. |
| 6 | Pods renamed after restarts ("cartservice-2 (deleted)"). | Medium | The suffix " (deleted)" is stripped: same pod entity. |
| 7 | No JVM metrics and no Kubernetes events: jvm-gc and pod_kill rely on indirect evidence. | Medium | Expected weak types; reported per type. No compensation by tuning. |
| 8 | A7 trace selection keeps every failing trace and 20% of the others, for the window AND the history alike. | Low | Error rates are compared like-for-like; absolute counts are not used across cases. Disclosed. |
| 9 | Missing series for some pods on some days. | Low | Missing data = no observation (as on RCA100), never imputed. |
| 10 | The scorer is a re-implementation; TA embedding fallback unknown. | Medium | TA reported as a lower bound; scorer code published with the paper's artefacts. |

A8 History cleaning: for case i, every metric / log / trace sample in [start_j, end_j + 10 min] of any other case j of input.json is
   dropped before calibration (the README notes that effects may outlast end_time). The window of case i itself is untouched.

## Implementation amendments (2026-11-05, before any run on AIOps2025; written from the frozen code and a synthetic release only)

Implemented in `adapters/aiops2025.py` (case), `data/dataset_profile.py` + `data/aiops2025_profile.py` (profile, selected by
`DATASET_PROFILE=aiops2025`; RCA100 unchanged: golden snapshot of `tests/test_gama.py` identical), `gama/belief.py` (A4 rule, gated by
the case's memberships), `scripts/aiops2025_run.py` (run / label-free check), `tests/test_aiops2025.py` (synthetic release in the real
layout: a CPU fault in one pod is located at the pod, in every pod at the service; ground-truth files are never opened).

| # | Item | Change and reason |
|---|---|---|
| I1 | A8 history cleaning | **Not applied.** The frozen core tests a window only when its history covers >= 80% of B_SPAN window lengths (`pipeline/fault_v4.MIN_COVERAGE`). Faults are injected hourly at hh:10 for ~22 min, so masking [start_j, end_j + 10 min] of the preceding case removes ~32 of the ~88 min of every real window's history (coverage ~64%): no continuous signal would be testable. Shown label-free on the synthetic release (`--check` with `AIOPS_HISTORY_MASK=drop`: the case with a masked predecessor loses all continuous tests). Contamination is instead absorbed by the frozen statistics: the history scale is median / MAD (robust while < 50% of the history is a previous fault; here <= ~36%), peer re-standardisation rejects shifts shared by the entities of a kind, and a previous fault inside the pseudo-windows makes the empirical null heavier, i.e. conservative (fewer signals hold, never more). The label-free check reports, per entity kind and signal, the share of testable pairs and the share that hold. |
| I2 | A5 fingerprints | `msg_error` is not an observation of the frozen core (the log family 'error' is not emitted by `entity_signals`); it is dropped from jvm-exception, erroneous-code and io-fault. Everything else as in the A5 table (`data/aiops2025_profile.py`). Fallback type when only the unspecified anomaly is testable: erroneous-code (service level), node_cpu (node level). No residual types (uniform priors). |
| I3 | A7 logs | Only the log lines the core reads are loaded: those matching the core's text families (`pipeline/fault_fingerprint._TEXT_ANY`) plus the first line of each hourly file (it dates the log history). Equivalent input for the statistical core (the only arm run); an LLM arm would need the full logs. |
| I4 | A2 metric families | AIOps2025 patterns added to the RCA100 ones: rrt -> latency, timeout -> error, network bytes / packets -> net, TiKV MB/s and filesystem usage -> disk, *_is_up -> ready. Excluded as configuration or bookkeeping, not behaviour: capacities, totals, sizes, uptime, PD region / leader / store counts, socket counts, and the APM `*_ratio` columns (derived from the counts that are kept). Infra-node series are per device / mountpoint. Duplicate samples (renamed pods) are averaged per minute. |
| I5 | A2 trace status | A span fails when its Jaeger status.code is not 0 (canonical status "2"); the gRPC code is carried as `rpc.grpc.status_code` and HTTP codes as `http.status_code`, so the frozen HTTP / gRPC readers apply. Span service = the pod (process tag name / podName). |
| I6 | A3 topology | Also tidb-tidb -> tidb-tikv and tidb-tidb -> tidb-pd (published TiDB architecture: the SQL layer reads TiKV and gets timestamps from PD). |
| I7 | A6 output | One JSON line per case: uuid, component, reason (taxonomy phrase first, then "at <component>" and up to three holding evidence items), time (window start), reasoning_trace = up to 5 steps {step, action: LoadMetrics / TraceAnalysis / LogSearch(<entity>), observation: the evidence text}. |

Run order: (1) `python scripts/aiops2025_run.py --data <release> --check --cases first:5` (adapter coverage, run time, testable / hold
shares); (2) the full run in shards, once: `--shard k/4`; (3) the scorer (the only code that opens groundtruth.jsonl).
