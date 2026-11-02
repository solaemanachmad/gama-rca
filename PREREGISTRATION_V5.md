# Pre-registration v5 — frozen configuration, odd/even protocol and multi-backbone study

Written 2026-10-31, BEFORE any v5 run. Amendments are appended with a date; nothing above an amendment is edited.

## 1. Status of the data

RCA100 (103 cases) is DEVELOPMENT data: v3 and v4 were designed and evaluated on all 103 cases and their per-case errors were
inspected (DEVLOG_V4.md). Scores on RCA100 are therefore optimistic and are reported as development results.

## 2. Frozen configuration A

GAMA-RCA = `belief_v4` with `V4_PARTS=all` (= `DEFAULT_PARTS`: score, open, twostage, fp, peer, prop, events, avail, onset, em,
selfnorm, llmtype, conseq, retry), V3_LOG_MODE=template, ALERT_T0_MODE=window, hyperparameters as in the code at the commit that adds
this file. Reference results: V4 + Qwen3-8B (Final 50.4) and V4_RULE (Final 50.0), DEVLOG_V4.md dev run 6.

## 3. Label-leakage audit (2026-10-31)

- Agent-facing code reads only the case's telemetry, topology, alert and task window; `answer_key/` is read only by
  `evaluation/scoring.py` and by diagnostics. Enforced by `tests/test_v5.py::agent_code_never_reads_labels`.
- The fault-type list and definitions are static schema text (`data/taxonomy.py`), identical for every system.
- No component is fitted to labels. Fixed constants are listed in Appendix B of the paper.
- Design leakage remains: fingerprint additions (fp, conseq) and the choice of parts were informed by errors on RCA100. This is the
  reason for sections 4 and 6.
- LLM contamination: all backbones (Qwen2.5 2024, Llama 3.1 2024, Gemma 2 2024, Qwen3 2025) were released before the RCA100
  benchmark paper (2026) and answer keys are distributed separately; memorisation of answers is unlikely. The LLM never receives an
  answer, a label or a score.

## 4. Protocol for new changes (v5): odd / even split

- DEV_ODD = cases with an odd number (t001, t003, …; 52 cases). VAL_EVEN = even numbers (51 cases). Defined in scripts/kaggle_run.py.
- New changes are designed by looking ONLY at odd cases (scripts/diagnostics/entity_errors_dev.py refuses even cases).
- Each change is stated as a mechanism (not a case), with its constants fixed before the first run, and recorded as an amendment.
- After the design is frozen, VAL_EVEN is run ONCE for configuration A and A + changes, and both halves are reported.
- Acceptance (decided now): a change is kept only if Final on VAL_EVEN improves by >= 2.0 points over A and the one-sided paired
  Wilcoxon test gives p < 0.05. Otherwise A stays the final configuration and the change is reported as a negative result.
- Disclosure: per-case results of even cases from v4 runs exist and were seen; the split protects the design of the NEW changes only.

Planned changes (mechanisms; exact form fixed in an amendment after the odd-case diagnosis):
- C1 chronically anomalous entities: an entity whose signals are flagged in the baseline pseudo-windows of the same case is less
  informative when flagged in the window (label-free; measured on pseudo-windows only). Motivation: recommendation is predicted in
  22 cases and is never the true root.
- C2 node versus service: a node fault should show at several services hosted on that node at the same time; a single anomalous
  service on a node favours the service.

## 5. Multi-backbone study (after the configuration is final)

Arms on all 103 cases, each with the backbones qwen3_8b_local, llama31_8b, gemma2_9b, qwen25_7b_v5 (optional qwen3_4b):
- V4 (GAMA-RCA), EVIDENCE_LLM (the LLM decides from the same calibrated evidence, one call), DIRECT_LLM (alert only).
- Identical prompts, greedy decoding, bfloat16, thinking off; a template without a system role gets the system text prepended to the
  user message (agents/kaggle_client.render_chat), recorded per run.
Hypotheses: H1 for every backbone, Final(V4) > Final(EVIDENCE_LLM) (paired Wilcoxon, Holm-corrected over backbones);
H2 the Final scores of V4 across backbones lie within 3 points of each other (the statistical core carries the result).

## 6. Analysis plan

Metrics: EL, FI, RP (node-match), Final = 0.4 EL + 0.3 FI + 0.3 RP; cases with exact entity / exact type. 95% percentile bootstrap
intervals (5,000 resamples). Paired Wilcoxon signed-rank tests; Holm correction within each family of comparisons. All runs are
reported, including negative ones.

## Amendment 1 (2026-10-31, before any run of these parts): multimodal alignment

Disclosure: the motivation (recommendation predicted 22 times and never the true root; 26 entity-right / type-wrong cases) came from
an aggregate error analysis of the v4 run over all 103 cases. The FORM and the CONSTANTS below were fixed from the mechanism only,
without per-case inspection, and are not changed after the odd-case run except through a further dated amendment.

- Part `mm` (pipeline/belief_v4.py: modality, mm_loglik). Modalities: metrics, events (ev_*), traces (http*, error_spans, db/redis/
  queue span outcomes), logs (log_*_rise, msg_*). For a specific hypothesis (e*, type), k = number of modalities in which a
  key / support / may signal of the type holds at e* (generic symptoms excluded). If k >= 2 the log-posterior gains
  log(THETA_M / beta_M), THETA_M = 0.6, beta_M = max((n2 + 10 * 0.1) / (n + 10), 0.1) over the case's specific hypotheses
  (n2 of them with k >= 2). k < 2 is uninformative. Unspecified-anomaly and null hypotheses are unaffected.
- Part `xcard` (pipeline/llm_typing.py): each evidence line in the typing prompt is tagged with its modality; each shortlisted type
  shows its expected cross-modal signature read from the fingerprint; one guidance line asks to prefer the candidate matched in more
  modalities in a mechanism-consistent order. Acceptance rules (shortlist + verbatim quote) unchanged.
- Runs: V4_RULE_MM on DEV_ODD (compare with V4_RULE, results-2, same cases). If mm helps on the odd half, V4_RULE_MM on VAL_EVEN
  once; acceptance as in section 4 (>= +2.0 Final on VAL_EVEN, one-sided Wilcoxon p < 0.05). V4_MM_XCARD / V4_XCARD (GPU) follow the
  same protocol against V4 (dev run 6).

## Result of the odd-case run of `mm` and Amendment 2 (2026-11-01)

V4_RULE_MM on DEV_ODD (commit ae9d3a7) vs V4_RULE (configuration A, same 52 cases): EL 56.4 -> 57.9, FI 15.4 -> 19.2,
RP 71.6 -> 71.6, Final 48.7 -> 50.4; 3 cases changed (2 better: t053, t071; 1 unchanged score: t011; 0 worse); Wilcoxon p = 0.60.
`mm` is safe but rarely decisive; it does not remove the recommendation attractor (8 of the odd-case entity errors).

Odd-case trace analysis (decision trails, odd cases only): recommendation shows svc_cpu_rise in 32/52 and svc_request_rise in 29/52
incidents, ad svc_request_rise in 34/52, frontend op_request_rise in 36/52. The extra load enters at the user-facing services and
flows DOWNSTREAM to their callees (the opposite direction of error propagation), and matches the load fingerprints (trafficSurge,
cpuFullLoad, cpuDeadLoop, trafficHotspot) of the callees.

Part `loadx` (fixed before any run): for a service whose own request count rises AND one of whose callers' request count rises,
its svc_request_rise, op_request_rise and svc_cpu_rise observations get background THETA_P = 0.6 instead of ALPHA (they are
expected under no fault there); fingerprint roles for that entity are unchanged. No new constant.
Runs: V4_RULE_LOADX and V4_RULE_MM_LOADX on DEV_ODD, compared with V4_RULE and V4_RULE_MM. Risk stated in advance: genuine
trafficSurge / hotspot faults injected through the entry service may lose evidence. VAL_EVEN once for the best odd-half variant.

## Amendment 3 (2026-11-01, before any run of `ratio`): relational signals

Motivation (odd-case traces): 6 of the 8 recommendation errors are trafficSurge decisions driven by a request rise that flows down
from the entry; raw rises cannot tell load from fault. Disclosure: the per-type list of true root entities (all 103 cases) was also
looked at when checking which services are genuine load-fault roots.

Part `ratio` (pipeline/fault_v4.ratio_signals; fixed before any run, no new constant): three continuous signals, scored and calibrated
like all others: svc_cpu_per_req_rise = CPU / (requests + 1); svc_err_rate_rise = errors / (requests + 1);
svc_req_share_rise = requests / (sum of callers' requests + 1), equal to the own request rise for services without caller series.
Fingerprints with `ratio`: cpuFullLoad and cpuDeadLoop key = CPU per request (raw CPU rise becomes support); trafficSurge key =
request share (raw request rise becomes support); httpError5xx and codeDefect gain error rate as support.
Runs: V4_RULE_RATIO and V4_RULE_MM_RATIO on DEV_ODD; compared with V4_RULE, V4_RULE_MM, V4_RULE_LOADX, V4_RULE_MM_LOADX on the same
cases. Only the best odd-half variant is run once on VAL_EVEN.

## Result of `ratio` (odd half) and Amendment 4 (2026-11-01): trace structure

V4_RULE_MM_RATIO on DEV_ODD (commit ceba18a): EL 54.4, FI 15.4, RP 71.0, Final 47.7 (V4_RULE_MM: 50.4; A: 48.7); 4 cases better,
9 worse than mm alone. Cause: the per-request ratios rise when a fault lowers throughput (denominator effect: memoryPressure and
slowSQL cases became CPU types) and the request share rises when callers fail (trafficSurge everywhere). `ratio` is REJECTED as
pre-registered and reported as a negative result; it is not patched further on the same cases.

Odd-case trace analysis: in 19 of 20 entity errors with a resolvable true root, the true root HAS evidence; the failure is attribution
(which entity starts the problem), not detection.

Part `torigin` (pipeline/trace_structure.py; constants fixed before any run: MIN_TRACES 3, ORIGIN_SHARE 0.3, SLOW_Q 0.9, MIN_GROUP 10,
SLOW_CONTRAST 0.2): per service, error origin = deepest failing spans (no failing child) in >= 30% and >= 3 of the failing traces;
slow origin = largest self time (duration minus children's covered time) in the slowest 10% of traces of a root operation, at least 20
points more often than in the other traces, in >= 3 slow traces. Categorical observations; own evidence (THETA_P) of a root at the
service or at its hosting node; absence uninformative; never a propagated symptom. Only OpenTelemetry fields are used.
Also: data/dataset_profile.py collects the dataset-specific knowledge (metric families, injector workloads, constant metrics, error
status, entity types); the patterns are unchanged. EVIDENCE_LLM receives the trace origins too when 'torigin' is on.
Runs: V4_RULE_MM_TORIGIN and V4_RULE_TORIGIN on DEV_ODD. Decision rule unchanged (section 4). This is the last design change before
the VAL_EVEN run and the external benchmark.

## Result of `torigin` (odd half) and Amendment 5 (2026-11-01): trace structure v2 / tier decision

V4_RULE_MM_TORIGIN on DEV_ODD (commit a4b733c): EL 58.8, FI 17.3, RP 71.6, Final 50.2 (mm: 50.4); exact entities 26 (mm 25);
vs mm 3 better (t001, t007 via large error origins: payment 1329/1500, inventory 763 traces; t065), 4 worse. Defects found:
(1) a node received the origin of ANY hosted service, so a single small origin (t039: frontend-proxy, 37 traces) moved the answer to
a node (t039, t099, t003); (2) origins from 2-3 traces were admitted; (3) cpuFullLoad / cpuDeadLoop have near-identical fingerprints,
so small posterior changes flip between them (t067), independent of torigin.
Decision: the user continues architecture development on the odd half (VAL_EVEN still unseen; it remains the confirmatory test,
together with an external benchmark).

Part `torigin2` (fixed before any run): as torigin with MIN_ERR_TRACES 10, MIN_SLOW_TRACES 5, and a node supported by trace origins
only when >= 2 of the services it hosts are origins (a node fault hits co-located services together). torigin (v1) is unchanged.
Run: V4_RULE_MM_TORIGIN2 on DEV_ODD; compared with V4_RULE_MM and V4_RULE_MM_TORIGIN.

## Amendment 6 (2026-11-01, before any run of these parts): origin messages and CPU parsimony

Part `omsg` (needs torigin / torigin2; constant OMSG_SHARE 0.3 fixed a priori): for each error-origin service, the status messages
and HTTP codes of its ORIGIN spans, plus log lines that carry the trace id of one of its origin traces (32-hex id in the line), give
omsg_<category> / ohttp<code> when >= 30% of them carry it. Role 'may' for every type whose fingerprint names the same message
category or status code. Rationale (odd cases: 26 entity-right / type-wrong): a caller relays its callee's failure message, so
message evidence counted per service misleads; read at the origin it names the mechanism where it happens.
Part `cpuspec`: cpuDeadLoop (a loop starving its own process) needs starvation evidence: key = svc_cpu_rise & svc_request_drop; a
CPU rise alone is cpuFullLoad (parsimony; the two fingerprints were otherwise near-identical and flipped on small changes).
Runs on DEV_ODD: V4_RULE_MM_T2_OMSG_CPU (all), V4_RULE_MM_T2_OMSG, V4_RULE_MM_T2_CPU, against V4_RULE_MM_TORIGIN2.

## Results of `omsg` / `cpuspec` (odd half) and Amendment 7 (2026-11-01)

Commit bf3b3a1, DEV_ODD (52 cases), Final (EL / FI): A 48.7 (56.4 / 15.4); mm 50.4 (57.9 / 19.2); mm+torigin 50.2 (58.8 / 17.3);
mm+torigin2+omsg 53.3 (66.0 / 17.3), exact entities 29 (mm 25), 8 better / 5 worse vs mm (p = 0.47);
mm+torigin2+omsg+cpuspec 52.7 (66.0 / 15.4). The V4_RULE_MM_TORIGIN2 arm alone was not run, so torigin2 and omsg are not separated.
`cpuspec` is REJECTED: the CPU stress also lowers throughput, so the request drop does not separate deadLoop from fullLoad (t067,
t071, t093 became cpuDeadLoop). The two CPU types remain indistinguishable with the available signals (a limitation to report).
Remaining systematic error (5 cases: t033, t065, t081, t087, t095): when a callee has no pods (replicaScaleDown, podPending), the
caller's client span fails with 'connection refused' and has no child span, so the CALLER (checkout) is the deepest failing span,
the error origin, and its 'refused' message makes it networkPolicyIsolation.

Part `torigin3` (= torigin2 + caller-to-callee attribution; no new constant): a deepest failing span with an 'unreachable' message
(refused / unavailable / no such host / no healthy upstream / no endpoints / connection reset) and no child span, or with a timeout
message, is attributed to its callee: the service named in the span name or attributes (longest known service name, normalised,
other than its own), or for a timeout with children, the longest child. Otherwise the span's own service keeps the origin. Counts
of moved origins are stored (v5_relayed). Run: V4_RULE_MM_T3_OMSG on DEV_ODD vs V4_RULE_MM_T2_OMSG.

## Amendment 8 (2026-11-01, before any run of these parts): fault-type identification

Odd-case analysis of mm+torigin2+omsg (entity right in 29 cases, type right in 7): (a) the load types (cpuFullLoad, cpuDeadLoop,
trafficSurge, trafficHotspot) alone explained a node_cpu_rise at each host of the root, one THETA_P factor per host, so memory,
GC, thread and application faults that also raise CPU lost (t057, t061, t047: memoryPressure -> CPU types); (b) an error origin
with HTTP 5xx at the root counted equally for every type, so CPU types beat application types (t029, t035, t083:
nullPointerException -> cpuDeadLoop); (c) cpuFullLoad and cpuDeadLoop are often EXACTLY tied (0.444 / 0.444) and an unstable sort
picked either (t027, t041, t071, t093).
Parts (no new constants): `hostcpu`: a host node_cpu_rise is explained by any hypothesis at a root whose own svc_cpu_rise holds;
`errclass`: an error origin at the root is expected only for types whose fingerprint names an error signal (http*, msg_*,
error_spans, svc_error_rise, *_err), background for the others (slow origins unchanged; node rule unchanged); `tiebreak`: stable
sort, so an exact tie goes to the type listed first (the general cpuFullLoad before its special case cpuDeadLoop).
Disclosure: (c) uses the fingerprint order, which lists the general type first; cpuFullLoad is also the more frequent type in
RCA100 (seen in the labels), which favours this rule on this benchmark.
Runs: V4_RULE_MM_T3_OMSG_FI on DEV_ODD (CPU) vs V4_RULE_MM_T3_OMSG; V4_MM_T3_OMSG_FI (GPU, + xcard LLM typing) afterwards.

## Amendment 9 (2026-11-01, before any run of this part): consequences of a slow dependency

Odd-case analysis of mm+torigin3+omsg (DB-type failures): in t021 and t097 db_slow holds at the true root (inventory), yet
memoryPressure / cpuFullLoad win, because inventory's svc_thread_rise, svc_cpu_rise and svc_gc_rise also hold and these signals
are explained only by resource types; each unexplained holding signal costs the dependency hypotheses a background factor.
A thread pool blocked on a slow database or cache grows its thread count, and the waiting/retrying work raises CPU and GC, so these
are consequences of the dependency fault, not evidence against it.
Part `depconseq` (no new constants): for F010 slowSQL, F018 dbNetworkLatency, F029 redisUnavailable, F009 cacheBreakdown and
F005 messageQueueBacklog, add svc_thread_rise, svc_cpu_rise and svc_gc_rise with role 'may' (explained when present, THETA_U;
uninformative when absent). Key / support / contra roles of every type are unchanged; other types are unchanged.
Expected failure mode (disclosed before the run): when db_slow does not hold (t015, t085: no DB spans detected), the part cannot
help; when a resource fault ALSO slows DB calls, the dependency types may now win wrongly.
Diagnostic (odd cases only): scripts/diagnostics/db_spans.py prints the DB-like spans of a service, how fault_calibrated
classifies them, and the in-window duration test behind db_slow.
Run: V4_RULE_MM_T3_OMSG_FI_DEP on DEV_ODD (CPU) vs V4_RULE_MM_T3_OMSG_FI.

## Amendment 10 (2026-11-01, before any run of this part): dependency latency against the service's own history

Diagnostic scripts/diagnostics/db_spans.py on odd cases t015, t085, t021, t097 (dependency-type faults, all at inventory):
- db_slow compared the window's DB spans with the service's OTHER window spans. A query span (median 0.4-0.5 ms) is much shorter
  than the server span containing it (1.2-1.9 ms), so the test is weak by construction. t015 (DB calls time out at 5 s, only 4 DB
  spans in the window): p = 0.36 today, p = 0.0003 against the same spans before the window. t097: holds today only through the tail.
- HikariDataSource.getConnection (connection-pool wait, ~9,000 spans per case) was classified "other", i.e. it entered the comparison
  group instead of the database group.
- t085 (slowSQL): DB span durations are unchanged (0.4 / 0.4 ms) under both tests -> not detectable from traces; expected to stay wrong.
- t085 cart redis spans: median and p90 unchanged, yet a rank test against history gives p = 1.9e-23 (12,000 vs 4,000 spans): a rank
  test alone over-fires on large span counts -> an effect-size condition is required.
- RCA100 records no redis client spans in most cases (t097: none) -> redis faults rely on messages / logs (unchanged).
Part `depbase` (pipeline/dep_latency.py; no new constants): per service and dependency kind, <dep>_slow holds iff window durations >
history durations (one-sided Mann-Whitney, p <= ALPHA) AND mean of the slowest 10% in the window >= (1 + SLOW_CONTRAST) x that of the
history; history = the same kind of span of the same service in the B_SPAN window lengths before the window; connection-pool spans
(hikari, c3p0, dbcp, druid, getConnection) count as database calls. It replaces db_slow / redis_slow / queue_slow; no observation
without dependency spans. The tail-mean (not the 0.9 quantile) was chosen while writing the unit tests, because durations are
recorded in 0.1 ms steps and a quantile jumps between steps.
Disclosure: designed after inspecting four odd (development) cases with labels; even cases unseen.
Run: V4_RULE_MM_T3_OMSG_FI_DEPB (= FI parts + depconseq + depbase) on DEV_ODD (CPU) vs V4_RULE_MM_T3_OMSG_FI_DEP.

### Result of amendments 9-10 (V4_RULE_MM_T3_OMSG_FI_DEPB, DEV_ODD, commit 4a8e7a6)
Final 53.3 / EL 66.0 / FI 17.3 / RP 72.3 (entities 30, types 9) vs V4_RULE_MM_T3_OMSG 54.5 / 66.0 / 21.2 / 72.6 (30, 11); Wilcoxon
p = 0.28. Gains: memoryPressure right in t047, t057, t061 (previously CPU types). Losses:
- `depconseq` REJECTED: dependency types became sinks at services without that dependency (redisUnavailable at inventory, which
  has no redis spans: t027, t067, t071; messageQueueBacklog t007, t083; cacheBreakdown t023, t045, t091). Their keys are untestable
  there, so they paid nothing, while the 'may' roles explained every JVM / CPU rise.
- `depbase` is sensitive but not specific: db_slow now holds at inventory in t015, t021, t097 (DB faults) but also in 11 non-DB
  cases (t017, t029, t035, t037, t039, t047, t055, t077, t089, t099, ...): a starved service stretches its own DB calls as well
  (t099 httpError5xx -> slowSQL).

## Amendment 11 (2026-11-01, before any run of this part): where the extra time goes

Part `depshare` (with depbase; pipeline/dep_latency.py; no new constants): <dep>_slow additionally requires that the share of the
service's request time spent in calls of that dependency kind rises by >= SLOW_CONTRAST: share = sum of the dependency span
durations / sum of the service's entry spans (spans whose parent is not a span of the same service), window vs history. A slow
dependency moves the extra time into the calls; a starved service (CPU, GC, threads) also stretches its calls but spends most of the
extra time in itself. Same principle as the slow origin of amendment 4, applied inside one service. No observation change otherwise.
Run: V4_RULE_MM_T3_OMSG_FI_DBS (= FI parts + depbase + depshare, WITHOUT depconseq) on DEV_ODD (CPU), compared with
V4_RULE_MM_T3_OMSG_FI (to be run as well: it separates the FI parts from the dependency parts). Decision rule: if FI_DBS does not
improve over FI, the dependency parts are dropped and recorded as negative results.

### Result of amendment 11 (V4_RULE_MM_T3_OMSG_FI_DBS, DEV_ODD, commit d0d89fe)
Final 53.7 (shard 0 RP-nodes taken from T3_OMSG for identical entities; plain Final 45.0 vs 45.6, Wilcoxon p = 0.82) / EL 64.3 /
FI 21.2 / group 44.2 (entities 29, types 11) vs V4_RULE_MM_T3_OMSG 54.5 / 66.0 / 21.2 / 44.2 (30, 11). The redis / queue sinks of
depconseq are gone. depshare keeps db_slow at the DB faults (t015 share 99%+, t021 99% vs 20%) and removes most non-DB holdings, but
still passes t031, t039, t055 (share 18-22% -> 22-25%). Detection is no longer the bottleneck: t015 and t021 hold db_slow and are
still typed memoryPressure, because four resource signals at inventory (memory, CPU, GC, threads) outweigh one dependency key.
Type changes vs T3_OMSG: +t007 threadExhaustion, +t047 / t057 / t061 memoryPressure; -t053 / t067 / t071 cpuFullLoad (-> fullGC /
memoryPressure), -t091. CPU, memory, GC and dependency faults at the same JVM service raise the same set of signals; their counts do
not separate them. Next: diagnostic scripts/diagnostics/signal_onsets_dev.py (which signal moves first, odd cases only).

### Decision on amendments 9-11
The diagnostic run of V4_RULE_MM_T3_OMSG_FI (signal_onsets_dev.py, same parts) types 11 odd cases right, as FI_DBS; FI_DBS also
lost one entity (t039). Per the decision rule of amendment 11, the dependency parts (depconseq, depbase, depshare) are dropped from
the configuration and reported as negative results (detection works, decision unchanged).

## Amendment 12 (2026-11-02, before any run of this part): which mechanism signal moves first

Diagnostic signal_onsets_dev.py (odd cases, labels read for the analysis only): at the true root, the onsets of the four resource
signals of a process (CPU, memory, GC, threads; 30 s sampling) separate the JVM-type faults that share them:
- cpuFullLoad (t027, t041, t053, t067, t071, t093): CPU moves first in all 6; GC 30-60 s later in all 6.
- memoryPressure (t009, t063): memory and GC first, CPU later; (t047, t057, t061): memory holds without a measured onset (gradual),
  CPU and GC tie.
- dependency faults (t015, t021, t025, t085, t097): CPU (and latency) first, GC / threads later -> the rule cannot help them.
Part `sigorder`: for h = (e*, type) whose key names one of MECHANISM_SIGNALS = (svc_cpu_rise, svc_mem_rise, svc_gc_rise,
svc_thread_rise), when at least one of them has a measured onset at e*: THETA_T if a key of the type is among the first movers
(earliest onset, ties included), 0.5 if the key holds without a measured onset, 1 - THETA_T otherwise. THETA_T = 0.9 as the
between-entity onset rule; no new constants.
Disclosure: before this amendment, the rule was simulated offline on the saved odd-case posteriors of FI_DBS / FI_DEPB (top-8 types
at the decided entity, multiplied by the factor). FI_DBS: +5 types (t027, t053, t067, t071, t093 -> cpuFullLoad), -1 (t007
threadExhaustion -> codeDefect: memory moved first). A variant that leaves types whose key does not hold neutral gained nothing and
was not adopted. Even cases unseen.
Runs: V4_RULE_MM_T3_OMSG_FI_ORD (CPU) on DEV_ODD; then V4_MM_T3_OMSG_FI_ORD (GPU, + xcard LLM typing).

### Result of amendment 12 (V4_RULE_MM_T3_OMSG_FI_ORD, DEV_ODD, commit 4e2ffd8)
Final 55.3 / EL 62.6 / FI 28.8 / RP 72.0 / group 51.9 (entities 28, types 15) vs V4_RULE_MM_T3_OMSG 54.5 / 66.0 / 21.2 / 72.6 / 44.2
(30, 11) and V4_RULE_MM_T3_OMSG_FI (shards 0-2, 39 cases: types 6 vs ORD 10). Wilcoxon vs T3_OMSG p = 0.90. Types gained:
cpuFullLoad t027, t041, t093 (CPU moved first) and memoryPressure t047, t057, t061 kept; lost: t007 (threadExhaustion -> codeDefect,
memory moved first, as simulated). Entities lost: t039 (also in FI / FI_DBS: FI parts) and t099 (inventory 0.398 vs recommendation
0.395 before; the order rule lowered inventory's type mass and moved the decision to recommendation): the rule meant to TYPE the root
also moved the root.

## Amendment 13 (2026-11-02, before any run of this part): the order rule types the root, it does not choose it

Part `sigorder2` = the rule of amendment 12 applied only in the second stage of the two-stage decision: the root is chosen from the
entity marginals WITHOUT the rule; then the types of that root are re-weighted by it (renormalised within the root). No new constants.
Offline check on the FI arm's saved posteriors (39 odd cases, onsets from signal_onsets_dev.py): types 6 -> 11 (ORD: 10); t099 keeps
inventory / httpError5xx.
Runs: V4_RULE_MM_T3_OMSG_FI_ORD2 (CPU) on DEV_ODD; then V4_MM_T3_OMSG_FI_ORD2 (GPU, + xcard LLM typing, which reads the re-weighted
type ranking).

### Result of amendment 13 (V4_RULE_MM_T3_OMSG_FI_ORD2, DEV_ODD, commit 3a98b08)
Final 56.6 [bootstrap 95% CI 48.8, 64.4] / EL 64.3 / FI 30.8 / RP 72.3 / group 53.8 (entities 29, types 16), exactly the offline
check: vs ORD only t099 changed (back to inventory / httpError5xx). Wilcoxon on Final: vs configuration A (48.7) p = 0.014; vs
V4_RULE_MM_T3_OMSG (54.5) p = 0.61. Current best CPU configuration: A + mm + torigin3 + omsg + hostcpu + errclass + tiebreak +
sigorder2. Remaining weakness: entity localisation of node faults (0 of 5), K8s lifecycle (1 of 6) and the recommendation attractor.

## Amendment 14 (2026-11-02): entity errors of ORD2, and the never-run `loadx`

Diagnostic entity_errors_dev.py (odd, ORD2 parts; 23 entity errors), with a new per-signal count of how many of the entity's OWN
pseudo-windows moved as much as the real window:
- Idea "the attractor is a noisy service" REJECTED by the data: almost every holding signal has self 0/5 (0/2, 0/3); the
  recommendation rises are real in the window (workload 1 -> 583, 30 -> 2800, 28 -> 1138 requests), not noise. Per-entity
  calibration would not remove it.
- recommendation is still chosen in 8 errors (t005, t013, t017, t019, t059, t073, t077, t097): rateLimiting, node and DB faults
  elsewhere. Its evidence is load (request, operation request, CPU, latency rise). In 4 of them it cannot even explain the alert.
- Node faults: the node signal exists in t003 (memory 19 -> 83 %) and t005 (CPU 9 -> 100 %) but the hosted service wins; t069, t073
  (nodeDown): no node signal at all (a down node stops reporting).
- Scaled-down services (quote t033 / t081, shipping t065): no or one holding signal; the caller with 'connection refused' wins.
Amendment 2 (`loadx`, load entering at the user-facing services flows downstream; a callee's request / CPU rise together with a
caller's request rise is expected without a fault there) was fixed but NEVER RUN. It targets exactly the recommendation attractor.
It is run now, unchanged, on top of the current best: V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX on DEV_ODD (CPU), vs ORD2.
Pilot (4 odd cases, commit 70a9f82): load_explained = recommendation, frontend, ad, product-catalog (the frontend-proxy -> frontend edge
exists); t067 unchanged (cpuFullLoad, correct); t013, t017, t005 stay at recommendation (entity p 0.99 / 0.998 / 0.88), now carried by
its latency rise and other terms. Decision trail of t017 (ORD2): the recommendation hypothesis also explains frontend / frontend-proxy
HTTP 5xx and error spans (as upstream propagation from a callee) and the CPU rise of its three host nodes (hostcpu), although the
traces attribute all 190 error traces to inventory (HTTP 429). The full loadx run is postponed; entity_errors_dev.py now prints, per
wrong case, the per-observation log-likelihood difference between the best hypothesis of the predicted and of the true root.

## Amendment 15 (2026-11-02, before any run of this part): load and what load causes (`loadx2`)

Diagnostic entity_errors_dev.py with per-observation comparison (ORD2 + loadx, odd cases; log-likelihood difference between the best
hypothesis of the predicted and of the true root): with loadx the request rises no longer count, but the load-carrying service still
wins through what the load CAUSES: (a) its latency rise (+2.48: t005, t013, t017, t077, t097); (b) the CPU rise of the nodes that host
it, credited to it through hostcpu (+7.44 in t017 and t077, +5.0 in t097 vs the true root's own hosts); (c) its early onset (load
arrives before the fault's effects), which penalises the true root through the between-entity onset rule (t013: +2.29 non-likelihood).
Other findings, not addressed here: node faults whose node shows no signal (t019, t069, t073: data limit); scaled-down services
(t033, t065, t081) beaten by the caller's 'connection refused' evidence; t003 node memory vs the memory rise of a hosted service.
Part `loadx2` (no new constants): loadx with (i) svc_latency_rise added to the load signals of a service whose request rise is
load-explained, (ii) node_cpu_rise of the hosts of a service whose CPU rise is load-explained is load-explained too (background
THETA_P for every hypothesis), (iii) load-explained signals are excluded when dating an entity's onset.
Expected from the per-observation differences (not a guarantee: EM and the mm term also move): t005, t017, t097 may flip to the true
root; t013, t077 probably not (memory rise / remaining margin); risk: true roots whose own request rise is load-explained lose their
latency / host-CPU evidence.
Run: V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX2 on DEV_ODD (CPU), vs ORD2.

## Amendment 16 (2026-11-02, before any run of this part): the endpoint named in the failure message (`msgtarget`)

Diagnostic silent_dev.py (odd t033, t065, t081, replicaScaleDown; no labels read by the script):
- t033 / t081 (quote scaled down): checkout's failing span is the client call /oteldemo.ShippingService/GetQuote; shipping WAS
  reached (it has a child span), so torigin3 keeps checkout as the origin. The message carries the real failure:
  "error sending request for url (http://quote:8080/getquote): ... connection refused" (shipping could not reach quote).
  quote emits no spans in the window (t033: 3513 before, 0 in), so it also had no trace observation at all.
- t065 (shipping scaled down): "dial tcp 192.168.132.109:8080: ... connection refused" (an IP, names no service); torigin3 already
  relays to shipping (no child span). Unchanged by this amendment.
- "Silence" is not specific: when checkout fails, payment / email / accounting / fraud-detection also emit no spans (never called).
  No silence signal is added.
Part `msgtarget` (with torigin3; no new constants): for a deepest failing span whose message is of the unreachable / timeout kind,
if the message names the endpoint (host of a URL, of 'dial tcp <host>:' or of a DNS lookup) and that host's first DNS label is a known
service other than the span's own, that service is the error origin (and receives the origin messages); services that become
origins this way get trace observations even without spans of their own. An IP address names no service.
Run: V4_RULE_MM_T3_OMSG_FI_ORD2_MSGT on DEV_ODD (CPU), vs ORD2; separately from loadx2, to attribute each.

### Result of amendment 15 (V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX2, DEV_ODD, commit efdb33b)
Final 62.3 [bootstrap 95% CI 54.4, 69.7] / EL 74.6 / FI 34.6 / RP 73.5 (entities 35, types 18) vs ORD2 56.6 / 64.3 / 30.8 / 72.3
(29, 16). Wilcoxon on Final: vs ORD2 p = 0.025; vs V4_RULE_MM_T3_OMSG p = 0.008; vs configuration A p = 0.0003. 12 cases changed:
entity AND type right now in t013, t017, t089 (rateLimiting at inventory, previously recommendation / a node); entity right in t039,
t059, t097; t065 0.25 -> 0.5 hop credit; one loss: t049 (trafficHotspot type lost; its entity was already wrong) -- the risk stated
in advance. The recommendation attractor is gone in the rateLimiting cases; t077 (rateLimiting) and t073 (nodeDown) remain.
By fault group (entity / type right): Resource 6/6 6/6, JVM 8/8 5/8, Middleware & DB 7/8 1/8, Application logic 12/19 5/19,
K8s lifecycle 2/6 1/6, Cloud resource 0/5 0/5.
Next: msgtarget on top of loadx2 (V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX2_MSGT) instead of on ORD2 alone, since loadx2 is now the base.

### Result of amendment 16 (V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX2_MSGT, DEV_ODD, commit d4ef239)
Final 63.4 [bootstrap 95% CI 55.4, 71.0] / EL 77.1 / FI 34.6 / RP 74.0 (entities 37, types 18) vs LOADX2 62.3 / 74.6 / 34.6 / 73.5
(35, 18); Wilcoxon vs LOADX2 p = 0.59 (3 cases changed), vs configuration A p = 0.0003. The relay worked where designed: t033 and t081
relay 2419 / 1666 error origins to quote. t081: quote / replicaScaleDown, entity and type right (0.23 -> 0.87). t095: entity now right
(shipping; type replicaScaleDown instead of podPendingUnschedulable). t033: worse (0.16 -> 0.07): quote and shipping are both error
origins there and a node wins (cn-hongkong.10.0.1.107). K8s lifecycle entities 2/6 -> 4/6.

## Design freeze (2026-11-02) -- CPU core
Configuration GAMA-RCA v5 (core) = A + mm + torigin3 + omsg + hostcpu + errclass + tiebreak + sigorder2 + loadx2 + msgtarget
(arm V4_RULE_MM_T3_OMSG_FI_ORD2_LOADX2_MSGT). No further change to the core is made on the odd cases. Known failure modes recorded
above (node faults without node signals; t033 node attraction; DB slowness invisible in traces). Next, in order: GPU run of the same
core + LLM (log sensor + typing) on DEV_ODD with one backbone, to check that the LLM stage is not harmful; then VAL_EVEN once (core,
and per backbone DIRECT_LLM / EVIDENCE_LLM / GAMA-RCA + LLM), with the arms aligned to this configuration before any even-case run.

## Amendment 17 (2026-11-02, before any run of this part; reopens the frozen core on the odd half only): text evidence (`textx`)

Per-type analysis of the frozen core (odd half): 18 cases have the right entity and the wrong type; in about 13 the key signal of the
true type is not detected at the root. Diagnostic error_text_dev.py (odd cases, case:service pairs given, no labels read):
- nullPointerException (t029, t035, t083): the span status message is empty, but every failing span carries the exception in the
  OTel attributes (exception.type / exception.message) and the service logs 549-1280 'java.lang.NullPointerException' lines in the
  window vs 0-7 before. The core reads only statusMessage for msg_*, and log_npe_rise did not hold.
- httpError5xx at gRPC services (t045 ad): no http.status_code; rpc.grpc.status_code on the failing spans and 'Status{code=UNAVAILABLE}'
  in the log. The core reads only http.status_code.
- dbNetworkLatency (t015, t025): logs 'SQL Error', 'HikariPool - Interrupted during connection acquisition', mysql exceptions (0 before);
  span exception 'No operations allowed after connection closed.'
- Kubernetes events ARE mapped to services (inventory probe failures / restarts, shipping FailedScheduling / scale-down); unmapped
  events belong to DaemonSets and prometheus only. No change needed there.
- t039, t087 (shipping): no failing spans, no logs, no events of shipping in the window: data limit.
Part `textx` (read in fault_calibrated when V4_PARTS names it; no new constants): (i) the text of a failing span = status message +
OTel exception attributes (exception.type / message / stacktrace, vendor excep.name / message / stack); (ii) a failing span's
rpc.grpc.status_code without an HTTP status is mapped to its HTTP class by the standard gRPC->HTTP table (2/13 -> 500, 14 -> 503,
8 -> 429, 7 -> 403, 4 -> 504, 12 -> 501, 16 -> 401); (iii) msg_<family> also holds when log lines of that family are NEW in the window:
one-sided Poisson test of the window count against the same length of history (rate (h+1)/L per bin), at ALPHA.
Expected: t029, t035, t083 -> nullPointerException (key msg_npe); t045 may move (http503 is also a key of loadBalancerFailure); risk:
more msg_* at callers that log their callee's failures (handled by the categorical background rate).
Run: GAMA_CORE_TX (= frozen core + textx) on DEV_ODD (CPU), vs GAMA_CORE (= LOADX2_MSGT result, Final 63.4).

### Result of amendment 17 (GAMA_CORE_TX, DEV_ODD, commit 2f9ec4e)
Final 63.2 / EL 76.6 / FI 34.6 / RP 74.0 (entities 37, types 18) vs frozen core 63.4 / 77.1 / 34.6 / 74.0 (37, 18); Wilcoxon p = 0.95.
Gains as predicted: nullPointerException right in t029, t035, t083 (exception attributes + new NPE log text); t007 threadExhaustion right.
Losses: cpuFullLoad -> threadExhaustion in t027, t053, t067, t071; t065 entity 0.5 -> 0.25; t045 -> loadBalancerFailure (http503 from
gRPC UNAVAILABLE). Decision trails: in all four CPU cases the only new text evidence at inventory is 'timeout' log text (80 lines,
0 before) -> msg_timeout, which is a SUPPORT signal of threadExhaustion and unexplained under the CPU types. 'timeout' / 'refused'
report that a call did not complete -- an effect of almost any fault -- not a mechanism.

## Amendment 18 (2026-11-02, before any run of this part): only mechanism-naming text from logs (`textx2`)
textx with the log rule (iii) restricted to the families that name a mechanism: npe, ratelimit, dns, redis, oom, gc, thread, queue,
lb, netpol, sql, cache; 'timeout' and 'refused' log text no longer creates msg_* (span messages and exception attributes unchanged).
No new constants. Expected from the trails: the four CPU cases and t007 return to the frozen-core decision, the three NPE gains stay
(about 21 types right); t045 stays loadBalancerFailure.
Run: GAMA_CORE_TX2 on DEV_ODD (CPU).

### Result of amendment 18 (GAMA_CORE_TX2, DEV_ODD, commit 9cb48ea)
Final 65.0 [bootstrap 95% CI 56.6, 72.7] / EL 76.6 / FI 40.4 / RP 74.0 / group 53.8 (entities 37, types 21) vs frozen core 63.4 / 77.1 /
34.6 / 74.0 (37, 18), as predicted: nullPointerException right in t029, t035, t083; the four CPU cases unchanged; t045 -> loadBalancerFailure
(was codeDefect, both wrong); t065 0.5 -> 0.25; t073 type changed (entity wrong). Wilcoxon vs frozen core p = 0.14; vs
V4_RULE_MM_T3_OMSG p = 0.0015; vs configuration A p = 0.00014. textx2 joins the frozen configuration (V5_FINAL in scripts/kaggle_run.py;
arms GAMA_CORE / GAMA_LLM / EVIDENCE_LLM_FINAL now include it).

## Amendment 19 (2026-11-02, before any run of this part): the reasoning chain (`rpchain`; affects RP only)
RP = 0.5 chain-node match + 0.5 checkpoint hit rate. The chain is the topology path from the decided root to the alerted service; the
checkpoint texts were the infra evidence retrieved around the ALERT only, hence decision-independent (constant ~75.6).
Part `rpchain` (no new constants; the entity and type decisions are untouched): (i) among the shortest root -> alert paths, the one whose
intermediate services hold the most observations (the propagation the evidence supports) instead of an arbitrary one; (ii) the infra
evidence is also retrieved around the decided root (1-hop neighbourhood), deduplicated -- evidence a diagnostician retrieves to support
the chain. Disclosure: (ii) adds evidence text and can raise the checkpoint hit rate also through volume; retrieval precision /
recall are reported next to RP so that this is visible.
Run: GAMA_CORE_RP (= frozen core incl. textx2 + rpchain) on DEV_ODD (CPU); EL and FI must equal GAMA_CORE_TX2 exactly.

### Result of amendment 19 (GAMA_CORE_RP, DEV_ODD, commit 3101593) -- NOT adopted
EL and type identical to GAMA_CORE_TX2 in all 52 cases (check passed). (i) the symptom-supported path changed no chain (chain-node
match 72.1 -> 72.1). (ii) retrieving evidence around the decided root raised the checkpoint hit rate 76.0 -> 83.8 (12 cases up,
0 down), RP 74.0 -> 78.0, Final 65.0 -> 66.2 (Wilcoxon p = 0.002) -- but the gain is LARGER where the decided entity is wrong (+12.2
points) than where it is right (+6.1): the checkpoint test is not entity-specific, so more retrieved text hits more checkpoints whatever
the diagnosis. This measures retrieval volume, not reasoning quality. rpchain is not adopted; the frozen configuration stays
GAMA_CORE (V5_FINAL, incl. textx2; Final 65.0). The result is reported as a sensitivity analysis of the RP metric (checkpoint credit
can be raised by retrieval volume alone), in line with the benchmark authors' own truncation of evidence to suppress keyword stuffing.

## Final design freeze (2026-11-02)
CPU core = V5_FINAL = A + mm + torigin3 + omsg + hostcpu + errclass + tiebreak + sigorder2 + loadx2 + msgtarget + textx2
(arm GAMA_CORE). Odd-half result: Final 65.0 [56.6, 72.7], EL 76.6, FI 40.4, RP 74.0. No further change before VAL_EVEN.

## Amendment 20 (2026-11-02, before any run of this part): evidence at each chain step (`redchain`) and a step-anchored RP
Reference (Cai et al., RCA100 protocol): the ground truth is a typed chain cause -> propagation -> impact with numeric observability
checkpoints PER STEP (e.g. cause payment error_count >= 8829; propagation checkout error_count >= 8828; impact checkout::PlaceOrder);
"an agent must walk the right chain and cite the right evidence at each step". Our scorer matched checkpoints against ALL retrieved
text, which amendment 19 showed to reward volume.
(1) Metric, evaluation only, reported next to the existing ones: rp_checkpoint_hit_step = share of ground-truth checkpoints satisfied
by a predicted chain step that NAMES the checkpoint's step target (signal=value meeting the constraint); reasoning_process_step =
0.5 chain-node match + 0.5 rp_checkpoint_hit_step; final_score_step. Values at other steps or in retrieved tables do not count.
(2) Part `redchain` (no new constants beyond the citation format): every step of the rule chain cites up to 4 metric=value readings of
its own target -- golden signals request / error / latency for services and operations, cpu / memory / disk for nodes -- the window
extreme in the direction of the change from the baseline median, largest relative changes first. The same per-step table is given
to the LLM narrator in GAMA_LLM. Entity and type decisions are untouched.
Run: GAMA_CORE_RED on DEV_ODD (CPU); EL, FI, chain-node match must equal GAMA_CORE_TX2; compare RP, RP_step and the step-anchored
checkpoint rate where the entity is right vs wrong (it should rise mainly where the entity is right).

### Result of amendment 20 (GAMA_CORE_RED, DEV_ODD, commit dad4dba)
Control passed: EL, type, chain-node match and the old RP identical to GAMA_CORE_TX2 (RP 74.0, old checkpoint rate 76.0). New,
step-anchored: checkpoint rate 29.8 (TX2 chain cites no signal=value, ~0), reasoning_process_step 51.0, final_score_step 58.1; the
step-anchored rate is higher where the entity is right (32.5) than where it is wrong (23.3) -- the opposite of the volume effect of
amendment 19. Two weaknesses seen in the chains: (a) a CPU ratio 'deployment_cpu_usage_vs_requests' was cited as a request signal
(name contains 'request'): bug, fixed (resource words excluded for service / operation steps); (b) service-level metrics are named
'error' / 'latency' in the data while checkpoint signals may be 'error_count' / 'avg_request_latency_seconds' (e.g. t001 cites
error=8899 at payment). Diagnostic checkpoint_names_dev.py (odd cases; answer key read for evaluation analysis only) lists the
checkpoint signal names per step against the names our steps cite, before any decision on (b).

### Re-run of amendment 20 with the fix (commit 1e80418) and checkpoint-name diagnostic
Identical scores (step-anchored checkpoint 29.8, RP_step 51.0; right 32.5 vs wrong 23.3). checkpoint_names_dev.py over the four odd
shards (answer key read for evaluation analysis only): ground-truth checkpoints of the cause / propagation / impact steps are mostly
request_count, avg_request_latency_seconds and error_count (service / operation), plus, at CAUSE steps, mechanism metrics
(arms_jvm_gc_delta / gc_seconds_delta, arms_thread_pool_*, arms_jvm_mem_used_bytes, deployment_*_replicas, node_*_usage_rate,
node_ready_status, pod_restart_count, slow_count). Of the checkpoints: about 94 have no step for their target in our chain (entity
or path wrong); 140 are cited under the same name at the right step; 93 are not cited although the step exists -- mainly error_count
(our service-level reading is named 'error' in the data, values agree, e.g. payment error=8899 vs error_count >= 8829) and the
mechanism metrics of cause steps. No renaming is done (the names are the data's own); the error / error_count difference is reported
as a property of the metric.

## Amendment 21 (2026-11-03, before any run of this part): the cause step cites its mechanism readings
Within `redchain`: the cause step additionally cites, as metric=value, the readings behind the root's holding evidence (the metric
named in each holding observation's evidence text, with its window extreme), before the golden signals (at most 4 + 3 readings).
This is what the decision used, now written in the chain. No new constants. Affects RP only.
Run: GAMA_CORE_RED again on DEV_ODD (CPU); EL / FI / chain-node match must stay identical.

### Result of amendment 21 (GAMA_CORE_RED, DEV_ODD, commit 89d3345)
EL / type / chain-node match / old RP identical (control passed). Step-anchored checkpoint rate 29.8 -> 33.1 (9 cases up, 0 down),
RP_step 51.0 -> 52.6, final_score_step 58.1 -> 58.5 (Wilcoxon p = 0.007); higher where the entity is right (36.0) than where it is
wrong (25.9). redchain (amendments 20-21) is adopted: it changes no decision and makes the chain cite the evidence of each step.

## Final configuration (2026-11-03)
V5_FINAL = A + mm + torigin3 + omsg + hostcpu + errclass + tiebreak + sigorder2 + loadx2 + msgtarget + textx2 + redchain
(arms GAMA_CORE / GAMA_LLM / EVIDENCE_LLM_FINAL). Odd half: Final 65.0 [56.6, 72.7], EL 76.6, FI 40.4, RP 74.0;
step-anchored: RP_step 52.6, final_score_step 58.5 [50.4, 66.4]. Both RP variants are reported.

### GPU pilot of GAMA_LLM (Qwen2.5-7B, 6 odd cases chosen as hard / control cases, commit 89d3345)
Runs cleanly: 2-4 LLM calls and 7.2k-11.5k tokens per case, 160-244 s per case; log probes in 5 of 6 cases. Entities unchanged
(6/6 right). Types vs GAMA_CORE: t045 fixed (loadBalancerFailure -> httpError5xx, LLM typing), t067 broken (cpuFullLoad -> cpuDeadLoop:
an exact statistical tie that the core resolves to the general type, the LLM to the special one); t007, t055, t059 still wrong; net 0.
The LLM narrator lowered the reasoning score: step-anchored checkpoints 0 in 5 of 6 cases (rule chain 0.11-0.71), and the old RP
dropped in t059: the narrative cites fewer of the step readings and once attributes a node metric to a service.

## Amendment 22 (2026-11-03, before the full GPU run): the computed chain is the scored chain (`keepchain`)
In GAMA_LLM the scored reasoning chain is the computed chain (topology path + per-step readings, redchain); the LLM narrative is kept
in the output (llm_chain) for the human reader and scored separately on the same metrics (rp_chain_nodes_llm,
rp_checkpoint_hit_step_llm), so that hypothesis H3 (does a stronger LLM write a better chain?) can still be tested. Consistent with
principle P3: the LLM does not replace computed evidence. The LLM still reads logs (sensor) and types the root from the shortlist.
Run: GAMA_LLM (now with keepchain) on DEV_ODD, GPU, 4 shards.

## Amendment 23 (2026-11-03, before any full LLM run): the quote must support the chosen type (`typecheck`)
LLM typing answers were accepted when the type is in the statistical shortlist and the quoted evidence line exists verbatim; the quote
did not have to bear on the chosen type. Now (in GAMA_LLM) the quoted line must name a signal with a key / support / consequence
role in the chosen type's fingerprint, or be log text of a family the fingerprint names (msg_<f> / log_<f>_rise); otherwise the
statistical type is kept. A deterministic soundness check by code, in the spirit of the verifier of DiagGuard (Lu et al.) and the
belief sentinel of PoS (Luo et al.), which are LLM-run. No new constants.

## Evaluation runs (fixed now; no design change after them)
- GAMA_LLM (V5_FINAL + xcard + keepchain + typecheck; Qwen2.5-7B, temperature 0, seed 0, max 1024 new tokens, shortlist 5, sensor
  budget 5, stop at P >= 0.8) on ALL 103 cases (4 shards): the odd half compares with GAMA_CORE, the even half is validation.
- GAMA_CORE on the EVEN half (the odd half is done): the validation of the statistical core.
- Then, on the even half: EVIDENCE_LLM_FINAL and DIRECT_LLM (Qwen2.5-7B), and the cross-model study.

## VALIDATION RESULT -- GAMA_CORE on the even half (51 cases, never used for design; commit 96dacea)
Final 60.8 [bootstrap 95% CI 53.1, 68.6] / EL 61.0 / FI 47.1 / RP 74.2 / group 58.8 (entities 27, types 24); step-anchored RP 50.8,
final_score_step 53.8. Configuration A on the same even cases: 51.4 [44.0, 58.8] (EL 54.6, FI 25.5); paired Wilcoxon p = 0.059;
17 cases better, 6 worse, 28 equal. Odd half (development) 65.0; odd vs even difference not significant (Mann-Whitney p = 0.38).
All 103 cases: 62.9 [57.2, 68.4] vs A 50.0, Wilcoxon p = 4.6e-5.
By group (even): FI improves in every group except Resource and Middleware (equal); EL improves in Application logic, JVM, K8s and
Middleware, but DROPS for Cloud-resource (node) faults (0.41 -> 0.28). The even half has 9 node faults (odd: 5). All six regressions
vs A involve a node: the true node loses to another node (t008, t036: cn-hongkong.10.0.2.59, hosting several error / slow origins,
wins with no node signal of its own), or a node wins over the service (t042, t058, t088). Probable cause (post hoc, not acted on):
the node support of the trace-origin module (a node hosting >= 2 origin services) -- visible only now because the odd half had few
node faults. Reported as a limitation; the leave-one-module-out ablations on the even half will measure it. No design change.

## Leave-one-module-out ablations (even half, CPU; arms GAMA_CORE_NO_{MM, TRACE, TYPE, LOAD, TEXT})
Each = V5_FINAL without one module: MM (mm), TRACE (torigin3, omsg, msgtarget), TYPE (hostcpu, errclass, tiebreak, sigorder2),
LOAD (loadx2), TEXT (textx2). Reported as measured, whatever the outcome.

## Refactor of the frozen method (2026-11-03) -- not an amendment

The frozen configuration V5_FINAL (tag `v5-frozen` = 012ab6e) is re-implemented as the package `gama/` with one switch per ablation
module: M1 `mm`, M2 `trace` (torigin3, omsg, msgtarget), M3 `typing` (hostcpu, errclass, tiebreak, sigorder2), M4 `load` (loadx2),
M5 `text` (textx2), M6 `chain` (redchain), M7 `llm` (sensor, llmtype, xcard, keepchain, typecheck). Development parts that were not
adopted are not in `gama/` (they stay reproducible from the tag). No decision rule, constant or prompt changes.

Equivalence (required before `gama` replaces the development arms in any reported run):
1. synthetic: `tests/test_gama.py` -- identical outputs (entity, type, confidence, chain, evidence, all stats, decision trail) for
   core, each leave-one-out, base, and full with a deterministic stub LLM (identical prompts as well); 4/4 passed (commit d3d4f14);
2. real cases: `python scripts/kaggle_job.py --check core,llm --cases all --shard k/4` (CPU) -- result to be recorded here.
