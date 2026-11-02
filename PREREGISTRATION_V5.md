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
