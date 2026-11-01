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
