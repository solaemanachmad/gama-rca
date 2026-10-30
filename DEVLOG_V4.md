# v4 development log (design: claude/RCA_v4_design.md in the project)

Status: RCA100 is DEVELOPMENT data for v4. Its labels were seen in the V3_RAWLOG run (6/103 joint; error analysis in
claude/V3_RAWLOG_results_103.md); this is disclosed. The final test of v4 is on an unseen benchmark, frozen and pre-registered before it runs.

## Inputs to the design (no further label use)
- Label-free data audit (scripts/diagnostics/data_audit.py, 103 cases): coverage is not the problem (node, APM, JVM metrics are read);
  injector workloads (l3reg-*, chaosblade) are present and must stay excluded; node .16.66 has sparse series; JVM metrics exist for one
  service; deployment_availability_rate and APM 'slow' were unmapped.
- Mechanism found in the code: rank-test saturation (v3 scored the minimum Mann-Whitney p; pseudo-window drifts reach the same floor).

## Changes (each switchable via V4_PARTS for ablation)
| Part | Change | Synthetic test (tests/test_v4.py) |
|---|---|---|
| score | robust shift z instead of a rank test; coverage rule (80% of history); zero-filled event counts; constants and injector workloads excluded; small null pools fall back to the entity-kind pool (MIN_POOL 20) | shift_size_is_scored_not_only_order, sparse_series_is_not_tested, constants_and_injectors_excluded, strong_node_cpu_detected_despite_drifting_null, sparse_node_gives_no_signal, single_service_jvm_signal_can_reach_alpha |
| open | "anomaly of unspecified type" hypothesis per entity | unfingerprinted_anomaly_is_still_localised |
| twostage | entity by marginal posterior (equal prior per entity), then type given entity; no "no fault" answer | node_cpu_fault_is_answered_at_the_node |
| fp | replicaScaleDown + svc_avail_drop (key) and svc_request_drop; memoryPressure + svc_cpu_rise; cpuDeadLoop + svc_request_drop | silent_scaled_down_callee_beats_its_caller, memory_pressure_beats_cpu_when_heap_and_gc_rise |
v3 code paths are unchanged (v3_engine_untouched_when_v4_unused; make test).
Fixed constants (a priori): REL_FLOOR 5% of the baseline level, ABS_FLOOR 1e-6, COUNT_FLOOR 1 event/bin, MIN_COVERAGE 0.8, MIN_POOL 20.

## Dev run 1 (CPU, no LLM, 103 cases, commit b5293ef)
| Arm | joint | entity | type | group | mean EL |
|---|---|---|---|---|---|
| V3_RULE | 6 | 32 | 9 | 28 | 0.424 |
| V4_RULE | 4 | 38 | 7 | 38 | 0.493 |
| V4_RULE_NOSCORE | 7 | 30 | 11 | 30 | 0.427 |
| V4_RULE_NOTWOSTAGE | 4 | 37 | 7 | 36 | 0.493 |
Localisation improves (EL +0.07) but joint accuracy does not. Remaining failure modes:
1. Node CPU faults (12): still no node_cpu_rise at any node in most cases; h0 ~0.6-0.7. Open question: did the fault start before the
   alert window (then the baseline contains the fault and no shift is visible)?
2. node .16.66 / nodeMemoryOOM is still the default answer in 18 cases: its mass comes from categorical evidence (events), which has no
   baseline (chronic events look new).
3. memoryPressure (12, inventory located in 11): cpuDeadLoop wins. The fp addition "cpuDeadLoop + svc_request_drop" also fires under memory
   pressure, and svc_mem_rise does not hold. That addition was a design error and should be revisited.
4. replicaScaleDown (6): svc_avail_drop does not hold at any service.
Next step: per-case timelines (scripts/diagnostics/v4_case_dump.py) for selected dev cases of each failure mode.

## Case timelines (scripts/diagnostics/v4_case_dump.py; dev cases t008 t036 t069 t024 t047 t065 t081 t004)
Mechanisms seen in the raw timelines (5-minute medians), not guessed from scores:
1. **Non-stationary history with common-mode steps.** The 55 min before a window contains other injected faults (t008: node 10.0.1.107
   at 100% CPU 35 min earlier, 10.0.1.49 at 100% 50 min earlier; t069: 10.0.1.49 and 10.0.1.56 at 100% in the history) and a load step
   that raises request count and CPU of every service ~15 min before the window. The pseudo-window null then contains 100%-CPU
   episodes, so the real fault in the window (t008: 10.0.1.56 2% -> 100%; t069: 10.0.1.107 5% -> 100%) gets calibrated p 0.04-0.06.
   Every service shows the same load step, so per-entity history comparisons see a "rise" everywhere.
2. **Absent propagated symptoms penalise wide hypotheses.** A node (or an upstream service) can explain symptoms at many services;
   with THETA_P = 0.6 every one of them that shows no symptom costs log(0.4/0.95). Real faults reach only some replicas or callers.
3. **Chronic events count as new evidence.** Node 10.0.16.66 has EvictionThresholdMet since 2026-03-15 (count ~247,000); a prometheus
   pod has OOMKilling/BackOff since the day before. They overlap every window and give node .66 its nodeMemoryOOM mass.
4. **Availability is exported as a state, sparsely.** t081: quote's deployment_available_replicas exists only inside the window (value 0);
   the coverage rule makes it untestable, although "available < desired" is unavailability by definition and needs no history.
5. memoryPressure vs CPU/GC types: inventory shows heap, GC, thread, CPU (to 97% of limit) and latency rising together; the CPU of
   its host nodes rises too, which the LOAD_TYPES rule credits to the CPU types. Overlapping mechanisms; no fix proposed from these cases.

## v4.1 (from the case timelines; synthetic tests in tests/test_v4.py, 18/18)
| Part | Change | Test |
|---|---|---|
| peer | common-mode rejection: per window, z re-standardised against the same signal of the other entities of the kind (median, MAD, floor 1) | common_mode_load_step_cancels_with_peers |
| rnull | Efron's empirical null by central matching (normal fitted to the median and MAD of the pseudo-window scores) | earlier_fault_in_history_does_not_mask_the_real_one |
| prop | absent symptom at a reachable entity = uninformative (partial propagation) | absent_symptoms_do_not_penalise_a_wide_hypothesis |
| events | an event counts only if its window rate exceeds its own long-run rate (Poisson, alpha 0.05) | chronic_events_are_not_new_but_fresh_ones_are |
| avail | svc_unavailable: available < desired or availability < full in the window, not chronic; key of replicaScaleDown | unavailability_is_a_state |
Not changed: memoryPressure vs CPU/GC types (overlapping mechanisms; left to the text probes, reported as a limitation).

**Stopping rule (agreed before the run):** v4.1 gets ONE more development run on RCA100 (CPU arms, no LLM). After it, v4 is frozen
whatever the result; no further changes are motivated by RCA100 labels. Final claims rest on the frozen method.

## Dev run 2 = the last RCA100 development run (CPU, no LLM, 103 cases, commit d97a61b, two Kaggle accounts)
| Arm | joint | entity | type | group | mean EL | Final |
|---|---|---|---|---|---|---|
| V3_RULE (run 1) | 6 | 32 | 9 | 28 | 0.424 | 0.337 |
| V4.0_RULE (run 1) | 4 | 38 | 7 | 38 | 0.493 | 0.358 |
| **V4.1_RULE (all parts)** | 7 | 36 | 10 | 24 | 0.547 | 0.390 |
| V4.1 without peer | 8 | 35 | 12 | 25 | 0.512 | 0.382 |
| V4.1 without rnull | **9** | 34 | **16** | 34 | **0.577** | **0.420** |
| V4.1 without prop | 4 | 39 | 8 | 32 | 0.491 | 0.361 |
Observations:
- Node faults are now found (Cloud resource: 3 joint, was 0).
- Partial propagation (prop) is the part that matters most (removing it: joint 7 -> 4).
- The central-matching null (rnull) hurts on every measure.
- A new attractor appears: node cn-hongkong.10.0.2.59 is the answer in 19 cases (13 without rnull), mostly Application-logic ground
  truth. Once absent symptoms are free, a wide (node) hypothesis is cheap. Application logic (38 cases) falls to 6 exact entities.
- The differences between the V4.1 variants (7-9 of 103) are within noise for McNemar.
Per the stopping rule, no further changes are made from RCA100 labels.

## Scoring bug found while reading run 2 (2026-10-30)
evaluation/scoring.py rolled every predicted id up to an apm.service by BFS (max 3 hops) before comparing. A k8s.node reaches a service
through node -> pod -> instance -> service, so a node answer could inherit an arbitrary hosted service and score EL = 1.0 (and
rcl_strict = 1). In run 2, 18 node answers (V4.1 without rnull) scored EL = 1.0 against a service ground truth. **joint accuracy
(rcl_exact AND ft_binary) uses the exact id and was never affected**, nor the V3 pre-registered analysis. EL, rcl_strict and Final of
all runs that predicted nodes are inflated, so the EL gains of V4.1 are partly an artifact.
Fix: nodes, clusters and namespaces are not rolled up (test scoring_does_not_roll_a_node_up_to_a_service). scripts/rescore_el.py
recomputes EL / rcl_strict / Final of existing CSVs (needs the dataset; CPU, minutes).

## v4.2 (development plan ideas 1-3; synthetic tests 22/22)
| Part | Change | Test |
|---|---|---|
| onset | temporal precedence: onset = first sample >= 3 robust SDs (twice) from one window before w0, history ending there; for h = (e*, .), every service e* can explain with its own onset contributes P(onset_e >= onset_e* - 60 s) = 0.9 vs 0.5 | onset_order_separates_root_from_later_victim |
| em | propagation probability pi estimated per case by EM (posterior-weighted share of reachable symptoms that hold; clamped to [0.06, 0.95]); replaces the all-or-nothing THETA_P / neutral choice | em_estimates_a_propagation_probability_and_penalises_wide_quiet_hypotheses |
| ops | operation-level signals op_error_rise / op_latency_rise (the most affected operation; zero-filled error counts); key of httpError5xx, support of codeDefect / NPE / rateLimiting; own and propagated symptoms | failing_operation_is_seen_when_the_service_total_is_diluted |
Default parts = all except rnull (it hurt on every measure in dev run 2: a model-selection step on development data, disclosed).
EL / Final from this run on use the corrected scorer (no node roll-up): compare with earlier runs on joint accuracy only.

## Chain format (v4 only; does not change entity or type)
The benchmark's chain is cause -> propagation -> impact, the impact step naming the alerted apm.operation (benchmark paper, Fig. 6).
v4 now gives the narrator the alerted operation and the topology call path (root -> ... -> alerted service) and asks for one step
per path entity; without an LLM, rule_chain builds the same chain deterministically. v3 prompts unchanged.
Test: chain_follows_the_topology_path_to_the_alerted_operation.

## Dev run 3 (CPU, no LLM, 103 cases, commit cec8b9d, four notebooks; corrected EL scorer)
| Arm | joint | entity | type | group | EL | RP (Jaccard proxy) | Final |
|---|---|---|---|---|---|---|---|
| V3_RULE (run 1) | 6 | 32 | 9 | 28 | (old scorer) | 0.470 | - |
| V4.1 without rnull (run 2) | 9 | 34 | 16 | 34 | (old scorer) | 0.475 | - |
| **V4.2** | 13 | 46 | 19 | 42 | 0.539 | 0.427 | 0.399 |
| V4.2 without onset | 11 | 44 | 17 | 40 | 0.522 | 0.426 | 0.386 |
| V4.2 without em | 9 | 34 | 15 | 30 | 0.423 | 0.423 | 0.340 |
| **V4.2 without ops** | **14** | **48** | **20** | **46** | **0.565** | 0.426 | **0.412** |
Observations:
- The EM propagation probability is the largest contributor (removing it: joint 13 -> 9, entity 46 -> 34).
- Onset precedence adds 2 joint and 2 entity.
- Operation-level evidence (ops) does not help: one case fewer on every measure.
- Joint accuracy has doubled from V3_RULE (6 -> 13/14 of 103). The answer is no longer a single node: the most frequent
  predicted entity is now inventory, which is also the most frequent ground-truth entity.
- The RP proxy fell from 0.475 to 0.427 because the new rule chain names evidence in every step. The repo's chain part was a
  word-Jaccard against "type: target", which penalises extra words. The benchmark paper defines it as a node match rate, so
  evaluation/scoring.py now also reports rp_chain_nodes / reasoning_process_nodes / final_score_nodes (node match + checkpoint hit
  rate). The Jaccard columns are kept for continuity.

## Audit of runs 1-3 (2026-10-30) and v4.3
Audit (V4.2 without ops, 103 cases):
- **Ground-truth concentration:** inventory is the root cause in 59 of 103 cases (57%). A trivial "always inventory" answer would
  localise 57%, more than V4.2 (48/103). Entity accuracy alone is therefore a weak measure on RCA100. Joint accuracy (entity AND
  type) is the primary outcome, and the paper reports this baseline. The alerted service is the root cause in 31 cases
  ("answer = alerted service" baseline: 31).
- Of the 34 entity-right / type-wrong cases, 17 are cpuDeadLoop answers, caused by the v4.0 support "request drop". In 19 of the 34,
  the right type is not among the top five statistical candidates.
- New attractor: recommendation is the answer in 25 cases (16 of them inventory ground truth), mostly as trafficSurge. It is a volatile
  service (its workload and CPU swing in normal periods).
- Joint accuracy is fragile across variants: 15 cases are joint-correct in at least one V4.2 arm, 5 in all four.
- Never joint-correct so far in any run: 21 of the 28 types (all K8s-lifecycle types, slowSQL, NPE, trafficHotspot, ...).
- The posterior is not calibrated as a confidence: joint 13/72 when p_type >= 0.5.
v4.3:
- fp: the cpuDeadLoop + svc_request_drop support is removed.
- selfnorm: each entity's z is normalised against its own pseudo-window z (leave-one-out).
- llmtype: an LLM typing stage (pipeline/llm_typing.py) picks the type for the chosen entity from all types of its level, given the
  evidence summary and the statistical ranking. It is accepted only with a candidate type and a verbatim evidence line.
- ops is opt-in.
Tests 26/26: volatile_entity_loses_to_one_that_changed_now, llm_typing_needs_a_candidate_and_a_verbatim_line.

## Dev run 4 (commit 6b1d853): v4.3 on CPU and the first GPU run (Qwen3-8B)
| Arm | joint | entity | type | group | final (node-match RP) | s/case |
|---|---|---|---|---|---|---|
| V4_RULE without selfnorm (CPU) | **16** | 44 | **22** | 44 | 0.492 | ~78 |
| V4_RULE (CPU) | 15 | 42 | 21 | 37 | 0.481 | 78 |
| **V4 (GPU: log probes + LLM typing + narrator)** | 13 | 40 | 17 | 38 | 0.463 | 158 |
Observations:
- Removing the cpuDeadLoop request-drop support helped (+2 joint, +2 type). selfnorm did not: the recommendation attractor
  stays at 25 cases. recommendation's request count genuinely rises during these faults (retries / load), and the model reads it
  as trafficSurge.
- **LLM typing stage (Qwen3-8B, no thinking): net harmful.** It changed 37 of 103 types: 5 right -> wrong, 1 wrong -> right,
  31 wrong -> other wrong. The frequent changes are cpuFullLoad / cpuDeadLoop -> podRestartFlapping (12) and trafficSurge ->
  dnsResolutionFailure (4).
  - The prompt told the LLM to prefer direct mechanism evidence (events, messages). Restart events, which are consequences, then
    win.
  - The evidence summary rarely contains what separates the true type.
- Log probes ran in 31 cases and changed the entity in 3 (2 of those for the worse). Narration gives the same node-match RP as the
  rule chain (0.71).

## Audit before run 5 (2026-10-30) and v4.4
Bugs found by rendering every v4 prompt on a synthetic case (not from labels):
1. **The v4 narrator was never called.** Its prompt was built but generate_json sat in the v3 branch only. GPU run 4 therefore used
   the rule chain, so its RP equals the CPU arm's. Fixed; test v4_narrator_is_called_and_rule_chain_skips_the_alert.
2. **rule_chain listed the alert pseudo-observation** as a propagation step's evidence. Fixed.
3. **A single new event never counted.** new_events tested a first-seen object against a pseudo-rate of 0.5/horizon, so one occurrence
   gave p = 0.12; one-off OOM kills were dropped. Now an object first seen within the horizon is new by definition; objects with a
   history keep the Poisson test. Test a_single_new_event_counts.
4. LLM typing v1 offered all types of the level and told the LLM to prefer events and messages (dev run 4: 12 restart answers that
   were consequences). Replaced by v2: a shortlist of the 5 most probable statistical types, an onset for every signal line, and
   explicit guidance that events, codes and messages are often consequences.
v4.4 parts:
- conseq: restart / probe-failure / OOM events get the role 'may' (explained when present, uninformative when absent) for the
  resource types. The OOM kill no longer excludes memoryPressure / fullGC. resourceLimitMisconfig loses the memory-rise support
  (it is OOM at normal usage).
- retry: a request rise at a reachable entity is a propagated symptom.
Tests 31/31. Each new part has a test that fails without it: restarts_after_cpu_saturation_are_cpu_types (without conseq:
podRestartFlapping, the run-4 error), oom_and_restarts_after_memory_growth_are_memory_pressure, and
retry_request_rise_upstream_is_explained_by_the_failing_callee.
Known, not changed: the narrator's evidence tables (v1 code) state a legacy window description; they are narrative context only.

## Dev run 5, CPU part (commit 999d665; V4_RULE_NOCONSEQ and the GPU arms pending)
| Arm | joint | entity | type | group | EL | final (node-match) |
|---|---|---|---|---|---|---|
| V4.4 V4_RULE | 15 | 47 | 21 | 39 | 0.555 | 0.500 |
| V4.4 without retry | **17** | 46 | **23** | 40 | 0.544 | 0.501 |
retry does not help (-2 joint). The recommendation attractor persists (22-24 cases). Seminar charts: results-2/grafik_seminar
(5 PNGs; numbers from runs 1-5, labelled as development results).

## Dev run 6 — V4 + local LLM (GPU, Qwen3-8B, commit 999d665, parts incl. retry)

Same parts as V4_RULE in results-2, so the pair is matched (only the LLM differs).

| | Joint | EL | FI | RP(nodes) | Final(nodes) | s/case |
|---|---|---|---|---|---|---|
| V4_RULE (CPU) | 15 | 55.5 | 20.4 | 72.4 | 50.0 | 88 |
| V4 + LLM (GPU) | 19 | 53.8 | 22.3 | 73.9 | 50.4 | 193 |

- Joint +4 (t027, t041, t051, t071), none lost. Final Δ +0.4, Wilcoxon p = 0.31 (not significant).
- LLM typing v2: accepted 99/103, changed 22: 5 wrong→right, 2 right→wrong (t049, t088: F004→F006).
  17 of the 22 changes are F012 cpuDeadLoop → F034 cpuFullLoad (near-duplicate classes); 3 correct.
- Text sensor probed in 31 cases; moved the entity in 4 cases, all worse (t020, t026, t046, t076) → EL −1.7.
- Cost: 2.3 LLM calls, ~8.5k tokens per case.
- Takeaway: the LLM helps typing slightly and never broke a joint hit, but the gain is within noise.
  The statistical core carries the result. Pending: V4_NOLLMTYPE (sensor+narrator only) to separate the two LLM roles.
