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
