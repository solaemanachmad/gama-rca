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
