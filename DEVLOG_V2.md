# v2 development log (dev-30 only; the 73 held-out cases are untouched)

Protocol: dev-30 is the development set. Design changes may use it, but only (a) general rules justified by the fault-type definitions
or by the method's own assumptions, never case-specific fixes or tuned numbers, (b) logged here with the evidence that motivated them,
(c) at most two rounds; after round 2 the method is frozen (tag) and pre-registered, whatever round 2 shows. The held-out set is used once.

## Round 0 — d9cdd4f, arm V2_RULE (Stages 1-3, no LLM), CPU
- EL mean 54.7 (v1 best: 54.4); FI exact 5/30 = 16.7% (v1: 13-20%); entity exact 8/30, strict (service roll-up) 13/30; abstain 4/30;
  ~100 s per case on CPU, no GPU.
- Of the 13 cases with the entity right (strict), the type was right in 5 (v1: 3 of 13).
- Failure pattern A (8 cases: t029 t055 t058 t070 t072 t078 t093 t100): the top hypothesis is a node-level type admitted on TWO OR
  MORE nodes at once (e.g. nodeCpuHigh on all nodes hosting inventory). A node fault concerns one node; a rise shared by several nodes
  is load. Because the root rule treated any host node with a hypothesis as an explaining dependency, these node hypotheses also
  demoted the correct service hypotheses (e.g. quote:replicaScaleDown in t078) to non-root.
- Failure pattern B (3 cases: t030 t076 t078): nodeDown admitted from a drop in running pods on a node, which a scale-down also causes.
- Note on scoring: the local EL rolls a predicted node up to a nearby service (find_service_ancestor), so a node answer can receive
  EL credit for a service fault (e.g. t029). Report rcl_exact next to EL.

## Round 1 — changes (this commit)
1. Node locality (pattern A): a node-level hypothesis admitted on >= 2 nodes for the same type is "diffuse": listed, but ranked
   with the residual types and not counted as an explaining dependency in the root rule. Ablation arm: V2_RULE_NOLOCAL.
   Justification: definitions of nodeCpuHigh / nodeMemoryOOM / nodeDown / diskIOHigh (one node).
2. nodeDown (pattern B): key = NotReady event or a drop of the ready status; the running-pod count only supports it; a scale-down
   event excludes it. Justification: definition of nodeDown ("node becomes unreachable").
Next: re-run V2_RULE on dev-30 (round 1 check), then GPU smoke of V2 on 4 cases, fingerprint review, pre-registration, freeze.

## Superseded
- PREREGISTRATION.md (FP arms on the v1 pipeline, tag fp-freeze-v1) was not run; replaced by the v2 design. The round-1 change to
  pipeline/fault_calibrated.py (nodeDown signals) therefore does not affect any reported experiment.

## Causal attribution prototype — label-free checks (no labels read or printed)
### Check 1 — 97088f3, scripts/diagnostics/causal_check.py on dev-30 inputs, ALERT_T0_MODE=window
- Calibration OK: pseudo-window false-alarm rate 55/961 = 5.7% (target 5%).
- Coverage: per case 19 services, 15 with a latency series, only ~3 with an error series, **0 with a request series**; 14 nodes with CPU.
- Baseline hold-out R^2 of the latency models: median -0.94 (worse than the mean), and explaining away did not narrow the anomalous
  entities (median 4 before / 4 after).
- Causes found without labels: (1) the APM request count is exported as `workload`, which no metric family matched, so the load
  driver was missing (this also silenced `svc_request_rise` in the fingerprints); (2) error samples exist only when errors >= 1
  (repo history 29e2a97), so error series looked absent; (3) models used every candidate parent, which over-fits 110 samples.
### Fixes (this commit)
1. `workload` added to the request family (fault_fingerprint._METRIC_FAMILIES; affects fault_calibrated as well).
2. Error count zero-filled where a request sample exists.
3. Parent selection by forward selection on the EARLY baseline (before the pseudo-windows; 70/30 time split, a parent is added only
   if it lowers the validation error; at most 3); variables where no parent helps get no explaining away.
Synthetic tests unchanged (20/20, 20/20, 19/20). Next: re-run causal_check on dev-30 inputs.
### Check 2 — 8d343b8, same script and inputs
Criteria fixed BEFORE the run: request series present for most services; median selection skill > 0; fewer anomalous entities after
explaining away; false alarms ~5%.
- Request present 15/19 (met); false alarms 75/1323 = 5.7% (met).
- Median skill of the selected models vs intercept-only = 0.0 (NOT met): parents rarely predict a service's latency/errors on
  held-out baseline data; hold-out R^2 median -0.73.
- Anomalous entities before / after explaining away: median 5 / 5 (NOT met).
**Decision (as fixed beforehand): the per-case linear causal model is not used in v2.** Interpretation for the paper: the ~55 min
of normal operation contains too little co-variation between connected services for a per-case regression to learn propagation
(propagation mainly shows under the fault itself), so regression-based explaining away (CIRCA-style) cannot separate victims
from origins on RCA100; v2 keeps the structural root rule over admitted hypotheses. The module and its checks stay in the repo as
this negative result. The `workload` fix (request family) stays: it is a data-mapping bug fix that also enables `svc_request_rise`.
From here on, dev-30 runs are summarised with scripts/diagnostics/v2_label_free_summary.py only (no label-derived column).
