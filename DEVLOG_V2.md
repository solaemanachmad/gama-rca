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
