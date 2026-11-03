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
