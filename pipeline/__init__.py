"""pipeline
========
Telemetry extraction, statistical calibration, and comparative baseline engines
supporting the GAMA-RCA framework.

Three Functional Clusters:
1. Evidence Extractors:
   - propagation_evidence: Topology neighborhood & window slicing
   - infra_evidence: Host CPU, memory, and container tables
   - trace_evidence: Per-service HTTP status codes & p95 latencies
   - trace_structure: Distributed trace error-origin identification
   - log_templates: Regex & Drain3 log template mining
   - log_preprocess: Poisson burst testing & Benjamini-Hochberg FDR control
   - semantic_probes: Text extraction for bounded LLM sensing

2. Calibration & Statistical Core:
   - fault_fingerprint: Taxonomy patterns for all 28 microservice fault types
   - fault_calibrated: Pseudo-window empirical null calibration & Mann-Whitney U tests
   - fault: Robust shift z-score scale estimation & event signals
   - belief_diagnosis: Bayesian hypothesis likelihood model
   - llm_typing: Fine-grained mechanism typing arbitration

3. Comparative Baselines:
   - baselines: System registry for main.py (--systems)
   - direct_llm: Alert-only LLM decision baseline
   - evidence_llm: Calibrated-evidence LLM baseline (without Bayesian core)
"""
