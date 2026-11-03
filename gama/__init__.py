"""GAMA-RCA: Graph-aware, Multimodal, Abductive root-cause analysis for microservices. Label-free: an abductive belief over
(entity x fault type) hypotheses is COMPUTED from calibrated multimodal evidence along the service / host topology; a bounded local
LLM reads text and selects among ranked options (optional).

Principles
  P1 calibrate, do not threshold: every observation is scored against the case's own history and peers; rates, not cut-offs
  P2 causes precede and propagate directionally: errors flow up the call graph, load flows down; what the root explains is not
     evidence against it, and what came from upstream is not evidence for a downstream root
  P3 the LLM reads text and chooses among ranked options; it never overrides calibrated evidence

Stages (files)                                          Modules (one switch each, gama/config.py; ablation unit of the paper)
  1 calibrated evidence      gama/evidence.py             M1 mm      cross-modal corroboration
  2 Bayesian belief + EM     gama/belief.py               M2 trace   trace structure (origins, origin messages, endpoint relay)
  3 two-stage decision       gama/decide.py               M3 typing  type stage (host CPU, error class, tie-break, signal order)
  4 bounded LLM (optional)   gama/llm.py                  M4 load    directional load propagation
  chain                      gama/chain.py                M5 text    mechanism-naming text
  one case -> RCAResult      gama/run.py                  M6 chain   evidence-cited reasoning chain
                                                          M7 llm     sensor, shortlist typing, narrator

Lower-level evidence extraction (dataset adapters and tests) stays in pipeline/: fault_calibrated.py (signal database, fingerprints),
fault_v4.py (shift scoring, peers, events), trace_structure.py, belief_diagnosis.py (constants, background, explainers, probe model),
log_preprocess.py / semantic_probes.py (text for the LLM), llm_typing.py; data/dataset_profile.py holds the only dataset-specific
knowledge. The development history (all amendments and their switches) is reproducible from git tag v5-frozen.
"""
