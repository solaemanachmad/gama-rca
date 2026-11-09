# `pipeline/` — Telemetry Extraction, Calibration & Baseline Engines

This directory contains the foundational telemetry extractors, statistical calibration layers, and comparative baseline engines that support the **GAMA-RCA** framework (`gama/`).

---

## 🏛️ Architecture & Three Functional Clusters

```
                     Raw Telemetry (Metrics, Logs, Traces, Events)
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 1. Telemetry & Evidence Extractors                                                │
│    - propagation_evidence.py   : Call-graph neighbourhood & directional traversal │
│    - infra_evidence.py         : Node and container CPU/memory telemetry tables   │
│    - trace_evidence.py         : Per-service HTTP status codes & p95 latencies    │
│    - trace_structure.py        : Distributed trace origin & deep error detection  │
│    - log_templates.py          : Regex & Drain3 log message template mining       │
│    - log_preprocess.py         : Poisson burst test against pre-incident baseline │
│    - semantic_probes.py        : Text extraction for the bounded LLM sensor       │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 2. Calibration & Statistical Core                                                 │
│    - fault_fingerprint.py      : Signal patterns for all 28 microservice faults   │
│    - fault_calibrated.py       : Non-parametric pseudo-window calibration & MWU   │
│    - fault.py                  : Shift scale estimation (MAD) & event signals     │
│    - belief_diagnosis.py       : Bayesian likelihood models & hypothesis states   │
│    - llm_typing.py             : Fine-grained mechanism typing arbitration        │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
                           Used by `gama/` Orchestrator
         (gama/evidence.py ──► gama/belief.py ──► gama/decide.py)

─────────────────────────────────────────────────────────────────────────────────────

┌───────────────────────────────────────────────────────────────────────────────────┐
│ 3. Comparative Benchmark Baselines (Table 4 in paper)                            │
│    - baselines.py              : Central registry for `main.py --systems`         │
│    - direct_llm.py             : Direct LLM baseline (Alert text only)            │
│    - evidence_llm.py           : Evidence -> LLM baseline (Telemetry without GAMA)│
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 File-by-File Reference

| File | Category | Responsibility | Section in Paper |
|:---|:---:|:---|:---:|
| [`propagation_evidence.py`](propagation_evidence.py) | Evidence | Slices time windows, discovers upstream callers and downstream callees. | §4.1 (Topology) |
| [`infra_evidence.py`](infra_evidence.py) | Evidence | Builds infrastructure tables (host CPU, memory, replica states). | §4.1 (Metrics) |
| [`trace_evidence.py`](trace_evidence.py) | Evidence | Aggregates span counts, error rates, and slowest operations. | §4.1 (Traces) |
| [`trace_structure.py`](trace_structure.py) | Evidence | Identifies deepest failing span to isolate true origin from victims. | §4.1 (Traces) |
| [`log_templates.py`](log_templates.py) | Evidence | Mines log templates via Drain3 without significance filtering. | §4.1 (Logs) |
| [`log_preprocess.py`](log_preprocess.py) | Evidence | Runs Poisson test & Benjamini-Hochberg FDR control on log bursts. | §4.1 (Logs) |
| [`semantic_probes.py`](semantic_probes.py) | Evidence | Prepares entity-scoped log/event quotes for LLM text sensing. | §4.3 (LLM Sensor) |
| [`fault_fingerprint.py`](fault_fingerprint.py) | Calibration | Formal definitions and discriminating signals for the 28 fault types. | §3.2 (Taxonomy) |
| [`fault_calibrated.py`](fault_calibrated.py) | Calibration | Empirical null distribution via pseudo-windows & Mann-Whitney U test. | §4.1 (Calibration) |
| [`fault.py`](fault.py) | Calibration | Robust shift z-score ($1.4826 \times \text{MAD}$) and service-level signals. | §4.1 (Calibration) |
| [`belief_diagnosis.py`](belief_diagnosis.py) | Calibration | Hypothesis prior/likelihood calculations and probe budget tracking. | §4.2 (Belief) |
| [`llm_typing.py`](llm_typing.py) | Calibration | One-call mechanism typing for ambiguous root causes. | §4.3 (Typing) |
| [`baselines.py`](baselines.py) | Baseline | System registry mapping `--systems` to baseline execution functions. | §5.2 (Baselines) |
| [`direct_llm.py`](direct_llm.py) | Baseline | Unaugmented LLM baseline deciding purely from alert text. | §5.2 (Table 4) |
| [`evidence_llm.py`](evidence_llm.py) | Baseline | LLM baseline deciding from calibrated evidence tables without Bayesian core. | §5.2 (Table 4) |

---

## 💡 How `pipeline/` Connects to `gama/`

The codebase separates **low-level telemetry operations** (`pipeline/`) from **high-level Bayesian reasoning** (`gama/`):
- `gama/evidence.py` calls `pipeline.fault_calibrated` and `pipeline.fault` to construct calibrated observations.
- `gama/belief.py` calls `pipeline.belief_diagnosis` to compute prior and posterior probabilities.
- `gama/chain.py` uses `pipeline.propagation_evidence` to trace the causal graph.
- `gama/llm.py` uses `pipeline.semantic_probes` and `pipeline.log_preprocess` for faithful quote grounding.
