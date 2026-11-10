# `pipeline/` — Telemetry Extraction & Statistical Calibration Engines

This directory contains the foundational multi-modal telemetry extractors and statistical calibration layers supporting the **GAMA-RCA** framework (`gama/`).

*(Comparative baselines such as Direct LLM, Evidence->LLM, and Belief->LLM are located in the dedicated [`baselines/`](../baselines) package).*

---

## 🏛️ Architecture & 4 Functional Subpackages

```
                     Raw Telemetry (Metrics, Logs, Traces, Events)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│ 1. metric/       │             │ 2. log/          │             │ 3. trace/        │
│  - infra.py      │             │  - preprocess.py │             │  - evidence.py   │
│  - propagation.py│             │  - templates.py  │             │  - structure.py  │
│                  │             │  - probes.py     │             │                  │
└────────┬─────────┘             └────────┬─────────┘             └────────┬─────────┘
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          ▼
                         ┌──────────────────────────────────┐
                         │ 4. calibration/                  │
                         │  - fault.py                      │
                         │  - calibrated.py                 │
                         │  - fingerprint.py                │
                         │  - belief.py                     │
                         │  - typing.py                     │
                         └────────────────┬─────────────────┘
                                          │
                                          ▼
                            Used by `gama/core/` Engine
             (gama.core.evidence ──► gama.core.belief ──► gama.core.decide)
```

---

## 📋 Subpackage & File Reference

### 1. `pipeline/metric/` — Metric & Topology Extractors
| File | Responsibility | Section in Paper |
|:---|:---|:---:|
| [`metric/infra.py`](metric/infra.py) | Builds infrastructure tables (host CPU, memory, replica states). | §4.1 (Metrics) |
| [`metric/propagation.py`](metric/propagation.py) | Slices time windows, discovers upstream callers and downstream callees. | §4.1 (Topology) |

### 2. `pipeline/log/` — Log Pre-processing & Text Probes
| File | Responsibility | Section in Paper |
|:---|:---|:---:|
| [`log/preprocess.py`](log/preprocess.py) | Runs Poisson test & Benjamini-Hochberg FDR control on log bursts. | §4.1 (Logs) |
| [`log/templates.py`](log/templates.py) | Mines log templates via Drain3 without significance filtering. | §4.1 (Logs) |
| [`log/probes.py`](log/probes.py) | Prepares entity-scoped log/event quotes for LLM text sensing. | §4.3 (LLM Sensor) |

### 3. `pipeline/trace/` — Distributed Trace Analyzers
| File | Responsibility | Section in Paper |
|:---|:---|:---:|
| [`trace/evidence.py`](trace/evidence.py) | Aggregates span counts, error rates, and slowest operations. | §4.1 (Traces) |
| [`trace/structure.py`](trace/structure.py) | Identifies deepest failing span to isolate true origin from victims. | §4.1 (Traces) |

### 4. `pipeline/calibration/` — Statistical Core & Fault Taxonomy
| File | Responsibility | Section in Paper |
|:---|:---|:---:|
| [`calibration/fault.py`](calibration/fault.py) | Robust shift z-score ($1.4826 \times \text{MAD}$) and service-level signals. | §4.1 (Calibration) |
| [`calibration/calibrated.py`](calibration/calibrated.py) | Empirical null distribution via pseudo-windows & Mann-Whitney U test. | §4.1 (Calibration) |
| [`calibration/fingerprint.py`](calibration/fingerprint.py) | Formal definitions and discriminating signals for the 28 fault types. | §3.2 (Taxonomy) |
| [`calibration/belief.py`](calibration/belief.py) | Hypothesis prior/likelihood calculations and probe budget tracking. | §4.2 (Belief) |
| [`calibration/typing.py`](calibration/typing.py) | One-call mechanism typing for ambiguous root causes. | §4.3 (Typing) |

---

## 💡 How `pipeline/` Connects to `gama/core/`

The codebase separates **telemetry operations** (`pipeline/`) from **Bayesian abduction** (`gama/core/`):
- `gama.core.evidence` calls `pipeline.calibration.calibrated` and `pipeline.calibration.fault`.
- `gama.core.belief` calls `pipeline.calibration.belief` to compute prior and posterior probabilities.
- `gama.core.chain` uses `pipeline.metric.propagation` to trace the causal graph.
- `gama.core.llm` uses `pipeline.log.probes` and `pipeline.log.preprocess` for quote grounding.
