# GAMARCA — Graph-aware, Multimodal, Abductive Root Cause Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Pre-registration: v6](https://img.shields.io/badge/Pre--registration-v6%20Frozen-success.svg)](PREREGISTRATION_V6.md)
[![Tests](https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen.svg)](tests/)

Official research code and reproducibility package for:
> **GAMA-RCA: Graph-aware, Multimodal, Abductive Root Cause Analysis for Microservices with a Computed Belief and a Bounded Language Model**  
> *Achmad Solaeman, Nur Rohman Rosyid, Ardhi Wicaksono, Prima Asmara Sejati, and Hamzah*  
> Universitas Gadjah Mada & PT Media Pangan Kreatif Indonesia.  
> Preprint manuscript available in [`paper/main.pdf`](paper/main.pdf).

---

## 📌 Overview

**GAMARCA** provides label-free, zero-shot root cause analysis (RCA) for complex microservice failures across multimodal telemetry (metrics, logs, distributed traces, and Kubernetes events) along service and host topologies.

Unlike purely generative LLM-agent approaches that consume millions of tokens per incident and suffer from anchoring bias, **GAMARCA computes an auditable abductive belief over root-cause hypotheses $(\text{entity} \times \text{fault type})$ on a standard CPU in seconds**. 

A bounded Large Language Model (e.g., Qwen2.5-7B or gpt-oss-120b) serves as an **entropy-gated adjudicator** and **evidence-cited narrator**, intervening selectively where statistical signals are ambiguous without overriding calibrated evidence.

```
                      [Multimodal Telemetry]
                 (Metrics, Logs, Traces, Events)
                                │
                                ▼
         ┌───────────────────────────────────────────────┐
         │ 1. Calibrated Evidence Engine (CPU)          │
         │    - History & peer calibration               │
         │    - Directional load explain-away            │
         │    - Cross-modal corroboration                │
         └──────────────────────┬────────────────────────┘
                                │
                                ▼
         ┌───────────────────────────────────────────────┐
         │ 2. Abductive Bayesian Belief (CPU)            │
         │    - Fingerprint likelihoods (K/U/M/C)        │
         │    - Expectation-Maximisation (EM)            │
         │    - Temporal onset precedence                │
         └──────────────────────┬────────────────────────┘
                                │
                    P(e*) >= 0.8? 
                     ├── YES ──► Computed Two-Stage Decision
                     └── NO  ──► G1 Entropy Gate: Bounded LLM Adjudication
                                │
                                ▼
         ┌───────────────────────────────────────────────┐
         │ 3. Evidence-Cited Causal Chain & Narration    │
         │    - Step-anchored metric & event citation    │
         │    - Prescriptive SRE explanation             │
         └───────────────────────────────────────────────┘
```

### Core Principles
1. **P1 Calibrate, do not threshold:** Every observation is evaluated against the incident's own historical baseline and peer distributions.
2. **P2 Causes precede and propagate directionally:** Latency and errors propagate up the call graph; load propagates down. Explaining symptoms away prevents false root attribution.
3. **P3 Bounded & Calibrated LLM Integration:** The LLM inspects unstructured text and arbitrates borderline cases where the statistical posterior is uncertain ($P(e^*) < 0.8$), deferring when discriminating evidence is absent.

---

## 🔬 Benchmark Results (RCA100)

Evaluated under strict pre-registration across all 103 microservice incidents in **RCA100**:

| Method / Arm | Compute | Entity Loc. (EL) | Fault Type (FI) | Reasoning Path (RP) | Final Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| Evidence $\rightarrow$ LLM (7B) | GPU / API | 61.2 | 20.4 | 39.8 | 42.7 |
| Belief $\rightarrow$ LLM (7B) | GPU / API | 70.9 | 43.7 | 45.1 | 55.0 |
| **GAMA Core (Statistical)** | **CPU only** | **78.6** | **43.7** | **61.4** | **62.9** [57.4, 68.5] |
| GAMA v5 Full (+ bounded 7B) | CPU + GPU | 76.7 | 39.8 | 61.0 | 59.2 |
| GAMA v5 Full (+ gpt-oss-120b) | CPU + API | 77.7 | 40.8 | 60.3 | 59.4 |
| **GAMA v6 (+ gate + defer)** | **CPU + API** | *In progress* | *In progress* | *In progress* | *Confirmatory* |

*Note: GAMA Core on CPU outperforms unguided LLM reasoning by **+20.2 points Final** ($p = 2.8 \times 10^{-10}$), while the bounded LLM significantly enhances explanation quality (narrator checkpoint rate $0.107 \rightarrow 0.284$).*

---

## 📂 Repository Structure

```
gama-rca/
├── gama/                      # Proposed framework: Bayesian abduction, explain-away, decision engine
│   ├── core/                  # Reasoning modules (evidence, belief, chain, decide, llm)
│   ├── config.py              # Hyperparameters, module switches, and ablation presets
│   └── run.py                 # Single-case pipeline coordinator
│
├── baselines/                 # Comparative benchmark baselines (Table 4 in paper)
│   ├── direct_llm.py          # Direct LLM baseline (Alert only)
│   ├── evidence_llm.py        # Evidence -> LLM baseline (Calibrated telemetry without GAMA)
│   └── belief_llm.py          # Belief -> LLM baseline (Unconstrained LLM on computed belief)
│
├── pipeline/                  # Foundation library: telemetry extractors & statistical calibration
│   ├── metric/                # Metric & infrastructure telemetry (infra, propagation)
│   ├── log/                   # Log pre-processing & text probes (preprocess, templates, probes)
│   ├── trace/                 # Trace span aggregations & deep origin analysis (evidence, structure)
│   └── calibration/           # Statistical shift scoring & taxonomy (fault, calibrated, fingerprint)
│
├── data/                      # Dataset modules & taxonomies
│   ├── rca100/                # RCA100 loader & 28-fault taxonomy
│   ├── aiops2025/             # AIOps2025 loader & 19-fault taxonomy
│   └── dataset_profile.py     # Profile router & metric regex patterns
│
├── evaluation/                # Official scoring protocols (EL, FI, RP, Final)
├── scripts/                   # Experiment runners (ablation, kaggle_job, aiops2025_run, analyze)
├── tests/                     # Unit tests & deterministic golden snapshots
└── paper/                     # LaTeX manuscript source, figures, tables, and preprint PDF
```

---

## ⚡ Quickstart & Reproducibility

### 1. Installation

Clone and install dependencies (Python 3.10+):

```bash
# Clone the repository
git clone https://github.com/solaemanachmad/gama-rca.git
cd gama-rca

# Recommended: setup virtual environment with uv (or standard venv)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### 2. Run Tests (Verification without Dataset or GPU)

All unit tests and golden-snapshot checks run in ~5 seconds on CPU:

```bash
make test
```

Expected output:
```
10/10 passed   (test_fingerprint.py)
11/11 passed   (test_calibrated.py)
8/8 passed     (test_logpre.py)
5/5 passed     (test_gama.py)
3/3 passed     (test_aiops2025.py)
3/3 passed     (test_api_client.py)
4/4 passed     (test_rules.py)
```

### 3. Run Leave-One-Out Ablation Study

Run the fast multi-module ablation suite (~2 seconds on CPU):

```bash
make ablation
```

### 4. Re-running Benchmark Analyses

To reproduce statistical analyses and figures from scored experiment runs:

```bash
# Generate Table A, Table B, and markdown report from run archives
python scripts/analyze.py \
  --arm GAMA_CORE="results/GAMA_CORE_*.zip" \
  --arm EVIDENCE_LLM="results/EVIDENCE_LLM_*.zip" \
  --out report.md

# Regenerate manuscript figures
cd paper && python make_figs.py
```

---

## 🛡️ Pre-registration & Scientific Integrity

To maintain strict scientific integrity, all experimental configurations, hypotheses, and analysis scripts were pre-registered before running test sets:
- [`PREREGISTRATION_V5.md`](PREREGISTRATION_V5.md): Final v5 evaluation protocol on RCA100.
- [`PREREGISTRATION_V6.md`](PREREGISTRATION_V6.md): v6 selective LLM rules (`gate`, `defer`, `calsensor`).
- [`PREREGISTRATION_AIOPS2025.md`](PREREGISTRATION_AIOPS2025.md): Zero-shot generalization protocol on AIOps2025.

---

## 📊 Datasets & Licensing Notice

This repository contains research code under the Apache 2.0 License. Benchmark datasets carry their own non-commercial research licenses:
- **RCA100**: Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Ground truth answer keys are not redistributed.
- **CCF AIOps 2025**: Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Telemetry is processed label-free.

---

## 📖 Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{solaeman2026gama,
  title     = {{GAMA-RCA}: Graph-aware, Multimodal, Abductive Root Cause Analysis for Microservices with a Computed Belief and a Bounded Language Model},
  author    = {Solaeman, Achmad and Rosyid, Nur Rohman and Wicaksono, Ardhi and Sejati, Prima Asmara and Hamzah},
  journal   = {Preprint},
  year      = {2026},
  url       = {https://github.com/solaemanachmad/gama-rca}
}
```

---

## 📄 License

Code is licensed under the [Apache License 2.0](LICENSE).
