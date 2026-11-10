# GAMARCA Empirical Results & Benchmarks

This directory organizes all experimental outputs, predictions, and empirical scorecards for **GAMARCA** across the standard microservice RCA benchmarks.

---

## 🏆 Benchmark Leaderboard

### 1. RCA100 (103 Real-world Microservice Incidents)
Evaluated across all 103 cases under frozen pre-registration (Cai et al., 2026):

| Arm / Method | Compute | Localization (EL) | Fault Type (FI) | Reasoning Path (RP) | Final Score | Exact (All 3) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Direct LLM (7B) | GPU / API | 28.0 | 14.6 | 10.9 | 18.8 | 0 / 103 |
| Evidence $\rightarrow$ LLM (7B) | GPU / API | 71.8 | 20.4 | 26.2 | 42.7 | 18 / 103 |
| Belief $\rightarrow$ LLM (7B) | GPU / API | 67.9 | 43.7 | 25.8 | 48.0 | 39 / 103 |
| **GAMA Core (Statistical)** | **CPU only** | **68.9** | **43.7** | **74.1** | **62.9** | **39 / 103** |
| GAMA Full (+ bounded 7B) | CPU + GPU | 66.5 | 35.0 | 73.7 | 59.2 | 30 / 103 |
| GAMA Full (+ gpt-oss-120b) | CPU + API | 65.7 | 36.9 | 73.7 | 59.4 | 32 / 103 |

> **Key takeaway**: GAMA Core on CPU outperforms unguided LLM reasoning by **+20.2 points Final** ($p = 2.8 \times 10^{-10}$) at **zero token cost**.

---

### 2. AIOps2025 (400 Zero-Shot Generalisation Incidents)
Zero-shot transfer to an unseen Kubernetes cluster with TiDB, Redis, and Chaos Mesh injections:

| Benchmark System | Topology | Modalities | Hardware | Throughput |
|:---|:---|:---|:---:|:---:|
| HipsterShop + TiDB + Redis | 14 services, 31 pods, 11 nodes | Metrics, Logs, Traces | CPU (4 cores) | 2–9s per case |

---

## 📂 Subfolder Layout

* [`rca100/`](rca100/):
  * [`core/summary.csv`](rca100/core/summary.csv): Full 103-case diagnosis metrics for GAMA Core.
  * [`ablation/ablation_summary.csv`](rca100/ablation/ablation_summary.csv): Quantitative evaluation of $M_1$ through $M_6$ module contributions.
* [`aiops2025/`](aiops2025/):
  * [`core/`](aiops2025/core/): Clean 12-column per-case diagnosis summaries, predictions JSONL, and empirical scorecards.

---

## 🚀 How to Reproduce

### Run AIOps2025 on CPU (Parallel):
```bash
python scripts/aiops2025_run.py \
  --data /kaggle/input/aiops2025 \
  --out results/aiops2025/core \
  --workers 4
```

### Run RCA100 on CPU:
```bash
python main.py --systems gama --cases all
```
