# AIOps2025 Benchmark Results

This directory contains evaluation outputs, predictions, and empirical scorecards for **GAMA-RCA** transferred zero-shot to the **AIOps2025** benchmark (CCF International AIOps Challenge; HipsterShop + TiDB + Redis on Kubernetes, Chaos Mesh faults, 400 cases; Cai et al., 2026).

---

## 🔬 Benchmark Overview

* **Zero-Shot Transfer**: Evaluates generalisation to an unseen microservice topology without retraining or tuning any statistical hyperparameters.
* **Evaluation Metrics**:
  * **Localization Accuracy (LA)**: Whether the predicted entity matches the ground truth failure root (Service, Pod, or Node).
  * **Type Accuracy (TA)**: Whether the predicted fault type matches the ground truth failure injection mechanism.
  * **Final Score**: Harmonic / weighted mean: $0.5 \times \text{LA} + 0.5 \times \text{TA}$.

---

## 📁 Directory Structure

```
results/aiops2025/
├── README.md                      # Benchmark instructions & guidelines
└── core/                          # Outputs from GAMA Core (CPU Only)
    ├── README.md                  # Output format specification
    ├── summary.csv                # 12-column per-case diagnosis summary
    ├── predictions.jsonl          # Submission JSONL format with step-anchored trace
    └── score.csv                  # Empirical accuracy breakdown against groundtruth
```

---

## ⚡ Running on CPU (Sequential & Parallel)

### 1. Single Command Multi-Core Execution (Fastest on Kaggle)
Kaggle CPU notebooks provide 4 vCPUs. To utilize all 4 CPU cores simultaneously:
```bash
python scripts/aiops2025_run.py \
  --data /kaggle/input/aiops2025 \
  --out results/aiops2025/core \
  --workers 4 \
  --gt /kaggle/input/aiops2025/groundtruth.jsonl
```

### 2. Parallel Sharding & Merging (Distributed / Multi-session)
If running multiple background shards:
```bash
# Run 4 parallel shards
for k in 0 1 2 3; do
  python scripts/aiops2025_run.py \
    --data /kaggle/input/aiops2025 \
    --out /kaggle/working/shards \
    --shard ${k}/4 &
done
wait

# Merge all shards and compute final score
python scripts/aiops2025_merge.py \
  --inputs "/kaggle/working/shards/*.jsonl" \
  --out results/aiops2025/core \
  --gt /kaggle/input/aiops2025/groundtruth.jsonl
```

---

## 📋 CSV Format (12 Clean Columns)

The output `summary.csv` provides clean, tabular diagnostics:
1. `case_id`: Incident identifier (UUID)
2. `component`: Identified root cause entity (pod / service / node)
3. `fault_type`: Identified fault mechanism (e.g. `cpu_stress`, `network_delay`)
4. `confidence`: Posterior Bayesian confidence $P(e^*)$
5. `reason`: Concise summary of the root cause and holding evidence
6. `n_steps`: Number of causal reasoning steps in the evidence chain
7. `n_metrics`: Total metric time-series points evaluated
8. `n_logs`: Total container log lines evaluated
9. `n_traces`: Total distributed trace spans evaluated
10. `load_time_s`: Telemetry data loading duration (seconds)
11. `pipeline_time_s`: GAMA evidence calibration & Bayesian reasoning duration (seconds)
12. `wall_s`: Total execution wall-clock time per case (seconds)
