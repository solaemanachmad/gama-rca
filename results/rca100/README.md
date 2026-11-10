# RCA100 Benchmark Results

This directory contains official evaluation results for **GAMA-RCA** on the **RCA100** benchmark (103 real-world microservice incident cases; Cai et al., 2026).

---

## 📊 Summary Table

Evaluated under strict pre-registration across all 103 microservice incidents:

| Method / Arm | Compute | Entity Loc. (EL) | Fault Type (FI) | Reasoning Path (RP) | Final Score | Exact (Ent/Type/Both) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct LLM (7B)** | GPU / API | 28.0 | 14.6 | 10.9 | **18.8** [15.3, 22.5] | 0 / 15 / 0 |
| **Evidence $\rightarrow$ LLM (7B)** | GPU / API | 71.8 | 20.4 | 26.2 | **42.7** [38.1, 47.4] | 66 / 21 / 18 |
| **Belief $\rightarrow$ LLM (7B)** | GPU / API | 67.9 | 43.7 | 25.8 | **48.0** [42.6, 53.6] | 63 / 45 / 39 |
| **GAMA Core (Statistical)** | **CPU only** | **68.9** | **43.7** | **74.1** | **62.9** [57.4, 68.5] | **64 / 45 / 39** |
| GAMA Full (+ bounded 7B) | CPU + GPU | 66.5 | 35.0 | 73.7 | 59.2 [53.9, 64.6] | 60 / 36 / 30 |
| GAMA Full (+ gpt-oss-120b) | CPU + API | 65.7 | 36.9 | 73.7 | 59.4 [54.0, 65.2] | 59 / 38 / 32 |

---

## 📁 Directory Structure

* [`core/summary.csv`](core/summary.csv): Full 103-case diagnosis metrics (Entity Localization, Fault Identification, Reasoning Path, Final Score, Runtime).
* [`ablation/ablation_summary.csv`](ablation/ablation_summary.csv): Quantitative impact of removing individual reasoning modules ($M_1$ through $M_6$).

---

## 🔄 Reproduction

To run the RCA100 benchmark locally:
```bash
python main.py --systems gama --cases all
```
