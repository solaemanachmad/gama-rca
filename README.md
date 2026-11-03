# GAMA-RCA — Graph-aware, Multimodal, Abductive Root Cause Analysis

Label-free root-cause analysis for microservices on RCA100. GAMA-RCA computes an **abductive belief** over root-cause hypotheses
(entity × fault type) from **calibrated multimodal evidence** (metrics, logs, traces, Kubernetes events, alerts) along the **service
and host topology**. A **bounded local LLM** (Qwen2.5-7B) is optional: it reads text where the statistics cannot and chooses only
among options the belief already ranked. No training, no labels, and no external API in the main configuration.

> **Status: private research code.** RCA100 is licensed CC BY-NC-SA 4.0; its answer key is read only by `evaluation/scoring.py`
> and must never be made public. Keep this repository, the Kaggle dataset and any W&B project private.

## Method

Principles:

- **P1 Calibrate, do not threshold.** Every observation is scored against the case's own history and its peers.
- **P2 Causes precede and propagate directionally.** Errors flow up the call graph and load flows down. What the root explains is not
  evidence against it; load that came from upstream is not evidence for a downstream root.
- **P3 The LLM reads text and selects among ranked options.** It never overrides calibrated evidence.

| Stage | Code |
|---|---|
| 1 Calibrated multimodal evidence | `gama/evidence.py` |
| 2 Abductive belief (Bayesian hypotheses, fingerprint roles, topology explain-away, EM, onset precedence) | `gama/belief.py` |
| 3 Two-stage decision: entity, then mechanism | `gama/decide.py` |
| 4 Bounded LLM (optional): EIG sensor and shortlist selector | `gama/llm.py` |
| Evidence-cited reasoning chain | `gama/chain.py` |
| One case → result; module switches | `gama/run.py`, `gama/config.py` |

Modules, one switch each (`GAMA_MODULES`; the unit of the ablation):

| Switch | Module |
|---|---|
| `mm` | M1 cross-modal corroboration |
| `trace` | M2 trace-origin attribution with endpoint relay |
| `typing` | M3 mechanism typing (intra-entity signal precedence) |
| `load` | M4 direction-aware load explain-away |
| `text` | M5 mechanism-naming text |
| `chain` | M6 evidence-cited chain |
| `llm` | M7 bounded LLM |

Spec strings: `core` = M1–M6; `full` = M1–M7; `core,-trace` = without M2.

## Run (Kaggle; the notebook never changes)

```
# Cell 1: clone (secret GITHUB_TOKEN; RCA100 attached; Internet on) -- see scripts/kaggle_job.py
# Cell 2: one command
!python scripts/kaggle_job.py --arm GAMA --cases all --shard 0/4                      # CPU, no LLM
!python scripts/kaggle_job.py --arm GAMA --modules core,-trace --cases all --shard 0/4 # ablation
!python scripts/kaggle_job.py --arm GAMA_FULL --cases all --shard 0/4                 # GPU, + local LLM
!python scripts/kaggle_job.py --check core,llm --cases all --shard 0/4                # equivalence with tag v5-frozen
```

Each run writes `<name>_all.zip`: the scored CSV (one row per case), the log, per-case decision trails, every LLM call (prompt,
output, input / output tokens, latency) and a manifest (commit, modules, model, hardware, versions, wall time).

The analysis (`PREREGISTRATION_V5.md`, "Analysis plan v5") reads those zips; nothing is re-run:

```
python scripts/analyze_v5.py --arm A="results/A_*.zip" --arm GAMA_CORE="results/GAMA_CORE_*.zip" ... --out report.md
```

## Repository

| Path | Contents |
|---|---|
| `gama/` | the method |
| `pipeline/` | evidence extraction used by `gama/` (signal database, fingerprints, trace structure, text preprocessing, LLM typing) and the baselines |
| `data/` | loader and the only dataset-specific settings (`dataset_profile.py`); taxonomy |
| `evaluation/` | scoring (EL, FI, RP, Final, step-anchored checkpoints) |
| `agents/` | local LLM client (`transformers`), call trace |
| `scripts/` | `kaggle_job.py` (runs), `kaggle_run.py` (arms, models), `equivalence.py`, `analyze_v5.py` |
| `tests/` | synthetic tests (no data, no model): `make test` |

The development history (all amendments and their switches) is reproducible from git tag `v5-frozen`.

## Reproducibility

- Deterministic: LLM temperature 0, seed 0, `PYTHONHASHSEED=0`; constants fixed a priori (`gama/config.py`,
  `pipeline/belief_diagnosis.py`).
- Every design change is a dated amendment in `PREREGISTRATION_V5.md`.
- API keys belong only in Kaggle Secrets or the environment, never in the repository.

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is a draft and has not been peer reviewed.
