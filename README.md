# GAMA-RCA — Graph-Augmented Multi-Agent Root Cause Analysis on RCA100

Code for the paper **"GAMA-RCA: A Graph-Augmented Multi-Agent Framework for Root Cause Analysis on RCA100"**
(manuscript: [`paper/GAMA-RCA_manuscript.docx`](paper/GAMA-RCA_manuscript.docx)).

> **Status: private research code.** RCA100 is licensed CC BY-NC-SA 4.0 and its answer key must never be public.
> This repository and the Kaggle dataset stay private. The only LLM backend is a local `transformers` model;
> no RCA100 data is sent to any external API. Results below are on the 30 **development** cases only.

**Design principle: code extracts evidence, the LLM decides.** Deterministic code builds evidence tables (rows are selected by entity *name*,
never by significance). No component is trained or tuned on RCA100 ground truth, and `answer_key/` is read only by `evaluation/scoring.py`.

## What it does

Given an alert, GAMA-RCA diagnoses one incident in these stages (see Figure 1 and Algorithm 1 in the paper):

1. **Alert Parser** extracts the alerted entity, symptoms and time.
2. **Topology-aware graph retrieval**: 3-hop BFS plus a bounded infrastructure search (≤ 6 hops), personalized PageRank (α = 0.85) on the undirected graph, and a temporal boost (×1.5 for entities whose evidence precedes the alert, ×0.8 if it only follows).
3. **Structural anchor** (one LLM call): a first guess of the origin from the call-graph table.
4. **Evidence**: vector retrieval (BGE-M3, top 20 per modality) fused with the graph score (`H = 0.5·g + 0.5·v`), plus deterministic evidence tables (call-graph propagation, infrastructure, trace with TWIST composite score, log templates via Drain3).
5. **Evidence Summarizer** groups evidence by modality.
6. **Four specialist agents** (Metrics, Logs, Trace, Topology) report findings on their modality.
7. **Chain-first Coordinator** writes a cause → propagation → impact chain with `signal=value` quotes, then names the root-cause entity and one of 28 fault types.

LLM: Qwen2.5-7B-Instruct, deterministic decoding (`LLM_TEMPERATURE=0`, `LLM_SEED=0`).

## Repository layout

```
config/             all knobs (config/__init__.py); CLI overrides in config/args.py; every flag is recorded in the run manifest
data/               loader.py (Case, parquet ingestion, entity resolution), taxonomy.py (28 fault types, definitions, knowledge block)
retrieval/          graph.py (BFS / PPR / temporal boost), vector.py (FAISS + BGE-M3), hybrid.py (score fusion)
pipeline/           pipeline.py (the proposed system), propagation_evidence.py, infra_evidence.py, trace_evidence.py,
                    log_templates.py, twist_scoring.py, evidence_summarizer.py; baselines.py / gala_lite.py (comparison systems)
agents/             multi_agent.py (4 specialists + Coordinator), kaggle_client.py (local transformers model), factory.py, llm_protocol.py
evaluation/         scoring.py (RCA100 score 0.4·EL + 0.3·FI + 0.3·RP, plus strict FT / RCL / Joint)
scripts/            kaggle_run.py (Kaggle cells, arm definitions), summarize_arms.py (paired bootstrap CIs)
scripts/diagnostics/  one-off data audits and debugging tools (read-only, not part of a run)
notebooks/          Kaggle notebooks for the dev-30 arms and held-out runs
tests/              prompt_snapshot.py (prompt-equivalence test; needs neither data nor a model) and its snapshots
paper/              manuscript
main.py             batch runner (CLI)        smoke_test.py   stage-by-stage check without an LLM
```

## Setup

```bash
pip install -r requirements.txt            # Python 3.10+, a GPU is needed for the 7B model
export RCA100_ROOT=/path/to/RCA100         # folder with cases/, answer_key/, manifest.txt (not included)
```

Obtain RCA100 from its authors (CC BY-NC-SA 4.0). Place a Qwen2.5-7B-Instruct checkpoint where `transformers` can load it, or set
`KAGGLE_MODEL_LOCAL_PATH` to a local path.

## How to use

**1. Check the data layer (no LLM):**

```bash
python smoke_test.py                       # stages 1-3: inspect, load, retrieval (no LLM)
python smoke_test.py --with-llm            # also stage 4, the full pipeline (needs the model)
```

**2. Run the pipeline locally:**

```bash
python main.py --case_ids t004 t005 --systems proposed_hybrid --save_path results/run1.csv
python main.py --n_cases 10 --start_case 0                # first 10 cases
```

Each flag below is read from the environment, so an arm is just a set of variables, e.g.

```bash
CHAIN_FIRST=1 CHAIN_ABSOLUTE_SIGNALS=1 USE_KNOWLEDGE_POS=1 ... python main.py --case_ids t032
```

**3. Run on Kaggle (how the experiments were run):** paste the cells of `scripts/kaggle_run.py` into a notebook (GPU, Internet on, secret
`GITHUB_TOKEN`, the RCA100 dataset and a Qwen2.5 model as inputs). Set `ARM`, `CASES` (`DEV30`, `SUBSET8`, `HELDOUT`, `ALL_IDS`) and optionally `SHARD`.
Cell 5 is resume-safe. Ready-made notebooks are in `notebooks/`.

**4. Compare arms (paired, with bootstrap intervals):**

```bash
python scripts/summarize_arms.py BASE=base.csv EVSC=evsc.csv --ref BASE --boot 10000
```

## Configurations (arms)

All arms are defined in `scripts/kaggle_run.py` (`ARMS`) on a fixed base; every flag is off by default.

| Arm | What changes |
|---|---|
| `A1rT_fixed_ABS_KP` | **Main configuration** (chain-first Coordinator, call-graph, infra and trace tables, log templates, knowledge block) |
| `..._P1` | grounded fault choice: the Coordinator lists `fault_evidence` before the type |
| `..._TYPE`, `..._TYPELEAN` | separate typing call on entity-scoped evidence; `LEAN` makes the Coordinator write only entity + chain |
| `..._EVSC`, `..._TYPELEAN_EVSC` | Kubernetes events scoped to the neighbourhood services and their nodes |
| `..._TYPECMP`, `..._TYPELEAN_CMP` | typing call compares up to 3 candidate types with supporting and contradicting rows |
| `..._NOSPEC`, `..._EVOBS` | ablations: no specialist calls / specialists copy observations only |
| `..._NOSTATS` | ablation: no change/z/rank/n appended to numeric rows |
| `..._COV`, `..._SEG`, `..._COV_SEG` | evidence-coverage and chain-segment candidates, not shown to help |
| `..._FPSHOW` | fault fingerprints: code-checked signal profile of the origin + candidate types shown to a typing call; any type allowed |
| `..._FPSL` | fault fingerprints, **shortlist**: the typing call must choose among the top-3 candidates (types whose key signal is present) |
| `..._FPRULE` | ablation: top fingerprint candidate, no typing LLM call |
| `..._FPRULE_FIX`, `..._FPSL_FIX` | ablation: fixed a-priori thresholds instead of the calibrated tests |
| `..._VER` | chain verifier: answered entity = cause target; quoted numbers must occur in the evidence, else one repair call |
| `..._FPSL_VER`, `..._FPSL_VER_LVL` | shortlist + verifier (candidate main configuration); `LVL`: a node-level type moves the answer to the hosting node |

Backbones are a separate setting (`MODEL` in Cell 4, `MODELS` in `scripts/kaggle_run.py`): `qwen25_7b` (default), `qwen3_8b`,
`qwen3_8b_think` (`ENABLE_THINKING=1`, `<think>` block stripped before JSON parsing), `qwen25_14b_4bit`, `qwen3_14b_4bit`, `qwen3_32b_4bit`.

### Fault fingerprints (`pipeline/fault_calibrated.py`; fixed-threshold ablation in `pipeline/fault_fingerprint.py`)

The dev-30 bottleneck is the evidence-to-label step, so the fault type gets its own, label-free evidence layer:

1. **Evidence.** Continuous signals (metric series, log-keyword rates) are tested against the entity's *own* baseline with a one-sided
   Mann-Whitney U test (p <= `FP_ALPHA` = 0.05; several series of one family combined by Bonferroni). Categorical observations that name a
   mechanism (K8s event reasons, HTTP 429/403/503/5xx, error-message patterns, failing DB/Redis/Kafka spans) are taken from the alert window.
   RCA100 exports traces only inside the window and ~15 min of logs before it (`check_baseline_length.py`), so trace evidence is categorical
   or compared within the window.
2. **Hypotheses (entity, type).** For the Coordinator's origin and its hosting nodes, a type is admitted when one of its key signals holds,
   its level matches the entity, and no excluding signal holds. Ranking: number of categorical observations, then Fisher's -2 sum ln p.
   Residual types (httpError5xx, codeDefect) only when nothing specific is admitted.
3. **Typing.** `show` / `shortlist` (LLM chooses among the top-3 hypotheses) / `rule` (top hypothesis, no LLM).

No threshold is chosen by looking at labels; the fingerprint table is written from the type definitions and frozen before evaluation.
`tests/test_calibrated.py` runs synthetic scenarios. `scripts/diagnostics/fp_oracle_coverage.py` (evaluation only, reads `answer_key/`)
reports the FI ceiling with the true entity and the label-free false-alarm rate of the rank tests. Commit e14ddff (rules adjusted after
looking at dev-30) was reverted in 1d3a84b and is not part of the method.

## Evaluation protocol for the next round

1. Develop and debug on dev-30 only (`fp_oracle_coverage.py` on dev-30 is allowed while developing).
2. Freeze code, arms, prompts and thresholds (git tag), then run the chosen arms ONCE on the 73 held-out cases (`CASES = HELDOUT`).
3. Report paired bootstrap intervals (`summarize_arms.py`) plus a paired test on the binary FI column, per semantic group and per level
   (node vs service), and the confusion matrix of fault types. On 30 cases one case is 3.3 FI points: dev-30 differences are not evidence.

## Reproducibility notes

- Decoding is deterministic (`LLM_TEMPERATURE=0`, `LLM_SEED=0`, `PYTHONHASHSEED=0`); both are written to the run manifest together with every flag.
  Runs from before commit `3ef1e24` sampled at temperature 0.1 without a seed and are not comparable with later runs.
- Compare arms only within one data version (RCA100 v1.1) and one commit.
- `python tests/prompt_snapshot.py compare tests/snapshots/prompts_v6_fp.json` checks that the agent prompts of all retained arms are byte-identical (v6 adds the FP and VER arms)
  to the snapshot; run it after any refactoring. Removed experiments live in git tag `pre-cleanup-3ef1e24`.
- API keys and tokens belong only in Kaggle Secrets or the environment, never in the repository.

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is a draft and has not been peer reviewed.
