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

## Metrics

`Final = 0.4·EL + 0.3·FI + 0.3·RP` (×100 in the paper). **EL** Entity Localization (topology-distance partial credit), **FI** Fault Identification
(exact over 28 classes), **RP** Reasoning Process. RP here is a **local approximation** of the official scorer, so absolute RP and Final values can
differ from official scores. The run CSV also has strict binary `ft_binary`, `rcl_strict`, `joint_strict` columns.

## Results so far (30 development cases, scale 0–100)

| Configuration | EL | FI | RP | Final |
|---|---|---|---|---|
| Anchor only † | 45.7 | 3.3 | 27.8 | 27.6 |
| + call-graph table † | 54.0 | 10.0 | 27.7 | 32.9 |
| Full pipeline † | 54.6 | 13.3 | 34.3 | 36.1 |
| + scoped events | 53.5 | 16.7 | 35.3 | 37.0 |
| + typing call + events | 56.7 | 13.3 | 34.6 | 37.0 |
| + typing call + candidates (best of several on dev) | 54.4 | 20.0 | 34.8 | 38.2 |

† sampled decoding (before commit `3ef1e24`); the other rows are deterministic. All paired differences against the full pipeline include zero.
The entity is exactly correct in 13 of 30 cases, but the fault type in only 3 of those 13: the main bottleneck is the mapping from evidence to
fault label. These are development results; held-out evaluation and comparison with external baselines remain to be done.

## Reproducibility notes

- Decoding is deterministic (`LLM_TEMPERATURE=0`, `LLM_SEED=0`, `PYTHONHASHSEED=0`); both are written to the run manifest together with every flag.
  Runs from before commit `3ef1e24` sampled at temperature 0.1 without a seed and are not comparable with later runs.
- Compare arms only within one data version (RCA100 v1.1) and one commit.
- `python tests/prompt_snapshot.py compare tests/snapshots/prompts_v5_cmp.json` checks that the agent prompts of all retained arms are byte-identical
  to the snapshot; run it after any refactoring. Removed experiments live in git tag `pre-cleanup-3ef1e24`.
- API keys and tokens belong only in Kaggle Secrets or the environment, never in the repository.

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is a draft and has not been peer reviewed.
