# gama-rca — Graph-Augmented Multi-Agent Root-Cause Analysis on RCA100

Private research code. Principle: **code extracts evidence, the LLM decides.** No component is trained or tuned on RCA100 ground truth;
`answer_key/` is read only by `evaluation/scoring.py`.

## Layout

```
config/             all knobs (config/__init__.py); CLI overrides in config/args.py; every flag is recorded in the run manifest
data/               loader.py (Case, parquet ingestion, entity resolution), taxonomy.py (28 fault types, definitions, knowledge block)
retrieval/          graph.py (BFS/PPR/temporal boost), vector.py (FAISS), hybrid.py (score fusion)
pipeline/           pipeline.py (the proposed system), baselines.py / gala_lite.py (comparison systems),
                    propagation_evidence.py, infra_evidence.py, trace_evidence.py, log_templates.py, twist_scoring.py, evidence_summarizer.py
agents/             multi_agent.py (4 specialists + Coordinator), kaggle_client.py (local transformers model), factory.py, llm_protocol.py
evaluation/         scoring.py (RCA100 score: 0.4 EL + 0.3 FI + 0.3 RP, plus strict FT/RCL/Joint)
scripts/            kaggle_run.py (Kaggle cells), summarize_arms.py, diagnostic tools
tests/              prompt_snapshot.py (prompt-equivalence test, no data or model needed)
main.py             batch runner (CLI)
```

## Decoding

Deterministic: `LLM_TEMPERATURE=0` and `LLM_SEED=0` (defaults; both are written to the manifest). Before commit 3ef1e24 the specialists and the
Coordinator sampled at temperature 0.1 without a seed; results from before that commit are not comparable with later runs.

## Running

Kaggle (the way the experiments are run): paste the cells of `scripts/kaggle_run.py` into a notebook (GPU, Internet on, secret `GITHUB_TOKEN`,
the RCA100 dataset and a Qwen2.5 model as inputs). Choose `ARM`, `CASES` (`DEV30`, `SUBSET8`, `HELDOUT`, `ALL_IDS`) and optionally `SHARD`.

Locally (needs a GPU and the dataset):

```bash
pip install -r requirements.txt
export RCA100_ROOT=/path/to/RCA100          # folder with cases/, answer_key/, manifest.txt
python smoke_test.py                        # stages 1-3, no LLM
python main.py --case_ids t004 t005 --systems proposed_hybrid
```

Main configuration (arm `A1rT_fixed_ABS_KP`): `CHAIN_FIRST=1 USE_PROPAGATION_EVIDENCE=1 USE_INFRA_EVIDENCE=1 USE_TRACE_EVIDENCE=1 USE_LOG_TEMPLATES=1
USE_CHANGE_FACTOR=1 USE_CHANGE_STATS=1 ENTITY_SOURCE=coordinator STRICT_MODALITY_ROUTING=1 SPECIALIST_ENGLISH=1 CHAIN_ABSOLUTE_SIGNALS=1 USE_KNOWLEDGE_POS=1`.
Candidate additions, not yet shown to help: `INFRA_ALL_ENTITIES` + `USE_EVENT_DIGEST` (evidence coverage), `SPEC_CHAIN_SEGMENTS` (specialists write chain segments).

## Tests

`python tests/prompt_snapshot.py compare tests/snapshots/prompts_pre_cleanup2.json` checks that the agent prompts of the retained configurations are
byte-identical to the snapshot taken before the 2026-10-13 cleanup. Removed experiments live in git tag `pre-cleanup-3ef1e24`.

## Data and license

RCA100 is CC BY-NC-SA 4.0. The answer key must never be public: this repository and the Kaggle dataset stay private. The only LLM backend is a local
transformers model; no RCA100 data is sent to any external API.
