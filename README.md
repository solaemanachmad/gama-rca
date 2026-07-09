# Graph-Augmented Multi-Agent RCA — Local / Kaggle Prototype

Plain `.py` files, no notebook. Import order matters — either paste each
file's contents into its own Kaggle notebook cell in this order, or on your
own PC just run the scripts below (Python imports resolve the order for you
as long as all files sit in the same folder).

## Running locally on your own PC

```bash
cd graphrag_rca
pip install -r requirements.txt

export RCA100_ROOT=/path/to/RCA100      # folder containing cases/, answer-key/, manifest.txt
# Windows PowerShell: $env:RCA100_ROOT="C:\path\to\RCA100"

# 1. Start Ollama in a separate terminal first:
ollama serve
ollama pull qwen2.5:7b        # or a smaller model for a first pass, e.g. qwen2.5:3b

# 2. Validate the pipeline stage by stage BEFORE the full experiment:
python smoke_test.py                 # stages 1-3: ingestion + retrieval, no LLM needed
python smoke_test.py --with-llm      # stage 4: one full case through the LLM pipeline

# 3. Only once smoke_test.py passes, run the batch experiment:
python run_experiment.py --n_cases 5                          # small run first
python run_experiment.py --n_cases 103                        # full benchmark
python run_experiment.py --n_cases 10 --systems direct_llm standard_rag   # subset of systems
```

`run_experiment.py` is a CLI script — `python run_experiment.py --n_cases 10`
runs all 5 systems (4 baselines + proposed hybrid) on 10 cases and writes
`graphrag_rca_work/results/results.csv`. You do NOT have to run it against
all 103 cases or all 5 systems at once — both `--n_cases` and `--systems`
are there specifically so you can start small.

## Kaggle notebook cell order

Paste each file's contents into its own Kaggle notebook cell **in this order**:

1. `config.py`
2. `schema.py`
3. `data_loader.py`
4. `graph_retrieval.py`
5. `vector_retrieval.py`
6. `hybrid_retrieval.py`
7. `evidence_summarizer.py`
8. `llm_client.py`
9. `agents.py`
10. `pipeline.py`
11. `evaluation.py`
12. `baselines.py`
13. `run_experiment.py` — then call `run_all(n_cases=5)` in a final cell

## Setup cell (before cell 1)

```python
!pip install -q pandas pyarrow networkx faiss-cpu sentence-transformers langgraph requests

# Local LLM via Ollama
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull qwen2.5:7b
```

## First thing to run after pasting config.py + data_loader.py

```python
inspect_case("t001")
```

This prints the real key names in `task.json`, `topology.json`, and the
column names of each parquet file for one case. The loaders in
`data_loader.py` and `evaluation.py` guess common key names
(`CANDIDATE_*_KEYS` lists at the top of each file) — if `inspect_case`
shows different names, edit those lists. That is the only place real-schema
drift should require a code change.

## Then sanity-check ingestion

```python
c = Case("t001")
print(c)
print(c.alert)
```

## Then run one full pipeline pass before the batch experiment

```python
llm = LLMClient()
pipeline = GraphRAGPipeline(llm=llm)
result = pipeline.run("t001")
print(result)
```

## Then the full ablation study

```python
df = run_all(n_cases=10)   # raise to 103 for the full benchmark once stable
```

## Notes

- `answer-key/` (ground truth) is only ever imported by `evaluation.py`.
  No other module reads it — this is deliberate, so the framework can never
  leak answers into its own retrieval/reasoning path.
- `HYBRID_ALPHA` / `HYBRID_BETA` in `config.py` are your RQ2 knob (graph vs.
  vector weight) — sweep them for the ablation table.
- Swap LLMs by changing `config.LLM_MODEL_NAME` only (`qwen2.5:7b`,
  `deepseek-r1`, `llama3`, `gemma2`, ...).
