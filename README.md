# Graph-Augmented Multi-Agent RCA — Package Layout

## Folder structure

```
graphrag_rca/
├── config.py                  # shared config, all knobs (HYBRID_ALPHA/BETA, hop limits, etc.)
├── schema.py                  # shared dataclasses (Entity, Observation, RCAResult, ...)
├── main.py          # CLI entry point for the batch ablation study
├── smoke_test.py              # stage-by-stage pipeline sanity check
├── requirements.txt
│
├── data/                      # ingestion + domain vocabulary
│   ├── loader.py              #   (was data_loader.py) Case, parquet loaders, entity resolution
│   └── taxonomy.py            #   fault-type label vocabulary + shortlist embedding
│
├── retrieval/                 # Modules 2–4 of the architecture
│   ├── graph.py               #   (was graph_retrieval.py) BFS + PPR + cluster expansion
│   ├── vector.py              #   (was vector_retrieval.py) FAISS index, embedding singleton
│   └── hybrid.py              #   (was hybrid_retrieval.py) graph+vector score fusion
│
├── agents/                    # LLM clients + multi-agent coordination
│   ├── llm_client.py          #   Ollama-backed client
│   ├── gemini_client.py       #   Gemini API-backed client (same interface, drop-in swap)
│   └── multi_agent.py         #   (was agents.py) Metrics/Logs/Trace/Topology/Coordinator agents
│
├── pipeline/                  # end-to-end systems being compared
│   ├── evidence_summarizer.py
│   ├── pipeline.py            #   (was pipeline.py) GraphRAGPipeline — the proposed_hybrid system
│   └── baselines.py           #   direct_llm / standard_rag / graphrag_only / multi_agent_only
│
├── evaluation/
│   └── scoring.py             #   (was evaluation.py) ground-truth loading + RCA100 scoring.
│                               #   Only this module ever reads answer_key/ — deliberate, so no
│                               #   other module can leak answers into retrieval/reasoning.
│
└── scripts/                   # one-off diagnostic tools (not imported by the pipeline itself)
    ├── debug_retrieval.py     #   per-case retrieval tracer (subgraph membership, rank, overlap)
    ├── check_values.py        #   raw parquet column VALUES (not just names)
    ├── check_k8s_domain.py    #   why domain=="k8s" metrics fail to resolve an entity
    ├── check_metric_field.py  #   where node identity is (or isn't) encoded in metrics
    ├── check_node_reachability.py  # true shortest-path distance + edge trace
    └── diagnose_schema.py
```

Every file under `data/`, `retrieval/`, `agents/`, `pipeline/`, `evaluation/`
is a real Python package (`__init__.py` present) — import with the full
path, e.g. `from retrieval.graph import GraphRetriever`,
`from pipeline.proposed import GraphRAGPipeline`.

`scripts/*.py` are standalone tools, not part of the package graph. Each has
a small `sys.path` bootstrap at the top so they run correctly regardless of
your current directory:
```bash
python scripts/debug_retrieval.py t003 t005 t006 t007 t008
```

## Running locally on your own PC

```bash
cd graphrag_rca
pip install -r requirements.txt
export HF_HUB_OFFLINE=1                 # skip HF network checks on every run, once the
                                         # embedding model is already cached locally

export RCA100_ROOT=/path/to/RCA100      # folder containing cases/, answer_key/, manifest.txt
# Windows PowerShell: $env:RCA100_ROOT="C:\path\to\RCA100"

# 1. Start Ollama in a separate terminal first (skip if using GeminiClient instead):
ollama serve
ollama pull qwen2.5:7b

# 2. Validate the pipeline stage by stage BEFORE the full experiment:
python smoke_test.py                 # stages 1-3: ingestion + retrieval, no LLM needed
python smoke_test.py --with-llm      # stage 4: one full case through the LLM pipeline

# 3. Only once smoke_test.py passes, run the batch experiment:
python main.py --n_cases 5
python main.py --n_cases 103
python main.py --n_cases 10 --systems direct_llm standard_rag
```

## Performance notes (why this used to feel slow)

- **Embedding model is now a process-wide singleton** (`retrieval/vector.py`
  -> `get_embedding_model()`). It previously reloaded `SentenceTransformer`
  from disk on every single `VectorIndex()` call -- i.e. once per case per
  system, 500+ times across a full `--n_cases 103` run across 5 systems.
  It now loads exactly once per process. `data/taxonomy.py` shares the same
  singleton instead of loading a second copy.
- Set `HF_HUB_OFFLINE=1` (see above) to stop Hugging Face Hub network
  checks on every run once the model is cached -- this is what caused the
  `HTTP 504` retry storms you may have seen.
- `MAX_RESOLVED_PER_MODALITY` in `config.py` caps how many resolved
  observations get embedded per modality per case, as a hard safety net
  against OOM on cases with unusually large candidate subgraphs.
- To use Gemini API instead of local Ollama for the LLM reasoning step
  (much faster wall-clock per call, at the cost of no longer being fully
  offline/local): `pip install google-genai`, set `GEMINI_API_KEY`, and
  swap `LLMClient()` -> `GeminiClient(model_name="gemini-3.1-flash-lite")`
  in `pipeline/pipeline.py` / `pipeline/baselines.py`. Same interface, zero
  other code changes needed.

## Notes

- `HYBRID_ALPHA` / `HYBRID_BETA` in `config.py` are your RQ2 knob (graph vs.
  vector weight) -- sweep them for the ablation table.
- `GRAPH_HOP_LIMIT` / `INFRA_SEARCH_HOP_CAP` in `config.py` control the
  two-phase graph retrieval (Phase A: shallow service-neighborhood BFS;
  Phase B: targeted infra-layer search for `k8s.node`/`k8s.cluster`
  root causes) -- see the Findings section of the paper for why this is
  two separate parameters rather than one blanket radius.
- Swap LLMs by changing `config.LLM_MODEL_NAME` only (`qwen2.5:7b`,
  `deepseek-r1`, `llama3`, `gemma2`, ...), or swap the client class
  entirely for `GeminiClient` (see above).