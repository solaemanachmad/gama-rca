"""
config.py
=========
Central configuration for the Graph-Augmented Multi-Agent RCA framework.

Adjust ROOT / DATASET_ROOT to match your Kaggle input mount point.
Everything downstream (data_loader, retrieval, agents, evaluation) imports
paths and constants from here, so this is the ONLY file you should need to
edit when moving between environments (Kaggle -> local -> server).
"""

import os

# ---------------------------------------------------------------------------
# Paths — EDIT THIS for your machine, or set the RCA100_ROOT env var instead
# of editing the file (e.g. `export RCA100_ROOT=/home/you/data/RCA100`).
# Falls back to Kaggle's mount path only if nothing else is set.
# ---------------------------------------------------------------------------
DATASET_ROOT = os.environ.get(
    "RCA100_ROOT",
    "C:/Users/achsoe/Developments/gama-rca/RCA100",
)
 
CASES_DIR = os.path.join(DATASET_ROOT, "cases")
ANSWER_KEY_DIR = os.path.join(DATASET_ROOT, "answer_key")   # underscore, per AIOps_README.md — NEVER read during retrieval/reasoning
MANIFEST_PATH = os.path.join(DATASET_ROOT, "manifest.txt")
SUMMARY_PATH = os.path.join(DATASET_ROOT, "summary.json")
 
# Working directory (writable). Vector index, logs, results go here.
WORK_DIR = os.environ.get("RCA100_WORK_DIR", os.path.join(os.getcwd(), "graphrag_rca_work"))
INDEX_DIR = os.path.join(WORK_DIR, "vector_index")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
 
for d in (WORK_DIR, INDEX_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
 
# ---------------------------------------------------------------------------
# Per-case file names (as shipped inside cases/t###/)
# ---------------------------------------------------------------------------
FILE_TASK = "task.json"
FILE_TOPOLOGY = "topology.json"
FILE_METRICS = "metrics.parquet"
FILE_LOGS = "logs.parquet"
FILE_TRACES = "traces.parquet"
FILE_EVENTS = "events.parquet"
FILE_ALERTS = "alerts.parquet"
 
# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
GRAPH_HOP_LIMIT = 3                # BFS radius around alert entity (3, not 2: apm.operation-level
                                    # alerts may need operation->instance->service->caller, i.e. 3 hops,
                                    # to reach a root cause in another service like payment)
PPR_ALPHA = 0.85                   # Personalized PageRank damping factor
PPR_TOP_K = 15                     # top-k entities kept after PPR ranking
 
VECTOR_TOP_K = 20                  # top-k text chunks per modality per query
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
 
# Hybrid score: HybridScore = ALPHA * GraphScore + BETA * VectorScore
HYBRID_ALPHA = 0.5
HYBRID_BETA = 0.5
 
TIME_WINDOW_MINUTES = 15           # +/- window around alert timestamp for slicing
 
# ---------------------------------------------------------------------------
# DEV / QUICK-TEST MODE — turn this OFF before your real 103-case experiment.
# The single biggest cost driver right now is embedding unresolved-entity
# observations (mostly logs that failed entity resolution, ~600K rows/case)
# that get kept via the "entity_id is None" safety net in pipeline.py's
# graph filter. This caps that specific bucket for fast iteration; entities
# that DID resolve into the graph-retrieved subgraph are never capped, so
# correctness of the "does graph-aware retrieval work" check is unaffected.
# ---------------------------------------------------------------------------
DEV_QUICK_TEST = True
DEV_MAX_UNRESOLVED_PER_MODALITY = 300   # only used when DEV_QUICK_TEST is True
 
# ---------------------------------------------------------------------------
# LLM settings (local inference via Ollama; swappable)
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL_NAME = "qwen2.5:7b"       # swap to "deepseek-r1", "llama3", "gemma2", etc.
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
 
# ---------------------------------------------------------------------------
# Evaluation weights (RCA100 official protocol, Section 5.4 of the paper)
# ---------------------------------------------------------------------------
WEIGHT_ENTITY_LOCALIZATION = 0.40
WEIGHT_FAULT_IDENTIFICATION = 0.30
WEIGHT_REASONING_PROCESS = 0.30
 
RANDOM_SEED = 42