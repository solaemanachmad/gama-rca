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

# Load .env (if present) so HF_TOKEN, GEMINI_API_KEY, RCA100_ROOT, etc. don't
# need to be manually exported every terminal session. Requires:
#   pip install python-dotenv --break-system-packages
# Falls back silently to plain os.environ if python-dotenv isn't installed
# or no .env file exists -- so this is safe even without either.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
GRAPH_HOP_LIMIT = 3                # Phase A: shallow service-neighborhood BFS radius.
                                    # Kept small on purpose -- Phase B (INFRA_SEARCH_HOP_CAP
                                    # below) handles reaching node-level entities via a
                                    # targeted search instead of raising this blanket radius,
                                    # which was tried (hop_limit=6) and made subgraphs balloon
                                    # to near-whole-topology size (145-232 nodes), causing an
                                    # OOM crash when embedding the resulting observation set.
INFRA_SEARCH_HOP_CAP = 6           # Phase B: max service-hops to search (scratch space, not
                                    # added to subgraph wholesale) for the nearest apm.instance
                                    # entry point into the infra layer (instance/pod/node/
                                    # cluster). Measured 3-4 hops needed in practice; 6 gives
                                    # headroom without the Phase-A blast-radius cost.
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
MAX_RESOLVED_PER_MODALITY = 5000        # hard safety cap, always applied (not just
                                         # DEV_QUICK_TEST): even resolved observations
                                         # within the candidate subgraph are capped
                                         # before embedding, so a large subgraph can
                                         # never again feed an unbounded row count into
                                         # sentence-transformers and OOM like it did
                                         # when GRAPH_HOP_LIMIT=6 produced 145-232 node
                                         # subgraphs. Prioritizes rows nearest the alert
                                         # window if truncation is needed.
 
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