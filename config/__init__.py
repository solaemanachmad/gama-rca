"""
config/__init__.py
====================
Central configuration for the Graph-Augmented Multi-Agent RCA framework.

This used to be a single flat config.py -- it's now a package so CLI
overrides (config/args.py) can live alongside it without cluttering this
file. Import contract is UNCHANGED: every other module still does
`import config` and reads `config.SOME_CONSTANT` exactly as before --
this __init__.py is what actually runs when they do that, since Python
treats a package's __init__.py as the package's own namespace.

Adjust DATASET_ROOT to match your machine, or set RCA100_ROOT in .env.
Everything downstream (data/, retrieval/, agents/, evaluation/) imports
paths and constants from here, so this + config/args.py are the only places
you should need to touch when moving between environments or switching
backends (local Ollama <-> Gemini API, local embeddings <-> HF API, etc).
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
# Paths -- EDIT THIS for your machine, or set the RCA100_ROOT env var instead
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
PPR_ALPHA = 0.85                   # Personalized PageRank damping factor
PPR_TOP_K = 15                     # top-k entities kept after PPR ranking (whole-graph
                                    # fallback path only -- the seeded path returns the full
                                    # ranked list; see retrieval/graph.py)

VECTOR_TOP_K = 20                  # top-k text chunks per modality per query
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local")
                                    # "local" (default): run SentenceTransformer on this
                                    # machine (CPU/GPU). "hf_api": offload to Hugging Face's
                                    # hosted Inference API using HF_TOKEN, no local compute --
                                    # trades local speed for network round-trips per text and
                                    # HF's own rate limits. Also settable via
                                    # --embedding-backend on the CLI (config/args.py).
HF_API_PROVIDER = "hf-inference"   # InferenceClient provider; see
                                    # https://huggingface.co/docs/inference-providers
EMBEDDING_CACHE_ENABLED = os.environ.get("EMBEDDING_CACHE_ENABLED", "1") != "0"
                                    # Persistent disk cache (retrieval/vector.py) keyed by
                                    # (backend, model, text) hash -- survives across separate
                                    # `python main.py` runs, not just within one process. Set
                                    # to 0/false to disable (e.g. while actively tuning
                                    # EMBEDDING_MODEL and you don't want stale-looking cache
                                    # hits from before, though the cache key already includes
                                    # the model name so this is rarely necessary).

# Hybrid score: HybridScore = ALPHA * GraphScore + BETA * VectorScore
HYBRID_ALPHA = 0.5

TEMPORAL_BOOST_ENABLED = os.environ.get("TEMPORAL_BOOST_ENABLED", "1") != "0"
                                    # Pre-processing (not just display): boosts graph_scores for
                                    # entities whose earliest evidence precedes the alert
                                    # (candidate causes) and discounts entities whose evidence
                                    # only appears after (likely downstream effects). See
                                    # retrieval/graph.py's apply_temporal_boost(). Toggle off for
                                    # an ablation comparison against the text-only relative-time
                                    # label in evidence_summarizer.py.
TEMPORAL_BOOST_BEFORE = 1.5         # multiplier for entities with evidence before the alert
TEMPORAL_PENALTY_AFTER_ONLY = 0.8   # multiplier for entities whose evidence ONLY appears after
HYBRID_BETA = 0.5

TIME_WINDOW_MINUTES = 15           # +/- window around alert timestamp for slicing

# ---------------------------------------------------------------------------
# DEV / QUICK-TEST MODE -- turn this OFF before your real 103-case experiment
# (or pass --no-dev-quick-test on the CLI).
# ---------------------------------------------------------------------------
DEV_QUICK_TEST = True
DEV_MAX_UNRESOLVED_PER_MODALITY = 300   # only used when DEV_QUICK_TEST is True
MAX_RESOLVED_PER_MODALITY = 5000        # hard safety cap, always applied (not just
                                         # DEV_QUICK_TEST): even resolved observations
                                         # within the candidate subgraph are capped
                                         # before embedding. Prioritizes rows nearest the
                                         # alert window if truncation is needed.
MAX_OBSERVATIONS_PER_INDEX = 10000       # universal cap enforced inside VectorIndex.add()
                                         # itself (retrieval/vector.py) -- covers
                                         # build_case_index() (topology-BLIND standard_rag
                                         # baseline + smoke_test.py) too, which previously
                                         # had NO cap at all (measured: 692,604 embedded
                                         # observations for a single case).

# ---------------------------------------------------------------------------
# LLM settings -- swap backend/model without touching any other file.
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")   # "ollama" (local), "gemini" (API),
                                                          # or "kaggle" (Kaggle Models via
                                                          # kagglehub + transformers, no server).
                                                          # Also settable via --llm-backend.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "qwen2.5:7b")   # Ollama model name;
                                                                   # swap to "deepseek-r1",
                                                                   # "llama3", "gemma2", etc.
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
KAGGLE_MODEL_HANDLE = os.environ.get("KAGGLE_MODEL_HANDLE",
                                      "google/gemma-4/transformers/gemma-4-12b-it")
                                      # Confirmed instruction-tuned variant (per official model
                                      # card). Requires transformers with Gemma 4 "Unified"
                                      # architecture support -- `pip install -U transformers`
                                      # if you hit "KeyError: 'gemma4_unified'".
KAGGLE_MODEL_LOCAL_PATH = os.environ.get("KAGGLE_MODEL_LOCAL_PATH", None)
                                      # PREFERRED on Kaggle: if set, points directly at a model
                                      # already mounted via the notebook's Add Input > Models
                                      # panel (e.g. "/kaggle/input/models/google/gemma-4/
                                      # transformers/gemma-4-12b-it/2") -- skips
                                      # kagglehub.model_download()'s network call entirely,
                                      # which is far more reliable than repeated downloads.
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Evaluation weights (RCA100 official protocol, Section 5.4 of the paper)
# ---------------------------------------------------------------------------
WEIGHT_ENTITY_LOCALIZATION = 0.40
WEIGHT_FAULT_IDENTIFICATION = 0.30
WEIGHT_REASONING_PROCESS = 0.30

RANDOM_SEED = 42