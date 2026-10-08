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
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")     # (env-overridable 2026-10-07 so a small/fast model can be selected
                                    # without code edits, e.g. EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
                                    # Multilingual (100+ languages incl. Chinese), chosen
                                    # because RCA100's alert_title/subject fields are
                                    # predominantly Chinese (e.g. "checkout响应时间突增告警")
                                    # and some log content embeds Chinese phrases too --
                                    # confirmed by directly inspecting cases/*/task.json and
                                    # logs.parquet. all-MiniLM-L6-v2 (the previous default) is
                                    # English-only and embeds that text as near-noise, which
                                    # degrades vector retrieval AND zero-shot fault-type
                                    # matching (comparing Chinese-contaminated evidence
                                    # embeddings against English FAULT_DEFINITIONS). bge-m3 is
                                    # state-of-art on multilingual/cross-lingual retrieval
                                    # benchmarks specifically for this kind of code-switched
                                    # EN+ZH technical text. Heavier than MiniLM (568M params,
                                    # 1024-dim vs. 384-dim) -- retrieval/vector.py's
                                    # VectorIndex now infers its FAISS index dimension from
                                    # the model directly (_infer_embedding_dim), so
                                    # EMBEDDING_DIM below is an informational fallback only,
                                    # not load-bearing.
EMBEDDING_DIM = 1024

# Speed knobs (all default to the previous behaviour, i.e. no change unless set).
EMBEDDING_MAX_SEQ_LENGTH = int(os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "0"))  # 0 = model default
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "256"))
EMBEDDING_FP16 = os.environ.get("EMBEDDING_FP16", "0") == "1"   # only applied when device is cuda
# Ablation switch: USE_VECTOR_RETRIEVAL=0 skips building the FAISS index and the
# vector half of hybrid retrieval; evidence then comes from graph-direct retrieval
# only. Used to measure whether vector retrieval contributes at all.
USE_VECTOR_RETRIEVAL = os.environ.get("USE_VECTOR_RETRIEVAL", "1") != "0"

EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
                                    # "cpu" (default): keeps the embedding model off the
                                    # GPU entirely so it never competes with the LLM
                                    # client for VRAM -- on a single Kaggle T4/P100,
                                    # Qwen2.5-7B-Instruct alone already occupies ~14GB
                                    # of the ~14.56GB card, leaving no room for bge-m3's
                                    # forward pass (confirmed: CUDA OOM in practice).
                                    # The persistent embedding cache (see
                                    # EMBEDDING_CACHE_ENABLED below) means CPU is a
                                    # one-time cost per unique text, not per run. Set to
                                    # "cuda" only if running with a spare GPU (e.g. 2x T4
                                    # with the LLM on one) or a non-GPU LLM backend.
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
USE_LLM_GRAPH_ANCHOR = os.environ.get("USE_LLM_GRAPH_ANCHOR", "1") == "1"
USE_LLM_SPECIALIST_AGENTS = os.environ.get("USE_LLM_SPECIALIST_AGENTS", "1") == "1"
                                      # Exploratory ablation: set either to "0" to replace that
                                      # component's LLM call with a rule-based/structural
                                      # alternative. The Coordinator ALWAYS stays LLM-based
                                      # (it's the one component whose job -- synthesizing a
                                      # coherent narrative -- genuinely benefits from natural
                                      # language generation). Default "1" (LLM) preserves the
                                      # validated Pass 2 configuration.
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
# SEA-RCA baseline (Single-Shot Evidence-Augmented RCA) -- see
# pipeline/baselines.py's sea_rca(). Deliberately SEPARATE knobs from
# MAX_OBSERVATIONS_PER_INDEX above: an 8-case head-to-head (results,
# 2026-10-02) found graphrag_only beats proposed_hybrid on
# entity_localization and ties on fault_identification, but scores ~0 on
# reasoning_process/explainability because it sees no evidence text at all.
# SEA_RCA tests whether a SMALL amount of hybrid-ranked evidence folded into
# ONE prompt (vs. proposed_hybrid's 6 LLM calls) recovers most of that
# reasoning_process value. Its cost profile should stay far below
# proposed_hybrid's regardless of how MAX_OBSERVATIONS_PER_INDEX is tuned
# for that pipeline's own needs -- hence its own small cap here rather than
# reusing the constant above.
SEA_RCA_MAX_OBSERVATIONS_PER_INDEX = 2000
SEA_RCA_TOP_K_EVIDENCE = 15

# ---------------------------------------------------------------------------
# Proposal 2: TWIST-to-Text Evidence Synthesis -- see
# pipeline/twist_scoring.py's twist_scores_to_observations() and
# method-proposals.md (project doc) for the full writeup. Motivated by the
# sea_rca negative result (2026-10-02): entity_localization suffered because
# retrieval_precision/recall were 0.0 for most cases -- the hybrid vector
# index never embeds raw trace observations (build_index_from_observations()
# defaults to logs/metrics/events only), so TWIST's trace-derived anomaly
# signal (c1..c4) was only ever visible to the LLM as a fixed top-5 text
# block, never retrievable by the alert-text/keyword queries that drive
# vector_based_items. This turns each service's TWIST scores into one
# synthesized sentence and feeds it into the SAME vector index as a new
# "twist_synth" modality, so semantic retrieval can surface high-twist
# services even when the raw span text itself doesn't lexically/semantically
# match the query. Toggle OFF for a clean with/without ablation (same cap,
# same cost elsewhere) -- see method-proposals.md's validation plan, which
# calls for breaking results down by gt_fault_group, not just an overall mean.
USE_TWIST_TEXT_EVIDENCE = os.environ.get("USE_TWIST_TEXT_EVIDENCE", "1") == "1"
TWIST_TEXT_TOP_N_SERVICES = 15   # cap on how many services get a synthesized
                                  # sentence, highest twist_score first -- RCA100
                                  # cases rarely have more than a few dozen
                                  # services, so this is a safety bound, not a
                                  # routinely-hit cap.

# ---------------------------------------------------------------------------
# Self-Consistency ensemble for the Coordinator's final diagnosis -- see
# agents/multi_agent.py's _self_consistency_sample() and
# coordinator_node(). Wang et al., ICLR 2023, "Self-Consistency Improves
# Chain of Thought Reasoning in Language Models": sample N independent
# completions, majority-vote predicted_fault_type, keep the
# highest-confidence sample among those agreeing with the majority.
# Training-free, no GT dependency -- pure test-time compute, aimed directly
# at fault_identification + reasoning_process (the Coordinator's own
# accuracy), unlike Proposals 2/3 which target entity-localization evidence
# quality instead. Scoped to ONLY the Coordinator call (see that function's
# docstring for why) -- Graph Anchor and Specialist Agents are NOT re-sampled.
#
# NEGATIVE RESULT (2026-10-04, 8-case sample): self_consistency_agreement
# was 1.0 (all 3 samples identical) in 7/8 cases even at
# SELF_CONSISTENCY_TEMPERATURE=0.7 -- predicted_fault_type did not change in
# ANY of the 8 cases vs. the single-shot run before this feature existed.
# This rules out sampling-variance as the bottleneck: the model is already
# highly consistent across samples, so voting has nothing to correct. The
# real driver turned out to be COORDINATOR_ANCHOR_CONFIRM_BIAS below
# (predicted_fault_type matched graph_anchor_fault_type in 7/8 cases --
# a prompt-level anchoring effect, not sampling noise). Defaulted OFF to
# stop paying ~80s/case for zero measured benefit while that's investigated;
# flip back to "1" to re-test once the anchor-bias ablation below has run,
# in case reducing the anchor's pull also increases sample-to-sample
# disagreement (making voting useful again).
USE_SELF_CONSISTENCY = os.environ.get("USE_SELF_CONSISTENCY", "0") == "1"
SELF_CONSISTENCY_SAMPLES = 3        # N completions to sample and vote over.
                                      # Cost is linear in this (3x Coordinator
                                      # calls only, not the whole 6-call
                                      # pipeline) -- 3 is the standard
                                      # small-N starting point in the
                                      # self-consistency literature.
SELF_CONSISTENCY_TEMPERATURE = 0.7   # temporarily overrides LLM_TEMPERATURE
                                      # (default 0.1) for just these N calls --
                                      # see docstring for why 0.1 is too low
                                      # for the samples to disagree meaningfully.

# ---------------------------------------------------------------------------
# Coordinator anchor-bias ablation -- see agents/multi_agent.py's
# coordinator_node() anchor_block construction. True (default, legacy,
# unchanged from the original design) tells the Coordinator to CONFIRM the
# Stage-0.5 Graph Anchor's fault-type guess unless specialist findings
# clearly contradict it. False swaps in neutral wording that explicitly
# tells the Coordinator the anchor hasn't seen any evidence and should be
# weighed on equal footing with the specialist findings, not confirmed by
# default. Motivated by the 2026-10-04 finding above: predicted_fault_type
# matched graph_anchor_fault_type in 7/8 cases on the 8-case sample, and the
# one case the Coordinator DID override the anchor (t077), it was correct.
# Free to test either way -- this only changes prompt wording, no added
# LLM calls.
COORDINATOR_ANCHOR_CONFIRM_BIAS = os.environ.get("COORDINATOR_ANCHOR_CONFIRM_BIAS", "1") == "1"

# ---------------------------------------------------------------------------
# Two-stage Coordinator (order-of-exposure anchoring-bias mitigation) -- see
# agents/multi_agent.py's _two_stage_coordinate(). Deeper fix than
# COORDINATOR_ANCHOR_CONFIRM_BIAS: that toggle only changes wording AFTER
# the anchor is already in the prompt; this instead elicits an independent,
# evidence-only diagnosis (specialist findings + keyword hints + zero-shot/
# taxonomy hint, NO anchor) in a first Coordinator call, then reveals the
# anchor in a second call and asks for an explicit, cited reconciliation.
# Standard anchoring-bias mitigation (Tversky & Kahneman): a judgment formed
# before exposure to an anchor resists it regardless of the anchor's
# wording. Default False because it costs one extra Coordinator call per
# case (two total vs. one) -- opt in to test. Takes priority over
# COORDINATOR_ANCHOR_CONFIRM_BIAS when both apply (the single-call anchor
# wording question becomes moot once there's no single call with the
# anchor baked in from the start) and is mutually exclusive with
# USE_SELF_CONSISTENCY in the current implementation (test one mechanism at
# a time -- combining them would be 2x the self-consistency sample count in
# calls, and a confounded experiment).
COORDINATOR_TWO_STAGE_ANCHOR = os.environ.get("COORDINATOR_TWO_STAGE_ANCHOR", "0") == "1"

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
                                      "Qwen/Qwen2.5-7B-Instruct")
                                      # Plain Hugging Face Hub repo id (not a Kaggle-catalog
                                      # handle) -- transformers downloads it directly, no
                                      # kagglehub / Kaggle-page license click needed since the
                                      # weights are open. Chosen over the larger Gemma-4-12B-it
                                      # for reproducibility: fits comfortably on a single Kaggle
                                      # T4/P100 (no 2xT4 or quantization required), runs faster
                                      # per call across ~600+ calls/103-case run, and is easy
                                      # for someone else to rerun on the free tier. Swap to a
                                      # Kaggle-catalog handle (org/model/framework/variant, e.g.
                                      # "google/gemma-4/transformers/gemma-4-12b-it") or a
                                      # KAGGLE_MODEL_LOCAL_PATH mount for a larger-model run.
KAGGLE_MODEL_LOCAL_PATH = os.environ.get("KAGGLE_MODEL_LOCAL_PATH", None)

# --- Optional: SEPARATE model for the Coordinator only -----------------------
# Everything else (specialist agents, Graph Anchor) keeps using
# KAGGLE_MODEL_HANDLE/KAGGLE_MODEL_LOCAL_PATH above. Leave unset (None) to
# use the SAME model everywhere (the default configuration) -- only set
# this for a deliberate, clearly-labeled ablation comparing a stronger
# Coordinator against the baseline Coordinator, not as a silent swap that
# would invalidate comparability with the baseline systems. (An earlier
# version of this comment cited a fault_group_identification split by the
# since-removed GT-trained classifier's confidence as motivation -- that
# classifier was deleted for benchmark-leakage reasons, so that specific
# number no longer applies; the option itself is still useful on its own
# merits for a Coordinator-strength ablation.)
COORDINATOR_MODEL_HANDLE = os.environ.get("COORDINATOR_MODEL_HANDLE", None)
COORDINATOR_MODEL_LOCAL_PATH = os.environ.get("COORDINATOR_MODEL_LOCAL_PATH", None)
COORDINATOR_USE_4BIT = os.environ.get("COORDINATOR_USE_4BIT", "0") == "1"
                                      # bitsandbytes 4-bit quantization, NOT GPTQ (which hit an
                                      # unresolved optimum/auto-gptq compatibility bug this
                                      # session) -- lets a larger Coordinator-only model (e.g.
                                      # Qwen2.5-32B-Instruct) fit in Kaggle's GPU memory.
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

# --- Propagation-aware localization (2026-10-07) ----------------------------
# Audit over all 103 cases: alerts mostly mark the VICTIM; the origin is a
# downstream callee in ~47% of cases. USE_PROPAGATION_EVIDENCE=1 gives the
# anchor LLM and the agents a call-graph/trace table (evidence only, no
# filtering) and replaces the "most topologically central" anchor instruction
# with an origin-vs-victim instruction. Default off = previous behaviour.
USE_PROPAGATION_EVIDENCE = os.environ.get("USE_PROPAGATION_EVIDENCE", "0") == "1"
# Which component's entity becomes predicted_entity_ids:
#   "anchor"      (previous behaviour: Stage-0.5 anchor overrides the Coordinator)
#   "coordinator" (the Coordinator's own entity, falling back to the anchor)
ENTITY_SOURCE = os.environ.get("ENTITY_SOURCE", "anchor")
