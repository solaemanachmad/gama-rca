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
# Default = RCA100v1.1 next to the repo (v1.1 is the data version all our results use; the
# repo's .gitignore already excludes RCA100*/). Override with RCA100_ROOT.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get("RCA100_ROOT", os.path.join(_REPO_DIR, "RCA100v1.1"))

CASES_DIR = os.path.join(DATASET_ROOT, "cases")
ANSWER_KEY_DIR = os.path.join(DATASET_ROOT, "answer_key")   # underscore, per AIOps_README.md — NEVER read during retrieval/reasoning
MANIFEST_PATH = os.path.join(DATASET_ROOT, "manifest.txt")
SUMMARY_PATH = os.path.join(DATASET_ROOT, "summary.json")

# Dataset version guard: all experiments are compared within ONE data version (v1.1).
# DATASET_VERSION is read from summary.json (informational; logged per case and printed at start).
EXPECTED_DATASET_VERSION = os.environ.get("EXPECTED_DATASET_VERSION", "1.1")


def _read_dataset_version():
    try:
        import json as _json
        with open(SUMMARY_PATH, "r", encoding="utf-8") as _f:
            v = _json.load(_f).get("version")
        if v:
            return str(v)
    except Exception:
        pass
    # fall back to the folder name (e.g. RCA100v1.1)
    import re as _re
    m = _re.search(r"v(\d+\.\d+)", os.path.basename(os.path.normpath(DATASET_ROOT)))
    return m.group(1) if m else "unknown"


DATASET_VERSION = _read_dataset_version()

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
# Main (all-agents) client: optional 4-bit NF4 and compute dtype. Defaults reproduce earlier runs
# T0 rule: "legacy" = tag alert_time, else alert_window.start (default; reproduces earlier runs);
# "alert" = always the alert-fired time (uniform across cases).
ALERT_T0_MODE = os.environ.get("ALERT_T0_MODE", "legacy")  # legacy | alert | window (window: T0 = task alert_window.start; evidence window = [start, end])
# exactly (bf16, no quantisation). For a 14B run on T4 GPUs use KAGGLE_USE_4BIT=1 and
# KAGGLE_DTYPE=float16 (T4 has no native bf16).
KAGGLE_USE_4BIT = os.environ.get("KAGGLE_USE_4BIT", "0") == "1"
KAGGLE_DTYPE = os.environ.get("KAGGLE_DTYPE", "bfloat16")   # bfloat16 | float16
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

# --- P2: numeric infra evidence + chain-first Coordinator (2026-10-09) -------
# USE_INFRA_EVIDENCE=1 adds pipeline/infra_evidence.py tables (evidence only).
# CHAIN_FIRST=1 makes the Coordinator emit a structured chain
# ("step_type: target ...") and derives entity + fault type from its cause step
# (fault type no longer copied from the Stage-0.5 anchor).
USE_INFRA_EVIDENCE = os.environ.get("USE_INFRA_EVIDENCE", "0") == "1"
CHAIN_FIRST = os.environ.get("CHAIN_FIRST", "0") == "1"

# --- P3: log-template evidence layer (2026-10-10) ---------------------------
# USE_LOG_TEMPLATES=1 replaces raw log + metric rows in the vector index with
# one entry per log template (regex masking; optional Drain3 stage via
# LOG_TEMPLATE_ENGINE=drain3). Metrics are served by the numeric operators
# (USE_INFRA_EVIDENCE) instead of embeddings. Default off = previous behaviour.
USE_LOG_TEMPLATES = os.environ.get("USE_LOG_TEMPLATES", "0") == "1"

# P4 (operator-style staged analysis; all default off = previous behaviour).
# USE_COLOCATION_EVIDENCE: per-node table of hosted services' error/latency and node
#   cpu/mem/disk (pipeline/colocation_evidence.py; evidence only).
# LAYER_AGENT: one extra LLM call before the Coordinator that decides at which level
#   (service / runtime / dependency / pod-deployment / node / cloud) the fault
#   originates, citing the tables; its output is a soft hypothesis for the Coordinator
#   (the full taxonomy stays visible, nothing is filtered).
# CHAIN_FIRST_ENTITY: with CHAIN_FIRST, take the final entity from the "coordinator"
#   chain (R5D behaviour) or from the topology "anchor" (R5D showed anchor EL 0.575 vs
#   coordinator 0.479 on dev-30; chain still decides the fault type).
USE_COLOCATION_EVIDENCE = os.environ.get("USE_COLOCATION_EVIDENCE", "0") == "1"
LAYER_AGENT = os.environ.get("LAYER_AGENT", "0") == "1"
CHAIN_FIRST_ENTITY = os.environ.get("CHAIN_FIRST_ENTITY", "coordinator")
# Debug: when set to a directory, pipeline writes one JSON per case with the exact prompts and
# raw outputs of every LLM call (anchor, agents, layer, coordinator). No ground truth inside.
DEBUG_DUMP_DIR = os.environ.get("DEBUG_DUMP_DIR", "")

# USE_CHANGE_FACTOR=1: append "change=xN" (window max / baseline median) to every numeric row of the
# infra and co-location tables. Pure evidence formatting (no threshold, no ranking). Motivation (A3
# debug dump, 2026-10-11): the 7B model misread "base/med/max" triplets, e.g. ad CPU base=0.0008
# max=0.56 (x670) vs memory flat, yet predicted memoryPressure.
USE_CHANGE_FACTOR = os.environ.get("USE_CHANGE_FACTOR", "0") == "1"


# USE_KEYWORD_HINT=0: do NOT show the coordinator the "KEYWORD-DETECTED FAULT-TYPE CANDIDATES" block
# nor the zero-shot embedding shortlist (both are deterministic evidence->fault-type suggestions). Research rule: the LLM decides the fault type;
# deterministic code may only extract evidence. Default "1" keeps legacy (archived R0-R5D) behaviour;
# all operator-ladder arms (A0-A3) set it to 0. Keyword candidates are still LOGGED as a diagnostic.
USE_KEYWORD_HINT = os.environ.get("USE_KEYWORD_HINT", "1") == "1"

# USE_CHANGE_STATS=1 (needs USE_CHANGE_FACTOR=1): also append robust z-score (MAD), baseline rank
# (% of baseline samples below the window max) and baseline sample count n next to the ratio.
USE_CHANGE_STATS = os.environ.get("USE_CHANGE_STATS", "0") == "1"

# Arm K -- USE_KNOWLEDGE_PROMPT=1: add a block of "how to tell similar fault types apart" (written from
# standard SRE knowledge + the public taxonomy definitions; no case-specific text, no GT) to the
# chain-first Coordinator prompt. The LLM still decides; nothing is filtered. See data.taxonomy.knowledge_prompt_block.
USE_KNOWLEDGE_PROMPT = os.environ.get("USE_KNOWLEDGE_PROMPT", "0") == "1"

# Arm V -- VERIFY_CHAIN=1: after the chain-first Coordinator, one verifier call checks the chain against the
# evidence tables (verbatim quotes, at most 3 issues, absence of evidence is not contradiction); if it reports
# issues the Coordinator revises once. A deterministic pre-check only lists cited numbers that appear in no table
# (grounding aid, decides nothing). Adds 1 call (+1 when revised).
VERIFY_CHAIN = os.environ.get("VERIFY_CHAIN", "0") == "1"

# USE_TRACE_EVIDENCE=1: per-service table (call-graph neighbourhood of the alert service) of HTTP status-code
# counts, masked error messages and slowest operations, alert window vs baseline (pipeline/trace_evidence.py;
# evidence extraction only). Meant to separate fault types that differ in what the failing calls say.
USE_TRACE_EVIDENCE = os.environ.get("USE_TRACE_EVIDENCE", "0") == "1"

# USE_DIRECTION_EVIDENCE=1 (arm D2): the infra tables also show the LOW side of every row (window minimum with its
# ratio/z/rank, and the window-median ratio), lead each service/operation line with traffic then errors, and use larger
# per-section budgets. Needs USE_CHANGE_FACTOR=1. Evidence formatting only. Default off = prompts unchanged.
USE_DIRECTION_EVIDENCE = os.environ.get("USE_DIRECTION_EVIDENCE", "0") == "1"

# USE_ALERT_LIFECYCLE=1 (arm L): one condensed block of the alert stream (pipeline/alert_lifecycle.py): repeated alert
# rows are run-length-encoded (same condition values -> one run), T axis = first alert firing, only rows up to the
# task's alert_window.end are used, final state (still firing / RECOVERED at T+x) is stated. Evidence extraction only.
# Shown in the chain-first Coordinator prompt (needs CHAIN_FIRST=1). Default off = prompts unchanged.
USE_ALERT_LIFECYCLE = os.environ.get("USE_ALERT_LIFECYCLE", "0") == "1"

# USE_EVENT_DIGEST=1 (arm E): the k8s events section becomes a time-labelled digest (one line per type/kind/reason,
# workload names, counts, first/last offset from the window start) instead of per-object rows cut alphabetically at
# CHAR_BUDGET["event"]. Every group is represented; no significance filter. Evidence extraction only.
USE_EVENT_DIGEST = os.environ.get("USE_EVENT_DIGEST", "0") == "1"

# USE_LOG_EVIDENCE=1 (arm G): per-burst log composition block in the Coordinator prompt (pipeline/log_evidence.py):
# container share of each export burst (window vs earlier bursts), error-like share, new/absent message templates
# for the alert service's call-graph neighbourhood. Raw counts and per-minute rates are not shown (export artefact).
USE_LOG_EVIDENCE = os.environ.get("USE_LOG_EVIDENCE", "0") == "1"

# USE_TRACE_EDGES=1 (arm H): per call-edge trace evidence per export burst in the Coordinator prompt
# (pipeline/trace_edges.py): edge share, error fraction, p50/p95 duration (window burst(s) vs earlier bursts), inherited
# vs leaf errors per service, orphan share. Raw span counts are not shown (export artefact). Independent of
# USE_TRACE_EVIDENCE (arm T, per-service HTTP codes / error messages), which is kept as is for comparison.
USE_TRACE_EDGES = os.environ.get("USE_TRACE_EDGES", "0") == "1"

# USE_METRIC_STRUCTURE=1 (arm M): (1) identity-free K8s aggregates (statistics across nodes/deployments at each timestamp;
# the export carries no node/deployment names) and (2) call-edge flow ratios callee/caller workload (pipeline/metric_structure.py),
# in the Coordinator prompt after the infra tables. Default off.
USE_METRIC_STRUCTURE = os.environ.get("USE_METRIC_STRUCTURE", "0") == "1"
# USE_INTERVAL_STRIP=1 (arm X, claude/design-arm-X-interval-strip-2026-10-12.md): time-aligned strip (30 s bins over the alert window)
# of service workload/error/latency (multiples of own baseline), span error share (covered bins only), alert stream and K8s events
# on neighbourhood pods, with export coverage stated (pipeline/interval_strip.py). Coordinator prompt only, after the infra tables.
USE_INTERVAL_STRIP = os.environ.get("USE_INTERVAL_STRIP", "0") == "1"
# SYMPTOM_FIRST=1 (arm S): the chain-first JSON asks for a per-signal-family symptom reading of the cause target before predicted_fault_type
# (agents/multi_agent.py CHAIN_FIRST_SCHEMA_HINT_S). Default off = prompt unchanged.
SYMPTOM_FIRST = os.environ.get("SYMPTOM_FIRST", "0") == "1"
# CHAIN_STEPS_FLEX=1: the chain-first Task text allows a fourth (propagation) step (12 of 103 RCA100 chains have 4 steps). Default off.
CHAIN_STEPS_FLEX = os.environ.get("CHAIN_STEPS_FLEX", "0") == "1"
# TOPOLOGY_NAMES=1 (arm T-names, claude/audit-top20-ranking-2026-10-12.md): the topology specialist sees service NAMES, an alphabetical list of
# the call-graph neighbourhood and the graph path as a neutral fact, instead of a Top-20 list of hex ids ranked by nearness (rank 1 = the
# alert's service in 103/103) and a path presented as its "primary reasoning basis". Default off = prompt unchanged.
TOPOLOGY_NAMES = os.environ.get("TOPOLOGY_NAMES", "0") == "1"
# STRICT_MODALITY_ROUTING=1 (audit 2026-10-12, claude/audit-agent-flow-2026-10-12.md, F2): the metrics/logs/trace specialists pick
# their bullets by SUBSTRING ("metric"/"log"/"span") on the bullet text. That leaked the call-graph table (it contains "spans="
# and e.g. "product-catalog") into the logs specialist, the twist summary and OpenTelemetry log lines (span_id=...) into the
# trace specialist. With the flag on, a bullet reaches a specialist by its explicit kind (Metric.../Log.../span-count) and the
# call-graph table (`_propagation`, derived from spans) goes to the trace specialist only; `_twist_scores` (cross-modality) goes
# to none. Default 0 = legacy routing, so older CSVs stay reproducible.
STRICT_MODALITY_ROUTING = os.environ.get("STRICT_MODALITY_ROUTING", "0") == "1"
# SPECIALIST_ENGLISH=1 (audit 2026-10-12, U4): the specialists' free-text fields (summary, supporting_evidence) are asked to be written
# in English. Without it, Qwen sometimes answers in Chinese (alert titles and some log messages are Chinese) and those characters
# enter the Coordinator prompt. Wording change only; evidence is untouched. Default 0 = legacy prompts.
SPECIALIST_ENGLISH = os.environ.get("SPECIALIST_ENGLISH", "0") == "1"

# USE_ONSET_EVIDENCE=1: compact per-minute robust-z table (alert-5min..+1min) of error/latency of the alert service's
# call-graph neighbourhood and cpu/memory of hosting nodes (pipeline/onset_evidence.py; evidence extraction only, ~1.8k chars).
# Shows WHEN each signal starts to deviate (origin before propagation). No threshold, no filtering.
USE_ONSET_EVIDENCE = os.environ.get("USE_ONSET_EVIDENCE", "0") == "1"
# Diagnostic only: log the Coordinator prompt size (chars) per case; PROMPT_CHAR_WARN just flags long prompts in stats.
PROMPT_CHAR_WARN = int(os.environ.get("PROMPT_CHAR_WARN", "24000"))

# Arm D -- USE_DOSSIER_FT=1: after the entity is chosen, a SEPARATE short call sees only the evidence rows that mention
# that entity (and its host) and picks the fault type among lettered options by next-token probability (two option
# orders averaged). Replaces the chain-first Coordinator's fault type; entity and chain are unchanged (agents/dossier.py).
USE_DOSSIER_FT = os.environ.get("USE_DOSSIER_FT", "0") == "1"
