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
backends (local embeddings <-> HF API).
"""

import os

# Load .env (if present) so HF_TOKEN, RCA100_ROOT, etc. don't
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

# Removed experiments (self-consistency, two-stage Coordinator, anchor-confirm wording and others): tag pre-cleanup-3ef1e24 keeps the code.

# ---------------------------------------------------------------------------
# LLM settings -- swap backend/model without touching any other file.
# ---------------------------------------------------------------------------
LLM_BACKEND = "kaggle"   # the only backend: local transformers model, no server, no external API
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
# Decoding. Default is deterministic (greedy). Until 2026-10-13 this was 0.1, which
# made the Kaggle client SAMPLE (do_sample = temperature > 0) with no seed, so the
# specialists and the Coordinator varied from run to run; only the Anchor call was
# forced to 0. Override with env LLM_TEMPERATURE only for a deliberate sampling study.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
LLM_SEED = int(os.environ.get("LLM_SEED", "0"))   # torch seed set before every generate() call
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))   # raise (e.g. 4096) when ENABLE_THINKING=1: the reasoning trace counts against it
# ENABLE_THINKING=1: pass enable_thinking=True to the chat template (Qwen3 / reasoning models). The <think>...</think> block is stripped
# before JSON parsing. Default off reproduces all earlier runs (Qwen2.5 templates ignore the flag).
ENABLE_THINKING = os.environ.get("ENABLE_THINKING", "0") == "1"

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

# CHAIN_FIRST_ENTITY: with CHAIN_FIRST, take the final entity from the "coordinator" chain or from the topology "anchor".
CHAIN_FIRST_ENTITY = os.environ.get("CHAIN_FIRST_ENTITY", "coordinator")
# Debug: when set to a directory, pipeline writes one JSON per case with the exact prompts and
# raw outputs of every LLM call (anchor, agents, layer, coordinator). No ground truth inside.
DEBUG_DUMP_DIR = os.environ.get("DEBUG_DUMP_DIR", "")

# USE_CHANGE_FACTOR=1: append "change=xN" (window max / baseline median) to every numeric row of the
# infra and co-location tables. Pure evidence formatting (no threshold, no ranking). Motivation (A3
# debug dump, 2026-10-11): the 7B model misread "base/med/max" triplets, e.g. ad CPU base=0.0008
# max=0.56 (x670) vs memory flat, yet predicted memoryPressure.
USE_CHANGE_FACTOR = os.environ.get("USE_CHANGE_FACTOR", "0") == "1"


# USE_CHANGE_STATS=1 (needs USE_CHANGE_FACTOR=1): also append robust z-score (MAD), baseline rank
# (% of baseline samples below the window max) and baseline sample count n next to the ratio.
USE_CHANGE_STATS = os.environ.get("USE_CHANGE_STATS", "0") == "1"

# USE_KNOWLEDGE_POS=1 (arm KP): same slot as arm K, text paraphrased from the public PoS agent catalog/methodology (MIT, cited in
# data.taxonomy.knowledge_prompt_block_pos).
USE_KNOWLEDGE_POS = os.environ.get("USE_KNOWLEDGE_POS", "0") == "1"


# USE_TRACE_EVIDENCE=1: per-service table (call-graph neighbourhood of the alert service) of HTTP status-code
# counts, masked error messages and slowest operations, alert window vs baseline (pipeline/trace_evidence.py;
# evidence extraction only). Meant to separate fault types that differ in what the failing calls say.
USE_TRACE_EVIDENCE = os.environ.get("USE_TRACE_EVIDENCE", "0") == "1"


# USE_EVENT_DIGEST=1 (arm E): the k8s events section becomes a time-labelled digest (one line per type/kind/reason,
# workload names, counts, first/last offset from the window start) instead of per-object rows cut alphabetically at
# CHAR_BUDGET["event"]. Every group is represented; no significance filter. Evidence extraction only.
USE_EVENT_DIGEST = os.environ.get("USE_EVENT_DIGEST", "0") == "1"


# CHAIN_ABSOLUTE_SIGNALS=1: the chain-first Task asks for absolute request count / average latency (s) / error count and finest-level targets. Default off.
CHAIN_ABSOLUTE_SIGNALS = os.environ.get("CHAIN_ABSOLUTE_SIGNALS", "0") == "1"
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

# Diagnostic only: log the Coordinator prompt size (chars) per case; PROMPT_CHAR_WARN just flags long prompts in stats.
PROMPT_CHAR_WARN = int(os.environ.get("PROMPT_CHAR_WARN", "24000"))


# INFRA_ALL_ENTITIES=1 (2026-10-13): the node and deployment tables list EVERY entity (rows are shortened: med, rank, n dropped) instead of
# cutting the alphabetical tail at a character budget. Audit: the tables were cut in 30/30 dev cases (7-8 of ~15 nodes, 1-15 deployments missing).
INFRA_ALL_ENTITIES = os.environ.get("INFRA_ALL_ENTITIES", "0") == "1"

# SPEC_CHAIN_SEGMENTS=1 (2026-10-13, arm SEG): specialists write chain segments of their own modality instead of one suspect entity; the Coordinator
# merges them (system prompt without "weigh agreement heavily"). Same LLM call count. See claude/audit-fault-choice-and-design-seg-2026-10-13.md.
SPEC_CHAIN_SEGMENTS = os.environ.get("SPEC_CHAIN_SEGMENTS", "0") == "1"

# GROUNDED_FAULT=1 (arm P1, claude/design-arm-P-grounded-fault-choice-2026-10-12.md): the chain-first Task gets one uniform rule (the fault type must
# follow from the cause step; copy the signal=value that the chosen type's definition requires) and the JSON a 'fault_evidence' field BEFORE
# predicted_fault_type. No LLM call added, no type filtered, same rule for every case. Default off.
GROUNDED_FAULT = os.environ.get("GROUNDED_FAULT", "0") == "1"

# Arms of 2026-10-13 (claude/prereg-arms-TYPE-NOSPEC-EVOBS-2026-10-13.md); all default off, no extra data, no ground truth.
# ENTITY_TYPING=1 (arm TYPE): after the chain-first Coordinator names the origin entity, ONE extra short LLM call chooses the fault type from the chain plus the
#   evidence rows that mention that entity's NAME (deterministic selection by name, never by significance) and the taxonomy.
# NO_SPECIALISTS=1 (arm NOSPEC): the four specialist calls are skipped; the Coordinator sees only the tables (ablation of the specialists).
# SPEC_EVIDENCE_ONLY=1 (arm EVOBS): specialists copy the observations of their own modality and do not name a suspect, summary or confidence.
ENTITY_TYPING = os.environ.get("ENTITY_TYPING", "0") == "1"
NO_SPECIALISTS = os.environ.get("NO_SPECIALISTS", "0") == "1"
SPEC_EVIDENCE_ONLY = os.environ.get("SPEC_EVIDENCE_ONLY", "0") == "1"
# COORD_LEAN=1 (arm TYPELEAN, needs ENTITY_TYPING=1): the chain-first Coordinator no longer sees the taxonomy, the KP block or a fault-type field; it writes only
#   the chain and the origin entity. The fault type is chosen afterwards by the typing call (entity-scoped evidence + taxonomy + KP). A failed typing call leaves 'unknown'.
COORD_LEAN = os.environ.get("COORD_LEAN", "0") == "1"
# INFRA_EVENTS_SCOPED=1 (arm EVSC): the K8s event section lists only event groups whose object NAME refers to a service of the alert neighbourhood (as the
#   tables already define it) or to a node that hosts one of those services; groups on other objects are not listed, only counted in one line. Selection by name,
#   never by significance; same rule for every case. Default (off): all groups, Warning first, alphabetical, cut at the character budget.
INFRA_EVENTS_SCOPED = os.environ.get("INFRA_EVENTS_SCOPED", "0") == "1"

# TYPING_COMPARE=1 (arm TYPECMP, needs ENTITY_TYPING=1): the typing call first proposes up to three candidate types and copies the supporting and the contradicting
# row for each, then chooses (one call, same evidence rows, nothing excluded by code). Default off.
TYPING_COMPARE = os.environ.get("TYPING_COMPARE", "0") == "1"

# FAULT_FINGERPRINT (arm FP; pipeline/fault_fingerprint.py): code checks a fixed list of discriminating signals for the origin entity and maps them
# to the fault types whose key signal is present (thresholds fixed in advance, never tuned on RCA100). '' = off (default, reproduces earlier runs);
# show = profile + candidates shown to the typing call, any type allowed; shortlist = the typing call must pick one of the top FP_TOP_K
# candidates; rule = top candidate, no LLM call. Replaces the ENTITY_TYPING call when set (it runs after the chain-first Coordinator).
FAULT_FINGERPRINT = os.environ.get("FAULT_FINGERPRINT", "").strip().lower()
FP_TOP_K = int(os.environ.get("FP_TOP_K", "3"))
# FP_SCORING: calibrated (default; pipeline/fault_calibrated.py: rank tests against each entity's own baseline, categorical observations,
# hypotheses (entity, type)) | fixed (pipeline/fault_fingerprint.py: the first version with a-priori fixed thresholds; ablation only).
FP_SCORING = os.environ.get("FP_SCORING", "calibrated").strip().lower()
FP_ALPHA = float(os.environ.get("FP_ALPHA", "0.05"))
# FP_LEVEL_SWITCH=1: when the chosen type is node-level (nodeCpuHigh, nodeMemoryOOM, nodeDown, diskIOHigh) but the origin is a service, answer the
# hosting node that carries the signal as the root-cause entity (the type's level decides the entity level). Default off.
FP_LEVEL_SWITCH = os.environ.get("FP_LEVEL_SWITCH", "0") == "1"
# VERIFY_CHAIN=1 (arm VER): deterministic soundness check of the final answer (cause target == answered entity; every number quoted in the chain
# occurs in the evidence text the Coordinator saw). On failure, ONE repair call rewrites the chain with only values present in the rows.
VERIFY_CHAIN = os.environ.get("VERIFY_CHAIN", "0") == "1"
VERIFY_MIN_GROUNDED = float(os.environ.get("VERIFY_MIN_GROUNDED", "0.5"))   # repair when fewer than this share of quoted numbers is grounded

# SAVE_TYPING_STATE_DIR (infrastructure, no effect on predictions): after the agent graph, write per case the inputs of the typing step
# (alert text, evidence summary, call-graph / infra / trace tables, the Coordinator's final_result) as JSON. scripts/retype.py re-runs only
# the typing step of the FP arms on this state, so the shared Coordinator stage is not recomputed for every arm (GPU budget).
SAVE_TYPING_STATE_DIR = os.environ.get("SAVE_TYPING_STATE_DIR", "")
