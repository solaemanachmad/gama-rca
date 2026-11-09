# Run definitions read by scripts/kaggle_job.py (the notebook never changes: clone + one kaggle_job command). The old notebook cells
# and the development arms are in archive/scripts/kaggle_run.py (git tag v5-frozen).

# %% Cell 3 -- definitions only (BASE, FLAGS, MODELS, ARMS, case lists); runs NO model
import os
DATA = globals().get("DATA", os.environ.get("RCA100_ROOT", "/kaggle/input/rca100"))   # kaggle_job.definitions() injects DATA
LOCAL_MODEL = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
BASE = dict(
    LLM_BACKEND="kaggle", CHUNKED_PREFILL="1", MAX_INPUT_TOKENS="16000", PREFILL_CHUNK="1024",
    LLM_TEMPERATURE="0", LLM_SEED="0",                              # deterministic decoding (recorded in the manifest)
    USE_LOG_TEMPLATES="1", LOG_TEMPLATE_ENGINE="drain3", DRAIN_SIM_TH="0.8",
    USE_PROPAGATION_EVIDENCE="1", ENTITY_SOURCE="coordinator", CHAIN_FIRST_ENTITY="coordinator",
    USE_INFRA_EVIDENCE="1", CHAIN_FIRST="1",
    USE_CHANGE_FACTOR="1", USE_CHANGE_STATS="1", USE_TRACE_EVIDENCE="1",
    PYTHONHASHSEED="0", RCA100_ROOT=DATA, PYTHONUNBUFFERED="1",   # unbuffered: per-case lines reach the .log at once (progress counter)
    **({"KAGGLE_MODEL_LOCAL_PATH": LOCAL_MODEL} if LOCAL_MODEL else {}))
FIXED = {"STRICT_MODALITY_ROUTING": "1", "SPECIALIST_ENGLISH": "1"}   # strict routing + English specialist text (the base A1rT_fixed)
# Backbone presets (MODEL in Cell 4). Each is a set of env variables on top of BASE; "" clears the local path so the HF handle is used.
# Prefer attaching the model via Add Input > Models and putting its mounted folder in KAGGLE_MODEL_LOCAL_PATH (no download). T4 has no native
# bf16, so the 4-bit presets compute in float16. Qwen3-32B in 4-bit needs ~20 GB of GPU memory (2xT4) and ~65 GB of disk for the bf16 download:
# attach a pre-quantised copy rather than downloading.
MODELS = {
    "qwen25_7b": {},                                                                                 # the main configuration so far (BASE)
    "qwen3_8b": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen3-8B"},
    "qwen3_8b_local": {"KAGGLE_MODEL_LOCAL_PATH": "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1"},   # v3 backbone (amendment 1); thinking OFF
    "qwen3_8b_think": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen3-8B", "ENABLE_THINKING": "1", "LLM_MAX_TOKENS": "4096"},
    "qwen25_14b_4bit": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen2.5-14B-Instruct", "KAGGLE_USE_4BIT": "1", "KAGGLE_DTYPE": "float16"},
    "qwen3_14b_4bit": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen3-14B", "KAGGLE_USE_4BIT": "1", "KAGGLE_DTYPE": "float16"},
    "qwen3_32b_4bit": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen3-32B", "KAGGLE_USE_4BIT": "1", "KAGGLE_DTYPE": "float16"},
    # v5 multi-backbone study (PREREGISTRATION_V5.md): same prompts, greedy decoding, bfloat16, thinking off. Attach each model via
    # Add Input > Models and put its mounted folder in KAGGLE_MODEL_LOCAL_PATH (Llama and Gemma are gated on Hugging Face).
    "qwen25_7b_v5": {"KAGGLE_MODEL_LOCAL_PATH": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"},
    "llama31_8b": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "meta-llama/Llama-3.1-8B-Instruct"},
    # one large API model (F4 / H3 / H4): gpt-oss-120b, reasoning effort fixed at medium; the key comes from the Kaggle Secret named
    # in API_KEY_SECRET (scripts/kaggle_job.py); provider data controls: zero data retention. CPU session (no GPU needed).
    "gptoss120b_groq": {"LLM_BACKEND": "api", "API_BASE_URL": "https://api.groq.com/openai/v1", "API_MODEL": "openai/gpt-oss-120b",
                        "API_KEY_SECRET": "GROQ_API_KEY", "API_REASONING_EFFORT": "medium", "LLM_MAX_TOKENS": "8192"},
    "gptoss120b_cerebras": {"LLM_BACKEND": "api", "API_BASE_URL": "https://api.cerebras.ai/v1", "API_MODEL": "gpt-oss-120b",
                            "API_KEY_SECRET": "CEREBRAS_API_KEY", "API_REASONING_EFFORT": "medium", "LLM_MAX_TOKENS": "8192",
                            "API_MIN_INTERVAL_S": "13"},          # free trial: 5 requests / minute
    "gemma2_9b": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "google/gemma-2-9b-it"},
    "qwen3_4b": {"KAGGLE_MODEL_LOCAL_PATH": "", "KAGGLE_MODEL_HANDLE": "Qwen/Qwen3-4B"},
}
FLAGS = ("ALERT_T0_MODE", "USE_EVENT_DIGEST", "CHAIN_ABSOLUTE_SIGNALS", "INFRA_ALL_ENTITIES", "SPEC_CHAIN_SEGMENTS",
         "STRICT_MODALITY_ROUTING", "SPECIALIST_ENGLISH", "USE_KNOWLEDGE_POS", "USE_LLM_SPECIALIST_AGENTS", "GROUNDED_FAULT",
         "ENTITY_TYPING", "NO_SPECIALISTS", "SPEC_EVIDENCE_ONLY", "COORD_LEAN", "INFRA_EVENTS_SCOPED", "TYPING_COMPARE", "USE_CHANGE_STATS", "USE_CHANGE_FACTOR",
         "FAULT_FINGERPRINT", "FP_TOP_K", "FP_LEVEL_SWITCH", "FP_SCORING", "FP_ALPHA", "VERIFY_CHAIN", "VERIFY_MIN_GROUNDED",
         "ENABLE_THINKING", "LLM_MAX_TOKENS", "KAGGLE_MODEL_HANDLE", "KAGGLE_USE_4BIT", "KAGGLE_DTYPE",
         "V2_NO_TOPO", "V2_NO_EXCLUDE", "V2_UNCALIBRATED", "V2_NO_VERIFY", "V2_NO_LOCALITY", "V3_LOG_MODE", "V4_PARTS", "GAMA_MODULES")
# Parts of the frozen configuration in the development code's names; the Evidence->LLM baseline reads them (trace origins, text).
V5_FINAL_PARTS = ("score,open,twostage,fp,peer,prop,events,avail,onset,em,selfnorm,llmtype,conseq,retry,"
                  "mm,torigin3,omsg,hostcpu,errclass,tiebreak,sigorder2,loadx2,msgtarget,textx2,redchain")
W = {"ALERT_T0_MODE": "window"}
ARMS = {                                                          # name -> extra env, on BASE + model preset + FIXED
    "GAMA": {"_SYSTEM": "gama", **W, "GAMA_MODULES": "core"},                    # CPU; kaggle_job --modules core,-trace ... overrides
    "GAMA_FULL": {"_SYSTEM": "gama", **W, "GAMA_MODULES": "full"},               # GPU: + bounded local LLM (M7)
    "BELIEF_LLM": {"_SYSTEM": "belief_llm", **W, "V3_LOG_MODE": "template"},     # GPU: GAMA's belief, the LLM decides freely
    "EVIDENCE_LLM_FINAL": {"_SYSTEM": "evidence_llm", **W, "V3_LOG_MODE": "template", "V4_PARTS": V5_FINAL_PARTS},   # GPU baseline
    "DIRECT_LLM": {"_SYSTEM": "direct_llm"},                                     # GPU baseline: alert only
    # names of the reported runs; identical outputs to the frozen development arms (PREREGISTRATION_V5.md, equivalence 206 / 206)
    "A": {"_SYSTEM": "gama", **W, "GAMA_MODULES": "base"},                       # configuration A (M0 only)
    "GAMA_CORE": {"_SYSTEM": "gama", **W, "GAMA_MODULES": "core"},
    "GAMA_LLM": {"_SYSTEM": "gama", **W, "GAMA_MODULES": "full"},
    **{f"GAMA_CORE_NO_{name}": {"_SYSTEM": "gama", **W, "GAMA_MODULES": f"core,-{mod}"}
       for name, mod in (("MM", "mm"), ("TRACE", "trace"), ("TYPE", "typing"), ("LOAD", "load"), ("TEXT", "text"), ("CHAIN", "chain"))},
}
_MF = os.path.join(DATA, "manifest.txt")                  # absent when only the presets are read (scripts/aiops2025_run.py --model)
ALL_IDS = [l.strip() for l in open(_MF) if l.strip()] if os.path.exists(_MF) else []
DEV_ODD = [c for c in ALL_IDS if int(c[1:]) % 2 == 1]     # internal record of the v5 development protocol (PREREGISTRATION_V5.md)
VAL_EVEN = [c for c in ALL_IDS if int(c[1:]) % 2 == 0]

# %% Cell end
