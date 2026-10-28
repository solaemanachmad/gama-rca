# Kaggle runner. Paste each "# %% Cell N" block into its own notebook cell, in order (Cells 1-6, self-contained).
# Needs: Secret GITHUB_TOKEN, the RCA100 dataset as input, Accelerator = GPU, Internet ON, a Qwen2.5 model attached (or HF download).
# Cell 5 is resume-safe (progress is read from the dump files). Decoding is deterministic: temperature 0 and a fixed seed (see Cell 3).

# %% Cell 1 -- code (private repo) and data
import os, subprocess

REPO = "solaemanachmad/gama-rca"                      # private; no leading slash
DATA = "/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1"   # folder that contains cases/ and manifest.txt
WORK = "/kaggle/working/gama-rca"

from kaggle_secrets import UserSecretsClient          # Add-ons > Secrets: GITHUB_TOKEN (read-only token for the private repo)
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["rm", "-rf", WORK])
r = subprocess.run(["git", "clone", "--depth", "60", f"https://{tok}@github.com/{REPO}.git", WORK],
                   capture_output=True, text=True)
print("clone rc =", r.returncode, (r.stderr or "").replace(tok, "***")[-300:])
os.chdir(WORK)
COMMIT = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
print("COMMIT", COMMIT); print(subprocess.run("git log --oneline | head -5", shell=True, capture_output=True, text=True).stdout)
assert os.path.isdir(os.path.join(DATA, "cases")), f"dataset path wrong: {DATA}"
os.environ["RCA100_ROOT"] = DATA
os.environ["PYTHONHASHSEED"] = "0"

# %% Cell 2 -- dependencies (skip what the notebook image already has)
import sys, importlib
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "sentence-transformers", "networkx", "drain3", "faiss-cpu", "langgraph",
                "bitsandbytes"])                     # bitsandbytes: only the 4-bit model presets (14B/32B) need it
for m in ("drain3", "faiss", "sentence_transformers", "networkx", "langgraph", "torch", "transformers"):
    try:
        importlib.import_module(m); print("ok  ", m)
    except Exception as e:
        print("MISSING", m, "->", e)

# %% Cell 3 -- definitions only (BASE, FLAGS, ARMS, run helpers); runs NO model
import time, glob
# Model source. Local copy attached via Add Input > Models (qwen-lm / qwen2.5 / transformers / 7b-instruct / 1); None = download the HF default.
LOCAL_MODEL = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
BASE = dict(
    LLM_BACKEND="kaggle", CHUNKED_PREFILL="1", MAX_INPUT_TOKENS="16000", PREFILL_CHUNK="1024",
    LLM_TEMPERATURE="0", LLM_SEED="0",                              # deterministic decoding (recorded in the manifest)
    USE_LOG_TEMPLATES="1", LOG_TEMPLATE_ENGINE="drain3", DRAIN_SIM_TH="0.8",
    USE_PROPAGATION_EVIDENCE="1", ENTITY_SOURCE="coordinator", CHAIN_FIRST_ENTITY="coordinator",
    USE_INFRA_EVIDENCE="1", CHAIN_FIRST="1",
    USE_CHANGE_FACTOR="1", USE_CHANGE_STATS="1", USE_TRACE_EVIDENCE="1",
    PYTHONHASHSEED="0", RCA100_ROOT=DATA,
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
}
FLAGS = ("ALERT_T0_MODE", "USE_EVENT_DIGEST", "CHAIN_ABSOLUTE_SIGNALS", "INFRA_ALL_ENTITIES", "SPEC_CHAIN_SEGMENTS",
         "STRICT_MODALITY_ROUTING", "SPECIALIST_ENGLISH", "USE_KNOWLEDGE_POS", "USE_LLM_SPECIALIST_AGENTS", "GROUNDED_FAULT",
         "ENTITY_TYPING", "NO_SPECIALISTS", "SPEC_EVIDENCE_ONLY", "COORD_LEAN", "INFRA_EVENTS_SCOPED", "TYPING_COMPARE", "USE_CHANGE_STATS", "USE_CHANGE_FACTOR",
         "FAULT_FINGERPRINT", "FP_TOP_K", "FP_LEVEL_SWITCH", "FP_SCORING", "FP_ALPHA", "VERIFY_CHAIN", "VERIFY_MIN_GROUNDED",
         "ENABLE_THINKING", "LLM_MAX_TOKENS", "KAGGLE_MODEL_HANDLE", "KAGGLE_USE_4BIT", "KAGGLE_DTYPE",
         "V2_NO_TOPO", "V2_NO_EXCLUDE", "V2_UNCALIBRATED", "V2_NO_VERIFY", "V2_NO_LOCALITY", "V3_LOG_MODE", "V4_PARTS")
ARMS = {                                                          # name -> extra env, all on the fixed base
    "A1rT_fixed": {},
    "A1rT_fixed_ABS_KP": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1"},          # main configuration
    "A1rT_fixed_ABS_KP_P1": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "GROUNDED_FAULT": "1"},    # + grounded fault choice (fault_evidence before the type)
    "A1rT_fixed_ABS_KP_TYPE": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1"},        # fault type from entity-scoped evidence (+1 short call)
    "A1rT_fixed_ABS_KP_TYPELEAN": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1", "COORD_LEAN": "1"},   # Coordinator: entity + chain only; typing call: taxonomy + KP + entity evidence
    "A1rT_fixed_ABS_KP_TYPELEAN_EVOBS": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1", "COORD_LEAN": "1", "SPEC_EVIDENCE_ONLY": "1"},
    "A1rT_fixed_ABS_KP_EVSC": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_EVENTS_SCOPED": "1"},      # events only for the neighbourhood services and their nodes
    "A1rT_fixed_ABS_KP_TYPELEAN_EVSC": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1", "COORD_LEAN": "1", "INFRA_EVENTS_SCOPED": "1"},
    "A1rT_fixed_ABS_KP_TYPECMP": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1", "TYPING_COMPARE": "1"},   # typing call proposes <=3 candidates with supporting/contradicting rows, then chooses
    "A1rT_fixed_ABS_KP_TYPELEAN_CMP": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "ENTITY_TYPING": "1", "COORD_LEAN": "1", "TYPING_COMPARE": "1"},
    "A1rT_fixed_ABS_KP_NOSTATS": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "USE_CHANGE_STATS": "0", "USE_CHANGE_FACTOR": "0"},   # ablation: no change/z/rank/n appended to numeric rows
    "A1rT_fixed_ABS_KP_NOSPEC": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "NO_SPECIALISTS": "1"},       # ablation: no specialist calls
    "A1rT_fixed_ABS_KP_EVOBS": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "SPEC_EVIDENCE_ONLY": "1"},    # specialists copy observations, no verdict
    # Fault fingerprints (default scoring: pipeline/fault_calibrated.py; _FIX = pipeline/fault_fingerprint.py fixed thresholds) and the
    # chain verifier; all on the main configuration
    "A1rT_fixed_ABS_KP_FPSHOW": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "show"},        # profile + candidates shown, any type allowed
    "A1rT_fixed_ABS_KP_FPSL": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "shortlist"},     # typing call constrained to the top-3 candidates
    "A1rT_fixed_ABS_KP_FPRULE": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "rule"},        # ablation: top candidate, no typing LLM call
    "A1rT_fixed_ABS_KP_VER": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "VERIFY_CHAIN": "1"},                   # verifier only
    "A1rT_fixed_ABS_KP_FPSL_VER": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "shortlist", "VERIFY_CHAIN": "1"},   # candidate main configuration
    "A1rT_fixed_ABS_KP_FPRULE_FIX": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "rule", "FP_SCORING": "fixed"},   # ablation: fixed thresholds, no calibration
    "A1rT_fixed_ABS_KP_FPSL_FIX": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "shortlist", "FP_SCORING": "fixed"},
    "A1rT_fixed_ABS_KP_FPSL_VER_LVL": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "FAULT_FINGERPRINT": "shortlist", "VERIFY_CHAIN": "1", "FP_LEVEL_SWITCH": "1"},
    # v2 hypothesis-constrained RCA (pipeline/hypothesis_rca.py). "_SYSTEM" picks the system in main.py (not an env variable).
    # ALERT_T0_MODE=window: the benchmark's own alert_window (task.json) defines window and baseline.
    "V2": {"_SYSTEM": "hypothesis_rca", "ALERT_T0_MODE": "window", "USE_KNOWLEDGE_POS": "1"},              # main v2 (GPU, 1-2 calls/case)
    "V2_NOVERIFY": {"_SYSTEM": "hypothesis_rca", "ALERT_T0_MODE": "window", "USE_KNOWLEDGE_POS": "1", "V2_NO_VERIFY": "1"},
    "V2_RULE": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "window"},                                   # Stages 1-3 only (CPU)
    "V2_RULE_NOTOPO": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "window", "V2_NO_TOPO": "1"},         # ablations (CPU)
    "V2_RULE_NOEXCL": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "window", "V2_NO_EXCLUDE": "1"},
    "V2_RULE_UNCAL": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "window", "V2_UNCALIBRATED": "1"},
    "V2_RULE_NOLOCAL": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "window", "V2_NO_LOCALITY": "1"},
    "V2_RULE_LEGACYWIN": {"_SYSTEM": "hypothesis_rule", "ALERT_T0_MODE": "legacy"},                         # window definition of v1
    # v3 computed belief states (pipeline/belief_rca.py; PREREGISTRATION_V3.md)
    "V3": {"_SYSTEM": "belief_rca", "ALERT_T0_MODE": "window", "V3_LOG_MODE": "template"},                 # proposed (GPU; Amendment 2)
    "V3_RAWLOG": {"_SYSTEM": "belief_rca", "ALERT_T0_MODE": "window", "V3_LOG_MODE": "raw"},               # frozen v3 sensor input (ablation)
    "V3_B0": {"_SYSTEM": "belief_rca_b0", "ALERT_T0_MODE": "window"},                                      # no text sensor (GPU, narrator)
    "V3_RULE": {"_SYSTEM": "belief_rule", "ALERT_T0_MODE": "window"},                                      # no LLM at all (CPU)
    # v4 (pipeline/belief_v4.py; design claude/RCA_v4_design.md). RCA100 is DEVELOPMENT data for v4 (labels seen in V3_RAWLOG).
    "V4": {"_SYSTEM": "belief_v4", "ALERT_T0_MODE": "window", "V3_LOG_MODE": "template", "V4_PARTS": "all"},          # GPU
    "V4_B0": {"_SYSTEM": "belief_v4_b0", "ALERT_T0_MODE": "window", "V4_PARTS": "all"},                               # GPU (narrator)
    "V4_RULE": {"_SYSTEM": "belief_v4_rule", "ALERT_T0_MODE": "window", "V4_PARTS": "all"},                           # CPU
    **{f"V4_RULE_NO{p.upper()}": {"_SYSTEM": "belief_v4_rule", "ALERT_T0_MODE": "window",
                                  "V4_PARTS": ",".join(x for x in ("score", "open", "twostage", "fp") if x != p)}
       for p in ("score", "open", "twostage", "fp")},                                                                 # CPU ablations
    "DIRECT_LLM": {"_SYSTEM": "direct_llm"},                                                                # baseline (GPU, 1 call/case)
    "A1rT_fixed_ABS_KP_COV": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_ALL_ENTITIES": "1", "USE_EVENT_DIGEST": "1"},
    "A1rT_fixed_ABS_KP_SEG": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "SPEC_CHAIN_SEGMENTS": "1"},
    "A1rT_fixed_ABS_KP_COV_SEG": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_ALL_ENTITIES": "1", "USE_EVENT_DIGEST": "1", "SPEC_CHAIN_SEGMENTS": "1"},
}
DEV30 = ["t004","t005","t012","t014","t015","t018","t026","t028","t029","t030","t032","t036","t054","t055","t058",
         "t065","t070","t072","t076","t078","t082","t087","t091","t092","t093","t095","t097","t099","t100","t101"]
SUBSET8 = ["t005","t036","t082","t092","t065","t076","t078","t100"]   # node + replica cases, for a quick mechanics check
ALL_IDS = [l.strip() for l in open(os.path.join(DATA, "manifest.txt")) if l.strip()]
HELDOUT = [c for c in ALL_IDS if c not in DEV30]

# %% Cell 4 -- settings
import torch
ARM = "A1rT_fixed_ABS_KP"
MODEL = "qwen25_7b"           # a key of MODELS (Cell 3)
CASES = DEV30                 # DEV30 | SUBSET8 | HELDOUT | ALL_IDS
SHARD = None                  # None, or (k, n): run every n-th case starting at k (0-based), e.g. (0, 3) for notebook 1 of 3
TAG = {"qwen25_7b": "7B"}.get(MODEL, MODEL)   # "7B" keeps the file names of earlier runs
if SHARD:
    CASES = [c for i, c in enumerate(CASES) if i % SHARD[1] == SHARD[0]]
print(f"{ARM}: {len(CASES)} cases | GPUs free GB:", [round(torch.cuda.mem_get_info(i)[0] / 1e9, 1) for i in range(torch.cuda.device_count())])

# %% Cell 5 -- run (background process, log to file; re-run the cell to resume)
import sys
MODEL = globals().get("MODEL", "qwen25_7b")    # notebooks written before the MODEL setting keep the 7B base
if MODELS[MODEL].get("KAGGLE_USE_4BIT") == "1":
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "bitsandbytes"])
name = f"{ARM}_{TAG}" + ("" if MODEL == "qwen25_7b" or MODEL in TAG else f"_{MODEL}") + (f"_shard{SHARD[0]}of{SHARD[1]}" if SHARD else "")
clean = {k: v for k, v in os.environ.items() if k not in FLAGS}
SYSTEM = ARMS[ARM].get("_SYSTEM", "proposed_hybrid")
env = {**clean, **BASE, **MODELS[MODEL], **{k: v for k, v in ARMS[ARM].items() if not k.startswith("_")}, **FIXED,
       "DEBUG_DUMP_DIR": f"/kaggle/working/dump_{name}",
       "SAVE_TYPING_STATE_DIR": f"/kaggle/working/state_{name}"}   # inputs of the typing step, for scripts/retype.py (no effect on results)
out = f"/kaggle/working/{name}.csv"
open(f"/kaggle/working/{name}.commit.txt", "w").write(COMMIT + "\n")
cmd = ["python", "main.py", "--case_ids", *CASES, "--systems", SYSTEM, "--save_path", out]   # resume ON
log = open(f"/kaggle/working/{name}.log", "a")
p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
t0 = time.time()
while p.poll() is None:
    try: time.sleep(300)
    except KeyboardInterrupt: p.kill(); print("stopped; re-run to resume"); break
    try:                                  # progress from main.py's per-case lines (dump files exist only for the v1 pipeline)
        _lg = open(f"/kaggle/working/{name}.log", encoding="utf-8", errors="ignore").read()
        _done = _lg.count("entity_loc=") + _lg.count("[error]")
        _last = [l for l in _lg.splitlines() if l.strip()][-1:]
    except Exception:
        _done, _last = 0, []
    print(f"{(time.time() - t0) / 60:.0f} min, {_done}/{len(CASES)} cases done | {(_last or [''])[0][-160:]}", flush=True)
print("rc =", p.poll()); print(subprocess.run(f"tail -c 1500 /kaggle/working/{name}.log", shell=True, capture_output=True, text=True).stdout)

# %% Cell 6 -- health + sanity (dev-30 is for debugging only; this is not a verdict)
import pandas as pd
d = pd.read_csv(out)
cols = [c for c in ("case_id", "error", "coordinator_prompt_chars", "truncated_prompts", "oom_retries", "total_calls", "entity_localization",
                    "predicted_fault_type", "gt_fault_type", "fault_identification_exact", "fault_group_identification", "final_score",
                    "cause_entity_mismatch", "fp_candidates", "fp_choice_in_shortlist", "verify_grounded_before", "verify_repaired",
                    "total_pipeline_time_s") if c in d.columns]
print(d[cols].to_string(index=False))
print(d[[c for c in ("entity_localization", "fault_identification_exact", "fault_group_identification", "reasoning_process", "final_score",
                     "cause_entity_mismatch", "fp_candidates", "fp_choice_in_shortlist", "verify_grounded_before", "verify_repaired",
                    "total_pipeline_time_s") if c in d.columns]].mean(numeric_only=True))
print(d.predicted_fault_type.value_counts().to_string())

# %% Cell 7 -- retype: FP arms from the saved state of the reference arm (no Coordinator re-run; see scripts/retype.py)
# Does nothing unless RETYPE_STEP is set in the notebook ("prepare" | "run"), so notebooks that exec "Cell 6 to the end" are unaffected.
# Run the reference arm A1rT_fixed_ABS_KP first (Cells 4-6). Then, with CASES / TAG / SHARD / MODEL of THAT run in Cell 4:
#   RETYPE_STEP = "prepare" (CPU is enough: signal databases, no model), then RETYPE_STEP = "run" for the arms in RETYPE_ARMS
#   (rule arms need no GPU; show / shortlist arms load the model once and make one call per case).
RETYPE_STEP = globals().get("RETYPE_STEP")
RETYPE_ARMS = globals().get("RETYPE_ARMS") or ["A1rT_fixed_ABS_KP_FPRULE", "A1rT_fixed_ABS_KP_FPRULE_FIX",
                                                "A1rT_fixed_ABS_KP_FPSL", "A1rT_fixed_ABS_KP_FPSHOW"]
if RETYPE_STEP:
    import sys
    MODEL = globals().get("MODEL", "qwen25_7b")
    sfx = ("" if MODEL == "qwen25_7b" or MODEL in TAG else f"_{MODEL}") + (f"_shard{SHARD[0]}of{SHARD[1]}" if SHARD else "")
    ref_name = f"A1rT_fixed_ABS_KP_{TAG}{sfx}"
    REF, STATE = f"/kaggle/working/{ref_name}.csv", f"/kaggle/working/state_{ref_name}"
    renv = {**{k: v for k, v in os.environ.items() if k not in FLAGS}, **BASE, **MODELS[MODEL], **FIXED}
    if RETYPE_STEP == "prepare":
        r = subprocess.run(["python", "scripts/retype.py", "prepare", "--ref", REF, "--state", STATE], env=renv, capture_output=True, text=True)
        print("prepare rc =", r.returncode, (r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
    else:
        for a in RETYPE_ARMS:
            out = f"/kaggle/working/{a}_{TAG}_retyped{sfx}.csv"
            if os.path.exists(out):
                print("exists, skipped:", out)
                continue
            r = subprocess.run(["python", "scripts/retype.py", "run", "--ref", REF, "--state", STATE, "--arm", a, "--out", out],
                               env=renv, capture_output=True, text=True)
            print(a, "rc =", r.returncode, (r.stdout or "")[-600:], (r.stderr or "")[-1500:])
