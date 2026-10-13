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
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "sentence-transformers", "networkx", "drain3", "faiss-cpu", "langgraph"])
for m in ("drain3", "faiss", "sentence_transformers", "networkx", "langgraph", "torch", "transformers"):
    try:
        importlib.import_module(m); print("ok  ", m)
    except Exception as e:
        print("MISSING", m, "->", e)

# %% Cell 3 -- definitions only (BASE, FLAGS, ARMS, run helpers); runs NO model
import json, time, glob
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
FLAGS = ("ALERT_T0_MODE", "USE_EVENT_DIGEST", "CHAIN_ABSOLUTE_SIGNALS", "INFRA_ALL_ENTITIES", "SPEC_CHAIN_SEGMENTS",
         "STRICT_MODALITY_ROUTING", "SPECIALIST_ENGLISH", "USE_KNOWLEDGE_POS", "USE_LLM_SPECIALIST_AGENTS")
ARMS = {                                                          # name -> extra env, all on the fixed base
    "A1rT_fixed": {},
    "A1rT_fixed_ABS_KP": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1"},          # main configuration
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
CASES = DEV30                 # DEV30 | SUBSET8 | HELDOUT | ALL_IDS
SHARD = None                  # None, or (k, n): run every n-th case starting at k (0-based), e.g. (0, 3) for notebook 1 of 3
TAG = "7B"
if SHARD:
    CASES = [c for i, c in enumerate(CASES) if i % SHARD[1] == SHARD[0]]
print(f"{ARM}: {len(CASES)} cases | GPUs free GB:", [round(torch.cuda.mem_get_info(i)[0] / 1e9, 1) for i in range(torch.cuda.device_count())])

# %% Cell 5 -- run (background process, log to file; re-run the cell to resume)
name = f"{ARM}_{TAG}" + (f"_shard{SHARD[0]}of{SHARD[1]}" if SHARD else "")
clean = {k: v for k, v in os.environ.items() if k not in FLAGS}
env = {**clean, **BASE, **ARMS[ARM], **FIXED, "DEBUG_DUMP_DIR": f"/kaggle/working/dump_{name}"}
out = f"/kaggle/working/{name}.csv"
open(f"/kaggle/working/{name}.commit.txt", "w").write(COMMIT + "\n")
cmd = ["python", "main.py", "--case_ids", *CASES, "--systems", "proposed_hybrid", "--save_path", out]   # resume ON
log = open(f"/kaggle/working/{name}.log", "a")
p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
t0 = time.time()
while p.poll() is None:
    try: time.sleep(300)
    except KeyboardInterrupt: p.kill(); print("stopped; re-run to resume"); break
    print(f"{(time.time() - t0) / 60:.0f} min, {len(glob.glob(f'/kaggle/working/dump_{name}/*.json'))}/{len(CASES)} cases done")
print("rc =", p.poll()); print(subprocess.run(f"tail -c 1500 /kaggle/working/{name}.log", shell=True, capture_output=True, text=True).stdout)

# %% Cell 6 -- health + sanity (dev-30 is for debugging only; this is not a verdict)
import pandas as pd
d = pd.read_csv(out)
cols = [c for c in ("case_id", "error", "coordinator_prompt_chars", "truncated_prompts", "oom_retries", "total_calls", "entity_localization",
                    "predicted_fault_type", "gt_fault_type", "fault_identification_exact", "fault_group_identification", "final_score",
                    "cause_entity_mismatch", "total_pipeline_time_s") if c in d.columns]
print(d[cols].to_string(index=False))
print(d[[c for c in ("entity_localization", "fault_identification_exact", "fault_group_identification", "reasoning_process", "final_score",
                     "cause_entity_mismatch", "total_pipeline_time_s") if c in d.columns]].mean(numeric_only=True))
print(d.predicted_fault_type.value_counts().to_string())
