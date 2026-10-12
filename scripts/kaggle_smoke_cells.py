# Kaggle smoke test (2026-10-11): paste each "# %% Cell N" block into its own notebook cell, in order.
# Purpose: DEBUGGING ONLY (does the code run, are the prompts sane). Three cases are not a basis for choosing flags.
# v3 (2026-10-12, after smoke v2): base entity = Coordinator (U2), arms A1rT / A1rT_fixed (strict routing + English specialists) / A1rT_fixed_all_new.
# v2 (2026-10-12): Cell 3 no longer leaks flags into Cell 4 (the v1 reference arm was contaminated), Cell 4 builds a CLEAN env
# per arm, STRICT_MODALITY_ROUTING arm added, Cell 5 also prints specialist prompt sizes.
# Fill the two <...> placeholders in Cell 1. Nothing here reads the answer key except the scoring that main.py already does.

# %% Cell 1 -- code (private repo) and data
import os, subprocess

REPO = "<GITHUB_USER>/<REPO>"                       # e.g. "achmad/gama-rca"  (private)
DATA = "/kaggle/input/<DATASET>/RCA100v1.1"          # folder that contains cases/ and manifest.txt
WORK = "/kaggle/working/gama-rca"

from kaggle_secrets import UserSecretsClient          # Add-ons > Secrets: GH_TOKEN (read-only token for the private repo)
tok = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(["rm", "-rf", WORK])
r = subprocess.run(["git", "clone", "--depth", "60", f"https://{tok}@github.com/{REPO}.git", WORK],
                   capture_output=True, text=True)
print("clone rc =", r.returncode, (r.stderr or "").replace(tok, "***")[-300:])
os.chdir(WORK)
print(subprocess.run("git log --oneline | head -12", shell=True, capture_output=True, text=True).stdout)
assert os.path.isdir(os.path.join(DATA, "cases")), f"dataset path wrong: {DATA}"
os.environ["RCA100_ROOT"] = DATA
os.environ["PYTHONHASHSEED"] = "0"

# %% Cell 2 -- dependencies (skip what the notebook image already has)
# !pip -q install drain3 faiss-cpu sentence-transformers networkx
import importlib
for m in ("drain3", "faiss", "sentence_transformers", "networkx", "torch", "transformers"):
    try:
        importlib.import_module(m); print("ok  ", m)
    except Exception as e:
        print("MISSING", m, "->", e)

# %% Cell 3 -- evidence blocks only (NO LLM): do all new operators run and how long are they?
import sys, time
sys.path.insert(0, WORK)
_ENV_BEFORE = dict(os.environ)          # restored at the end of this cell so nothing leaks into Cell 4
os.environ.update(ALERT_T0_MODE="window", USE_CHANGE_FACTOR="1", USE_CHANGE_STATS="1",
                  USE_DIRECTION_EVIDENCE="1", USE_EVENT_DIGEST="1")
import config
from data.loader import Case
from pipeline.infra_evidence import compute_infra_evidence
from pipeline.alert_lifecycle import compute_alert_lifecycle
from pipeline.log_evidence import compute_log_evidence
from pipeline.trace_edges import compute_trace_edges
from pipeline.metric_structure import compute_metric_structure
from pipeline.interval_strip import compute_interval_strip

SMOKE = ["t004", "t018", "t054"]
blocks = {}
for cid in SMOKE:
    k = Case(cid)
    at, ent = k.alert.alert_timestamp, k.alert.entry_entity_id
    t = time.time()
    out = {"L alert lifecycle": compute_alert_lifecycle(k),
           "infra (D2 + E)": compute_infra_evidence(k, at, ent, 2),
           "G log evidence": compute_log_evidence(k, at, ent),
           "H trace edges": compute_trace_edges(k, at, ent),
           "M metric structure": compute_metric_structure(k, at, ent),
           "X interval strip": compute_interval_strip(k, at, ent)}
    blocks[cid] = out
    print(f"\n=== {cid}  T0={at}  ({time.time() - t:.1f}s)")
    tot = 0
    for name, r in out.items():
        n = len(r.get("text", "")); tot += n
        print(f"  {name:22s} {n:6d} chars   error={r.get('error')}")
    print(f"  {'TOTAL new+infra':22s} {tot:6d} chars")
    assert all(not r.get("error") for r in out.values()), "an operator raised an error"

# look at one block of one case (change the keys to inspect others)
print(blocks["t004"]["H trace edges"]["text"][:2500])
print(blocks["t018"]["X interval strip"]["text"][:4000])
os.environ.clear(); os.environ.update(_ENV_BEFORE)   # undo the flags set above

# %% Cell 4 -- full pipeline on 3 cases, two arms (LLM; ~3 x 3 min per arm on a T4/P100)
import json
# Model source: local copy of Qwen2.5-7B-Instruct (Add Input > Models > qwen-lm / qwen2.5 / transformers / 7b-instruct / 1); None = HF hub download.
LOCAL_MODEL = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
if LOCAL_MODEL:
    print("local model dir exists:", os.path.isdir(LOCAL_MODEL), os.listdir(LOCAL_MODEL)[:8] if os.path.isdir(LOCAL_MODEL) else "")
BASE = dict(                       # arm A1rT: the proposed configuration BEFORE the new evidence (pre-registered base)
    LLM_BACKEND="kaggle", CHUNKED_PREFILL="1", MAX_INPUT_TOKENS="16000", PREFILL_CHUNK="1024",
    USE_LOG_TEMPLATES="1", LOG_TEMPLATE_ENGINE="drain3", DRAIN_SIM_TH="0.8",
    USE_PROPAGATION_EVIDENCE="1", ENTITY_SOURCE="coordinator", CHAIN_FIRST_ENTITY="coordinator",   # 2026-10-12 (U2): the Coordinator sees all evidence; el_if_anchor is still logged
    USE_KEYWORD_HINT="0", USE_INFRA_EVIDENCE="1", CHAIN_FIRST="1",
    USE_CHANGE_FACTOR="1", USE_CHANGE_STATS="1", USE_TRACE_EVIDENCE="1",
    PYTHONHASHSEED="0", RCA100_ROOT=DATA,
    **({"KAGGLE_MODEL_LOCAL_PATH": LOCAL_MODEL} if LOCAL_MODEL else {}))
NEW = dict(ALERT_T0_MODE="window", USE_ALERT_LIFECYCLE="1", USE_DIRECTION_EVIDENCE="1", USE_EVENT_DIGEST="1",
           USE_LOG_EVIDENCE="1", USE_TRACE_EDGES="1", USE_METRIC_STRUCTURE="1")

FLAGS = ("ALERT_T0_MODE", "USE_ALERT_LIFECYCLE", "USE_DIRECTION_EVIDENCE", "USE_EVENT_DIGEST", "USE_LOG_EVIDENCE", "USE_TRACE_EDGES",
         "USE_METRIC_STRUCTURE", "USE_INTERVAL_STRIP", "SYMPTOM_FIRST", "CHAIN_STEPS_FLEX", "CHAIN_ABSOLUTE_SIGNALS", "INFRA_ALL_ENTITIES", "SPEC_CHAIN_SEGMENTS", "GROUNDED_FAULT", "VERIFY_FAULT_TYPE", "TOPOLOGY_NAMES", "STRICT_MODALITY_ROUTING", "SPECIALIST_ENGLISH", "USE_COLOCATION_EVIDENCE", "LAYER_AGENT", "USE_KNOWLEDGE_PROMPT", "USE_KNOWLEDGE_POS",
         "VERIFY_CHAIN", "USE_ONSET_EVIDENCE", "USE_DOSSIER_FT", "USE_LLM_SPECIALIST_AGENTS")

def run_arm(name, extra):
    clean = {k: v for k, v in os.environ.items() if k not in FLAGS}          # no flag can come from the notebook process
    env = {**clean, **BASE, **extra, "DEBUG_DUMP_DIR": f"/kaggle/working/dump_{name}"}
    out = f"/kaggle/working/smoke_{name}.csv"
    cmd = ["python", "main.py", "--case_ids", *SMOKE, "--systems", "proposed_hybrid", "--save_path", out, "--no-resume"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(name, "rc =", r.returncode); print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return out

FIXED = {"STRICT_MODALITY_ROUTING": "1", "SPECIALIST_ENGLISH": "1"}   # A1rT-fixed = base for the registered arms (U1 + U4)
ARMS = {                                                                # name -> extra env (all on the fixed base)
    "A1rT_fixed": {},                                                   # reference (results of the 2026-10-12 run already exist; rerun only if needed)
    "A1rT_fixed_ABS": {"CHAIN_ABSOLUTE_SIGNALS": "1"},                  # absolute request count / avg latency (s) / error count, finest-level targets
    "A1rT_fixed_KP": {"USE_KNOWLEDGE_POS": "1"},                        # knowledge block paraphrased from the public PoS catalog (MIT)
    "A1rT_fixed_ABS_KP": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1"},
    "A1rT_fixed_ABS_KP_ALL": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_ALL_ENTITIES": "1"},   # + every node and deployment listed (compact rows)
    "A1rT_fixed_ABS_KP_COV": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_ALL_ENTITIES": "1", "USE_EVENT_DIGEST": "1"},   # coverage: every node/deployment/JVM row + event digest (no group omitted)
    "A1rT_fixed_ABS_KP_SEG": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "SPEC_CHAIN_SEGMENTS": "1"},   # specialists write chain segments; Coordinator merges
    "A1rT_fixed_ABS_KP_COV_SEG": {"CHAIN_ABSOLUTE_SIGNALS": "1", "USE_KNOWLEDGE_POS": "1", "INFRA_ALL_ENTITIES": "1", "USE_EVENT_DIGEST": "1", "SPEC_CHAIN_SEGMENTS": "1"},
    "A1rT_fixed_C": {"VERIFY_FAULT_TYPE": "1"},                         # short definition check of the chosen type (+ re-choice if rejected)
    "A1rT_fixed_P1": {"GROUNDED_FAULT": "1"},
}
# PAUSED 2026-10-12 (not run in the current rounds; code and flags are kept, this is NOT a verdict on them):
PAUSED = {
    "A1rT": {},
    "A1rT_fixed_X": {"ALERT_T0_MODE": "window", "USE_INTERVAL_STRIP": "1"},
    "A1rT_fixed_K": {"USE_KNOWLEDGE_PROMPT": "1"},
    "A1rT_fixed_S": {"SYMPTOM_FIRST": "1"},
    "A1rT_fixed_KS": {"USE_KNOWLEDGE_PROMPT": "1", "SYMPTOM_FIRST": "1"},
    "A1rT_fixed_TN": {"TOPOLOGY_NAMES": "1"},
    "A1rT_fixed_all_new": {**NEW},
}
ACTIVE = ["A1rT_fixed_ABS", "A1rT_fixed_KP"]            # edit this list to choose what runs (each arm ~10 min on 7B for the 3 cases)
CSV = {n: run_arm(n, {**ARMS[n], **FIXED}) for n in ACTIVE}

# %% Cell 5 -- health check and chains (NOT a comparison of scores)
import pandas as pd, json
for name, p in CSV.items():
    d = pd.read_csv(p)
    cols = [c for c in ("case_id", "error", "coordinator_prompt_chars", "truncated_prompts", "oom_retries", "chunked_fallbacks", "total_calls",
                        "entity_localization", "predicted_fault_type", "gt_fault_type", "fault_identification_exact", "fault_group_identification",
                        "final_score", "cause_target", "cause_entity_mismatch", "typecheck_checked", "typecheck_satisfied", "typecheck_revised") if c in d.columns]
    print("\n==", name); print(d[cols].to_string(index=False))
    for cid in SMOKE:
        fr = json.load(open(f"/kaggle/working/dump_{name}/{cid}.json"))["final_result"]
        print("  ", cid, fr.get("predicted_fault_type"), "|", fr.get("reasoning_chain"))
