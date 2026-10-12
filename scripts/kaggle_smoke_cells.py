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
BASE = dict(                       # arm A1rT: the proposed configuration BEFORE the new evidence (pre-registered base)
    LLM_BACKEND="kaggle", CHUNKED_PREFILL="1", MAX_INPUT_TOKENS="16000", PREFILL_CHUNK="1024",
    USE_LOG_TEMPLATES="1", LOG_TEMPLATE_ENGINE="drain3", DRAIN_SIM_TH="0.8",
    USE_PROPAGATION_EVIDENCE="1", ENTITY_SOURCE="coordinator", CHAIN_FIRST_ENTITY="coordinator",   # 2026-10-12 (U2): the Coordinator sees all evidence; el_if_anchor is still logged
    USE_KEYWORD_HINT="0", USE_INFRA_EVIDENCE="1", CHAIN_FIRST="1",
    USE_CHANGE_FACTOR="1", USE_CHANGE_STATS="1", USE_TRACE_EVIDENCE="1",
    PYTHONHASHSEED="0", RCA100_ROOT=DATA)
NEW = dict(ALERT_T0_MODE="window", USE_ALERT_LIFECYCLE="1", USE_DIRECTION_EVIDENCE="1", USE_EVENT_DIGEST="1",
           USE_LOG_EVIDENCE="1", USE_TRACE_EDGES="1", USE_METRIC_STRUCTURE="1")

FLAGS = ("ALERT_T0_MODE", "USE_ALERT_LIFECYCLE", "USE_DIRECTION_EVIDENCE", "USE_EVENT_DIGEST", "USE_LOG_EVIDENCE", "USE_TRACE_EDGES",
         "USE_METRIC_STRUCTURE", "USE_INTERVAL_STRIP", "SYMPTOM_FIRST", "CHAIN_STEPS_FLEX", "GROUNDED_FAULT", "VERIFY_FAULT_TYPE", "TOPOLOGY_NAMES", "STRICT_MODALITY_ROUTING", "SPECIALIST_ENGLISH", "USE_COLOCATION_EVIDENCE", "LAYER_AGENT", "USE_KNOWLEDGE_PROMPT",
         "VERIFY_CHAIN", "USE_ONSET_EVIDENCE", "USE_DOSSIER_FT", "USE_LLM_SPECIALIST_AGENTS")

def run_arm(name, extra):
    clean = {k: v for k, v in os.environ.items() if k not in FLAGS}          # no flag can come from the notebook process
    env = {**clean, **BASE, **extra, "DEBUG_DUMP_DIR": f"/kaggle/working/dump_{name}"}
    out = f"/kaggle/working/smoke_{name}.csv"
    cmd = ["python", "main.py", "--case_ids", *SMOKE, "--systems", "proposed_hybrid", "--save_path", out, "--no-resume"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(name, "rc =", r.returncode); print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return out

csv_a = run_arm("A1rT", {})                                         # reference: legacy T0, no new evidence, legacy routing
FIXED = {"STRICT_MODALITY_ROUTING": "1", "SPECIALIST_ENGLISH": "1"}   # A1rT-fixed = base for the registered arms (U1 + U4)
csv_s = run_arm("A1rT_fixed", FIXED)                                  # reference + routing fix + English specialists
csv_x = run_arm("A1rT_fixed_X", {"ALERT_T0_MODE": "window", "USE_INTERVAL_STRIP": "1", **FIXED})   # arm X alone on the fixed base (window T0)
csv_k = run_arm("A1rT_fixed_K", {"USE_KNOWLEDGE_PROMPT": "1", **FIXED})          # arm K alone (type-discrimination knowledge block, code existed)
csv_sy = run_arm("A1rT_fixed_S", {"SYMPTOM_FIRST": "1", **FIXED})                 # arm S alone (symptom-first JSON)
csv_ks = run_arm("A1rT_fixed_KS", {"USE_KNOWLEDGE_PROMPT": "1", "SYMPTOM_FIRST": "1", **FIXED})
csv_tn = run_arm("A1rT_fixed_TN", {"TOPOLOGY_NAMES": "1", **FIXED})                # arm T-names: readable, order-neutral topology input
csv_b = run_arm("A1rT_fixed_all_new", {**NEW, **FIXED})   # every new flag on at once (smoke only; real arms are added one group at a time)

# %% Cell 5 -- health check (NOT a comparison of scores)
import pandas as pd
for name, p in (("A1rT", csv_a), ("A1rT_fixed", csv_s), ("A1rT_fixed_X", csv_x), ("A1rT_fixed_K", csv_k), ("A1rT_fixed_S", csv_sy), ("A1rT_fixed_KS", csv_ks), ("A1rT_fixed_TN", csv_tn), ("A1rT_fixed_all_new", csv_b)):
    d = pd.read_csv(p)
    cols = [c for c in ("case_id", "error", "coordinator_prompt_chars", "truncated_prompts", "oom_retries",
                        "chunked_fallbacks", "total_calls", "use_alert_lifecycle", "use_event_digest", "use_log_evidence",
                        "use_trace_edges", "use_metric_structure", "log_evidence_chars", "trace_edges_chars",
                        "metric_structure_chars", "use_interval_strip", "interval_strip_chars", "interval_strip_error", "log_evidence_error", "trace_edges_error", "metric_structure_error")
            if c in d.columns]
    print("\n==", name); print(d[cols].to_string(index=False))
# Pass criteria: error empty; truncated_prompts = 0; oom_retries = 0; *_error empty; coordinator_prompt_chars of the
# all-new arm is roughly +8k chars over A1rT (if it is far larger, one block is much longer than measured offline).
# Then open dump_A1rT_all_new/<case>*.json to read the real Coordinator prompt and check that the new blocks are present.

# specialist prompt sizes per arm (metrics, logs, trace, topology) and the Coordinator prompt, from the dumps
import json
for name in ("A1rT", "A1rT_fixed", "A1rT_fixed_X", "A1rT_fixed_K", "A1rT_fixed_S", "A1rT_fixed_KS", "A1rT_fixed_TN", "A1rT_fixed_all_new"):
    for cid in SMOKE:
        d = json.load(open(f"/kaggle/working/dump_{name}/{cid}.json"))
        print(f"{name:14s} {cid}", [len(t["prompt"]) for t in d["trace"]])
