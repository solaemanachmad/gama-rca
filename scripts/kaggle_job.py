#!/usr/bin/env python3
"""kaggle_job.py -- one command per run on Kaggle. The notebook never changes; only the arguments do.

Notebook (2 cells, the same for every run):

  # Cell 1 -- code (Secret GITHUB_TOKEN; dataset RCA100 attached; Internet ON)
  import os, subprocess
  from kaggle_secrets import UserSecretsClient
  tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
  subprocess.run("rm -rf /kaggle/working/gama-rca", shell=True)
  subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://{tok}@github.com/solaemanachmad/gamarca.git", "/kaggle/working/gama-rca"])
  os.chdir("/kaggle/working/gama-rca")

  # Cell 2 -- run (change only the arguments)
  !python scripts/kaggle_job.py --arm GAMA --cases all --shard 0/4

What it does: installs the dependencies, runs main.py for the arm (resume ON: re-running continues), streams the per-case lines,
prints the scores, and writes /kaggle/working/<name>_all.zip (csv, log, trace folder, commit) to download.
Arms, model presets and case lists are defined in scripts/kaggle_run.py (Cell 3), read from there: one definition only.
Arms: GAMA (CPU) | GAMA_FULL | BELIEF_LLM | EVIDENCE_LLM_FINAL | DIRECT_LLM (GPU); the names of reported runs (A, GAMA_CORE,
GAMA_LLM, GAMA_CORE_NO_<M>) map to the same code.

  --arm     a key of ARMS (python scripts/kaggle_job.py --list shows them)
  --cases   all (default) | t001,t003,...
  --shard   k/n   run every n-th case starting at k (0-based), e.g. 0/4 .. 3/4 on four accounts
  --model   a key of MODELS (default qwen25_7b; only LLM arms use it; gptoss120b_groq / gptoss120b_cerebras = API, CPU session,
            key from the Kaggle Secret GROQ_API_KEY / CEREBRAS_API_KEY)
  --wandb   log to W&B (Secret WANDB_API_KEY; keep the project private)
  --data    folder containing cases/ and manifest.txt
  --dry     print what would run, run nothing
  --modules GAMA module spec for the arms GAMA / GAMA_FULL (gama/config.py): core | full | core,-trace | ...
"""
import argparse
import os
import subprocess
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOCAL_DATA = os.path.expanduser("~/Developments/RCA100")
DEFAULT_DATA = os.environ.get("RCA100_ROOT", LOCAL_DATA if os.path.isdir(LOCAL_DATA) else "/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1")
OUT = os.environ.get("OUT_DIR", "./results" if not os.path.isdir("/kaggle") else "/kaggle/working")


def definitions(data):
    """Executes Cell 3 of kaggle_run.py (definitions only, no model) and returns its names."""
    src = open(os.path.join(HERE, "kaggle_run.py"), encoding="utf-8").read()
    block = src.split("# %% Cell 3 ")[1].split("# %% Cell ")[0].split("\n", 1)[1]
    ns = {"os": os, "subprocess": subprocess, "DATA": data}
    exec(block, ns)
    return ns


def install(model_env):
    pkgs = ["networkx", "drain3", "wandb"]
    if model_env.get("KAGGLE_USE_4BIT") == "1":
        pkgs.append("bitsandbytes")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *pkgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--model", default="qwen25_7b")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no_install", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--modules", default=None)
    a = ap.parse_args()

    D = definitions(a.data)
    if a.list:
        print("ARMS:", ", ".join(D["ARMS"]))
        print("MODELS:", ", ".join(D["MODELS"]))
        return
    assert os.path.isdir(os.path.join(a.data, "cases")), f"dataset not found: {a.data} (attach RCA100 or pass --data)"
    assert a.arm in D["ARMS"], f"unknown arm {a.arm!r}; --list shows the arms"
    assert a.model in D["MODELS"], f"unknown model {a.model!r}; --list shows the models"

    cases = D["ALL_IDS"] if a.cases == "all" else [c.strip() for c in a.cases.split(",") if c.strip()]
    shard = None
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= k < n, "--shard k/n needs 0 <= k < n"
        shard = (k, n)
        cases = [c for i, c in enumerate(cases) if i % n == k]

    commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True, cwd=REPO).stdout.strip()
    tag = {"qwen25_7b": "7B"}.get(a.model, a.model)
    mods = a.modules.replace(",", "_").replace("-", "no") if a.modules else ""
    name = (f"{a.arm}_{tag}" + (f"_{mods}" if mods else "")
            + ("" if a.model == "qwen25_7b" or a.model in tag else f"_{a.model}")) + (f"_shard{shard[0]}of{shard[1]}" if shard else "")
    arm = dict(D["ARMS"][a.arm])
    if a.modules:
        assert arm.get("_SYSTEM") == "gama", "--modules applies to the GAMA arms"
        arm["GAMA_MODULES"] = a.modules
    system = arm.get("_SYSTEM", "proposed_hybrid")
    env = {**{k: v for k, v in os.environ.items() if k not in D["FLAGS"]}, **D["BASE"], **D["MODELS"][a.model],
           **{k: v for k, v in arm.items() if not k.startswith("_")}, **D["FIXED"],
           "RCA100_ROOT": a.data, "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
           "DEBUG_DUMP_DIR": f"{a.out}/dump_{name}", "SAVE_TYPING_STATE_DIR": f"{a.out}/state_{name}"}
    out_csv = f"{a.out}/{name}.csv"
    cmd = [sys.executable, "main.py", "--case_ids", *cases, "--systems", system, "--save_path", out_csv]
    if a.wandb:
        from kaggle_secrets import UserSecretsClient
        env["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
        cmd += ["--wandb", "--wandb_project", "gama-rca", "--wandb_group", f"{a.arm}_{a.model}"]

    print(f"commit {commit} | arm {a.arm} ({system}) | model {a.model} | {len(cases)} cases | output {name}")
    print("V4_PARTS:", env.get("V4_PARTS", "-"), "| GAMA_MODULES:", env.get("GAMA_MODULES", "-"))
    print("cases:", " ".join(cases))
    if a.dry:
        return
    secret = D["MODELS"][a.model].get("API_KEY_SECRET")
    if secret:                                                # API backbone: key from environment or Kaggle Secret
        key = os.environ.get(secret) or os.environ.get("API_KEY")
        if not key:
          try:
              from kaggle_secrets import UserSecretsClient
              key = UserSecretsClient().get_secret(secret)
          except Exception:
              pass
        assert key, f"API key not found: please set environment variable {secret}=... (or export API_KEY=...)"
        env["API_KEY"] = key
    if not a.no_install and os.path.isdir("/kaggle"):
        install(D["MODELS"][a.model])

    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/{name}.commit.txt", "w").write(commit + "\n")
    manifest = {"commit": commit, "arm": a.arm, "system": system, "modules": env.get("GAMA_MODULES"),
                "v4_parts": env.get("V4_PARTS"), "model": a.model, "model_env": D["MODELS"][a.model], "cases": cases,
                "shard": a.shard, "python": sys.version.split()[0], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:                                                      # hardware and library versions (efficiency table of the paper)
        import platform
        manifest.update(cpu=platform.processor() or platform.machine(), cpu_count=os.cpu_count())
        import torch
        import transformers
        manifest.update(torch=torch.__version__, transformers=transformers.__version__,
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        gpu_count=torch.cuda.device_count())
    except Exception as e:
        manifest["env_error"] = repr(e)
    t0, done = time.time(), 0
    with open(f"{a.out}/{name}.log", "ab") as log:
        p = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in p.stdout:                                  # stream: every line to the log, case lines to the notebook
            log.write(raw)
            log.flush()
            line = raw.decode("utf-8", errors="replace").rstrip()
            if "entity_loc=" in line or "[error]" in line or "Traceback" in line or "Error" in line[:40]:
                done += ("entity_loc=" in line) + ("[error]" in line)
                print(f"[{(time.time() - t0) / 60:5.1f} min {done}/{len(cases)}] {line[-170:]}", flush=True)
        rc = p.wait()
    print("rc =", rc)
    import json
    manifest.update(rc=rc, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.time() - t0, 1))
    json.dump(manifest, open(f"{a.out}/{name}.manifest.json", "w"), indent=1)
    if rc != 0:
        print(open(f"{a.out}/{name}.log", "rb").read()[-3000:].decode("utf-8", errors="replace"))

    if os.path.exists(out_csv):
        import pandas as pd
        d = pd.read_csv(out_csv)
        cols = [c for c in ("entity_loc", "fault_type", "fault_group", "reasoning", "final_score", "duration_s",
                            "entity_localization", "fault_identification_exact", "fault_group_identification",
                            "reasoning_process", "total_pipeline_time_s") if c in d.columns]
        print(f"\n{len(d)} rows")
        show_cols = [c for c in ("case_id", "entity_loc", "pred_entity", "gt_entity", "pred_type", "gt_type",
                                 "final_score", "entity_localization", "predicted_fault_type", "gt_fault_type")
                     if c in d.columns]
        print(d[show_cols].to_string(index=False))
        print(d[cols].mean(numeric_only=True).round(4).to_string())
    z = f"{a.out}/{name}_all.zip"
    subprocess.run(f"cd {a.out} && rm -f {name}_all.zip && zip -qr {name}_all.zip {name}.csv {name}.jsonl {name}.log {name}.commit.txt {name}.manifest.json "
                   f"{name}_trace 2>/dev/null", shell=True)
    print("\ndownload:", z if os.path.exists(z) else "(zip failed)")


if __name__ == "__main__":
    main()
