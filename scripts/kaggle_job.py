"""kaggle_job.py -- one command per run on Kaggle. The notebook never changes; only the arguments do.

Notebook (2 cells, the same for every run):

  # Cell 1 -- code (Secret GITHUB_TOKEN; dataset RCA100 attached; Internet ON)
  import os, subprocess
  from kaggle_secrets import UserSecretsClient
  tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
  subprocess.run("rm -rf /kaggle/working/gama-rca", shell=True)
  subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://{tok}@github.com/solaemanachmad/gama-rca.git", "/kaggle/working/gama-rca"])
  os.chdir("/kaggle/working/gama-rca")

  # Cell 2 -- run (change only the arguments)
  !python scripts/kaggle_job.py --arm V4_RULE_MM_T3_OMSG_FI_DEPB --cases odd --shard 0/4

What it does: installs the dependencies, runs main.py for the arm (resume ON: re-running continues), streams the per-case lines,
prints the scores, and writes /kaggle/working/<name>_all.zip (csv, log, trace folder, commit) to download.
Arms, model presets and case lists are the ones of scripts/kaggle_run.py (Cell 3), read from there: one definition only.

  --arm     a key of ARMS (python scripts/kaggle_job.py --list shows them)
  --cases   odd | even | all | dev30 | t001,t003,...      (odd = development half, even = validation half: run once at the end)
  --shard   k/n   run every n-th case starting at k (0-based), e.g. 0/4 .. 3/4 on four accounts
  --model   a key of MODELS (default qwen25_7b; only GPU arms use it)
  --wandb   log to W&B (Secret WANDB_API_KEY; keep the project private)
  --data    folder containing cases/ and manifest.txt
  --dry     print what would run, run nothing
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_DATA = "/kaggle/input/datasets/solaemanachmad/rca100v1-1/RCA100v1.1"
OUT = "/kaggle/working"


def definitions(data):
    """Executes Cell 3 of kaggle_run.py (definitions only, no model) and returns its names."""
    src = open(os.path.join(HERE, "kaggle_run.py"), encoding="utf-8").read()
    block = src.split("# %% Cell 3 ")[1].split("# %% Cell ")[0].split("\n", 1)[1]
    ns = {"os": os, "subprocess": subprocess, "DATA": data}
    exec(block, ns)
    return ns


def install(model_env):
    pkgs = ["sentence-transformers", "networkx", "drain3", "faiss-cpu", "langgraph", "wandb"]
    if model_env.get("KAGGLE_USE_4BIT") == "1":
        pkgs.append("bitsandbytes")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", *pkgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--cases", default="odd")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--model", default="qwen25_7b")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no_install", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    assert os.path.isdir(os.path.join(a.data, "cases")), f"dataset not found: {a.data} (attach RCA100 or pass --data)"
    D = definitions(a.data)
    if a.list:
        print("ARMS:", ", ".join(D["ARMS"]))
        print("MODELS:", ", ".join(D["MODELS"]))
        return
    assert a.arm in D["ARMS"], f"unknown arm {a.arm!r}; --list shows the arms"
    assert a.model in D["MODELS"], f"unknown model {a.model!r}; --list shows the models"

    named = {"odd": D["DEV_ODD"], "even": D["VAL_EVEN"], "all": D["ALL_IDS"], "dev30": D["DEV30"]}
    cases = named.get(a.cases) or [c.strip() for c in a.cases.split(",") if c.strip()]
    shard = None
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= k < n, "--shard k/n needs 0 <= k < n"
        shard = (k, n)
        cases = [c for i, c in enumerate(cases) if i % n == k]

    commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True, cwd=REPO).stdout.strip()
    tag = {"qwen25_7b": "7B"}.get(a.model, a.model)
    name = f"{a.arm}_{tag}" + ("" if a.model == "qwen25_7b" or a.model in tag else f"_{a.model}") + \
           (f"_shard{shard[0]}of{shard[1]}" if shard else "")
    arm = D["ARMS"][a.arm]
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
    print("V4_PARTS:", env.get("V4_PARTS", "-"))
    print("cases:", " ".join(cases))
    if a.dry:
        return
    if not a.no_install:
        install(D["MODELS"][a.model])

    open(f"{a.out}/{name}.commit.txt", "w").write(commit + "\n")
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
    if rc != 0:
        print(open(f"{a.out}/{name}.log", "rb").read()[-3000:].decode("utf-8", errors="replace"))

    if os.path.exists(out_csv):
        import pandas as pd
        d = pd.read_csv(out_csv)
        cols = [c for c in ("entity_localization", "fault_identification_exact", "fault_group_identification", "reasoning_process",
                            "final_score", "total_pipeline_time_s") if c in d.columns]
        print(f"\n{len(d)} rows")
        print(d[[c for c in ("case_id", "entity_localization", "predicted_fault_type", "gt_fault_type", "final_score")
                 if c in d.columns]].to_string(index=False))
        print(d[cols].mean(numeric_only=True).round(4).to_string())
    z = f"{a.out}/{name}_all.zip"
    subprocess.run(f"cd {a.out} && rm -f {name}_all.zip && zip -qr {name}_all.zip {name}.csv {name}.log {name}.commit.txt "
                   f"{name}_trace 2>/dev/null", shell=True)
    print("\ndownload:", z if os.path.exists(z) else "(zip failed)")


if __name__ == "__main__":
    main()
