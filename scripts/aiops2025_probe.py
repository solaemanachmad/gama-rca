"""aiops2025_probe.py -- second label-free look at AIOps2025 (after aiops2025_audit.py): the fields the adapter still needs.

Prints (never opens groundtruth*): input.json (count, first entries, description patterns); for a few trace files the inventory of
span-tag keys, process-tag keys and the values of status / error / kind keys, serviceName -> host / pod tags; span logs keys; and
for logs a sample of error-like messages per pod.

    !python scripts/aiops2025_probe.py --data <folder with input.json> --out /kaggle/working/aiops2025_probe.txt
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

STATUS_KEYS = re.compile(r"status|error|code|kind|exception|level|otel|http\.|rpc\.|net\.peer|hostname|ip|name", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="aiops2025_probe.txt")
    ap.add_argument("--files", type=int, default=3)
    a = ap.parse_args()
    sys.stdout = open(a.out, "w", encoding="utf-8")
    import pyarrow.parquet as pq

    p = os.path.join(a.data, "input.json")
    d = json.load(open(p, encoding="utf-8"))
    print(f"# input.json: {type(d).__name__} of {len(d)}")
    for x in (d[:5] if isinstance(d, list) else list(d.items())[:5]):
        print("  " + json.dumps(x, ensure_ascii=False)[:800])
    if isinstance(d, list) and d and isinstance(d[0], dict):
        print("keys:", collections.Counter(k for x in d for k in x).most_common())
        texts = [str(v) for x in d for v in x.values()]
        print("timestamps per entry (first 5):", [re.findall(r"\d{4}-\d{2}-\d{2}[T ][\d:]+Z?", t) for t in texts[:10]])

    tfiles = sorted(f for f in glob.glob(os.path.join(a.data, "**", "trace-parquet", "trace_*.parquet"), recursive=True)
                    if not os.path.basename(f).startswith("._"))
    step = max(1, len(tfiles) // a.files)
    tag_keys, proc_keys, svc = collections.Counter(), collections.Counter(), collections.Counter()
    vals = collections.defaultdict(collections.Counter)
    proc_vals = collections.defaultdict(collections.Counter)
    log_keys = collections.Counter()
    refs = collections.Counter()
    for f in tfiles[::step][:a.files]:
        t = pq.read_table(f, columns=["tags", "process", "logs", "references", "operationName"]).slice(0, 200000).to_pylist()
        for r in t:
            for tg in r["tags"] or []:
                tag_keys[tg["key"]] += 1
                if STATUS_KEYS.search(tg["key"]) and len(vals[tg["key"]]) < 60:
                    vals[tg["key"]][str(tg["value"])[:80]] += 1
                elif STATUS_KEYS.search(tg["key"]):
                    vals[tg["key"]][str(tg["value"])[:80]] += 1 if str(tg["value"])[:80] in vals[tg["key"]] else 0
            pr = r["process"] or {}
            svc[pr.get("serviceName")] += 1
            for tg in pr.get("tags") or []:
                proc_keys[tg["key"]] += 1
                if len(proc_vals[tg["key"]]) < 40:
                    proc_vals[tg["key"]][str(tg["value"])[:60]] += 1
            for lg in r["logs"] or []:
                for fd in lg.get("fields") or []:
                    log_keys[fd["key"]] += 1
                    if STATUS_KEYS.search(fd["key"]) and len(vals["log." + fd["key"]]) < 40:
                        vals["log." + fd["key"]][str(fd["value"])[:120]] += 1
            refs[tuple(sorted({x["refType"] for x in r["references"] or []}))] += 1
        print(f"\n(read {f})")
    print("\n# trace span-tag keys:", tag_keys.most_common(80))
    print("\n# trace process serviceName:", svc.most_common())
    print("\n# trace process-tag keys:", proc_keys.most_common(40))
    for k, c in proc_vals.items():
        print(f"  process.{k}: {c.most_common(40)}")
    print("\n# trace span-log field keys:", log_keys.most_common(40))
    print("\n# reference types:", refs.most_common())
    print("\n# values of status / error / kind / peer keys")
    for k, c in sorted(vals.items()):
        print(f"  {k}: {c.most_common(30)}")

    lfiles = sorted(f for f in glob.glob(os.path.join(a.data, "**", "log-parquet", "log_*.parquet"), recursive=True)
                    if not os.path.basename(f).startswith("._"))
    err = re.compile(r"error|exception|fail|refused|timeout|timed out|unavailable|dns|no such host|deadline|oom|gc", re.I)
    per = collections.defaultdict(collections.Counter)
    for f in lfiles[::max(1, len(lfiles) // a.files)][:a.files]:
        t = pq.read_table(f, columns=["k8_pod", "message"]).to_pylist()
        for r in t:
            m = r["message"] or ""
            if err.search(m):
                per[r["k8_pod"]][re.sub(r"[0-9a-f]{8,}|\d+", "#", m)[:200]] += 1
    print("\n# error-like log messages (masked), top per pod")
    for pod, c in sorted(per.items()):
        print(f"  {pod}:")
        for msg, n in c.most_common(6):
            print(f"    {n:6d}  {msg}")


if __name__ == "__main__":
    main()
