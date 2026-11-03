"""aiops2025_audit.py -- label-free STRUCTURE audit of the AIOps2025 dataset (Cai et al. 2026), before any adapter is written.

Prints what the GAMA adapter needs to know: folder layout, file counts / sizes, the schema of every kind of parquet file, a few sample
rows, the inventory of metric names and entity names (services, pods, nodes, TiDB components), trace field names, log fields, time
ranges, and the first entries of the case INPUT file (anomaly description + time window = what the method may read).

It never opens a ground-truth file: any path whose name matches GT_PATTERN is listed by name and size only. Run once on Kaggle
(CPU), send the text file back:

    !python scripts/aiops2025_audit.py --data /kaggle/input/<aiops2025 folder> --out /kaggle/working/aiops2025_audit.txt
"""
import argparse
import collections
import json
import os
import re
import sys

GT_PATTERN = re.compile(r"ground|truth|answer|label|gt[_.\-]|_gt\b|solution|key_obs|fault_desc", re.I)
SAMPLE_ROWS, MAX_FILES_PER_KIND, MAX_VALUES = 3, 6, 400


def out(*a):
    print(*a, flush=True)


def kind_of(path):
    """Group parquet files by their path with digits / dates removed (e.g. metric-parquet/apm/pod/pod_*.parquet)."""
    rel = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", path)
    rel = re.sub(r"\d{2}-\d{2}-\d{2}|_\d{2}(?=\.parquet)", "<hh>", rel)
    rel = re.sub(r"\d+", "#", rel)
    return rel


def short(v, n=160):
    s = str(v)
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="aiops2025_audit.txt")
    a = ap.parse_args()
    sys.stdout = open(a.out, "w", encoding="utf-8")
    import pyarrow.parquet as pq

    files, skipped = [], []
    for root, _, names in os.walk(a.data):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, a.data)
            (skipped if GT_PATTERN.search(rel) else files).append(rel)
    out(f"# AIOps2025 structure audit of {a.data}")
    out(f"{len(files)} files read-able, {len(skipped)} ground-truth-like files NOT opened:")
    for s in sorted(skipped)[:50]:
        out(f"  [not opened] {s} ({os.path.getsize(os.path.join(a.data, s))} bytes)")

    out("\n## Layout (directories, file counts, sizes, extensions)")
    by_dir = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for f in files:
        d = os.path.dirname(f)
        by_dir[d][0] += 1
        by_dir[d][1] += os.path.getsize(os.path.join(a.data, f))
        by_dir[d][2][os.path.splitext(f)[1]] += 1
    for d in sorted(by_dir)[:200]:
        n, b, ext = by_dir[d]
        out(f"  {d or '.'}: {n} files, {b / 1e6:.1f} MB, {dict(ext)}")

    out("\n## Non-parquet files (inputs, docs): first lines")
    for f in sorted(f for f in files if not f.endswith(".parquet"))[:30]:
        p = os.path.join(a.data, f)
        out(f"\n### {f} ({os.path.getsize(p)} bytes)")
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                txt = fh.read(4000)
            if f.endswith(".json"):
                try:
                    obj = json.loads(open(p, encoding="utf-8").read())
                    if isinstance(obj, list):
                        out(f"list of {len(obj)}; first 3:")
                        for x in obj[:3]:
                            out("  " + short(json.dumps(x, ensure_ascii=False), 600))
                        continue
                    out("keys: " + short(list(obj)[:30], 600))
                except Exception:
                    pass
            for line in txt.splitlines()[:8]:
                out("  " + short(line, 400))
        except Exception as e:
            out(f"  (unreadable: {e!r})")

    out("\n## Parquet kinds: schema, rows, samples, inventories")
    kinds = collections.defaultdict(list)
    for f in files:
        if f.endswith(".parquet"):
            kinds[kind_of(f)].append(f)
    for k in sorted(kinds):
        fl = sorted(kinds[k])
        out(f"\n### {k}  ({len(fl)} files; e.g. {fl[0]})")
        try:
            pf = pq.ParquetFile(os.path.join(a.data, fl[0]))
            out(f"rows in first file: {pf.metadata.num_rows}")
            out("schema: " + ", ".join(f"{fld.name}:{fld.type}" for fld in pf.schema_arrow))
            head = pf.read_row_group(0).slice(0, SAMPLE_ROWS).to_pylist()
            for r in head:
                out("  row: " + short(json.dumps(r, default=str, ensure_ascii=False), 700))
        except Exception as e:
            out(f"  (unreadable: {e!r})")
            continue
        inv = collections.defaultdict(collections.Counter)
        tmin, tmax = None, None
        for f in fl[:MAX_FILES_PER_KIND]:
            try:
                t = pq.read_table(os.path.join(a.data, f))
            except Exception:
                continue
            for col in t.column_names:
                lc = col.lower()
                if any(w in lc for w in ("name", "metric", "service", "pod", "node", "instance", "kpi", "object", "level", "status",
                                         "code", "kind", "operation", "cmdb", "namespace", "container", "job", "type")):
                    vals = t.column(col).to_pylist()
                    if vals and isinstance(vals[0], (str, int)) and len(set(map(str, vals[:20000]))) <= 5000:
                        inv[col].update(map(str, vals[:200000]))
                if lc in ("time", "timestamp", "@timestamp", "starttime", "start_time", "ts"):
                    try:
                        vs = [v for v in t.column(col).to_pylist() if v is not None]
                        lo, hi = min(vs), max(vs)
                        tmin = lo if tmin is None or str(lo) < str(tmin) else tmin
                        tmax = hi if tmax is None or str(hi) > str(tmax) else tmax
                    except Exception:
                        pass
        if tmin is not None:
            out(f"time range (first {MAX_FILES_PER_KIND} files): {tmin} .. {tmax}")
        for col, cnt in inv.items():
            items = cnt.most_common(MAX_VALUES)
            out(f"  {col}: {len(cnt)} distinct -> " + short(", ".join(f"{v}({c})" for v, c in items), 3000))
    out("\n(end of audit)")


if __name__ == "__main__":
    main()
