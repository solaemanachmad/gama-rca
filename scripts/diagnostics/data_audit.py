"""
data_audit.py -- LABEL-FREE audit of what RCA100 contains versus what the v3 evidence layer reads (input to the v4 design).
Never reads answer_key/. CPU only, pandas + json.

Reports, aggregated over the cases:
  A. topology: entity types, relation types (src_type -relation-> dst_type), names that exist under several entity types
     (the same logical entity in APM / K8s / cloud views), and which types the v3 answer mapper (_ids) can emit.
  B. metrics: per (domain, entity_set) rows, cases, entity names, metric names; whether entity_id is filled and which topology
     type it resolves to; how many rows the v3 signal database READS (k8s.node by exact name, else a name owned by an apm service)
     and how many it SKIPS; metric names that match no v3 metric family.
  C. node naming: k8s.node names in topology vs metrics, and entity sets / topology types that look like hosts or cloud instances.
  D. JVM / memory: metric names in JVM-like sets and memory/heap-like names, and the v3 family each maps to.
  E. alerts / task: alert entity types; events: involvedObject kinds.

Usage (RCA100_ROOT set):  python scripts/diagnostics/data_audit.py [--out audit.json] [--max-cases N] [t001 ...]
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

import config
from pipeline.fault_fingerprint import _owner, _METRIC_FAMILIES

HOSTLIKE = re.compile(r"node|host|ecs|instance|vm|machine|cloud|acs", re.I)
JVMLIKE = re.compile(r"jvm|java|heap|gc", re.I)
MEMLIKE = re.compile(r"mem|heap|rss|working_set", re.I)


def family(metric):
    m = str(metric).lower()
    for f, rx in _METRIC_FAMILIES:
        if rx.search(m):
            return f
    return None


def ids():
    return [l.strip() for l in open(os.path.join(config.DATASET_ROOT, "manifest.txt")) if l.strip()]


def main(argv):
    out_path, maxc = None, None
    if "--out" in argv:
        i = argv.index("--out"); out_path = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if "--max-cases" in argv:
        i = argv.index("--max-cases"); maxc = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    cases = [a for a in argv if not a.startswith("--")] or ids()
    if maxc:
        cases = cases[:maxc]

    ent_types, rel_types = Counter(), Counter()
    multi_name = Counter()                      # (type set) -> cases
    mapper_types = Counter()                    # type the v3 mapper would emit for a service / node name
    ms = defaultdict(lambda: {"rows": 0, "cases": set(), "names": set(), "metrics": Counter(), "eid_filled": 0,
                              "eid_types": Counter(), "read": 0, "skip": 0})
    unmapped = Counter()
    jvm_metrics, mem_metrics = Counter(), Counter()
    node_topo, node_metric = Counter(), Counter()
    hostlike_types = Counter()
    alert_types, event_kinds = Counter(), Counter()
    errors = []

    for k, cid in enumerate(cases, 1):
        d = os.path.join(config.CASES_DIR, cid)
        try:
            topo = json.load(open(os.path.join(d, config.FILE_TOPOLOGY), encoding="utf-8"))
            ents = topo.get("entities", [])
            by_id = {e.get("id"): e for e in ents}
            names_by = defaultdict(set)
            for e in ents:
                t = e.get("type", "?")
                ent_types[t] += 1
                names_by[e.get("name")].add(t)
                if HOSTLIKE.search(t):
                    hostlike_types[t] += 1
            for n, ts in names_by.items():
                if len(ts) > 1:
                    multi_name[" + ".join(sorted(ts))] += 1
            for e in topo.get("edges", []):
                st = e.get("src_type") or by_id.get(e.get("src"), {}).get("type", "?")
                dt_ = e.get("dst_type") or by_id.get(e.get("dst"), {}).get("type", "?")
                rel_types[f"{st} -{e.get('relation', '?')}-> {dt_}"] += 1
            services = sorted({e.get("name") for e in ents if e.get("type") in ("apm.service", "apm.svc")} - {None})
            nodes = {e.get("name") for e in ents if e.get("type") == "k8s.node"}
            for nme in nodes:
                node_topo[nme] += 1
            first = {}
            for e in ents:                                           # the v3 mapper: first entity of these types per name
                if e.get("type") in ("apm.service", "apm.svc", "k8s.node"):
                    first.setdefault(e.get("name"), e.get("type"))
            for t in first.values():
                mapper_types[t] += 1

            cols = pd.read_parquet(os.path.join(d, config.FILE_METRICS), columns=None).columns
            use = [c for c in ("entity_id", "entity_name", "service", "domain", "entity_set", "metric") if c in cols]
            df = pd.read_parquet(os.path.join(d, config.FILE_METRICS), columns=use)
            for c in use:
                df[c] = df[c].astype(str)
            g = df.groupby(use, dropna=False).size().reset_index(name="n")
            for r in g.itertuples(index=False):
                r = r._asdict()
                key = f"{r.get('domain', '')}|{r.get('entity_set', '')}"
                s = ms[key]
                s["rows"] += r["n"]; s["cases"].add(cid)
                ename = r.get("entity_name") if r.get("entity_name") not in ("", "None", "nan") else r.get("service")
                s["names"].add(ename); s["metrics"][r["metric"]] += r["n"]
                eid = r.get("entity_id")
                if eid and eid not in ("", "None", "nan"):
                    s["eid_filled"] += r["n"]
                    s["eid_types"][by_id.get(eid, {}).get("type", "not-in-topology")] += r["n"]
                read = (r.get("entity_set") == "k8s.node" and ename in nodes) or \
                       (r.get("entity_set") != "k8s.node" and _owner(ename, services) is not None)
                s["read" if read else "skip"] += r["n"]
                if family(r["metric"]) is None:
                    unmapped[f"{key}::{r['metric']}"] += r["n"]
                if JVMLIKE.search(r.get("entity_set", "")) or JVMLIKE.search(r["metric"]):
                    jvm_metrics[f"{r['metric']} -> {family(r['metric'])}"] += 1
                if MEMLIKE.search(r["metric"]):
                    mem_metrics[f"{key}::{r['metric']} -> {family(r['metric'])}"] += 1
                if r.get("entity_set") == "k8s.node":
                    node_metric[ename] += 1

            try:
                task = json.load(open(os.path.join(d, config.FILE_TASK), encoding="utf-8"))
                ae = task.get("alert_entity") or {}
                txt = json.dumps(task)
                m = re.search(r'entity_type=\\?"([^"\\]+)', txt)
                alert_types[ae.get("entity_type") or (m.group(1) if m else "none")] += 1
            except Exception:
                alert_types["unreadable"] += 1
            try:
                ev = pd.read_parquet(os.path.join(d, config.FILE_EVENTS), columns=["eventId"])
                for v in ev["eventId"]:
                    try:
                        o = json.loads(v) if isinstance(v, str) else (v or {})
                        event_kinds[(o.get("involvedObject") or {}).get("kind", "?")] += 1
                    except Exception:
                        event_kinds["unparsed"] += 1
            except Exception:
                pass
        except Exception as e:
            errors.append((cid, repr(e)[:200]))
        print(f"[{k}/{len(cases)}] {cid}", flush=True)

    P = lambda *a: print(*a, flush=True)
    P("\n===== A. TOPOLOGY =====")
    P("entity types (entities summed over cases):"); [P(f"  {t}: {n}") for t, n in ent_types.most_common()]
    P("relations (src_type -relation-> dst_type):"); [P(f"  {t}: {n}") for t, n in rel_types.most_common(40)]
    P("names present under several entity types (type set: count over cases):"); [P(f"  {t}: {n}") for t, n in multi_name.most_common(20)]
    P("types the v3 answer mapper emits:", dict(mapper_types))
    P("\n===== B. METRICS by domain|entity_set =====")
    tot_read = sum(s["read"] for s in ms.values()); tot = sum(s["rows"] for s in ms.values())
    P(f"rows read by v3: {tot_read}/{tot} = {tot_read / max(tot, 1):.1%}")
    for key, s in sorted(ms.items(), key=lambda kv: -kv[1]["rows"]):
        P(f"- {key}: rows {s['rows']}, cases {len(s['cases'])}, names {len(s['names'])}, metrics {len(s['metrics'])}, "
          f"READ {s['read'] / max(s['rows'], 1):.0%}, entity_id filled {s['eid_filled'] / max(s['rows'], 1):.0%} -> {dict(s['eid_types'].most_common(4))}")
        P(f"    names e.g. {sorted(map(str, s['names']))[:8]}")
        P(f"    metrics: {[m for m, _ in s['metrics'].most_common(25)]}")
    P("\nmetric names matching NO v3 family (rows):"); [P(f"  {m}: {n}") for m, n in unmapped.most_common(40)]
    P("\n===== C. NODES =====")
    P(f"k8s.node names in topology: {len(node_topo)} distinct; in k8s.node metrics: {len(node_metric)} distinct; "
      f"in both: {len(set(node_topo) & set(node_metric))}")
    P(f"  topology e.g. {sorted(node_topo)[:6]}"); P(f"  metrics e.g. {sorted(map(str, node_metric))[:6]}")
    P("host/cloud-like topology entity types:", dict(hostlike_types))
    P("\n===== D. JVM / MEMORY =====")
    P("JVM-like metric names -> v3 family:"); [P(f"  {m}") for m, _ in jvm_metrics.most_common(40)]
    P("memory-like metric names -> v3 family:"); [P(f"  {m}") for m, _ in mem_metrics.most_common(40)]
    P("\n===== E. ALERTS / EVENTS =====")
    P("alert entity types:", dict(alert_types)); P("event involvedObject kinds:", dict(event_kinds))
    if errors:
        P("\nERRORS:", errors[:10])
    if out_path:
        dump = {"entity_types": ent_types, "relations": rel_types, "multi_name": multi_name, "mapper_types": mapper_types,
                "metrics": {k: {**{kk: vv for kk, vv in v.items() if kk not in ("cases", "names", "metrics", "eid_types")},
                                "cases": len(v["cases"]), "names": sorted(map(str, v["names"]))[:200],
                                "metrics": dict(v["metrics"]), "eid_types": dict(v["eid_types"])} for k, v in ms.items()},
                "unmapped": dict(unmapped), "jvm": dict(jvm_metrics), "mem": dict(mem_metrics), "node_topo": node_topo,
                "node_metric": node_metric, "hostlike_types": hostlike_types, "alert_types": alert_types,
                "event_kinds": event_kinds, "errors": errors}
        json.dump(dump, open(out_path, "w"), indent=1, default=list)
        P("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
