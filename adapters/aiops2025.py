"""AIOps2025 -> the canonical case of gama/ (PREREGISTRATION_AIOPS2025.md A1-A4, A7, A8). Dataset knowledge only.

Never opens ground truth: every file whose path matches GT_PATTERN is skipped when the release is indexed. input.json (the agent's
input) gives the case window. Run with DATASET_PROFILE=aiops2025 (data/dataset_profile.py), which selects the AIOps2025 metric
families and fingerprints; this module only builds the case.

Canonical case (same interface as data/loader.Case):
  topology   networkx DiGraph; node attr 'entity' (schema.Entity): services, pods and TiDB components as 'apm.service'
             (A4: pods are entities of the service kind), nodes as 'k8s.node'. Edges (schema.Relation): 'calls' service -> service
             (parent / child spans of different services; cartservice -> redis-cart from net.peer.name; productcatalogservice ->
             tidb-tidb -> tidb-tikv / tidb-pd from the published architecture), 'calls' service -> its pods (A4 membership: a pod's
             fault explains its service's symptoms), 'contains' node -> pod (hosting, from infra_pod `instance` and span / log tags).
  memberships {pod: service}
  alert      no alert entity; text = the anomaly description; window [start, end] from it (UTC)
  observations metrics: entity_set / entity_name / metric / value; logs: _container_name_ / content; traces: traceId / spanId /
             parentSpanId / serviceName (= pod) / duration [ns] / statusCode ("2" error, "1" ok) / statusMessage / attributes (JSON) /
             spanName
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from schema import AlertContext, Entity, Observation, Relation

GT_PATTERN = re.compile(r"ground.?truth|answer.?key|label", re.I)
HISTORY = dt.timedelta(hours=5)             # A1: (K_PSEUDO + B_SPAN) window lengths of history and margin
CST = dt.timedelta(hours=8)                 # folder / file names are China Standard Time; the data are UTC
SAMPLE_MOD = 5                              # A7: traces without a failing span: the 1 in 5 selected by a hash of the traceID
DELETED = re.compile(r"\s*\(deleted\)\s*$")
POD_RE = re.compile(r"^(?P<svc>.+)-(?P<i>\d+)$")
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
TIDB = {"tidb": "tidb-tidb", "tikv": "tidb-tikv", "pd": "tidb-pd"}
STATIC_CALLS = [("productcatalogservice", "tidb-tidb"), ("tidb-tidb", "tidb-tikv"), ("tidb-tidb", "tidb-pd")]
APM_META = {"time", "object_id", "object_type"}
INFRA_META = {"time", "cf", "device", "instance", "kpi_key", "kpi_name", "kubernetes_node", "mountpoint", "namespace", "object_type",
              "pod", "sql_type", "type"}
SPAN_TAGS = ("span.kind", "status.code", "status.message", "http.status_code", "http.response.status_code", "grpc.status_code",
             "rpc.system", "rpc.service", "rpc.method", "db.system", "net.peer.name", "peer.service", "error")
PROC_TAGS = ("name", "podName", "node_name", "nodeName")
UTC = dt.timezone.utc


def _clean(name) -> str:
    return DELETED.sub("", str(name or "")).strip()


def service_of(pod: str) -> str:
    m = POD_RE.match(pod or "")
    return m.group("svc") if m else pod


def _utc(s) -> dt.datetime:
    return pd.Timestamp(s).tz_convert("UTC").to_pydatetime() if pd.Timestamp(s).tzinfo else \
        pd.Timestamp(s).tz_localize("UTC").to_pydatetime()


# ------------------------------------------------------------------------------------------------------------------------------
# release index (ground truth never opened)
# ------------------------------------------------------------------------------------------------------------------------------
_HOURLY = re.compile(r"^(?P<kind>log_filebeat-server|trace_jaeger-span)_(?P<d>\d{4}-\d{2}-\d{2})_(?P<h>\d{2})-\d{2}-\d{2}\.parquet$")
_DAILY = re.compile(r"_(?P<d>\d{4}-\d{2}-\d{2})\.parquet$")


class Release:
    """File index of the release: hourly logs / traces and daily metric files, by their UTC coverage."""

    def __init__(self, root: str):
        self.root = root
        self.inputs_path, self.hourly, self.daily = None, defaultdict(list), defaultdict(list)
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith("._") and not GT_PATTERN.search(x)]
            for f in files:
                if f.startswith("._") or GT_PATTERN.search(f):
                    continue
                p = os.path.join(d, f)
                if f == "input.json" and self.inputs_path is None:
                    self.inputs_path = p
                    continue
                m = _HOURLY.match(f)
                if m:
                    t0 = dt.datetime.fromisoformat(f"{m.group('d')}T{m.group('h')}:00:00").replace(tzinfo=UTC) - CST
                    self.hourly["log" if m.group("kind").startswith("log") else "trace"].append((t0, t0 + dt.timedelta(hours=1), p))
                    continue
                m = _DAILY.search(f)
                if m and "metric-parquet" in p:
                    if f.startswith("pod_ns_"):                    # namespace total: not an entity
                        continue
                    t0 = dt.datetime.fromisoformat(f"{m.group('d')}T00:00:00").replace(tzinfo=UTC) - CST
                    self.daily[m.group("d")].append((t0, t0 + dt.timedelta(days=1), p))

    def cases(self) -> List[Dict]:
        """input.json: [{uuid, description, start, end}] (UTC)."""
        assert self.inputs_path, f"input.json not found under {self.root}"
        out = []
        for r in json.load(open(self.inputs_path, encoding="utf-8")):
            desc = r.get("Anomaly Description") or ""
            ts = TS_RE.findall(desc)
            out.append({"uuid": r["uuid"], "description": desc, "start": _utc(ts[0]), "end": _utc(ts[1])})
        return out

    def files(self, kind: str, lo: dt.datetime, hi: dt.datetime) -> List[str]:
        if kind == "metric":
            items = [x for v in self.daily.values() for x in v]
        else:
            items = self.hourly[kind]
        return sorted(p for a, b, p in items if a < hi and b > lo)


# ------------------------------------------------------------------------------------------------------------------------------
# per-file canonical tables (cached: neighbouring cases share most hourly / daily files)
# ------------------------------------------------------------------------------------------------------------------------------
class _LRU(OrderedDict):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.n:
            self.popitem(last=False)
        return v


_CACHE = _LRU(int(os.environ.get("AIOPS_CACHE_FILES", "16")))


def _tag_table(col, keys) -> pd.DataFrame:
    """list<struct<key, type, value>> column -> DataFrame (row index x selected keys), first value per key."""
    flat = pc.list_flatten(col)
    idx = pc.list_parent_indices(col).to_numpy()
    df = pd.DataFrame({"i": idx, "k": flat.field("key").to_numpy(zero_copy_only=False),
                       "v": flat.field("value").to_numpy(zero_copy_only=False)})
    df = df[df["k"].isin(keys)].drop_duplicates(["i", "k"])
    return df.pivot(index="i", columns="k", values="v")


def trace_table(path: str) -> pd.DataFrame:
    def load():
        t = pq.read_table(path, columns=["traceID", "spanID", "operationName", "references", "startTime", "duration", "tags", "process"])
        n = t.num_rows
        tags = _tag_table(t.column("tags").combine_chunks(), SPAN_TAGS).reindex(range(n))
        proc = t.column("process").combine_chunks()
        ptags = _tag_table(proc.field("tags"), PROC_TAGS).reindex(range(n))
        refs = t.column("references").combine_chunks()
        rflat = pc.list_flatten(refs)
        rdf = pd.DataFrame({"i": pc.list_parent_indices(refs).to_numpy(),
                            "type": rflat.field("refType").to_numpy(zero_copy_only=False),
                            "sid": rflat.field("spanID").to_numpy(zero_copy_only=False)})
        rdf = rdf[rdf["type"] == "CHILD_OF"].drop_duplicates("i").set_index("i")["sid"]
        df = pd.DataFrame({"traceId": t.column("traceID").to_numpy(zero_copy_only=False),
                           "spanId": t.column("spanID").to_numpy(zero_copy_only=False),
                           "spanName": t.column("operationName").to_numpy(zero_copy_only=False),
                           "start_us": t.column("startTime").to_numpy(zero_copy_only=False),
                           "duration": t.column("duration").to_numpy(zero_copy_only=False).astype("float64") * 1000.0,   # us -> ns
                           "procService": proc.field("serviceName").to_numpy(zero_copy_only=False)})
        df["parentSpanId"] = rdf.reindex(range(n)).to_numpy()
        col = lambda tab, k: tab[k].astype(object) if k in tab.columns else pd.Series([None] * n, index=range(n), dtype=object)
        first = lambda a, b: a.where(a.notna(), b)
        df["serviceName"] = first(first(col(ptags, "name"), col(ptags, "podName")), df["procService"]).map(_clean)
        df["node"] = first(col(ptags, "node_name"), col(ptags, "nodeName"))
        code = col(tags, "status.code").map(lambda x: "0" if x is None or x != x or x == "" else str(x))
        df["statusCode"] = (code != "0").map({True: "2", False: "1"})
        df["statusMessage"] = col(tags, "status.message").map(lambda x: x if isinstance(x, str) else "")
        recs = tags.to_dict("records")
        attrs = []
        for r, c in zip(recs, code):
            a = {k: v for k, v in r.items() if isinstance(v, str) and k not in ("status.code", "status.message")}
            if a.get("rpc.system") == "grpc" or "grpc.status_code" in a:
                a["rpc.grpc.status_code"] = c                    # canonical OTel name of the gRPC status (A2)
            attrs.append(json.dumps(a, separators=(",", ":"), sort_keys=True))
        df["attributes"] = attrs
        return df
    return _CACHE.get_or(("trace", path), load)


def log_table(path: str) -> pd.DataFrame:
    """Lines the core reads (pipeline/fault_fingerprint._TEXT_ANY: every text family of the evidence), plus the first line of the
    file (it dates the start of the log history). A7 amendment: equivalent input for the statistical core."""
    from pipeline.fault_fingerprint import _TEXT_ANY

    def load():
        df = pd.read_parquet(path, columns=["@timestamp", "k8_pod", "message", "k8_node_name"])
        df["ts"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts", kind="stable")
        keep = df["message"].fillna("").str.contains(_TEXT_ANY.pattern, case=False, regex=True)
        if len(df):
            keep.iloc[0] = True
        out = df[keep][["ts", "k8_pod", "message", "k8_node_name"]].copy()
        out["k8_pod"] = out["k8_pod"].map(_clean)
        hosting = df[["k8_pod", "k8_node_name"]].dropna().drop_duplicates()
        hosting = [(_clean(p), str(n)) for p, n in zip(hosting["k8_pod"], hosting["k8_node_name"])]
        return out, hosting
    return _CACHE.get_or(("log", path), load)


def metric_table(path: str) -> pd.DataFrame:
    """One daily metric file -> rows (time, entity_set, entity_name, metric, value) and hosting pairs (pod, node)."""
    def load():
        df = pd.read_parquet(path)
        f = os.path.basename(path)
        rows, hosting = [], []
        df["t"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if "object_id" in df.columns:                              # APM (pod_<pod>_<date> / service_<svc>_<date>)
            ent = df["object_id"].map(_clean)
            for c in df.columns:
                if c in APM_META or c == "t" or not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                rows.append(pd.DataFrame({"t": df["t"], "entity_set": "apm.service", "entity_name": ent, "metric": c,
                                          "value": df[c].astype(float)}))
        else:
            kpi = str(df["kpi_key"].iloc[0]) if "kpi_key" in df.columns and len(df) else None
            vcol = kpi if kpi in df.columns else next((c for c in df.columns if c not in INFRA_META and c != "t"), None)
            if vcol is None:
                return pd.DataFrame(columns=["t", "entity_set", "entity_name", "metric", "value"]), []
            otype = df["object_type"].astype(str) if "object_type" in df.columns else pd.Series("", index=df.index)
            if f.startswith("infra_node_"):
                dev = df.get("device", pd.Series("null", index=df.index)).astype(str)
                mnt = df.get("mountpoint", pd.Series("null", index=df.index)).astype(str)
                suffix = [("@" + d) if d not in ("null", "", "None", "nan") else (("@" + m) if m not in ("null", "", "None", "nan") else "")
                          for d, m in zip(dev, mnt)]
                rows.append(pd.DataFrame({"t": df["t"], "entity_set": "k8s.node", "entity_name": df["kubernetes_node"].astype(str),
                                          "metric": [vcol + s for s in suffix], "value": df[vcol].astype(float)}))
            elif f.startswith("infra_pod_"):
                pod = df["pod"].map(_clean)
                rows.append(pd.DataFrame({"t": df["t"], "entity_set": "k8s.pod", "entity_name": pod, "metric": vcol,
                                          "value": df[vcol].astype(float)}))
                hp = pd.DataFrame({"p": pod, "n": df["instance"].astype(str)}).drop_duplicates()
                hosting = [(p, n) for p, n in zip(hp["p"], hp["n"]) if n not in ("null", "", "None")]
            else:                                                  # TiDB SQL layer, TiKV, PD
                comp = otype.map(lambda x: TIDB.get(x.lower(), ""))
                if not comp.any():
                    comp = pd.Series(next((v for k, v in TIDB.items() if f.startswith(f"infra_{k}_")), ""), index=df.index)
                rows.append(pd.DataFrame({"t": df["t"], "entity_set": "tidb", "entity_name": comp, "metric": vcol,
                                          "value": df[vcol].astype(float)}))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["t", "entity_set", "entity_name", "metric", "value"])
        out = out[(out["entity_name"] != "") & out["t"].notna() & out["value"].notna()]
        return out, hosting
    return _CACHE.get_or(("metric", path), load)


# ------------------------------------------------------------------------------------------------------------------------------
# case
# ------------------------------------------------------------------------------------------------------------------------------
def _sampled(tid: str) -> bool:
    try:
        return int(str(tid)[-7:], 16) % SAMPLE_MOD == 0
    except ValueError:
        return False


class AIOpsCase:
    def __init__(self, release: Release, spec: Dict, all_windows: Optional[List[Tuple[dt.datetime, dt.datetime]]] = None,
                 history_mask: str = None):
        t_start = time.time()
        self.case_id = spec["uuid"]
        self.case_dir = release.root
        w0, w1 = spec["start"], spec["end"]
        lo, hi = w0 - HISTORY, w1
        mask = (history_mask or os.environ.get("AIOPS_HISTORY_MASK", "none")).lower()
        drops = []
        if mask == "drop":                                         # A8 as first written (see the amendment: infeasible)
            drops = [(a, b + dt.timedelta(minutes=10)) for a, b in (all_windows or []) if not (a == w0 and b == w1) and a < hi and b > lo]

        def keep_t(ts: pd.Series) -> pd.Series:
            k = (ts >= lo) & (ts <= hi)
            for a, b in drops:
                k &= ~((ts >= a) & (ts <= b))
            return k

        self.load_times = {}
        hosting = set()
        # metrics
        t0 = time.time()
        parts = []
        for p in release.files("metric", lo, hi):
            m, h = metric_table(p)
            parts.append(m[keep_t(m["t"])])
            hosting.update(h)
        met = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "entity_set", "entity_name", "metric", "value"])
        met = met.groupby(["entity_set", "entity_name", "metric", "t"], as_index=False)["value"].mean()   # duplicates (renamed pods)
        pods_inf = met[met["entity_set"] == "k8s.pod"].copy()
        if len(pods_inf):                                          # A4: a service's infra series = mean of its pods' series
            pods_inf["entity_name"] = pods_inf["entity_name"].map(service_of)
            agg = pods_inf.groupby(["entity_set", "entity_name", "metric", "t"], as_index=False)["value"].mean()
            met = pd.concat([met, agg], ignore_index=True)
        self.load_times["load_metrics_s"] = time.time() - t0
        # traces
        t0 = time.time()
        parts = [trace_table(p) for p in release.files("trace", lo, hi)]
        tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["traceId", "start_us"])
        if len(tr):
            tr["ts"] = pd.to_datetime(tr["start_us"], unit="us", utc=True)
            tr = tr[keep_t(tr["ts"])]
        calls = set()
        if len(tr):
            sv = tr[["traceId", "spanId", "serviceName"]].copy()
            sv["svc"] = sv["serviceName"].map(service_of)
            par = tr[["traceId", "parentSpanId", "serviceName"]].merge(
                sv.rename(columns={"spanId": "parentSpanId", "svc": "psvc", "serviceName": "_p"}), on=["traceId", "parentSpanId"])
            par["csvc"] = par["serviceName"].map(service_of)
            calls |= {(a, b) for a, b in zip(par["psvc"], par["csvc"]) if a and b and a != b}
            peer = tr["attributes"].str.extract(r'"net\.peer\.name":"([^"]+)"')[0]
            calls |= {(service_of(s), p) for s, p in zip(tr["serviceName"], peer) if isinstance(p, str) and service_of(s) != p}
            failing = set(tr.loc[tr["statusCode"] == "2", "traceId"])
            tr = tr[tr["traceId"].isin(failing) | tr["traceId"].map(_sampled)]       # A7
            hosting.update((p, n) for p, n in zip(tr["serviceName"], tr["node"]) if isinstance(n, str) and n)
        self.load_times["load_traces_s"] = time.time() - t0
        # logs
        t0 = time.time()
        parts = []
        for p in release.files("log", lo, hi):
            lg, h = log_table(p)
            parts.append(lg[keep_t(lg["ts"])])
            hosting.update(h)
        logs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "k8_pod", "message"])
        self.load_times["load_logs_s"] = time.time() - t0

        # entities
        pods = set()
        for name in set(met.loc[met["entity_set"] != "k8s.node", "entity_name"]) | set(tr["serviceName"] if len(tr) else []) | \
                set(logs["k8_pod"]) | {p for p, _ in hosting}:
            if name and POD_RE.match(name) and name not in TIDB.values():
                pods.add(name)
        services = {service_of(p) for p in pods} | set(met.loc[met["entity_set"].isin(("apm.service", "tidb")), "entity_name"]) | \
            {s for e in calls for s in e} - pods
        services = {s for s in services if s and s not in pods}
        nodes = set(met.loc[met["entity_set"] == "k8s.node", "entity_name"]) | {n for _, n in hosting if n}
        nodes -= services | pods
        self.memberships = {p: service_of(p) for p in pods}
        g = nx.DiGraph()
        for s in sorted(services | pods):
            g.add_node(s, entity=Entity(entity_id=s, entity_type="apm.service", name=s,
                                        attributes={"kind": "pod" if s in pods else "service"}))
        for n in sorted(nodes):
            g.add_node(n, entity=Entity(entity_id=n, entity_type="k8s.node", name=n, attributes={}))
        for a, b in sorted(calls | {e for e in STATIC_CALLS if e[0] in g and e[1] in g}):
            if a in g and b in g and a != b:
                g.add_edge(a, b, relation=Relation(a, b, "calls"))
        for p, s in self.memberships.items():
            if s in g:
                g.add_edge(s, p, relation=Relation(s, p, "calls"))
        for p, n in sorted(hosting):
            if p in pods and n in nodes:
                g.add_edge(n, p, relation=Relation(n, p, "contains"))
        self.topology = g
        from data.loader import build_name_index
        self.name_index = build_name_index(g)

        iso = lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.alert = AlertContext(case_id=self.case_id, alert_text=spec["description"], alert_timestamp=w0, entry_entity_id=None,
                                  raw={"alert_window": {"start": iso(w0), "end": iso(w1)}, "_alert_fire_time": None})
        # observations (canonical payloads)
        mo = [Observation(entity_id=e, timestamp=t.to_pydatetime(), modality="metrics", text=f"[metric] {e} {m}={v}",
                          payload={"entity_set": es, "entity_name": e, "metric": m, "value": v})
              for t, es, e, m, v in zip(met["t"], met["entity_set"], met["entity_name"], met["metric"], met["value"])]
        to = []
        if len(tr):
            cols = ["traceId", "spanId", "parentSpanId", "serviceName", "duration", "statusCode", "statusMessage", "attributes", "spanName"]
            for ts, *vals in zip(tr["ts"], *[tr[c] for c in cols]):
                p = dict(zip(cols, vals))
                if not isinstance(p["parentSpanId"], str):
                    p["parentSpanId"] = ""
                to.append(Observation(entity_id=p["serviceName"], timestamp=ts.to_pydatetime(), modality="traces",
                                      text=f"[span] {p['serviceName']}.{p['spanName']} duration={p['duration'] / 1e6}ms", payload=p))
        lo_ = [Observation(entity_id=pod, timestamp=ts.to_pydatetime(), modality="logs", text=f"[log] {pod}: {msg}",
                           payload={"_container_name_": pod, "_time_": ts, "content": msg})
               for ts, pod, msg in zip(logs["ts"], logs["k8_pod"], logs["message"].fillna(""))]
        self.observations = {"metrics": mo, "logs": lo_, "traces": to, "events": [], "alerts": []}
        self.unresolved_entities = set()
        self.load_times["load_total_s"] = time.time() - t_start

    def all_observations(self) -> List[Observation]:
        return [o for v in self.observations.values() for o in v]

    def coverage(self) -> Dict:
        """Label-free adapter coverage (PREREGISTRATION_AIOPS2025.md 'Checks allowed before the run')."""
        types = defaultdict(int)
        for n, d in self.topology.nodes(data=True):
            types[d["entity"].attributes.get("kind") or d["entity"].entity_type] += 1
        rel = defaultdict(int)
        for _, _, d in self.topology.edges(data=True):
            rel[d["relation"].relation_type] += 1
        return {"entities": dict(types), "edges": dict(rel), **{f"n_{k}": len(v) for k, v in self.observations.items()},
                **{k: round(v, 1) for k, v in self.load_times.items()}}

    def __repr__(self):
        return f"<AIOpsCase {self.case_id}: {self.topology.number_of_nodes()} entities, {sum(map(len, self.observations.values()))} obs>"
