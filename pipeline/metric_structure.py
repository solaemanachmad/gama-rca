"""Metric structure evidence (evidence extraction ONLY -- no decision, no significance filter).

Two blocks, both computed from metrics.parquet + topology + task window, no ground truth:

(1) Identity-free K8s aggregates (audit 2026-10-11, claude/audit-metrics-2026-10-11.md): entity_id/entity_name are empty for
    100% of the k8s.node / k8s.deployment / k8s.namespace / k8s.cluster rows in all 103 cases, so a node or deployment
    cannot be named from metrics (the old infra tables silently skipped them). What stays well-defined without identity are
    order statistics ACROSS the entities at one timestamp (max / median / min / sum / count) and cluster-level series.
    For each fixed (metric, statistic) pair below the per-timestamp statistic is computed first, then its series over
    time is summarised as baseline median -> window min / median / max. The list of pairs is a fixed table (same for
    every case); nothing is chosen by significance.

(2) Call-edge flow ratios (operator D): for every `calls` edge between services of the alert service's neighbourhood
    (chosen by NAME), the ratio callee workload / caller workload at the same timestamps, baseline median vs window median,
    next to each service's own sustained change (window median / baseline median). A stable ratio means the callee follows
    its caller; a ratio that moves is specific to that edge. Medians are used so that sustained drops are not hidden by
    the window maximum.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from pipeline.export_bursts import neighbourhood, sec
from pipeline.propagation_evidence import _calls_graph, _node_name, _node_type, window_bounds

MAX_EDGES = 14

# (entity_set, metric, statistic, label). statistic: one (single series) | sum | max | med | min over the rows of one timestamp.
# The namespace pod totals are left out: they equal the cluster-level series.
K8S_AGG: List[Tuple[str, str, str, str]] = [
    ("k8s.cluster", "cluster_node_count", "one", "cluster node count"),
    ("k8s.cluster", "cluster_failed_node_count", "one", "cluster failed node count"),
    ("k8s.cluster", "cluster_number_of_pod", "one", "cluster pods"),
    ("k8s.cluster", "cluster_number_of_not_ready_pods", "one", "cluster not-ready pods"),
    ("k8s.cluster", "apiserver_non_2xx_read_request_qps", "one", "apiserver non-2xx read qps"),
    ("k8s.node", "node_ready_status", "sum", "nodes ready (count)"),
    ("k8s.node", "node_not_ready_status", "sum", "nodes not ready (count)"),
    ("k8s.node", "node_cpu_usage_rate", "max", "node cpu usage rate, max over nodes"),
    ("k8s.node", "node_cpu_usage_rate", "med", "node cpu usage rate, median over nodes"),
    ("k8s.node", "node_memory_usage_rate", "max", "node memory usage rate, max over nodes"),
    ("k8s.node", "node_memory_usage_rate", "med", "node memory usage rate, median over nodes"),
    ("k8s.node", "node_disk_usage_rate", "max", "node disk usage rate, max over nodes"),
    ("k8s.node", "node_pod_running_count", "sum", "running pods, sum over nodes"),
    ("k8s.node", "node_pod_running_count", "min", "running pods, min over nodes"),
    ("k8s.deployment", "deployment_desired_replicas", "sum", "desired replicas, sum over deployments"),
    ("k8s.deployment", "deployment_ready_replicas", "sum", "ready replicas, sum over deployments"),
    ("k8s.deployment", "deployment_available_replicas", "sum", "available replicas, sum over deployments"),
    ("k8s.deployment", "deployment_availability_rate", "min", "availability rate, min over deployments"),
    ("k8s.deployment", "deployment_cpu_usage_vs_limits", "max", "cpu usage vs limits, max over deployments"),
    ("k8s.deployment", "deployment_cpu_usage_vs_requests", "max", "cpu usage vs requests, max over deployments"),
    ("k8s.deployment", "deployment_memory_usage_total", "max", "memory usage, max over deployments"),
]
_ES = {"k8s.cluster", "k8s.namespace", "k8s.node", "k8s.deployment"}


def compute_metric_structure(case, alert_ts: Optional[dt.datetime], entry_entity_id: Optional[str] = None,
                             max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0, "stats": {}}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _to_sec(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    m = float(v.dropna().abs().median()) if v.notna().any() else 0.0
    div = 1e9 if m > 1e17 else 1e6 if m > 1e14 else 1e3 if m > 1e11 else 1.0
    return v / div


def _fmt(x) -> str:
    if x is None or x != x:
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x:.3g}"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.3g}"


def _agg(group: pd.DataFrame, how: str) -> pd.Series:
    """per-timestamp statistic across the rows of one metric (identity-free)."""
    g = group.groupby("t")["value"]
    if how in ("one", "sum"):
        return g.sum()
    if how == "max":
        return g.max()
    if how == "min":
        return g.min()
    return g.median()


def _compute(case, alert_ts, entry_entity_id, max_hops):
    if alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = window_bounds(case, at)
    w0s, w1s = sec(w0), sec(w1)
    fp = os.path.join(case.case_dir, config.FILE_METRICS)
    d = pd.read_parquet(fp, columns=["time", "entity_set", "entity_name", "metric", "value"])
    d["t"] = _to_sec(d["time"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["t", "value"])
    d = d[d["t"] <= w1s]

    # ---- (1) identity-free K8s aggregates ----
    k8s = d[d["entity_set"].isin(_ES)]
    lines1: List[str] = []
    for es, metric, how, label in K8S_AGG:
        g = k8s[(k8s["entity_set"] == es) & (k8s["metric"] == metric)]
        if g.empty:
            continue
        s = _agg(g, how)
        base = s[s.index < w0s]
        win = s[(s.index >= w0s) & (s.index <= w1s)]
        if win.empty:
            continue
        lines1.append(f"- {label}: base median {_fmt(base.median() if len(base) else None)} -> window "
                      f"min {_fmt(win.min())} / median {_fmt(win.median())} / max {_fmt(win.max())}")

    # ---- (2) call-edge flow ratios ----
    lines2: List[str] = []
    cand, svc0 = neighbourhood(case, entry_entity_id, max_hops)
    if cand:
        topo = case.topology
        names = {e: _node_name(topo, e) for e in cand}
        alert_name = names.get(svc0)
        wl = d[(d["entity_set"] == "apm.service.legacy") & (d["metric"] == "workload")]
        ser = {n: g.groupby("t")["value"].sum() for n, g in wl.groupby("entity_name") if n in set(names.values())}
        calls = _calls_graph(topo)
        edges = []
        for u, v in calls.edges:
            if u in names and v in names and _node_type(topo, u) == "apm.service" and _node_type(topo, v) == "apm.service":
                edges.append((names[u], names[v]))
        edges = sorted(set(edges), key=lambda e: (alert_name not in e, e))

        def own(sv):
            b, w = sv[sv.index < w0s], sv[(sv.index >= w0s) & (sv.index <= w1s)]
            if b.empty or w.empty or b.median() == 0:
                return None, None, None
            return b.median(), w.median(), w.median() / b.median()

        for a, c in edges[:MAX_EDGES]:
            if a not in ser or c not in ser:
                continue
            ra = ser[a].replace(0, np.nan)
            ratio = (ser[c] / ra).dropna()
            rb, rw = ratio[ratio.index < w0s], ratio[(ratio.index >= w0s) & (ratio.index <= w1s)]
            if rb.empty or rw.empty:
                continue
            _, _, fa = own(ser[a])
            _, _, fc = own(ser[c])
            lines2.append(f"- {a}>{c}: callee/caller workload ratio base {_fmt(rb.median())} -> window "
                          f"{_fmt(rw.median())}; sustained change (window median / base median): caller "
                          f"x{_fmt(fa)}, callee x{_fmt(fc)}")
    out = []
    if lines1:
        out.append("Identity-free K8s aggregates (node, deployment, namespace and cluster metrics carry no entity names "
                   "in this export, so only statistics across entities at each timestamp are shown; they cannot say "
                   "WHICH node or deployment). baseline = samples before the window; window = [start, end].")
        out.extend(lines1)
    if lines2:
        out.append("Call-edge flow (workload of the callee divided by workload of the caller at the same timestamps; "
                   "medians, so sustained changes are visible):")
        out.extend(lines2)
    text = "\n".join(out)
    return {"text": text, "rows": len(lines1) + len(lines2),
            "stats": {"metric_struct_agg_lines": len(lines1), "metric_struct_flow_edges": len(lines2)}}
