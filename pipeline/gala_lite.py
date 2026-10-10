"""GALA-lite: a reduced re-implementation of the GALA workflow (Tian et al.,
arXiv 2508.12472) used ONLY as a comparison baseline on RCA100, with the SAME
LLM and the SAME inputs as gama-rca. It is deliberately independent of gama-rca's
own contributions (no propagation table, no infra tables, no chain-first,
no Graph Anchor, no vector retrieval).

Mapping to GALA's four phases and the deviations (state these in the paper):
  Phase I   Initial hypotheses. GALA: metrics causal DAG + Random Walk/PageRank,
            and TWIST over traces. Lite: TWIST (same module as gama-rca, i.e.
            GALA's own four-score formula) + a z-score metrics ranking (no causal
            discovery -- not available/robust at RCA100 scale).
  Phase II  Diagnostic bundle per candidate: temporal performance profile
            (baseline vs window numbers of the candidate's own service metrics),
            dependency subgraph (predecessors/successors on the call graph),
            error-centric log abstraction (error-level lines, de-duplicated,
            seeded random sample).
  Phase III Iterative Re-ranking Agent + Deep-Dive Agent loop. GALA: ReAct, up to
            6 iterations. Lite: up to GALA_MAX_ITERS (default 3) to bound cost.
  Phase IV  Remediation agent: omitted (RCA100 does not score it). The final JSON
            uses gama-rca's shared FINAL_SCHEMA_HINT so scoring is identical.
No ground truth is read. Fault type is chosen by the LLM from the full taxonomy.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import re
import statistics
import time
from collections import defaultdict
from typing import Dict, List

import config
from agents.llm_client import LLMClient
from agents.multi_agent import FINAL_SCHEMA_HINT
from data.loader import Case, find_service_ancestor, resolve_entity_by_name
from data.taxonomy import taxonomy_prompt_block
from pipeline.evidence_summarizer import ERROR_PATTERN
from pipeline.propagation_evidence import (WINDOW_AFTER_S, WINDOW_BEFORE_S,
                                           _calls_graph, _node_name, _node_type)
from pipeline.twist_scoring import compute_twist_scores
from schema import RCAResult

GALA_MAX_ITERS = int(os.environ.get("GALA_MAX_ITERS", "3"))
GALA_TOP_PER_LIST = 5

RERANK_SYSTEM = ("You are the Re-ranking Agent of an RCA workflow. Given candidate root-cause "
                 "services with initial rankings and the summaries gathered so far, decide the "
                 "next candidate to analyze in depth, or finish. Respond ONLY with JSON: "
                 '{"ranking": ["svc1", "svc2", ...], "next": "<service name or FINISH>"}')
DEEPDIVE_SYSTEM = ("You are the Deep-Dive Agent. Summarize in 3-4 sentences whether this service "
                   "looks like the ORIGIN of the incident or only a victim, citing the numbers "
                   "given. Respond ONLY with JSON: " '{"summary": "..."}')
FINAL_SYSTEM = ("You are an SRE finishing a root cause analysis. Respond ONLY with valid JSON: "
                f"{FINAL_SCHEMA_HINT}")


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _metric_z_ranking(case: Case, at, services: set):
    w0, w1 = at - dt.timedelta(seconds=WINDOW_BEFORE_S), at + dt.timedelta(seconds=WINDOW_AFTER_S)
    series = defaultdict(list)
    for o in case.observations.get("metrics", []):
        p = o.payload or {}
        if p.get("entity_set") != "apm.service.legacy" or p.get("entity_name") not in services:
            continue
        ts = _naive(o.timestamp)
        if ts is None or ts > w1 or p.get("value") is None:
            continue
        series[(p["entity_name"], p.get("metric"))].append((ts, float(p["value"])))
    best, profile = defaultdict(float), defaultdict(list)
    for (svc, metric), pts in series.items():
        base = [v for t, v in pts if t < w0]
        win = [v for t, v in pts if w0 <= t <= w1]
        if len(base) < 3 or not win:
            continue
        mu, sd = statistics.mean(base), statistics.pstdev(base)
        z = (max(win) - mu) / (sd + 1e-9)
        best[svc] = max(best[svc], z)
        profile[svc].append(f"{metric}: baseline_median={statistics.median(base):.4g}, window_max={max(win):.4g}")
    return sorted(best, key=lambda s: -best[s]), profile


def _error_logs(case: Case, svc: str, at, k: int = 8) -> List[str]:
    w0, w1 = at - dt.timedelta(seconds=WINDOW_BEFORE_S), at + dt.timedelta(seconds=WINDOW_AFTER_S)
    tag = f" {svc}:"
    seen, out = set(), []
    for o in case.observations.get("logs", []):
        if tag not in o.text:
            continue
        ts = _naive(o.timestamp)
        if ts is None or not (w0 <= ts <= w1) or not ERROR_PATTERN.search(o.text):
            continue
        key = re.sub(r"\d+", "#", o.text)[:160]
        if key not in seen:
            seen.add(key)
            out.append(o.text[:220])
    random.Random(0).shuffle(out)
    return out[:k]


def gala_lite(case_id: str, llm: LLMClient, cases_dir: str = config.CASES_DIR) -> RCAResult:
    from pipeline.pipeline import parse_alert
    from data.loader import normalize_entity_ids
    llm.reset_usage()
    t0 = time.time()
    case = Case(case_id, cases_dir=cases_dir)
    parsed = parse_alert(case)
    at = _naive(case.alert.alert_timestamp)
    topo, calls = case.topology, _calls_graph(case.topology)
    svc_nodes = {_node_name(topo, n): n for n in topo.nodes if _node_type(topo, n) == "apm.service"}

    # Phase I
    twist = compute_twist_scores(case.observations.get("traces", []), None, case.name_index)
    twist_rank = sorted(twist, key=lambda s: -twist[s]["twist_score"])
    metric_rank, profile = _metric_z_ranking(case, at, set(svc_nodes)) if at else ([], {})
    cands = []
    for s in twist_rank[:GALA_TOP_PER_LIST] + metric_rank[:GALA_TOP_PER_LIST]:
        if s in svc_nodes and s not in cands:
            cands.append(s)
    entry = find_service_ancestor(parsed["entry_entity_id"], topo) if parsed.get("entry_entity_id") else None
    if entry and _node_name(topo, entry) not in cands:
        cands.append(_node_name(topo, entry))
    header = (f"Alert: {parsed['alert_text']}\n"
              f"Trace-based initial ranking (TWIST): {twist_rank[:GALA_TOP_PER_LIST]}\n"
              f"Metrics-based initial ranking (z-score): {metric_rank[:GALA_TOP_PER_LIST]}\n")

    # Phase II + III
    summaries: Dict[str, str] = {}
    ranking = list(cands)
    nxt = cands[0] if cands else None
    for _ in range(GALA_MAX_ITERS):
        if not nxt or nxt not in svc_nodes or nxt in summaries:
            break
        n = svc_nodes[nxt]
        preds = sorted(_node_name(topo, p) for p in calls.predecessors(n))
        succs = sorted(_node_name(topo, s) for s in calls.successors(n))
        logs = _error_logs(case, nxt, at) if at else []
        tw = twist.get(nxt)
        bundle = (f"Service {nxt}\n"
                  f"Temporal profile: {'; '.join(profile.get(nxt, [])[:8]) or 'n/a'}\n"
                  f"TWIST: {tw if tw else 'n/a'}\n"
                  f"Callers: {preds}; Callees: {succs}\n"
                  f"Error-centric logs: {logs or 'none in window'}\n")
        d = llm.generate_json(bundle, system=DEEPDIVE_SYSTEM)
        summaries[nxt] = str(d.get("summary", ""))[:600]
        done = "\n".join(f"- {k}: {v}" for k, v in summaries.items())
        r = llm.generate_json(header + f"Current ranking: {ranking}\nSummaries so far:\n{done}\n"
                              f"Candidates not yet analyzed: {[c for c in cands if c not in summaries]}",
                              system=RERANK_SYSTEM)
        ranking = [s for s in (r.get("ranking") or []) if s in svc_nodes] or ranking
        nxt = r.get("next")
        if not nxt or str(nxt).upper() == "FINISH":
            break

    done = "\n".join(f"- {k}: {v}" for k, v in summaries.items()) or "(none)"
    final = llm.generate_json(
        header + f"Final ranking: {ranking}\nDeep-dive summaries:\n{done}\n\n{taxonomy_prompt_block()}\n\n"
        "Task: give the root cause entity (service name, the ORIGIN of the fault), the fault type, "
        "and a cause -> propagation -> impact reasoning chain citing the numbers above.",
        system=FINAL_SYSTEM)
    ents = final.get("predicted_entity_ids", []) or []
    ents = normalize_entity_ids(ents, topo, case.name_index) or \
        ([svc_nodes[ranking[0]]] if ranking and ranking[0] in svc_nodes else [])
    stats = {"total_pipeline_time_s": time.time() - t0, "gala_iters": len(summaries),
             "gala_candidates": len(cands), **llm.usage_stats()}
    return RCAResult(case_id=case_id, predicted_entity_ids=ents,
                     predicted_fault_type=final.get("predicted_fault_type", "unknown"),
                     reasoning_chain=final.get("reasoning_chain", []) or [],
                     confidence=float(final.get("confidence", 0.0) or 0.0),
                     agent_findings=[], retrieval_stats=stats, evidence_items=[])
