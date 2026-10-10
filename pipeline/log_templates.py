"""Log template layer (evidence extraction ONLY).

Raw log volume per RCA100 case is 0.5-0.7M lines but only a few hundred distinct
message templates once variable parts (timestamps, ids, IPs, numbers) are masked.
Instead of embedding a positional 5,000-line slice, every template is kept with
its counts, so the whole log modality is covered:

  template = (service, level, masked message)
  per template: count before the alert window (baseline), count inside the
  window, per-minute rates, first/last seen, one raw example.

No significance filter, no threshold, no ranking by a decision rule. Templates are
emitted as Observations (modality "log_template") so the existing hybrid retrieval
can embed them; the whole set is small enough to index without a cap.
Uses telemetry + alert time only (no GT, no training). Masking is a hand-written
regex, the same preprocessing step Drain/Drain3 apply before clustering.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from pipeline.propagation_evidence import WINDOW_AFTER_S, WINDOW_BEFORE_S

_KEEP_KEYS = ("level", "severity", "msg", "message", "error", "err", "reason", "status",
              "statuscode", "code", "event", "method", "path", "route", "type", "name")
_JSON_STR = re.compile(r'("([A-Za-z_.][\w.\-]*)"\s*:\s*)"((?:[^"\\]|\\.)*)"')


def _mask_json_values(text: str) -> str:
    """Structured (JSON) log lines: mask string VALUES except for message-like keys,
    so per-request payloads (names, cities, ids) do not create new templates."""
    def rep(m):
        return m.group(0) if m.group(2).lower() in _KEEP_KEYS else m.group(1) + '"<S>"'
    return _JSON_STR.sub(rep, text)


_MASKS = [
    (re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*?-[a-z0-9]{8,10}-[a-z0-9]{5}\b"), "<POD>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
    (re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8,}\b"), "<ID>"),
    (re.compile(r"(?<![A-Za-z_<])\d+(?:\.\d+)?"), "<N>"),
    (re.compile(r"\s+"), " "),
]
_JSON_LEVEL = re.compile(r'"level"\s*:\s*"?(\w+)')
_HEAD = re.compile(r"^\[log:([^\]]*)\]\s*([^:]+):\s*(.*)$", re.S)
MAX_TEMPLATE_CHARS = 300
ENGINE = os.environ.get("LOG_TEMPLATE_ENGINE", "regex")        # "regex" | "drain3"
DRAIN_SIM_TH = float(os.environ.get("DRAIN_SIM_TH", "0.4"))     # Drain3 default
DRAIN_DEPTH = int(os.environ.get("DRAIN_DEPTH", "4"))           # Drain3 default


def mask(text: str) -> str:
    if '":' in text:
        text = _mask_json_values(text)
    for pat, rep in _MASKS:
        text = pat.sub(rep, text)
    return text.strip()[:MAX_TEMPLATE_CHARS]


def _naive(ts):
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def compute_log_templates(case, alert_ts: Optional[dt.datetime]) -> Dict:
    """Returns {"templates": [dict], "observations": [Observation], "stats": {...}}."""
    from schema import Observation
    empty = {"templates": [], "observations": [], "stats": {"log_rows": 0, "log_templates": 0}}
    try:
        return _compute(case, alert_ts, Observation) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


def _compute(case, alert_ts, Observation):
    logs = case.observations.get("logs", [])
    name_index = getattr(case, "name_index", None)
    if not logs or alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = at - dt.timedelta(seconds=WINDOW_BEFORE_S), at + dt.timedelta(seconds=WINDOW_AFTER_S)
    tmin = min((_naive(o.timestamp) for o in logs if o.timestamp is not None), default=None)
    base_min = max(1.0, ((w0 - tmin).total_seconds() / 60.0) if tmin else 1.0)
    win_min = (WINDOW_BEFORE_S + WINDOW_AFTER_S) / 60.0

    agg: Dict[tuple, dict] = {}
    n_rows = 0
    for o in logs:
        m = _HEAD.match(o.text or "")
        level, svc, msg = (m.group(1), m.group(2).strip(), m.group(3)) if m else ("", "?", o.text or "")
        if not level:
            lm = _JSON_LEVEL.search(msg)
            level = lm.group(1).lower() if lm else ""
        key = (svc, level, mask(msg))
        a = agg.get(key)
        if a is None:
            a = agg[key] = {"base": 0, "win": 0, "after": 0, "first": None, "last": None, "ex": (o.text or "")[:240]}
        n_rows += 1
        ts = _naive(o.timestamp)
        if ts is None:
            continue
        if ts < w0:
            a["base"] += 1
        elif ts <= w1:
            a["win"] += 1
        else:
            a["after"] += 1
        if a["first"] is None or ts < a["first"]:
            a["first"] = ts
        if a["last"] is None or ts > a["last"]:
            a["last"] = ts

    if ENGINE == "drain3":
        agg = _drain3_merge(agg)

    templates, obs = [], []
    for (svc, level, tpl), a in sorted(agg.items()):
        row = {"service": svc, "level": level, "template": tpl, "count_base": a["base"],
               "count_window": a["win"], "rate_base_per_min": a["base"] / base_min,
               "rate_window_per_min": a["win"] / win_min, "first": a["first"], "last": a["last"],
               "example": a["ex"]}
        templates.append(row)
        text = (f"[logtpl:{level}] {svc}: {tpl} "
                f"(baseline {a['base']} lines = {row['rate_base_per_min']:.3g}/min; "
                f"alert window {a['win']} lines = {row['rate_window_per_min']:.3g}/min)")
        eid = None
        if name_index:
            from data.loader import resolve_entity_by_name
            eid = resolve_entity_by_name(svc, name_index)
        obs.append(Observation(entity_id=eid, timestamp=at, modality="log_template", text=text,
                               payload={"service": svc, "level": level, **{k: row[k] for k in
                                        ("count_base", "count_window")}}, source_file="log_templates"))
    return {"templates": templates, "observations": obs,
            "stats": {"log_rows": n_rows, "log_templates": len(templates),
                      "log_compression": round(n_rows / max(1, len(templates)), 1)}}


def _drain3_merge(agg: Dict[tuple, dict]) -> Dict[tuple, dict]:
    """Stage 2 (optional): cluster the regex-masked UNIQUE lines with Drain
    (He et al., ICWS 2017; drain3 implementation), one parse tree per service.
    Counting is exact: each unique masked line is fed once (most frequent first,
    deterministic order) and its counts are added to its final cluster."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    def make_miner():
        cfg = TemplateMinerConfig()
        cfg.drain_sim_th = DRAIN_SIM_TH
        cfg.drain_depth = DRAIN_DEPTH
        cfg.drain_max_clusters = None
        cfg.parametrize_numeric_tokens = False   # already masked by the regex stage
        return TemplateMiner(config=cfg)

    miners: Dict[str, object] = {}
    assign: Dict[tuple, int] = {}
    order = sorted(agg.items(), key=lambda kv: (-(kv[1]["base"] + kv[1]["win"] + kv[1]["after"]), kv[0]))
    for (svc, level, tpl), a in order:
        m = miners.get(svc)
        if m is None:
            m = miners[svc] = make_miner()
        assign[(svc, level, tpl)] = m.add_log_message(tpl)["cluster_id"]

    merged: Dict[tuple, dict] = {}
    for (svc, level, tpl), a in agg.items():
        cl = miners[svc].drain.id_to_cluster.get(assign[(svc, level, tpl)])
        final = cl.get_template() if cl is not None else tpl
        key = (svc, level, final[:MAX_TEMPLATE_CHARS])
        b = merged.get(key)
        if b is None:
            merged[key] = dict(a)
            continue
        b["base"] += a["base"]; b["win"] += a["win"]; b["after"] += a["after"]
        for k, f in (("first", min), ("last", max)):
            if a[k] is not None:
                b[k] = a[k] if b[k] is None else f(b[k], a[k])
    return merged
