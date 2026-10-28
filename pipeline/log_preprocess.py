"""v3 log preprocessing for the text sensor (Amendment 2 of PREREGISTRATION_V3.md). Deterministic, CPU only, no labels, no LLM.

The raw sensor input (semantic_probes.collect_texts) keeps the first 20 distinct keyword-matching lines per entity in the alert window.
That selection depends on vocabulary, ignores frequency and time order, and cannot tell chronic errors from new ones. Here every log
line of a service is reduced to a template and tested against the service's own pre-window rate:

  1. Template: regex masking (pipeline.log_templates.mask: ids, IPs, numbers, timestamps) and Drain (He et al., ICWS 2017; drain3, its
     default similarity 0.4 and depth 4), one parse tree per service. No keyword filter.
  2. Counts per template: n_b lines before the window (the log baseline, ~15 min in RCA100) and n_w lines inside it.
  3. Test: one-sided Poisson test of n_w against the baseline rate, lambda = (n_b + 0.5) * T_w / T_b (0.5: Jeffreys-type continuity so a
     template unseen before the window is testable); Benjamini-Hochberg over all window templates of the case at q = 0.05.
  4. Sensor input per entity, same budget as the raw input (20 items, 200 characters): Kubernetes events and error-span messages of the
     window (no baseline exists for them: traces are exported only for the window), then the significant templates by p-value. Each item
     carries one raw example line (the quote must come from it) and its counts.
Entities without any selected item are not probed (a chronic error log no longer makes an entity probe-able)."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from pipeline.fault_fingerprint import _owner, _naive
from pipeline.log_templates import mask

MAX_ITEMS, MAX_CHARS = 20, 200          # = semantic_probes.MAX_LINES / MAX_CHARS (same budget as the raw input)
BH_Q = 0.05
CONT = 0.5
SIM_TH, DEPTH = 0.4, 4                  # drain3 defaults
_WS = re.compile(r"\s+")


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k), X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    logterm = lambda i: -lam + i * math.log(lam) - math.lgamma(i + 1)
    if k > lam:                                          # upper tail: sum from k while terms matter
        s, i = 0.0, k
        while True:
            t = math.exp(logterm(i))
            s += t
            if t < 1e-17 * max(s, 1e-300) or i > k + 100000:
                break
            i += 1
        return min(1.0, s)
    return max(0.0, 1.0 - sum(math.exp(logterm(i)) for i in range(k)))


def bh(pvals: List[float], q: float = BH_Q) -> List[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for r, i in enumerate(order, 1):
        if pvals[i] <= q * r / m:
            kmax = r
    keep = [False] * m
    for r, i in enumerate(order, 1):
        keep[i] = r <= kmax
    return keep


def _clean(s) -> str:
    return _WS.sub(" ", str(s)).strip()[:MAX_CHARS]


def _drain_groups(lines: Dict[str, dict]) -> Dict[str, str]:
    """masked line -> Drain template (one tree; most frequent first, then lexical: deterministic)."""
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th, cfg.drain_depth, cfg.drain_max_clusters = SIM_TH, DEPTH, None
    cfg.parametrize_numeric_tokens = False
    m = TemplateMiner(config=cfg)
    ids = {}
    for ml, a in sorted(lines.items(), key=lambda kv: (-(kv[1]["b"] + kv[1]["w"]), kv[0])):
        ids[ml] = m.add_log_message(ml)["cluster_id"]
    return {ml: m.drain.id_to_cluster[c].get_template() for ml, c in ids.items()}


def preprocess_texts(case, db) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], dict]:
    """Returns (texts, notes, stats): texts[entity] = example lines (quotable), notes[entity] = aligned count notes."""
    from data.loader import event_time_bounds, _parse_ts_iso
    w0, w1 = db["w0"], db["w1"]
    services, nodes = list(db["services"]), set(db["nodes"])
    t_w = max((w1 - w0).total_seconds() / 60.0, 1e-9)
    items = defaultdict(list)                            # entity -> [(rank_key, text, note)]

    # Kubernetes events overlapping the window (as in the raw input), counted
    ev = defaultdict(lambda: [0, ""])
    for o in case.observations.get("events", []):
        p = o.payload or {}
        try:
            e = json.loads(p.get("eventId")) if isinstance(p.get("eventId"), str) else (p.get("eventId") or {})
        except Exception:
            e = {}
        _f, _l = event_time_bounds(e)
        first, last = _naive(_parse_ts_iso(_f)), _naive(_parse_ts_iso(_l))
        if first is None or last is None or last < w0 or first > w1:
            continue
        io = e.get("involvedObject") or {}
        name = io.get("name") or ""
        ent = name if (io.get("kind") == "Node" or name in nodes) else _owner(name, services)
        if ent:
            txt = _clean(f"{e.get('reason', '')}: {e.get('message', '')}")
            k = (ent, mask(txt))
            ev[k][0] += int(e.get("count") or 1)
            ev[k][1] = ev[k][1] or txt
    for (ent, _), (n, txt) in ev.items():
        items[ent].append(((0, -n, txt), txt, f"Kubernetes event, count {n}"))

    # error-span messages in the window (no baseline exists for traces), counted
    sp = defaultdict(lambda: [0, ""])
    for o in case.observations.get("traces", []):
        p = o.payload or {}
        ts = _naive(o.timestamp)
        if ts is None or ts < w0 or ts > w1 or str(p.get("statusCode")) != "2" or not p.get("statusMessage"):
            continue
        s = _owner(p.get("serviceName"), services)
        if s:
            txt = _clean(p["statusMessage"])
            k = (s, mask(txt))
            sp[k][0] += 1
            sp[k][1] = sp[k][1] or txt
    for (ent, _), (n, txt) in sp.items():
        items[ent].append(((1, -n, txt), txt, f"error span message, {n} spans in the window (traces exist only for the window)"))

    # logs: template, counts before / inside the window, Poisson test, BH
    logs = case.observations.get("logs", [])
    tmin = min((_naive(o.timestamp) for o in logs if o.timestamp is not None), default=None)
    t_b = (w0 - tmin).total_seconds() / 60.0 if tmin is not None and tmin < w0 else 0.0
    per_svc = defaultdict(dict)                          # svc -> masked line -> {"b","w","ex"}
    cache = {}
    n_lines = 0
    for o in logs:
        ts = _naive(o.timestamp)
        if ts is None or ts > w1:
            continue
        p = o.payload or {}
        s = _owner(p.get("_container_name_"), services)
        content = p.get("content") or ""
        if not s or not content:
            continue
        n_lines += 1
        ml = cache.get(content)
        if ml is None:
            ml = cache[content] = mask(content)
        a = per_svc[s].get(ml)
        if a is None:
            a = per_svc[s][ml] = {"b": 0, "w": 0, "ex": None}
        if ts < w0:
            a["b"] += 1
        else:
            a["w"] += 1
            if a["ex"] is None:
                a["ex"] = _clean(content)
    tpls = []                                            # (svc, template, n_b, n_w, example)
    for s, lines in per_svc.items():
        g = _drain_groups(lines)
        agg = {}
        for ml, a in lines.items():
            t = agg.setdefault(g[ml], {"b": 0, "w": 0, "ex": None, "exw": -1})
            t["b"] += a["b"]
            t["w"] += a["w"]
            if a["ex"] is not None and a["w"] > t["exw"]:         # example: the most frequent window variant
                t["ex"], t["exw"] = a["ex"], a["w"]
        tpls += [(s, k, t["b"], t["w"], t["ex"]) for k, t in sorted(agg.items()) if t["w"] > 0]
    stats = {"logpre_lines": n_lines, "logpre_templates": len(tpls), "logpre_baseline_min": round(t_b, 2)}
    if tpls and t_b >= t_w:                              # testable only with at least one window length of baseline
        pv = [poisson_sf(nw, (nb + CONT) * t_w / t_b) for _, _, nb, nw, _ in tpls]
        keep = bh(pv)
        for (s, _, nb, nw, ex), p, k in zip(tpls, pv, keep):
            if k:
                note = (f"log template, {nw} lines in the window ({nw / t_w:.3g}/min) vs {nb} before ({nb / t_b:.3g}/min)"
                        + ("; new in the window" if nb == 0 else ""))
                items[s].append(((2, p, -nw, ex), ex, note))
        stats["logpre_significant"] = sum(keep)
    else:
        stats["logpre_significant"] = 0
        stats["logpre_untestable"] = 1
    texts, notes = {}, {}
    for ent, its in items.items():
        seen, tx, nt = set(), [], []
        for _, txt, note in sorted(its, key=lambda x: x[0]):
            if txt and txt not in seen and len(tx) < MAX_ITEMS:
                seen.add(txt)
                tx.append(txt)
                nt.append(note)
        if tx:
            texts[ent], notes[ent] = tx, nt
    stats["logpre_entities"] = len(texts)
    return texts, notes, stats
