"""Log evidence per export burst (evidence extraction ONLY -- no decision, no significance filter).

Why this module exists (audit 2026-10-11, claude/audit-logs-2026-10-11.md): the log export is a set of bursts of
<= ~100,000 lines that are dense for ~1 minute every ~5 minutes (grid anchored at alert_window.start). Raw line counts and
per-minute rates are therefore export artefacts (the old log_templates rates divide by minutes that contain no
samples). What survives the export is the COMPOSITION of a burst: which container produced what share of the lines,
which share of a container's lines is error-like, and which message templates appear in one burst but not in another.

Rules (uniform for every case, nothing tuned on ground truth):
  * bursts = maximal runs of log lines whose consecutive timestamps are <= BURST_GAP_S apart;
  * window bursts = bursts that intersect the evidence window (window_bounds); if none intersects, the latest
    earlier burst is used and the text says so; baseline bursts = all other bursts that start before the window end;
  * per container: share of lines in a burst (baseline: median and range over baseline bursts; window: pooled),
    error-like share (level error/fatal/... or an error word in the masked template), number of templates, number of
    templates seen in the window bursts but in no baseline burst ("new") and the reverse ("absent");
  * the containers of the call-graph neighbourhood of the alert service (same set as trace/infra evidence, chosen by
    NAME) get template detail: the NEW_N most frequent new templates, the ERR_N most frequent error-like template of
    the window bursts (whether new or not), counts only -- no rate;
  * every other container gets one compact line (share and error-like share), so no container is hidden.
Uses logs + topology + task window only (no ground truth, no training)."""
from __future__ import annotations

import datetime as dt
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from pipeline.log_templates import _HEAD, _JSON_LEVEL, mask, _naive
from pipeline.propagation_evidence import window_bounds

BURST_GAP_S = 60.0
NEW_N = 2
ERR_N = 1
TPL_CHARS = 140
MAX_NEIGH = 8   # neighbourhood containers with template detail (alphabetical order if more; the rest get compact lines)

_ERR_LEVELS = {"error", "err", "fatal", "critical", "severe", "panic", "crit", "emerg", "alert"}
# second-stage normalisation, used ONLY for grouping in this module: mask() leaves tokens such as "@3fa9c1" or
# "a3c1" (digits glued to letters) which would make every line its own template.
_MIXED = re.compile(r"@[0-9A-Za-z]+|[A-Za-z_]*\d[A-Za-z0-9_]*")
_NN = re.compile(r"(?:<N>){2,}")
_ERR_WORD = re.compile(r"error|exception|fail|timeout|timed out|refused|unavailable|denied|fatal|panic|oom|crash",
                       re.I)


_BOILER = [re.compile(p) for p in (r"^<TS>\s*", r"^\[[^\]]*\]\s*", r"\btrace_?[iI]d:?\s*<HEX>\s*", r"\bspan_?[iI]d:?\s*<HEX>\s*",
                                  r"\s\[,\]", r"\s{2,}")]


def _show(tpl: str) -> str:
    """Display form of a masked template: boilerplate (timestamp, thread, trace/span ids) removed, cut to TPL_CHARS;
    if the error word lies beyond the cut, the cut is moved to start shortly before it. Display only."""
    t = tpl
    for rx in _BOILER:
        t = rx.sub(" " if rx.pattern == r"\s{2,}" else "", t)
    t = t.strip()
    if len(t) <= TPL_CHARS:
        return t
    m = _ERR_WORD.search(t)
    if m and m.start() > TPL_CHARS - 30:
        a = max(0, m.start() - 50)
        return "..." + t[a:a + TPL_CHARS]
    return t[:TPL_CHARS] + "..."


def compute_log_evidence(case, alert_ts: Optional[dt.datetime], entry_entity_id: Optional[str] = None,
                         max_hops: int = 2) -> Dict:
    empty = {"text": "", "rows": 0, "stats": {}}
    try:
        return _compute(case, alert_ts, entry_entity_id, max_hops) or empty
    except Exception as e:  # evidence must never break the pipeline
        empty["error"] = repr(e)
        return empty


_EPOCH = dt.datetime(1970, 1, 1)


def _sec(t: dt.datetime) -> float:
    return (t - _EPOCH).total_seconds()   # naive UTC datetimes


def _off(sec: float, w0_sec: float) -> str:
    d = int(round(sec - w0_sec))
    sign = "-" if d < 0 else "+"
    d = abs(d)
    return f"T{sign}{d // 60}:{d % 60:02d}"


def _neighbourhood(case, entry_entity_id, max_hops):
    """Set of topology entity ids of the alert service, its callees (<= max_hops) and direct callers."""
    import networkx as nx
    from data.loader import find_service_ancestor
    from pipeline.propagation_evidence import _calls_graph, _node_type
    topo = case.topology
    if not entry_entity_id or entry_entity_id not in topo:
        return set(), None
    svc0 = find_service_ancestor(entry_entity_id, topo) or entry_entity_id
    calls = _calls_graph(topo)
    cand = {svc0}
    if svc0 in calls:
        for n in nx.single_source_shortest_path_length(calls, svc0, cutoff=max_hops):
            if _node_type(topo, n) == "apm.service":
                cand.add(n)
        for p in calls.predecessors(svc0):
            if _node_type(topo, p) == "apm.service":
                cand.add(p)
    return cand, svc0


def _compute(case, alert_ts, entry_entity_id, max_hops):
    logs = case.observations.get("logs", [])
    if not logs or alert_ts is None:
        return None
    at = _naive(alert_ts)
    w0, w1 = window_bounds(case, at)
    w0s, w1s = _sec(w0), _sec(w1)

    cache: Dict[str, str] = {}
    rows = []   # (t, svc, tpl, is_err)
    for o in logs:
        if o.timestamp is None:
            continue
        ts = _naive(o.timestamp)
        m = _HEAD.match(o.text or "")
        level, svc, msg = (m.group(1), m.group(2).strip(), m.group(3)) if m else ("", "?", o.text or "")
        if not level:
            lm = _JSON_LEVEL.search(msg)
            level = lm.group(1).lower() if lm else ""
        tpl = cache.get(msg)
        if tpl is None:
            tpl = _MIXED.sub(lambda m: "@<H>" if m.group(0).startswith("@") else "<N>", mask(msg))
            tpl = cache[msg] = _NN.sub("<N>", tpl)
        rows.append((_sec(ts), svc, tpl, level.lower() in _ERR_LEVELS or bool(_ERR_WORD.search(tpl))))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])

    # bursts: runs with gaps <= BURST_GAP_S
    bid, b = [], 0
    for i, r in enumerate(rows):
        if i and r[0] - rows[i - 1][0] > BURST_GAP_S:
            b += 1
        bid.append(b)
    n_b = b + 1
    span = {}
    for r, k in zip(rows, bid):
        s = span.setdefault(k, [r[0], r[0]])
        s[1] = r[0]
    usable = [k for k in range(n_b) if span[k][0] <= w1s]
    win = [k for k in usable if span[k][1] >= w0s]
    note = ""
    if not win:
        earlier = [k for k in usable if span[k][1] < w0s]
        if earlier:
            win = [earlier[-1]]
            note = (f" No burst intersects the evidence window; the latest earlier burst "
                    f"({_off(span[win[0]][0], w0s)}..{_off(span[win[0]][1], w0s)}) is used instead.")
    base = [k for k in usable if k not in win]
    if not win:
        return None
    win_set, base_set = set(win), set(base)

    tot = Counter()
    n_sb = Counter()          # (burst, svc) -> lines
    e_sb = Counter()
    for r, k in zip(rows, bid):
        if k in win_set or k in base_set:
            tot[k] += 1
            n_sb[(k, r[1])] += 1
            if r[3]:
                e_sb[(k, r[1])] += 1
    svcs = sorted({s for (_, s) in n_sb})
    wtot = sum(tot[k] for k in win)

    def share(k, s):
        return n_sb[(k, s)] / tot[k] if tot[k] else 0.0

    stat = {}
    for s in svcs:
        w_n = sum(n_sb[(k, s)] for k in win)
        w_e = sum(e_sb[(k, s)] for k in win)
        bs = [share(k, s) for k in base]
        b_n = sum(n_sb[(k, s)] for k in base)
        b_e = sum(e_sb[(k, s)] for k in base)
        stat[s] = {"w_share": w_n / wtot if wtot else 0.0, "w_n": w_n, "w_err": (w_e / w_n) if w_n else None,
                   "b_med": statistics.median(bs) if bs else None, "b_min": min(bs) if bs else None,
                   "b_max": max(bs) if bs else None, "b_err": (b_e / b_n) if b_n else None}

    # templates per container, window vs baseline
    tw: Dict[str, Counter] = defaultdict(Counter)
    tb: Dict[str, Counter] = defaultdict(Counter)
    twe: Dict[str, Counter] = defaultdict(Counter)
    for r, k in zip(rows, bid):
        if k in win_set:
            tw[r[1]][r[2]] += 1
            if r[3]:
                twe[r[1]][r[2]] += 1
        elif k in base_set:
            tb[r[1]][r[2]] += 1

    cand, svc0 = _neighbourhood(case, entry_entity_id, max_hops)
    neigh = []
    if cand:
        from data.loader import resolve_entity_by_name, find_service_ancestor
        idx = getattr(case, "name_index", None) or {}
        for s in svcs:
            eid = resolve_entity_by_name(s, idx) if idx else None
            if eid is not None:
                eid = find_service_ancestor(eid, case.topology) or eid
            if eid in cand:
                neigh.append(s)
    detail = neigh[:MAX_NEIGH]

    def pct(x):
        return "n/a" if x is None else (f"{100 * x:.1f}%" if x < 0.1 else f"{100 * x:.0f}%")

    def bstr(s):
        st = stat[s]
        if st["b_med"] is None:
            return "no baseline burst"
        rng = f" [{pct(st['b_min'])}-{pct(st['b_max'])}]" if len(base) > 1 else ""
        return f"base {pct(st['b_med'])}{rng}"

    def line(s):
        st = stat[s]
        return (f"{s}: {pct(st['w_share'])} of window-burst lines ({bstr(s)}); error-like {pct(st['w_err'])} "
                f"(base {pct(st['b_err'])}); {len(tw[s])} templates, new {len([t for t in tw[s] if t not in tb[s]])}, "
                f"absent {len([t for t in tb[s] if t not in tw[s]])}")

    nb = len(base)
    hdr = (f"Log evidence. Logs are exported as bursts (up to ~100,000 lines, about 1 minute every ~5 minutes), so raw "
           f"counts and per-minute rates carry no information; only each container's share of a burst does. "
           f"T = evidence-window start. Window burst(s): {len(win)} "
           f"({', '.join(f'{_off(span[k][0], w0s)}..{_off(span[k][1], w0s)}' for k in win)}); earlier bursts: {nb}.{note} "
           f"error-like = level error/fatal or an error word in the template; new = template present in the window "
           f"burst(s) but in no earlier burst; absent = the reverse. Nothing is filtered by significance.")
    out = [hdr]
    if detail:
        out.append("Call-graph neighbourhood of the alert service:")
        for s in detail:
            out.append("- " + line(s))
            new = sorted(((t, c) for t, c in tw[s].items() if t not in tb[s]), key=lambda x: (-x[1], x[0]))
            for t, c in new[:NEW_N]:
                out.append(f"    new x{c}: {_show(t)}")
            if len(new) > NEW_N:
                out.append(f"    (+{len(new) - NEW_N} more new templates)")
            if twe[s]:
                t, c = sorted(twe[s].items(), key=lambda x: (-x[1], x[0]))[0]
                bavg = tb[s].get(t, 0) / nb if nb else 0
                out.append(f"    most frequent error-like x{c} (earlier bursts: {bavg:.0f} per burst): {_show(t)}")
    rest = [s for s in svcs if s not in detail]
    if rest:
        out.append("Other containers: " + "; ".join(
            f"{s} {pct(stat[s]['w_share'])} (base {pct(stat[s]['b_med'])}), error-like {pct(stat[s]['w_err'])} "
            f"(base {pct(stat[s]['b_err'])})" for s in rest))
    text = "\n".join(out)
    return {"text": text, "rows": len(detail) + len(rest),
            "stats": {"log_bursts": n_b, "log_window_bursts": len(win), "log_neighbourhood": len(neigh),
                      "log_detail": len(detail), "log_containers": len(svcs), "chars": len(text)}}
