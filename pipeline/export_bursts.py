"""Shared helpers for the burst structure of the log and trace exports (evidence extraction only).

Audit 2026-10-11 (claude/audit-logs-2026-10-11.md, claude/audit-traces-time-2026-10-11.md): both the logs and the traces
are exported as bursts of <= ~100,000 records that are dense for about a minute every ~5 minutes (grid anchored at
alert_window.start; "first 100k per slot" is an inference, not verified at the source). Raw counts and per-minute rates are
therefore export artefacts. These helpers split records into bursts by time gaps and decide which bursts represent the
evidence window and which the earlier baseline. Uniform for every case; no ground truth is used."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

BURST_GAP_S = 60.0
MIN_BURST_FRAC = 0.01   # bursts smaller than this fraction of the largest burst are export fragments (2-16 records seen)
_EPOCH = dt.datetime(1970, 1, 1)


def sec(t: dt.datetime) -> float:
    """Naive UTC datetime -> epoch seconds."""
    return (t - _EPOCH).total_seconds()


def off_label(seconds: float, w0s: float) -> str:
    d = int(round(seconds - w0s))
    sign = "-" if d < 0 else "+"
    d = abs(d)
    return f"T{sign}{d // 60}:{d % 60:02d}"


def burst_ids(times_sorted: Sequence[float], gap: float = BURST_GAP_S) -> np.ndarray:
    """Burst id (0,1,2,...) for each time of an ascending array: a new burst starts when the gap exceeds `gap`."""
    t = np.asarray(times_sorted, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(([0], np.cumsum(np.diff(t) > gap))).astype(np.int64)


def burst_spans(times_sorted: Sequence[float], ids: np.ndarray) -> Dict[int, Tuple[float, float]]:
    t = np.asarray(times_sorted, dtype=float)
    out: Dict[int, Tuple[float, float]] = {}
    if t.size == 0:
        return out
    starts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
    ends = np.concatenate((starts[1:] - 1, [t.size - 1]))
    for s, e in zip(starts, ends):
        out[int(ids[s])] = (float(t[s]), float(t[e]))
    return out


def drop_fragments(span: Dict[int, Tuple[float, float]], ids: np.ndarray, frac: float = MIN_BURST_FRAC
                   ) -> Tuple[Dict[int, Tuple[float, float]], int]:
    """Remove bursts with fewer records than `frac` x the largest burst. Returns (kept spans, number dropped)."""
    if not span:
        return span, 0
    sizes = np.bincount(ids)
    cut = frac * sizes.max()
    kept = {k: v for k, v in span.items() if sizes[k] >= cut}
    return kept, len(span) - len(kept)


def classify_bursts(span: Dict[int, Tuple[float, float]], w0s: float, w1s: float
                    ) -> Tuple[List[int], List[int], str]:
    """(window bursts, baseline bursts, note). Window bursts intersect [w0s, w1s]; if none does, the latest earlier burst
    is used (the note says so). Baseline = the other bursts that start before the window end."""
    usable = [k for k in sorted(span) if span[k][0] <= w1s]
    win = [k for k in usable if span[k][1] >= w0s]
    note = ""
    if not win:
        earlier = [k for k in usable if span[k][1] < w0s]
        if earlier:
            win = [earlier[-1]]
            note = (f" No burst intersects the evidence window; the latest earlier burst "
                    f"({off_label(span[win[0]][0], w0s)}..{off_label(span[win[0]][1], w0s)}) is used instead.")
    base = [k for k in usable if k not in win]
    return win, base, note


def neighbourhood(case, entry_entity_id: Optional[str], max_hops: int = 2) -> Tuple[Set[str], Optional[str]]:
    """Topology entity ids of the alert service, its callees (<= max_hops) and its direct callers (chosen by structure,
    same set as trace/infra evidence)."""
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


def pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if x < 0.1 else f"{100 * x:.0f}%"


def ms(x: Optional[float]) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x / 1000:.1f}s" if x >= 1000 else f"{x:.0f}ms"
