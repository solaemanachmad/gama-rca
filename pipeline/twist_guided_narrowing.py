"""
twist_guided_narrowing.py
==========================
Pure, GT-free helper kept from the earlier TWIST-guided fault-group
narrowing experiment.

The module used to also contain `narrow_fault_group()`, a routing function
that mapped TWIST anomaly profiles (c1/c2/c3/c4) and metric-signal presence
to a predicted fault group. It was removed: its `_EXCLUSIVE_SIGNAL_GROUPS`
table and `_C1_ORIGIN_THRESHOLD` constant were hand-derived by inspecting
RCA100's OWN ground truth across the full 103-case corpus (see the removed
docstrings' "GT analysis (103-case corpus)" and "5-case spot analysis"
notes) -- not held out, not cross-validated, just read off the answer key.
That makes any prediction it produced benchmark-specific curve-fitting
rather than genuine transferable reasoning, so it was dropped for the same
reason as `fault_group_classifier.py`, `fault_type_classifier.py`, and
`case_based_reasoning.py`.

What remains here, `extract_metric_signals()`, only reads signal NAMES out
of a case's own metric observations (e.g. "node_cpu_usage_rate") -- no
RCA100 label is referenced anywhere in it, so it stays available for
whatever downstream use finds the raw signal set useful (e.g. as
descriptive context, or future analysis) without reintroducing leakage.

The TWIST c1-c4 SCORES themselves are computed in twist_scoring.py, which
was independently confirmed clean (pure statistics from each case's own
trace data) and continues to be surfaced to the Coordinator as raw
informational context -- see pipeline.py's TWIST scoring block.
"""

from typing import List, Set


def extract_metric_signals(observations: List) -> Set[str]:
    """Extract all unique metric signal names from metric observations.
    Pure structural extraction -- no ground-truth dependency."""
    signals = set()
    for o in observations:
        if o.modality == "metrics" and o.payload:
            sig = o.payload.get("metric") or o.payload.get("signal")
            if sig:
                signals.add(sig)
    return signals
