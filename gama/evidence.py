"""Stage 1 -- calibrated evidence: what changed in the incident window, per entity, against the case's own history.

Each continuous signal (request, error, latency, CPU, memory, GC, threads, disk ...) is scored by the size of its shift against the
same signal before the window, re-standardised against the peers of its kind (common-mode rejection) and against the entity's own
pseudo-windows (volatile entities) -- pipeline/fault.calibrate. Categorical evidence: Kubernetes events whose rate in the window
exceeds their own long-run rate (Poisson test), availability below desired, HTTP / gRPC status and messages of failing spans, log text
families. Every observation is binary (holds / absent) with a calibrated rate (P1: calibrate, do not threshold).

Modules: M2 trace adds the trace structure (pipeline/trace_structure.py); M5 text is read when the signal database is built
(pipeline/fault_calibrated.compute_signal_db(text=...)).
"""
from __future__ import annotations

from typing import List

from pipeline import belief_diagnosis as bd
from pipeline.fault_calibrated import compute_signal_db
from pipeline.fault import calibrate as calibrate_v4, new_events, unavailability, B_SPAN
from gama.config import Modules


def signal_db(case, alert_ts, m: Modules):
    return compute_signal_db(case, alert_ts, text=m.text)


def observations(db, callees, case, m: Modules, model) -> List[dict]:
    if case is not None:
        new_events(case, db, B_SPAN * (db["w1"] - db["w0"]).total_seconds())
    calibrate_v4(db, peer=True, robust_null=False, ops_level=False, selfnorm=True, ratio=False, callees=callees)
    obs = bd.observations(db)
    for s_, bad in unavailability(db).items():
        obs.append({"entity": s_, "signal": "svc_unavailable", "holds": bad, "kind": "cat",
                    "ev": "deployment available replicas below desired / availability below full in the window" if bad else ""})
    if m.trace and case is not None:
        from pipeline.trace_structure import trace_observations, origin_nodes
        obs += trace_observations(case, db, v2=True, omsg=True, v3=True, msgtarget=True)
        model.origin_nodes = origin_nodes(obs, db.get("hosts", {}))
    return obs
