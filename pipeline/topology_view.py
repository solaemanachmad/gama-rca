"""Arm T-names (TOPOLOGY_NAMES=1): readable, order-neutral topology input for the topology specialist.

Audit claude/audit-top20-ranking-2026-10-12.md: the specialist got a Top-20 list of hex ids ranked by structural nearness (rank 1 = the
alert's service in 103/103 cases; 69% of the slots are operations, instances, external clients, K8s objects) plus a graph path it was told
to use as its "primary reasoning basis". With the flag the specialist sees names, an alphabetical list of the call-graph neighbourhood
services (same set as the trace/infra evidence), and the path as a plain structural fact. Evidence extraction only: no ranking by anomaly,
no ground truth.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pipeline.export_bursts import neighbourhood
from pipeline.propagation_evidence import _node_name


def build_topology_view(case, entry_entity_id: Optional[str], neighbors: Dict[str, List[str]],
                        path: Optional[List[str]]) -> Optional[Dict]:
    try:
        topo = case.topology
        nm = lambda e: _node_name(topo, e) if e in topo else str(e)
        cand, svc0 = neighbourhood(case, entry_entity_id, 2)
        services = sorted({nm(e) for e in cand})
        return {
            "alert_service": nm(svc0) if svc0 else None,
            "upstream": sorted({nm(e) for e in (neighbors or {}).get("upstream", [])}),
            "downstream": sorted({nm(e) for e in (neighbors or {}).get("downstream", [])}),
            "services": services,
            "path": [nm(e) for e in path] if path else None,
        }
    except Exception:
        return None
