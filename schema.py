"""
schema.py
=========
Canonical internal data contracts. Every loader, retriever, and agent speaks
these dataclasses instead of raw parquet/JSON, so modules stay decoupled
from RCA100's on-disk format.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import datetime as dt


@dataclass
class Entity:
    """A node in the UModel topology graph."""
    entity_id: str
    entity_type: str                 # e.g. "apm.svc", "apm.pod", "k8s.node", "apm.operation"
    name: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """An edge in the UModel topology graph."""
    source_id: str
    target_id: str
    relation_type: str                # e.g. "calls", "hosted_on", "belongs_to"


@dataclass
class AlertContext:
    """Parsed content of task.json — the ONLY input the framework may see."""
    case_id: str
    alert_text: str
    alert_timestamp: Optional[dt.datetime]
    entry_entity_id: Optional[str]    # None for the 13 composite/no-alert-entity cases
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """A single normalized record from any modality (metric point, log line,
    span, event, or alert row)."""
    entity_id: Optional[str]
    timestamp: Optional[dt.datetime]
    modality: str                     # "metrics" | "logs" | "traces" | "events" | "alerts"
    text: str                         # human/LLM-readable rendering of the row
    payload: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""


@dataclass
class EvidenceItem:
    """A ranked piece of evidence surfaced by hybrid retrieval, ready for
    the Evidence Summarizer."""
    observation: Observation
    graph_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass
class AgentFinding:
    """Structured output of one specialist agent (Metrics / Logs / Trace / Topology)."""
    agent_name: str
    entity_id: Optional[str]
    summary: str
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RCAResult:
    """Final output of the Coordinator + LLM stage."""
    case_id: str
    predicted_entity_ids: List[str]
    predicted_fault_type: str
    reasoning_chain: List[str]
    confidence: float
    agent_findings: List[AgentFinding] = field(default_factory=list)
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    evidence_items: List[EvidenceItem] = field(default_factory=list)   # for evaluation.py's
                                                                        # retrieval_precision_recall
                                                                        # and checkpoint scoring

    def summary(self) -> str:
        """Human-readable view of ONLY what the LLM actually produced
        (predictions + reasoning + agent findings) plus a few key retrieval
        stats -- leaves out the full evidence_items dump (often 100+ raw
        observations, retrieved BEFORE the LLM was ever called, kept on this
        object for evaluation.py / debugging, not something the LLM wrote).

        Usage: print(result.summary())  -- print(result) still shows
        everything, unchanged, for when you actually need the raw evidence.
        """
        lines = [
            f"=== RCAResult: {self.case_id} ===",
            f"predicted_entity_ids : {self.predicted_entity_ids}",
            f"predicted_fault_type : {self.predicted_fault_type}",
            f"confidence           : {self.confidence:.2f}",
            "reasoning_chain:",
        ]
        for i, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"  {i}. {step}")

        lines.append("\nagent_findings:")
        for f in self.agent_findings:
            entity = f.entity_id or "(none)"
            summary_text = f.summary or "(empty)"
            lines.append(f"  [{f.agent_name}] entity={entity} confidence={f.confidence:.2f}")
            lines.append(f"    {summary_text}")
            if f.supporting_evidence:
                lines.append(f"    evidence: {f.supporting_evidence}")

        stats = self.retrieval_stats
        if stats:
            lines.append("\nkey stats:")
            for k in ("evidence_items_retrieved", "indexed_observations",
                      "total_calls", "total_tokens", "total_pipeline_time_s",
                      "used_entity_fallback"):
                if k in stats:
                    v = stats[k]
                    if isinstance(v, float):
                        v = round(v, 2)
                    lines.append(f"  {k}: {v}")

        lines.append(f"\n(evidence_items omitted -- {len(self.evidence_items)} raw items "
                      f"retrieved before the LLM was called; print(result) shows them all)")
        return "\n".join(lines)