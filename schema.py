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
