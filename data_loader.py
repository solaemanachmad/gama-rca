"""
data_loader.py
===============
Stage 0 of the pipeline: turns the raw 7-file RCA100 case folder into a
single in-memory `Case` object with a NetworkX topology graph, an
`AlertContext`, and normalized `Observation` lists per modality.

SCHEMA NOTE (confirmed from real t001 data, not guessed):
-----------------------------------------------------------
- task.json: alert_entity.{entity_id,entity_name,entity_type,entity_domain}
  is frequently NULL (as in t001). The real entity attribution lives inside
  prompt_text as an embedded <alert_event ... entity_id="..." entity_name="..."
  entity_type="..." /> tag. We parse both, preferring the structured field
  and falling back to the embedded tag via regex.
- topology.json: top-level key is "entities" (list of {id, type, name, props}).
  The edges/relations key name is NOT yet confirmed (topology.json was large
  enough that the diagnostic dump got truncated before reaching it) — see
  the TOPOLOGY_EDGE_KEY_CANDIDATES probing below and run diagnose_schema.py
  again with a higher char cap / grep for the edges key if BFS returns
  suspiciously few neighbors.
- metrics.parquet: entity_id is a DIRECT column, but is EMPTY STRING for
  k8s-domain/node-level rows (aggregate node metrics have no single entity).
  time is int64 MICROSECONDS since epoch (not ms, not ns).
- logs.parquet: NO entity_id column at all. Entity attribution must go
  through _container_name_ (service name string, e.g. "inventory") resolved
  against topology entity names. Timestamp is the ISO-with-offset _time_
  column. Severity/exception info lives inside the free-text `content` field
  (Java/Spring-style log lines) and must be regex-extracted.
- traces.parquet: entity via serviceName (string, resolved by name against
  topology). startTime/endTime/duration are NANOSECOND epoch/duration
  strings. statusCode is OTel numeric-as-string: "0"=UNSET, "1"=OK, "2"=ERROR.
- events.parquet: the column literally named `eventId` actually contains the
  FULL raw Kubernetes Event object as a JSON string (reason, message, type,
  firstTimestamp, lastTimestamp are nested inside it, not top-level columns).
  Entity attribution goes through pod_name resolved against topology.
- alerts.parquet: entity info is nested inside the `resource` JSON string
  (resource.entity.entity_id/entity_type/domain). timestamp is a
  MILLISECOND epoch string; `time` is also available as ISO-with-offset.
"""

from collections import defaultdict
import os
import re
import json
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd
import networkx as nx

import config
from schema import Entity, Relation, AlertContext, Observation

TOPOLOGY_ENTITY_KEY = "entities"
# Confirmed real key: "edges", with fields src/src_type/dst/dst_type/relation.
# Other candidates kept as fallback in case a different topology export variant is used.
TOPOLOGY_EDGE_KEY_CANDIDATES = ["edges", "relations", "links", "calls", "dependencies"]

ALERT_EVENT_TAG_RE = re.compile(r"<alert_event\b([^>]*)/?>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

LOG_LEVEL_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|Exception)\b")


def _first_valid(*candidates: Any) -> Optional[str]:
    """Null-coalescing that actually treats pandas NaN as empty, unlike
    Python's `a or b` (which returns `a` unchanged when `a` is NaN, since
    float('nan') is truthy). Returns the first candidate that is a non-empty
    string; None if none qualify."""
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c
    return None


def _parse_json_field(value: Any) -> Dict[str, Any]:
    """Many RCA100 columns store a JSON object serialized as a string
    (resources, attributes, resource, labels, annotations, eventId)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_ts_iso(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True).to_pydatetime()
    except Exception:
        return None


def _parse_ts_epoch(value: Any, unit: str) -> Optional[dt.datetime]:
    """unit in {'s', 'ms', 'us', 'ns'}."""
    if value in (None, "", "NaN"):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    divisor = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]
    try:
        return dt.datetime.utcfromtimestamp(v / divisor)
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def load_topology(case_dir: str) -> nx.DiGraph:
    path = os.path.join(case_dir, config.FILE_TOPOLOGY)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    g = nx.DiGraph()

    for n in raw.get(TOPOLOGY_ENTITY_KEY, []):
        eid = n.get("id")
        if eid is None:
            continue
        g.add_node(
            eid,
            entity=Entity(entity_id=eid, entity_type=n.get("type", "unknown"),
                           name=n.get("name"), attributes=n.get("props", {})),
        )

    edges_raw = None
    used_key = None
    for key in TOPOLOGY_EDGE_KEY_CANDIDATES:
        if key in raw and isinstance(raw[key], list):
            edges_raw = raw[key]
            used_key = key
            break

    if edges_raw is None:
        # No confirmed edge key yet — degrade gracefully to a node-only graph
        # rather than crashing. Graph retrieval will fall back to whole-graph
        # PageRank in this case. Run diagnose_schema.py and check topology.json
        # keys beyond "entities" to find the real edge key, then add it to
        # TOPOLOGY_EDGE_KEY_CANDIDATES above.
        return g

    for e in edges_raw:
        src = e.get("src") or e.get("source") or e.get("from") or e.get("source_id") or e.get("caller")
        dst = e.get("dst") or e.get("target") or e.get("to") or e.get("target_id") or e.get("callee")
        rtype = e.get("relation") or e.get("relation_type") or e.get("type") or "related_to"
        if src is not None and dst is not None:
            g.add_edge(src, dst, relation=Relation(src, dst, rtype))

    return g


def build_name_index(graph: nx.DiGraph) -> Dict[str, str]:
    """Maps lowercase service/pod name -> topology entity_id, so modalities
    that only carry a name (traces.serviceName, logs._container_name_,
    events.pod_name) can be resolved to a UModel entity ID. Falls back to
    substring containment for operation-level names like
    "checkout::/oteldemo.CheckoutService/PlaceOrder" vs ground-truth's
    shorthand "checkout::PlaceOrder"."""
    index = {}
    for node_id, data in graph.nodes(data=True):
        entity: Entity = data.get("entity")
        if entity is None:
            continue
        if entity.name:
            index[entity.name.lower()] = node_id
        pod_name = entity.attributes.get("pod_name") or entity.attributes.get("service")
        if pod_name:
            index.setdefault(str(pod_name).lower(), node_id)
    return index


def find_service_ancestor(entity_id: str, topology: nx.DiGraph, max_hops: int = 3) -> Optional[str]:
    """Rolls up a fine-grained entity (e.g. apm.instance/apm.operation/k8s.pod)
    to its nearest apm.service ancestor in the topology, via BFS on the
    undirected graph. Needed because telemetry (metrics/traces/logs) often
    tags entity_id at instance/pod granularity, while ground truth
    target_entity_ids are given at service granularity -- without this
    rollup, retrieval_precision/recall comparisons silently fail to match
    even when the retrieved evidence genuinely belongs to the right service."""
    if entity_id not in topology:
        return None
    entity_data = topology.nodes[entity_id].get("entity")
    if entity_data and entity_data.entity_type == "apm.service":
        return entity_id

    undirected = topology.to_undirected(as_view=True)
    visited = {entity_id}
    frontier = [entity_id]
    for _ in range(max_hops):
        next_frontier = []
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                neighbor_data = topology.nodes[neighbor].get("entity")
                if neighbor_data and neighbor_data.entity_type == "apm.service":
                    return neighbor
                next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return None


def build_service_membership_index(topology: nx.DiGraph) -> Dict[str, List[str]]:
    """Maps each apm.service entity_id -> list of entity_ids (instances,
    operations, pods, and the service itself) that roll up to it via
    find_service_ancestor(). Needed because raw telemetry (metrics/traces/
    logs) is almost always tagged at instance/operation granularity, never
    at the service ID itself -- so when a service ranks highly by graph
    score, looking for observations with entity_id == that exact service ID
    finds nothing. This index lets graph_direct_evidence look up all of a
    service's finer-grained children instead."""
    membership = defaultdict(list)
    for eid in topology.nodes:
        service_id = find_service_ancestor(eid, topology) or eid
        membership[service_id].append(eid)
    return membership


def resolve_entity_by_name(name: Optional[Any], name_index: Dict[str, str]) -> Optional[str]:
    """Resolves a plain service/pod name (e.g. "checkout") or a composite
    "service::operation" name (e.g. "checkout::PlaceOrder", as used in
    ground-truth target_entities) against the topology name index. Composite
    names are matched by requiring BOTH the service part and the operation
    part to appear as substrings, since the real topology operation name is
    longer/fully-qualified (e.g. "checkout::/oteldemo.CheckoutService/PlaceOrder").

    Defensively rejects non-string input: pandas represents missing values
    in object/str columns as float('nan'), and `nan or other_value` still
    evaluates to nan (nan is truthy in Python) — so callers doing
    `row.get("service") or row.get("entity_name")` can hand this function
    a float nan instead of None. Guard here once instead of at every call site."""
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.lower()

    if key in name_index:
        return name_index[key]

    if "::" in key:
        service_part, _, operation_part = key.partition("::")
        service_part = service_part.strip()
        operation_part = operation_part.strip().lstrip("/").split("/")[-1]  # last path segment
        for known_name, node_id in name_index.items():
            if service_part in known_name and operation_part in known_name:
                return node_id
        # fall back to service-only match if the operation part can't be pinned
        if service_part in name_index:
            return name_index[service_part]

    for known_name, node_id in name_index.items():
        if known_name in key or key in known_name:
            return node_id
    return None


# ---------------------------------------------------------------------------
# Alert / task contract
# ---------------------------------------------------------------------------
def load_alert_context(case_dir: str, case_id: str) -> AlertContext:
    path = os.path.join(case_dir, config.FILE_TASK)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    alert_text = raw.get("alert_title", "")

    entity = raw.get("alert_entity") or {}
    entity_id = entity.get("entity_id")
    entity_name = entity.get("entity_name")
    entity_type = entity.get("entity_type")

    if entity_id is None:
        # fall back to the embedded <alert_event .../> tag inside prompt_text
        m = ALERT_EVENT_TAG_RE.search(raw.get("prompt_text", ""))
        if m:
            attrs = dict(ATTR_RE.findall(m.group(1)))
            entity_id = attrs.get("entity_id")
            entity_name = attrs.get("entity_name")
            entity_type = attrs.get("entity_type")

    ts = None
    window = raw.get("alert_window") or {}
    if window.get("start"):
        ts = _parse_ts_iso(window["start"])

    return AlertContext(
        case_id=case_id,
        alert_text=str(alert_text),
        alert_timestamp=ts,
        entry_entity_id=entity_id,
        raw={**raw, "_resolved_entity_name": entity_name, "_resolved_entity_type": entity_type},
    )


# ---------------------------------------------------------------------------
# Modality loaders -> normalized Observation lists
# ---------------------------------------------------------------------------
def load_metrics(case_dir: str, name_index: Dict[str, str]) -> List[Observation]:
    fp = os.path.join(case_dir, config.FILE_METRICS)
    df = pd.read_parquet(fp)
    obs = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        eid = _first_valid(r.get("entity_id"))
        if not eid:
            eid = resolve_entity_by_name(_first_valid(r.get("service"), r.get("entity_name")), name_index)
        ts = _parse_ts_epoch(r.get("time"), "us")
        entity_label = _first_valid(r.get("entity_name"), r.get("service")) or ""
        text = f"[metric] {entity_label} {r.get('metric')}={r.get('value')}"
        obs.append(Observation(entity_id=eid, timestamp=ts, modality="metrics",
                                text=text, payload=r, source_file=fp))
    return obs


def load_logs(case_dir: str, name_index: Dict[str, str]) -> List[Observation]:
    fp = os.path.join(case_dir, config.FILE_LOGS)
    df = pd.read_parquet(fp)
    obs = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        service_name = r.get("_container_name_")
        eid = resolve_entity_by_name(service_name, name_index)
        ts = _parse_ts_iso(r.get("_time_"))
        content = _first_valid(r.get("content")) or ""
        level_match = LOG_LEVEL_RE.search(content)
        level = level_match.group(1) if level_match else ""
        text = f"[log:{level}] {service_name}: {content}"
        obs.append(Observation(entity_id=eid, timestamp=ts, modality="logs",
                                text=text, payload=r, source_file=fp))
    return obs


def load_traces(case_dir: str, name_index: Dict[str, str]) -> List[Observation]:
    fp = os.path.join(case_dir, config.FILE_TRACES)
    df = pd.read_parquet(fp)
    obs = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        eid = resolve_entity_by_name(r.get("serviceName"), name_index)
        ts = _parse_ts_epoch(r.get("startTime"), "ns")
        try:
            duration_ms = float(r.get("duration", 0)) / 1e6
        except (TypeError, ValueError):
            duration_ms = None
        status = r.get("statusCode")
        status_label = {"0": "UNSET", "1": "OK", "2": "ERROR"}.get(status, status)
        text = (f"[span] {r.get('serviceName')}.{r.get('spanName')} "
                f"duration={duration_ms}ms status={status_label}")
        if status_label == "ERROR" and r.get("statusMessage"):
            text += f" msg={r.get('statusMessage')}"
        obs.append(Observation(entity_id=eid, timestamp=ts, modality="traces",
                                text=text, payload=r, source_file=fp))
    return obs


def load_events(case_dir: str, name_index: Dict[str, str]) -> List[Observation]:
    fp = os.path.join(case_dir, config.FILE_EVENTS)
    df = pd.read_parquet(fp)
    obs = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        # NOTE: `eventId` column actually holds the full raw K8s Event JSON.
        k8s_event = _parse_json_field(r.get("eventId"))
        reason = k8s_event.get("reason", "")
        message = k8s_event.get("message", "")
        ts = _parse_ts_iso(k8s_event.get("lastTimestamp") or k8s_event.get("firstTimestamp"))

        pod_name = r.get("pod_name")
        eid = resolve_entity_by_name(pod_name, name_index)
        text = f"[event:{r.get('level', '')}] {reason} — {message}"
        obs.append(Observation(entity_id=eid, timestamp=ts, modality="events",
                                text=text, payload=r, source_file=fp))
    return obs


def load_alerts(case_dir: str, name_index: Dict[str, str]) -> List[Observation]:
    fp = os.path.join(case_dir, config.FILE_ALERTS)
    df = pd.read_parquet(fp)
    obs = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        resource = _parse_json_field(r.get("resource"))
        entity = resource.get("entity", {})
        eid = entity.get("entity_id")

        annotations = _parse_json_field(r.get("annotations"))
        message = annotations.get("message", "")

        ts = _parse_ts_epoch(r.get("timestamp"), "ms") or _parse_ts_iso(r.get("time"))
        text = f"[alert:{r.get('status')}] {r.get('subject')} — {message}"
        obs.append(Observation(entity_id=eid, timestamp=ts, modality="alerts",
                                text=text, payload=r, source_file=fp))
    return obs


# ---------------------------------------------------------------------------
# Case container
# ---------------------------------------------------------------------------
class Case:
    def __init__(self, case_id: str, cases_dir: str = config.CASES_DIR):
        self.case_id = case_id
        self.case_dir = os.path.join(cases_dir, case_id)
        if not os.path.isdir(self.case_dir):
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")

        self.topology: nx.DiGraph = load_topology(self.case_dir)
        self.name_index: Dict[str, str] = build_name_index(self.topology)
        self.alert: AlertContext = load_alert_context(self.case_dir, case_id)

        # If the alert's entity wasn't a direct topology-id hit, try resolving
        # it by name (entity_type apm.operation names often aren't literal
        # topology node IDs but the operation string is).
        if self.alert.entry_entity_id not in self.topology and self.alert.raw.get("_resolved_entity_name"):
            resolved = resolve_entity_by_name(self.alert.raw["_resolved_entity_name"], self.name_index)
            if resolved:
                self.alert.entry_entity_id = resolved

        self.observations: Dict[str, List[Observation]] = {
            "metrics": load_metrics(self.case_dir, self.name_index),
            "logs": load_logs(self.case_dir, self.name_index),
            "traces": load_traces(self.case_dir, self.name_index),
            "events": load_events(self.case_dir, self.name_index),
            "alerts": load_alerts(self.case_dir, self.name_index),
        }

        self._validate_entity_coverage()

    def _validate_entity_coverage(self):
        topo_ids = set(self.topology.nodes)
        self.unresolved_entities = set()
        for obs_list in self.observations.values():
            for o in obs_list:
                if o.entity_id and o.entity_id not in topo_ids:
                    self.unresolved_entities.add(o.entity_id)

    def all_observations(self) -> List[Observation]:
        out = []
        for lst in self.observations.values():
            out.extend(lst)
        return out

    def observations_in_window(self, center: dt.datetime,
                                minutes: int = config.TIME_WINDOW_MINUTES) -> List[Observation]:
        delta = dt.timedelta(minutes=minutes)
        lo, hi = center - delta, center + delta
        return [o for o in self.all_observations()
                if o.timestamp is not None and lo <= o.timestamp <= hi]

    def __repr__(self):
        n_resolved = sum(1 for lst in self.observations.values() for o in lst if o.entity_id)
        n_total = sum(len(lst) for lst in self.observations.values())
        return (f"<Case {self.case_id}: {self.topology.number_of_nodes()} entities, "
                f"{n_total} observations ({n_resolved} entity-resolved), "
                f"{len(self.unresolved_entities)} unresolved entity IDs>")


def list_case_ids() -> List[str]:
    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def inspect_case(case_id: str) -> None:
    case_dir = os.path.join(config.CASES_DIR, case_id)
    with open(os.path.join(case_dir, config.FILE_TASK), "r", encoding="utf-8") as f:
        print("task.json keys:", list(json.load(f).keys()))
    with open(os.path.join(case_dir, config.FILE_TOPOLOGY), "r", encoding="utf-8") as f:
        topo = json.load(f)
        print("topology.json top-level keys:", list(topo.keys()))
    for fname in [config.FILE_METRICS, config.FILE_LOGS, config.FILE_TRACES,
                  config.FILE_EVENTS, config.FILE_ALERTS]:
        df = pd.read_parquet(os.path.join(case_dir, fname)).head(1)
        print(f"{fname} columns:", list(df.columns))