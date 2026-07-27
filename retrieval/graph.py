"""
graph_retrieval.py
===================
Module 2 — Topology-aware Graph Retrieval.

Given the alert's entry entity, extract a candidate subgraph via:
  1. BFS up to GRAPH_HOP_LIMIT hops (upstream + downstream)
  2. Personalized PageRank seeded at the alert entity, to RANK entities
     within (and slightly beyond) that BFS frontier by propagation relevance.

Output is a ranked list of (entity_id, graph_score) plus the induced
NetworkX subgraph, both consumed by hybrid_retrieval.py.
"""

from typing import Dict, List, Optional, Tuple
import datetime as dt
import networkx as nx

import config


def compute_propagation_path(topology: nx.DiGraph, source_id: str, target_id: str) -> Optional[List[str]]:
    """Pure graph algorithm (nx.shortest_path, undirected) -- no LLM call,
    essentially free compared to a generation call. Finds the ACTUAL path
    in the topology between the alert's entry entity (source -- the
    symptom/impact) and a candidate root-cause entity (target -- from
    graph_anchor), returning a human-readable hop-by-hop description.

    This directly targets RCA100's cause -> propagation -> impact reasoning
    structure: the entry entity is the "impact" (what the alert fired on),
    the candidate is the "cause", and THIS PATH is the "propagation" --
    factually grounded in real topology edges, rather than an LLM inventing
    a plausible-sounding but potentially fabricated propagation narrative
    from just an upstream/downstream neighbor list."""
    if source_id not in topology or target_id not in topology:
        return None
    undirected = topology.to_undirected(as_view=True)
    try:
        path = nx.shortest_path(undirected, source_id, target_id)
    except nx.NetworkXNoPath:
        return None

    hops = []
    for node_id in path:
        entity = topology.nodes[node_id].get("entity")
        if entity is not None:
            hops.append(f"{entity.name} ({entity.entity_type})")
        else:
            hops.append(node_id)
    return hops


def apply_temporal_boost(graph_scores: Dict[str, float],
                          observations_by_modality: Dict[str, list],
                          topology: nx.DiGraph,
                          alert_timestamp: Optional[dt.datetime],
                          boost_before: float = None,
                          penalty_after_only: float = None) -> Dict[str, float]:
    """Pre-processing temporal signal (not just a text label): boosts
    graph_scores for entities whose EARLIEST evidence occurs BEFORE the
    alert trigger (candidate causes, per RCA's standard temporal-causality
    principle -- root cause precedes and propagates to the symptom), and
    discounts entities whose evidence only appears AFTER (more likely
    downstream effects). Applied to graph_scores directly, so it changes
    ranked_entities ordering and therefore which entity graph_anchor picks
    -- unlike evidence_summarizer's relative-time label, which only adds
    text for the LLM to optionally notice.

    An entity with no timestamped evidence at all is left unboosted
    (neutral), not penalized -- absence of evidence isn't evidence of
    absence, and we don't want to punish entities purely for being sparse
    in this case's telemetry."""
    if alert_timestamp is None:
        return graph_scores

    boost_before = boost_before if boost_before is not None else config.TEMPORAL_BOOST_BEFORE
    penalty_after_only = penalty_after_only if penalty_after_only is not None else config.TEMPORAL_PENALTY_AFTER_ONLY

    from data.loader import find_service_ancestor  # local import: avoid a
    # data.loader <-> retrieval.graph circular import at module load time

    at = alert_timestamp.replace(tzinfo=None) if alert_timestamp.tzinfo else alert_timestamp

    # Memoize find_service_ancestor by entity_id -- it's a graph traversal,
    # and case.observations is UNFILTERED raw data (can be 500k+ rows per
    # modality), with the same handful of entity_ids repeated across nearly
    # every row. Calling find_service_ancestor fresh per row (as an earlier
    # version of this function did) meant hundreds of thousands of redundant
    # graph traversals per case -- measured impact: pipeline time jumped
    # from ~70-90s to ~740-790s per case (10x) once this function started
    # being called. Resolving each unique entity_id once fixes this.
    ancestor_cache: Dict[str, str] = {}

    def _rolled(entity_id: str) -> str:
        if entity_id not in ancestor_cache:
            ancestor_cache[entity_id] = find_service_ancestor(entity_id, topology) or entity_id
        return ancestor_cache[entity_id]

    earliest_by_entity: Dict[str, dt.datetime] = {}
    for obs_list in observations_by_modality.values():
        for o in obs_list:
            if not o.entity_id or not o.timestamp:
                continue
            rolled = _rolled(o.entity_id)
            ts = o.timestamp.replace(tzinfo=None) if o.timestamp.tzinfo else o.timestamp
            if rolled not in earliest_by_entity or ts < earliest_by_entity[rolled]:
                earliest_by_entity[rolled] = ts

    boosted = {}
    for eid, score in graph_scores.items():
        earliest = earliest_by_entity.get(eid)
        if earliest is None:
            boosted[eid] = score  # no timestamped evidence -- leave neutral
        elif earliest < at:
            boosted[eid] = score * boost_before
        else:
            boosted[eid] = score * penalty_after_only
    return boosted


_INFRA_TYPES = {"apm.instance", "k8s.pod", "k8s.node", "k8s.cluster"}


def _entity_type(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    entity = graph.nodes[node_id].get("entity")
    return entity.entity_type if entity is not None else None


def bfs_candidate_subgraph(graph: nx.DiGraph, seed_entity: str,
                            hop_limit: int = config.GRAPH_HOP_LIMIT) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Two-phase candidate subgraph construction.

    Phase A -- shallow service-neighborhood BFS (bounded by hop_limit, kept
    small on purpose): captures the "semantic" blast radius of upstream
    callers / downstream dependencies around the alert.

    Phase B -- infra-layer search: node-level faults (CPU/memory/disk on a
    k8s.node) don't propagate along service-dependency edges, they propagate
    via shared scheduling/infra within a k8s.cluster -- but the nearest
    apm.instance/k8s.pod/k8s.node chain can be several service-hops deep
    (measured: 3-4 hops in RCA100 topologies, since only some services carry
    instance-level telemetry, and some of those instances are dead ends with
    no further pod/node edges). Rather than raising hop_limit globally
    (which pulls in every node within that radius across ALL branches and
    can balloon the subgraph to near-whole-topology size), Phase B explores
    up to INFRA_SEARCH_HOP_CAP hops across every branch as scratch space,
    but only keeps nodes that are actually infra-typed (apm.instance/
    k8s.pod/k8s.node/k8s.cluster) plus any k8s.cluster's k8s.node siblings --
    the apm.operation/apm.service nodes used purely to reach them are
    discarded, so subgraph size stays close to Phase A's.

    Returns (subgraph, boost_map) where boost_map maps each cluster-sibling
    k8s.node -> the k8s.cluster hub it was pulled in through (organic PPR
    starves these far-flung siblings to near-zero, so retrieve() uses
    boost_map to give them at least their hub's score)."""
    if seed_entity not in graph:
        return nx.DiGraph(), {}

    undirected = graph.to_undirected(as_view=True)

    # --- Phase A: shallow, bounded service-neighborhood BFS ---------------
    visited = {seed_entity}
    frontier = [seed_entity]
    for _ in range(hop_limit):
        next_frontier = []
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    # --- Phase B: infra-layer search (scratch traversal) ------------------
    # Explore up to INFRA_SEARCH_HOP_CAP hops across ALL branches (don't
    # stop at the first infra-type node found -- some apm.instance nodes
    # are dead ends with no further pod/node chain, so committing to the
    # first hit can miss a cluster hub that a different branch reaches).
    # Only infra-type nodes actually found get kept in the final subgraph;
    # every apm.operation/apm.service node used purely to reach them is
    # discarded from `search_visited`, so this doesn't reintroduce the
    # subgraph-bloat/OOM problem a blanket deep BFS caused.
    search_visited = {seed_entity}
    search_frontier = [seed_entity]
    infra_found = set()
    if _entity_type(graph, seed_entity) in _INFRA_TYPES:
        infra_found.add(seed_entity)
    for _ in range(config.INFRA_SEARCH_HOP_CAP):
        next_frontier = []
        for node in search_frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor in search_visited:
                    continue
                search_visited.add(neighbor)
                if _entity_type(graph, neighbor) in _INFRA_TYPES:
                    infra_found.add(neighbor)
                next_frontier.append(neighbor)
        search_frontier = next_frontier
        if not search_frontier:
            break

    visited |= infra_found

    boost_map: Dict[str, str] = {}
    cluster_hubs = [n for n in infra_found if _entity_type(graph, n) == "k8s.cluster"]
    for hub in cluster_hubs:
        for neighbor in undirected.neighbors(hub):
            if _entity_type(graph, neighbor) == "k8s.node":
                is_new = neighbor not in visited
                visited.add(neighbor)
                if is_new:
                    boost_map[neighbor] = hub

    return graph.subgraph(visited).copy(), boost_map


def personalized_pagerank_rank(graph: nx.DiGraph, seed_entity: str,
                                alpha: float = config.PPR_ALPHA,
                                top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Rank entities by propagation relevance from the seed alert entity.

    IMPORTANT: runs on the UNDIRECTED view of the graph, not the directed
    one. Service dependency graphs have many "dangling" nodes in the directed
    sense (e.g. a leaf pod/instance with only incoming calls, no outgoing
    edges). NetworkX's default dangling-node handling redistributes a
    dangling node's rank mass according to the personalization vector — which,
    with all our mass concentrated on the seed, means dangling mass loops
    straight back to the seed on every iteration. That starves every other
    node (including a real root cause 2+ hops away) down to ~0 score, no
    matter how relevant it structurally is. Running on the undirected view
    avoids this because every connected node has at least one "out-edge"
    (its neighbors), so there's no dangling-mass pathology to begin with —
    consistent with bfs_candidate_subgraph() already treating the graph as
    undirected for traversal."""
    if seed_entity not in graph or graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected(as_view=True)
    personalization = {n: 0.0 for n in undirected.nodes}
    personalization[seed_entity] = 1.0

    try:
        scores = nx.pagerank(undirected, alpha=alpha, personalization=personalization,
                             max_iter=200)
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(undirected, alpha=alpha, personalization=personalization,
                             max_iter=1000, tol=1e-4)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


def identify_upstream_downstream(graph: nx.DiGraph, seed_entity: str) -> Dict[str, List[str]]:
    """Split immediate neighbors into upstream (callers) vs downstream (dependencies)."""
    upstream = list(graph.predecessors(seed_entity)) if seed_entity in graph else []
    downstream = list(graph.successors(seed_entity)) if seed_entity in graph else []
    return {"upstream": upstream, "downstream": downstream}


class GraphRetriever:
    """High-level entry point used by the pipeline. Wraps BFS + PPR into a
    single call and normalizes scores to [0, 1] for hybrid fusion."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def retrieve(self, seed_entity: Optional[str]) -> Dict:
        boost_map: Dict[str, str] = {}
        if seed_entity is None or seed_entity not in self.graph:
            # Composite / no-entry-entity case: fall back to whole-graph PPR
            # seeded uniformly (equivalent to unpersonalized PageRank).
            subgraph = self.graph
            scores = nx.pagerank(self.graph) if self.graph.number_of_nodes() else {}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:config.PPR_TOP_K]
        else:
            subgraph, boost_map = bfs_candidate_subgraph(self.graph, seed_entity)
            ranked = personalized_pagerank_rank(subgraph, seed_entity)

            # Cluster-expansion siblings (e.g. a k8s.node reached only via a
            # shared k8s.cluster hub, not via service-dependency edges) get
            # diluted to near-zero by organic PPR -- give each at least its
            # hub's score so they aren't buried purely by graph distance.
            if boost_map:
                scores_by_id = dict(ranked)
                for node, hub in boost_map.items():
                    hub_score = scores_by_id.get(hub, 0.0)
                    if scores_by_id.get(node, 0.0) < hub_score:
                        scores_by_id[node] = hub_score
                ranked = sorted(scores_by_id.items(), key=lambda kv: kv[1], reverse=True)

        max_score = max([s for _, s in ranked], default=1.0) or 1.0
        normalized = {eid: (score / max_score) for eid, score in ranked}

        return {
            "subgraph": subgraph,
            "ranked_entities": ranked,           # [(entity_id, raw_ppr_score), ...]
            "graph_scores": normalized,          # {entity_id: normalized_score in [0,1]}
            "neighbors": identify_upstream_downstream(self.graph, seed_entity) if seed_entity else {},
            "boosted_entities": set(boost_map.keys()),  # cluster-expansion siblings --
                                                          # pass to graph_direct_evidence's
                                                          # force_include so they're never
                                                          # cut by the top_n_entities rank
                                                          # threshold, only by PPR score
                                                          # normally.
        }