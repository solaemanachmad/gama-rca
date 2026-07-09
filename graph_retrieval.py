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
import networkx as nx

import config


def bfs_candidate_subgraph(graph: nx.DiGraph, seed_entity: str,
                            hop_limit: int = config.GRAPH_HOP_LIMIT) -> nx.DiGraph:
    """Undirected-style BFS (traverses edges in both directions) to capture
    both upstream callers and downstream dependencies of the seed entity."""
    if seed_entity not in graph:
        return nx.DiGraph()

    undirected = graph.to_undirected(as_view=True)
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

    return graph.subgraph(visited).copy()


def personalized_pagerank_rank(graph: nx.DiGraph, seed_entity: str,
                                alpha: float = config.PPR_ALPHA,
                                top_k: int = config.PPR_TOP_K) -> List[Tuple[str, float]]:
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
    return ranked[:top_k]


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
        if seed_entity is None or seed_entity not in self.graph:
            # Composite / no-entry-entity case: fall back to whole-graph PPR
            # seeded uniformly (equivalent to unpersonalized PageRank).
            subgraph = self.graph
            scores = nx.pagerank(self.graph) if self.graph.number_of_nodes() else {}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:config.PPR_TOP_K]
        else:
            subgraph = bfs_candidate_subgraph(self.graph, seed_entity)
            ranked = personalized_pagerank_rank(subgraph, seed_entity)

        max_score = max([s for _, s in ranked], default=1.0) or 1.0
        normalized = {eid: (score / max_score) for eid, score in ranked}

        return {
            "subgraph": subgraph,
            "ranked_entities": ranked,           # [(entity_id, raw_ppr_score), ...]
            "graph_scores": normalized,          # {entity_id: normalized_score in [0,1]}
            "neighbors": identify_upstream_downstream(self.graph, seed_entity) if seed_entity else {},
        }