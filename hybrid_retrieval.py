"""
hybrid_retrieval.py
=====================
Module 4 — Hybrid Retrieval.

HybridScore(evidence) = ALPHA * GraphScore(entity) + BETA * VectorScore(text)

GraphScore comes from personalized-PageRank over the topology (graph_retrieval.py),
normalized to [0,1] and attached to every observation via its entity_id.
VectorScore comes from FAISS cosine similarity (vector_retrieval.py), already
in [0,1] after L2-normalized inner product.

Observations whose entity never appears in the graph-ranked set (e.g. an
unresolved entity, config.unresolved_entities) fall back to VectorScore only,
with graph_score = 0 — they are not discarded, since it is 98.06% not 100%
of GT entities that resolve, and false-negative pruning is worse than a
lower-ranked but retrievable item.
"""

from collections import defaultdict
from typing import Dict, List, Optional
import config
from schema import EvidenceItem, Observation


def fuse_scores(graph_scores: Dict[str, float],
                 vector_hits: List[tuple],  # List[(Observation, vector_score)]
                 alpha: float = config.HYBRID_ALPHA,
                 beta: float = config.HYBRID_BETA) -> List[EvidenceItem]:
    fused = []
    for obs, vscore in vector_hits:
        gscore = graph_scores.get(obs.entity_id, 0.0) if obs.entity_id else 0.0
        hybrid = alpha * gscore + beta * vscore
        fused.append(EvidenceItem(
            observation=obs,
            graph_score=gscore,
            vector_score=vscore,
            hybrid_score=hybrid,
        ))
    fused.sort(key=lambda e: e.hybrid_score, reverse=True)
    return fused


class HybridRetriever:
    """Combines a GraphRetriever result with a VectorIndex search, per query."""

    def __init__(self, graph_result: Dict, vector_index):
        self.graph_scores = graph_result["graph_scores"]
        self.vector_index = vector_index

    def retrieve(self, query: str, top_k: int = config.VECTOR_TOP_K) -> List[EvidenceItem]:
        vector_hits = self.vector_index.search(query, top_k=top_k)
        return fuse_scores(self.graph_scores, vector_hits)[:top_k]

    def retrieve_multi(self, queries: List[str], top_k: int = config.VECTOR_TOP_K) -> List[EvidenceItem]:
        """Run several queries (e.g. one per specialist agent) and merge,
        deduplicating by observation identity. Uses (source_file, entity_id,
        timestamp, text) rather than (source_file, text) alone, so rows that
        legitimately share identical text (repeated log lines, or blank text
        from a field-mapping issue) are not silently collapsed into one."""
        seen = set()
        merged: List[EvidenceItem] = []
        for q in queries:
            for item in self.retrieve(q, top_k=top_k):
                o = item.observation
                key = (o.source_file, o.entity_id, str(o.timestamp), o.text)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        merged.sort(key=lambda e: e.hybrid_score, reverse=True)
        return merged


def _index_by_entity(observations_by_modality: Dict[str, List[Observation]]) -> Dict[str, List[Observation]]:
    index = defaultdict(list)
    for obs_list in observations_by_modality.values():
        for o in obs_list:
            if o.entity_id:
                index[o.entity_id].append(o)
    return index


def graph_direct_evidence(observations_by_modality: Dict[str, List[Observation]],
                           graph_scores: Dict[str, float],
                           top_n_entities: int = 10, max_per_entity: int = 6,
                           service_membership: Optional[Dict[str, List[str]]] = None) -> List[EvidenceItem]:
    """Guarantees representation for the top-N graph-central entities even if
    their observations never surface in the vector search's top-k semantic
    hits. This matters specifically for a root cause 2-3 hops upstream of the
    alert entity (e.g. "payment" when the alert only mentions "checkout") —
    the alert text has no semantic reason to rank payment's observations
    highly, so pure vector search can miss it regardless of how good the
    embedding model is. These items get vector_score=0 (they weren't found
    via similarity — they're included for structural relevance), so their
    hybrid_score is purely `HYBRID_ALPHA * graph_score`.

    IMPORTANT: raw telemetry is tagged at instance/operation/pod granularity,
    almost never at the service-level entity ID itself. If a service ranks
    highly by graph score but service_membership isn't provided, looking up
    entity_index[service_id] directly will almost always come back empty.
    Pass build_service_membership_index(topology) as service_membership so
    this expands the lookup to the service's actual instance/operation
    children, where the telemetry really lives."""
    entity_index = _index_by_entity(observations_by_modality)
    top_entities = sorted(graph_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n_entities]

    items = []
    for entity_id, gscore in top_entities:
        member_ids = service_membership.get(entity_id, [entity_id]) if service_membership else [entity_id]
        count = 0
        for member_id in member_ids:
            for o in entity_index.get(member_id, []):
                if count >= max_per_entity:
                    break
                items.append(EvidenceItem(observation=o, graph_score=gscore, vector_score=0.0,
                                           hybrid_score=config.HYBRID_ALPHA * gscore))
                count += 1
            if count >= max_per_entity:
                break
    return items


def merge_evidence(*evidence_lists: List[EvidenceItem]) -> List[EvidenceItem]:
    """Merges multiple EvidenceItem lists (e.g. vector-based hybrid hits +
    graph-direct hits), deduplicating by observation identity and keeping
    the highest-scoring instance of each, sorted by hybrid_score descending."""
    best_by_key: Dict[tuple, EvidenceItem] = {}
    for lst in evidence_lists:
        for item in lst:
            o = item.observation
            key = (o.source_file, o.entity_id, str(o.timestamp), o.text)
            if key not in best_by_key or item.hybrid_score > best_by_key[key].hybrid_score:
                best_by_key[key] = item
    return sorted(best_by_key.values(), key=lambda e: e.hybrid_score, reverse=True)