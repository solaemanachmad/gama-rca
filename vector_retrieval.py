"""
vector_retrieval.py
====================
Module 3 — Vector Retrieval.

Embeds Observation.text for logs / metrics / traces / events and indexes
them with FAISS (in-process, zero-infra — ideal for Kaggle). A pgvector
backend is sketched at the bottom for when you move off Kaggle to a
persistent service; swap VectorIndex -> PgVectorIndex without touching
callers, since both implement the same `.add()` / `.search()` interface.
"""

from typing import Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config
from schema import Observation


class VectorIndex:
    """In-memory FAISS index, scoped to a single case (rebuilt per case —
    RCA100 cases are small enough that this costs <1s)."""

    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(embedding_model)
        self.dim = config.EMBEDDING_DIM
        self.index = faiss.IndexFlatIP(self.dim)   # cosine via normalized IP
        self.observations: List[Observation] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def add(self, observations: List[Observation]):
        if not observations:
            return
        texts = [o.text for o in observations]
        vecs = self._embed(texts)
        self.index.add(vecs)
        self.observations.extend(observations)

    def search(self, query: str, top_k: int = config.VECTOR_TOP_K) -> List[Tuple[Observation, float]]:
        if self.index.ntotal == 0:
            return []
        qvec = self._embed([query])
        scores, idxs = self.index.search(qvec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.observations[idx], float(score)))
        return results


def build_index_from_observations(observations: Dict[str, list], modalities=("logs", "metrics", "events")) -> VectorIndex:
    """Build a FAISS index from a pre-filtered observations dict (e.g. only
    observations whose entity_id falls within the graph-retrieved candidate
    subgraph). This is what makes retrieval genuinely "topology-aware": the
    vector search space is restricted by Module 2 BEFORE Module 3 runs, per
    the architecture diagram (Graph Retrieval -> Hybrid Evidence Retrieval),
    rather than searching the whole case and only re-weighting afterward."""
    index = VectorIndex()
    for m in modalities:
        index.add(observations.get(m, []))
    return index


def build_case_index(case, modalities=("logs", "metrics", "events")) -> VectorIndex:
    """Build a FAISS index over the requested modalities for the ENTIRE case
    (no topology filtering). Used by baselines that are deliberately
    topology-blind (standard_rag) — the proposed hybrid pipeline should use
    build_index_from_observations() with a graph-filtered subset instead."""
    return build_index_from_observations(case.observations, modalities)


# ---------------------------------------------------------------------------
# Optional: PostgreSQL + pgvector backend (same interface, for production use
# beyond Kaggle's ephemeral filesystem). Left as a documented stub.
# ---------------------------------------------------------------------------
class PgVectorIndex:
    """
    Sketch only — requires `psycopg2` + a running Postgres with pgvector.

    CREATE TABLE observations (
        id SERIAL PRIMARY KEY,
        case_id TEXT,
        entity_id TEXT,
        modality TEXT,
        text TEXT,
        embedding VECTOR(384)
    );
    CREATE INDEX ON observations USING ivfflat (embedding vector_cosine_ops);
    """

    def __init__(self, dsn: str, embedding_model: str = config.EMBEDDING_MODEL):
        import psycopg2  # local import: optional dependency
        self.conn = psycopg2.connect(dsn)
        self.model = SentenceTransformer(embedding_model)

    def add(self, case_id: str, observations: List[Observation]):
        vecs = self.model.encode([o.text for o in observations], convert_to_numpy=True)
        with self.conn.cursor() as cur:
            for o, v in zip(observations, vecs):
                cur.execute(
                    "INSERT INTO observations (case_id, entity_id, modality, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (case_id, o.entity_id, o.modality, o.text, v.tolist()),
                )
        self.conn.commit()

    def search(self, case_id: str, query: str, top_k: int = config.VECTOR_TOP_K):
        qvec = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT entity_id, modality, text, 1 - (embedding <=> %s::vector) AS score "
                "FROM observations WHERE case_id = %s "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (qvec, case_id, qvec, top_k),
            )
            return cur.fetchall()
