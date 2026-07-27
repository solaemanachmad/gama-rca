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

from typing import Dict, List, Optional, Tuple
import atexit
import hashlib
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config
from schema import Observation

_EMBEDDING_MODEL = None   # type: Optional[object]  -- SentenceTransformer or HFAPIEmbedder

# ---------------------------------------------------------------------------
# Persistent embedding cache
# ---------------------------------------------------------------------------
# Keyed by sha256(backend :: model_name :: text) -> np.ndarray. Unlike
# get_embedding_model()'s singleton (which only avoids reloading the MODEL
# within one process), this cache persists ACROSS separate `python main.py`
# invocations -- re-running the same case later reuses embeddings computed
# in a previous run instead of recomputing them, which matters a lot given
# how often the same case gets re-run during iterative debugging/tuning.
_EMBEDDING_CACHE: Optional[Dict[str, np.ndarray]] = None
_EMBEDDING_CACHE_NEW_ENTRIES = 0
_EMBEDDING_CACHE_FLUSH_EVERY = 500   # auto-save every N new entries, so a crash
                                      # mid-run loses at most this many, not everything


def _cache_path() -> str:
    return os.path.join(config.INDEX_DIR, "embedding_cache.pkl")


def _load_cache() -> Dict[str, np.ndarray]:
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is None:
        path = _cache_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    _EMBEDDING_CACHE = pickle.load(f)
            except Exception:
                _EMBEDDING_CACHE = {}   # corrupt/partial file -- start fresh rather than crash
        else:
            _EMBEDDING_CACHE = {}
    return _EMBEDDING_CACHE


def _save_cache() -> None:
    if _EMBEDDING_CACHE is not None:
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        with open(_cache_path(), "wb") as f:
            pickle.dump(_EMBEDDING_CACHE, f)


atexit.register(_save_cache)   # flush whatever's left when the process exits normally


def _cache_key(text: str) -> str:
    raw = f"{config.EMBEDDING_BACKEND}::{config.EMBEDDING_MODEL}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clear_embedding_cache() -> None:
    """Call this after changing EMBEDDING_MODEL/backend behavior in a way the
    cache key doesn't already capture, or if you suspect the cache is stale/
    corrupt. Normal model/backend switches are already safe without this --
    the cache key includes both, so switching backends just means cache
    misses for the new backend, not incorrect hits."""
    global _EMBEDDING_CACHE, _EMBEDDING_CACHE_NEW_ENTRIES
    _EMBEDDING_CACHE = {}
    _EMBEDDING_CACHE_NEW_ENTRIES = 0
    path = _cache_path()
    if os.path.exists(path):
        os.remove(path)


class HFAPIEmbedder:
    """Offloads embedding to Hugging Face's hosted Inference API instead of
    running SentenceTransformer locally. Same .encode() interface as
    SentenceTransformer so VectorIndex doesn't need to know which backend
    it's talking to.

    Trade-off to be explicit about: this does ONE HTTP round-trip per text
    (the feature-extraction task doesn't reliably batch multiple sentences
    server-side across all hosted models), so for the volumes this pipeline
    handles (thousands of observations per case, even after
    MAX_OBSERVATIONS_PER_INDEX capping) this can be SLOWER wall-clock than
    local encoding, not faster -- unlike the LLM-reasoning step, where one
    API call replaces one local generation call 1:1. Use this when you
    specifically want to avoid local compute/memory (e.g. a low-RAM
    machine), not as a default speed optimization.

    Some hosted feature-extraction models return per-TOKEN embeddings
    (shape [seq_len, dim]) rather than one pooled sentence vector -- this
    does mean-pooling client-side in that case, matching what
    SentenceTransformer does internally for the same model."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 api_key: Optional[str] = None, provider: str = config.HF_API_PROVIDER):
        from huggingface_hub import InferenceClient
        api_key = api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_BACKEND='hf_api' requires an HF_TOKEN (env var or .env file). "
                "Get one at https://huggingface.co/settings/tokens (Read role is enough)."
            )
        self.client = InferenceClient(provider=provider, api_key=api_key)
        self.model_name = model_name.removeprefix("sentence-transformers/") \
            if model_name.startswith("sentence-transformers/") else model_name
        # feature-extraction wants the bare model id most of the time; try
        # both forms below since providers vary in what they accept.
        self._full_model_name = model_name

    def _encode_one(self, text: str) -> np.ndarray:
        for name in (self._full_model_name, self.model_name):
            try:
                result = self.client.feature_extraction(text, model=name)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f"HF Inference API feature_extraction failed for model "
                                f"'{self._full_model_name}' -- check that this model is "
                                f"available via provider='{config.HF_API_PROVIDER}'.")
        arr = np.asarray(result, dtype="float32")
        if arr.ndim == 2:          # per-token embeddings -> mean-pool, like ST does internally
            arr = arr.mean(axis=0)
        return arr

    def encode(self, texts: List[str], convert_to_numpy: bool = True,
               show_progress_bar: bool = False, batch_size: int = 256) -> np.ndarray:
        vecs = [self._encode_one(t) for t in texts]
        return np.vstack(vecs)


def get_embedding_model(model_name: str = config.EMBEDDING_MODEL):
    """Process-wide singleton. Loading SentenceTransformer from disk takes a
    non-trivial fraction of a second even when cached locally (tokenizer +
    weights + device placement), and previously happened once per
    VectorIndex() call -- i.e. once per case per system, ~500+ times across
    a full 103-case x 5-system run. taxonomy.py already used this pattern
    for its own embedder; this consolidates both into ONE shared model
    instance so it's loaded exactly once per process, not twice.

    Backend picked via config.EMBEDDING_BACKEND ("local" default, or
    "hf_api" to offload to Hugging Face's Inference API instead)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        if config.EMBEDDING_BACKEND == "hf_api":
            _EMBEDDING_MODEL = HFAPIEmbedder(model_name)
        else:
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


class VectorIndex:
    """In-memory FAISS index, scoped to a single case (rebuilt per case —
    RCA100 cases are small enough that this costs <1s once the embedding
    model itself is cached via get_embedding_model())."""

    def __init__(self, embedding_model: str = config.EMBEDDING_MODEL,
                 max_observations: int = config.MAX_OBSERVATIONS_PER_INDEX):
        self.model = get_embedding_model(embedding_model)
        self.dim = config.EMBEDDING_DIM
        self.index = faiss.IndexFlatIP(self.dim)   # cosine via normalized IP
        self.observations: List[Observation] = []
        self.max_observations = max_observations

    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        # batch_size=256 (vs. sentence-transformers' default of 32) cuts
        # Python-loop/dispatch overhead substantially on both CPU and GPU --
        # this matters here specifically because a single VectorIndex.add()
        # call can carry tens of thousands of texts (topology-blind full-case
        # indexes routinely hit 500k+ observations; see max_observations cap
        # below, which bounds this from the other direction).
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False,
                                  batch_size=256)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not config.EMBEDDING_CACHE_ENABLED or not texts:
            return self._embed_raw(texts)

        global _EMBEDDING_CACHE_NEW_ENTRIES
        cache = _load_cache()
        keys = [_cache_key(t) for t in texts]
        miss_positions = [i for i, k in enumerate(keys) if k not in cache]

        if miss_positions:
            miss_vecs = self._embed_raw([texts[i] for i in miss_positions])
            for pos, vec in zip(miss_positions, miss_vecs):
                cache[keys[pos]] = vec
            _EMBEDDING_CACHE_NEW_ENTRIES += len(miss_positions)
            if _EMBEDDING_CACHE_NEW_ENTRIES >= _EMBEDDING_CACHE_FLUSH_EVERY:
                _save_cache()
                _EMBEDDING_CACHE_NEW_ENTRIES = 0

        return np.stack([cache[k] for k in keys]).astype("float32")

    def add(self, observations: List[Observation]):
        if not observations:
            return
        remaining_capacity = self.max_observations - len(self.observations)
        if remaining_capacity <= 0:
            return
        if len(observations) > remaining_capacity:
            # Universal safety cap: this used to be implemented ad hoc
            # per-caller (e.g. pipeline.py's MAX_RESOLVED_PER_MODALITY),
            # which meant build_case_index() -- used by the topology-BLIND
            # standard_rag baseline and by smoke_test.py -- had NO cap at
            # all and would embed an entire case's raw observations
            # (measured: 692,604 texts for a single case) on every call.
            # Enforcing it here means every caller benefits uniformly.
            # Prioritize the most recent observations relative to the alert
            # window, consistent with pipeline.py's own truncation logic.
            observations = sorted(observations, key=lambda o: o.timestamp or 0,
                                   reverse=True)[:remaining_capacity]
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