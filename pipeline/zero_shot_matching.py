"""
zero_shot_matching.py
========================
Zero-shot fault-type matching: compares case evidence text against each
candidate type's own DEFINITION text (from data.taxonomy.FAULT_DEFINITIONS),
not against labeled example cases. This works even for fault types with
ZERO training examples -- unlike the tier-2 RandomForest classifier (needs
several examples to learn a decision boundary) or Case-Based Reasoning
(needs >=1 labeled example to compare against), zero-shot matching only
needs the type's own textual definition, which exists for all 28 RCA100
fault types regardless of how many (if any) labeled cases exist for them.

Deliberately DIFFERENT from the earlier removed embedding-based "shortlist"
mechanism (fault_shortlist_prompt_block, which compared evidence against
all 28 definitions BEFORE the LLM saw evidence, and correlated with ~5x
worse fault_identification -- the correct type was frequently excluded
from the narrowed top-5 before the LLM ever got to reason about it). This
version differs in two ways that matter:
  1. SCOPED to only the group already identified by the tier-1 classifier
     (typically 3-7 candidates, not all 28) -- a much more precise
     candidate set to compare against.
  2. Used as an ADDITIONAL soft hint alongside the classifier/CBR hints,
     never as an exclusionary filter -- nothing gets narrowed away before
     the Coordinator reasons over it.

Grounded in zero-shot anomaly detection literature (e.g. CLIP/WinCLIP-style
zero-/few-shot classification: embed class descriptions, embed the query,
rank by cosine similarity -- no labeled examples of the target class
required).
"""

from typing import Optional, Tuple

import config

_DEFINITION_EMBEDDING_CACHE = {}  # group_name -> (candidate_type_list, embedding_matrix)


def _get_definition_embeddings(candidate_group: str):
    """Definition embeddings are static (don't depend on the case), so
    compute them once per group per process and reuse -- avoids
    re-embedding the same ~3-7 definition texts on every single case."""
    if candidate_group in _DEFINITION_EMBEDDING_CACHE:
        return _DEFINITION_EMBEDDING_CACHE[candidate_group]

    from data.taxonomy import build_fault_taxonomy, FAULT_DEFINITIONS, fault_group
    from retrieval.vector import get_embedding_model

    taxonomy = build_fault_taxonomy()
    candidates = [t for t in taxonomy if fault_group(t) == candidate_group]
    if not candidates:
        _DEFINITION_EMBEDDING_CACHE[candidate_group] = (None, None)
        return None, None

    definition_texts = [FAULT_DEFINITIONS.get(t, t) for t in candidates]
    model = get_embedding_model(config.EMBEDDING_MODEL)
    def_embeddings = model.encode(definition_texts)

    _DEFINITION_EMBEDDING_CACHE[candidate_group] = (candidates, def_embeddings)
    return candidates, def_embeddings


def zero_shot_type_match(evidence_text: str, candidate_group: Optional[str],
                          min_similarity: float = 0.3) -> Optional[Tuple[str, float]]:
    """Returns (matched_fault_type, similarity) if the scoped evidence text
    is similar enough to one of candidate_group's fault-type definitions,
    else None. candidate_group should be the tier-1 classifier's predicted
    group (or graph_anchor's fault_type group as a fallback) -- this
    function does NOT search across all 28 types, only within the group
    already identified, per the module docstring's rationale.

    evidence_text: should be evidence SCOPED to the suspected root-cause
    entity/entities (e.g. the same top-3-entity scoping used for keyword
    detection in pipeline.py), not the full unscoped evidence pool --
    comparing a huge diluted text blob against short definitions produces
    weak, unreliable similarity scores."""
    if not candidate_group or not evidence_text or not evidence_text.strip():
        return None

    candidates, def_embeddings = _get_definition_embeddings(candidate_group)
    if candidates is None:
        return None

    from retrieval.vector import get_embedding_model
    from sklearn.metrics.pairwise import cosine_similarity

    model = get_embedding_model(config.EMBEDDING_MODEL)
    evidence_embedding = model.encode([evidence_text])

    sims = cosine_similarity(evidence_embedding, def_embeddings)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])

    if best_sim < min_similarity:
        return None
    return candidates[best_idx], round(best_sim, 4)
