"""
zero_shot_matching.py
========================
Zero-shot fault-type matching: compares case evidence text against each
candidate type's own DEFINITION text (from data.taxonomy.FAULT_DEFINITIONS),
not against labeled example cases or any per-case ground-truth signal. This
works even for fault types with ZERO training examples, and -- unlike the
tier-1/tier-2 RandomForest classifiers and the case-based-reasoning module
(all removed; they were trained/tuned on RCA100's own ground-truth labels,
which is a form of answer-key leakage even under leave-one-out
cross-validation) -- it needs nothing but the public taxonomy's own
descriptions, which exist for all 28 RCA100 fault types independent of any
labeled case.

Deliberately DIFFERENT from the earlier removed embedding-based "shortlist"
mechanism (fault_shortlist_prompt_block, which compared evidence against
all 28 definitions BEFORE the LLM saw evidence, and correlated with ~5x
worse fault_identification -- the correct type was frequently excluded
from the narrowed top-5 before the LLM ever got to reason about it). This
version differs in two ways that matter:
  1. Used as an ADDITIONAL soft hint the Coordinator can weigh alongside
     the full 28-type taxonomy, never as an exclusionary filter -- nothing
     gets narrowed away before the Coordinator reasons over it.
  2. No dependency on any GT-derived group hint. `candidate_group` is
     optional: when given (e.g. from a leakage-free source such as the
     structural graph anchor's own guess), matching is scoped to that
     group's definitions for a sharper comparison; when omitted, matching
     searches across all 28 type definitions directly, which is the
     default mode now that the GT-trained group classifier and the
     GT-tuned TWIST narrowing table have both been removed.

Grounded in zero-shot anomaly detection literature (e.g. CLIP/WinCLIP-style
zero-/few-shot classification: embed class descriptions, embed the query,
rank by cosine similarity -- no labeled examples of the target class
required).
"""

from typing import Optional, Tuple

import config

# Cache key is the candidate_group string, or the sentinel below when
# matching across the full unscoped taxonomy.
_ALL_TYPES_KEY = "__all__"
_DEFINITION_EMBEDDING_CACHE = {}  # cache_key -> (candidate_type_list, embedding_matrix)


def _get_definition_embeddings(candidate_group: Optional[str] = None):
    """Definition embeddings are static (don't depend on the case), so
    compute them once per group (or once for the full taxonomy) per
    process and reuse -- avoids re-embedding the same definition texts on
    every single case."""
    cache_key = candidate_group if candidate_group else _ALL_TYPES_KEY
    if cache_key in _DEFINITION_EMBEDDING_CACHE:
        return _DEFINITION_EMBEDDING_CACHE[cache_key]

    from data.taxonomy import build_fault_taxonomy, FAULT_DEFINITIONS, fault_group
    from retrieval.vector import get_embedding_model

    taxonomy = build_fault_taxonomy()
    if candidate_group:
        candidates = [t for t in taxonomy if fault_group(t) == candidate_group]
    else:
        candidates = list(taxonomy)  # full 28-type taxonomy, unscoped

    if not candidates:
        _DEFINITION_EMBEDDING_CACHE[cache_key] = (None, None)
        return None, None

    definition_texts = [FAULT_DEFINITIONS.get(t, t) for t in candidates]
    model = get_embedding_model(config.EMBEDDING_MODEL)
    def_embeddings = model.encode(definition_texts)

    _DEFINITION_EMBEDDING_CACHE[cache_key] = (candidates, def_embeddings)
    return candidates, def_embeddings


def zero_shot_type_match(evidence_text: str, candidate_group: Optional[str] = None,
                          min_similarity: float = 0.3) -> Optional[Tuple[str, float]]:
    """Returns (matched_fault_type, similarity) if the scoped evidence text
    is similar enough to one of the candidate fault-type definitions, else
    None.

    candidate_group: optional. If given, matching is scoped to just that
    group's type definitions (sharper comparison, fewer distractors). If
    None (the default), matching searches across all 28 fault-type
    definitions directly -- this is the normal mode now that there is no
    leakage-free source of a group hint upstream.

    evidence_text: should be evidence SCOPED to the suspected root-cause
    entity/entities (e.g. the same top-3-entity scoping used for keyword
    detection in pipeline.py), not the full unscoped evidence pool --
    comparing a huge diluted text blob against short definitions produces
    weak, unreliable similarity scores."""
    if not evidence_text or not evidence_text.strip():
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


def zero_shot_type_match_topk(evidence_text: str, candidate_group: Optional[str] = None,
                               k: int = 3, min_similarity: float = 0.2):
    """Same matching as zero_shot_type_match(), but returns up to k
    (fault_type, similarity) pairs sorted by similarity descending, instead
    of just the single best. Now that this module is the ONLY structured
    fault-type signal reaching the Coordinator (the GT-trained classifier,
    CBR, and TWIST-narrowing sources were all removed for benchmark-leakage
    reasons), a ranked shortlist gives the Coordinator LLM more to weigh
    than a single forced guess, without narrowing anything away -- the full
    taxonomy is still shown alongside it.

    min_similarity defaults lower here (0.2 vs. 0.3) than the single-best
    matcher: a top-k list is presented as a soft ranking for the
    Coordinator to weigh, not a single asserted answer, so a slightly wider
    net is appropriate."""
    if not evidence_text or not evidence_text.strip():
        return []

    candidates, def_embeddings = _get_definition_embeddings(candidate_group)
    if candidates is None:
        return []

    from retrieval.vector import get_embedding_model
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = get_embedding_model(config.EMBEDDING_MODEL)
    evidence_embedding = model.encode([evidence_text])

    sims = cosine_similarity(evidence_embedding, def_embeddings)[0]
    order = np.argsort(sims)[::-1]

    results = []
    for idx in order[:k]:
        sim = float(sims[idx])
        if sim < min_similarity:
            break
        results.append((candidates[idx], round(sim, 4)))
    return results
