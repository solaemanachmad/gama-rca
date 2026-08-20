"""
case_based_reasoning.py
==========================
Case-Based Reasoning (CBR) fallback for fault types with only 1 training
example each -- these cannot be learned statistically by the RandomForest
tier-2 classifier (a single example provides zero learnable decision
boundary), so instead of trying to CLASSIFY them, we directly COMPARE a
new case's structured feature vector against every historical case via
1-nearest-neighbor, and surface a hint specifically when the closest match
happens to be one of these singleton types.

Motivated by MicroCBR (Liu, F., Wang, Y., Li, Z., Ren, R., Guan, H., Yu, X.,
Chen, X., Xie, G. "MicroCBR: Case-Based Reasoning on Spatio-temporal Fault
Knowledge Graph for Microservices Troubleshooting." ICCBR 2022, 224-239),
which uses exactly this case-based-reasoning paradigm -- "extracts a
spatio-temporal knowledge graph with only one sample for each fault" -- for
microservice fault diagnosis where per-fault-type history is too sparse
for a trained statistical model. This is a complement to, not a
replacement for, the RandomForest tier-2 classifier: for well-populated
types the classifier already works reasonably (LOOCV showed real signal
there); this module specifically fills the gap for the ~13 types the
classifier structurally cannot learn from.

Reuses the SAME training_X/training_y/training_case_ids already saved in
fault_type_classifier.pkl for leave-one-out purposes -- no separate data
storage needed.
"""

import os
from typing import Dict, Optional, Tuple

import config


def find_similar_case(stats: Dict, case_id: Optional[str] = None,
                       min_similarity: float = 0.8) -> Optional[Tuple[str, float, str]]:
    """Returns (suggested_fault_type, similarity, matched_case_id) if the
    single nearest-neighbor training case (by cosine similarity on the
    standardized feature vector) belongs to a SINGLETON fault type (only 1
    example in the whole training set) and similarity clears min_similarity.
    Returns None otherwise -- including when the nearest neighbor is a
    well-populated type, since the classifier already handles those.

    case_id: excludes this case from the comparison pool if it was part of
    training (leave-one-out correctness, same principle as the classifier
    fixes) -- always pass this in production use."""
    path = os.path.join(config.WORK_DIR, "fault_type_classifier.pkl")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        bundle = joblib.load(path)
    except Exception as e:
        print(f"[case_based_reasoning] failed to load {path}: {e}")
        return None

    if "training_X" not in bundle or "training_y" not in bundle:
        return None

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    from scripts.train_fault_group_classifier import FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS

    X_train = bundle["training_X"].reset_index(drop=True)
    y_train = bundle["training_y"].reset_index(drop=True)
    case_ids_raw = bundle.get("training_case_ids")
    case_ids_train = list(case_ids_raw) if case_ids_raw is not None else [None] * len(X_train)
    trained_columns = bundle["feature_columns"]

    # Which fault types have exactly 1 training example -- these are the
    # ONLY ones this module should ever suggest (well-populated types
    # already get real signal from the classifier).
    type_counts = y_train.value_counts()
    singleton_types = set(type_counts[type_counts == 1].index)
    if not singleton_types:
        return None

    # Leave-one-out: exclude the current case from the comparison pool if
    # it was part of training.
    if case_id is not None:
        keep_mask = [cid != case_id for cid in case_ids_train]
        X_train = X_train[keep_mask].reset_index(drop=True)
        y_train = y_train[keep_mask].reset_index(drop=True)
        case_ids_train = [cid for cid, k in zip(case_ids_train, keep_mask) if k]

    if len(X_train) == 0:
        return None

    # Build the new case's feature vector, same construction as the
    # classifiers use.
    numeric = {c: stats.get(c, 0) for c in FEATURE_COLUMNS}
    categorical_raw = {c: stats.get(c, "unknown") or "unknown" for c in CATEGORICAL_FEATURE_COLUMNS}
    row = dict(numeric)
    for c, v in categorical_raw.items():
        dummy_col = f"{c}_{v}"
        if dummy_col in trained_columns:
            row[dummy_col] = 1
    x_new = pd.DataFrame([{col: row.get(col, 0) for col in trained_columns}])

    # Standardize before distance computation -- numeric features have very
    # different scales (e.g. observations_before_graph_filter ~1e6 vs trend
    # counts ~10), which would otherwise dominate the similarity score.
    combined = pd.concat([X_train[trained_columns], x_new[trained_columns]], ignore_index=True)
    scaled = StandardScaler().fit_transform(combined.fillna(0))
    x_new_scaled = scaled[-1:]
    X_train_scaled = scaled[:-1]

    similarities = cosine_similarity(x_new_scaled, X_train_scaled)[0]
    best_idx = int(np.argmax(similarities))
    best_similarity = float(similarities[best_idx])
    best_type = y_train.iloc[best_idx]
    best_case_id = case_ids_train[best_idx]

    if best_type not in singleton_types:
        return None  # nearest neighbor is a well-populated type -- classifier already covers this
    if best_similarity < min_similarity:
        return None  # not close enough to trust

    return best_type, round(best_similarity, 4), best_case_id
