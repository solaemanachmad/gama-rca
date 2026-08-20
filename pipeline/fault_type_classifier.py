"""
fault_type_classifier.py
===========================
Live-inference side of the fine-grained (28-type) fault classifier. Unlike
the group-level classifier (pipeline/fault_group_classifier.py), this one's
raw flat-28-way predict_proba is masked down to only the types belonging to
the group ALREADY predicted (by the group classifier or otherwise) before
picking a winner.

Why masking, not just training a 28-way classifier directly: several
fine-grained types have only 1-2 examples in the full 103-case corpus (see
Table 4 in the RCA100 paper -- e.g. nodeDown=1, nodeMemoryOOM=1,
cacheBreakdown=1), making a flat 28-way classifier's raw predictions for
those types close to meaningless on their own. But once the group is
already known with ~0.64-0.68 accuracy (from the group classifier), the
remaining problem is choosing among just the handful of types IN that
group -- a much easier discrimination task, since all irrelevant
candidates outside the group are already ruled out regardless of how
little data exists for each individual type.

LEAKAGE FIX: same issue and same fix as fault_group_classifier.py -- the
saved model is fit on ALL cases in whatever CSV trained it, so scoring one
of those same cases with the un-modified model is data leakage. Always
pass `case_id` to predict_fault_type() so leave-one-out retraining kicks
in when the case was part of the training set.
"""

import os
from typing import Dict, Optional, Tuple

import config
from data.taxonomy import fault_group

_MODEL_CACHE = None
_LOO_MODEL_CACHE: Dict[str, object] = {}


def _model_path() -> str:
    return os.path.join(config.WORK_DIR, "fault_type_classifier.pkl")


_MISSING_FILE_WARNED = False


def load_classifier():
    """Returns None (not an error) if no trained classifier exists yet --
    callers should fall back to letting the Coordinator decide freely
    within the group in that case."""
    global _MODEL_CACHE, _MISSING_FILE_WARNED
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    path = _model_path()
    if not os.path.exists(path):
        if not _MISSING_FILE_WARNED:
            print(f"[fault_type_classifier] no classifier found at {path} -- "
                  f"running without type-level narrowing for this run. "
                  f"If you expected a trained classifier to be active, check "
                  f"that config.WORK_DIR points to where the .pkl was placed.")
            _MISSING_FILE_WARNED = True
        return None
    try:
        import joblib
        _MODEL_CACHE = joblib.load(path)
        return _MODEL_CACHE
    except Exception as e:
        print(f"[fault_type_classifier] failed to load {path}: {e} -- "
              f"falling back to no type-level classifier for this run")
        return None


def _get_model_for_case(bundle: Dict, case_id: Optional[str]):
    """Same leave-one-out logic as fault_group_classifier.py's
    _get_model_for_case() -- see that module's docstring for why this is
    necessary (the saved model is fit on ALL training cases; using it
    un-modified to score one of those same cases is data leakage)."""
    training_case_ids = bundle.get("training_case_ids")
    if case_id is None or training_case_ids is None:
        return bundle["model"]

    training_case_ids = list(training_case_ids)
    if case_id not in training_case_ids:
        return bundle["model"]

    if case_id in _LOO_MODEL_CACHE:
        return _LOO_MODEL_CACHE[case_id]

    from sklearn.ensemble import RandomForestClassifier
    X_train = bundle["training_X"]
    y_train = bundle["training_y"]
    mask = [cid != case_id for cid in training_case_ids]
    X_loo = X_train[mask]
    y_loo = y_train[mask]
    loo_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    loo_model.fit(X_loo, y_loo)
    _LOO_MODEL_CACHE[case_id] = loo_model
    return loo_model


def predict_fault_type(stats: Dict, known_group: Optional[str],
                        case_id: Optional[str] = None) -> Optional[Tuple[str, float]]:
    """Returns (predicted_fault_type, confidence) or None if no classifier
    is available or known_group is None. Predictions are masked to only
    the fault types belonging to `known_group` -- see module docstring for
    why this matters given the sparse per-type training data.

    case_id: pass the current case's ID so leave-one-out retraining kicks
    in if this case was part of the classifier's training set. Always pass
    this in production use -- see fault_group_classifier.py's module
    docstring for why omitting it reintroduces data leakage.

    confidence here is the RENORMALIZED probability among just the
    in-group candidates, not the raw (much lower, diluted-by-27-other-
    classes) predict_proba value -- this is the honest confidence for the
    decision actually being made (which type within this group), not for
    the strictly harder unmasked 28-way problem."""
    if not known_group:
        return None
    bundle = load_classifier()
    if bundle is None:
        return None

    import pandas as pd
    from scripts.train_fault_group_classifier import FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS

    model = _get_model_for_case(bundle, case_id)
    trained_columns = bundle["feature_columns"]

    numeric = {c: stats.get(c, 0) for c in FEATURE_COLUMNS}
    categorical_raw = {c: stats.get(c, "unknown") or "unknown" for c in CATEGORICAL_FEATURE_COLUMNS}
    row = dict(numeric)
    for c, v in categorical_raw.items():
        dummy_col = f"{c}_{v}"
        if dummy_col in trained_columns:
            row[dummy_col] = 1
    X = pd.DataFrame([{col: row.get(col, 0) for col in trained_columns}])

    proba = model.predict_proba(X)[0]
    classes = model.classes_

    # Mask to only classes whose group matches known_group.
    in_group_mask = [fault_group(c) == known_group for c in classes]
    if not any(in_group_mask):
        # No trained class belongs to this group at all (e.g. the group
        # classifier predicted a group with zero training examples for any
        # of its types) -- nothing sensible to return.
        return None

    masked_proba = {c: p for c, p, in_grp in zip(classes, proba, in_group_mask) if in_grp}
    total = sum(masked_proba.values())
    if total <= 0:
        return None
    best_type = max(masked_proba, key=masked_proba.get)
    renormalized_confidence = masked_proba[best_type] / total
    return best_type, float(renormalized_confidence)