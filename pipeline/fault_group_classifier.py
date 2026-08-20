"""
fault_group_classifier.py
============================
Live-inference side of the fault-group classifier: loads the model
persisted by scripts/train_fault_group_classifier.py and predicts a
fault_group for a single in-progress case, from whatever `stats` fields
are available BEFORE the Coordinator LLM call (see pipeline.py's
FEATURE_COLUMNS comment for why the feature set is restricted this way).

Used to narrow the Coordinator's taxonomy prompt to just the predicted
group's fault types (see agents/multi_agent.py's coordinator_node) --
motivated by the session finding that the classifier's fault_group
accuracy (0.66-0.68, LOOCV) is stable across runs where the LLM's own
end-to-end fault_group accuracy swung between 0.39 and 0.44 depending on
unrelated pipeline changes. See findings_session_20260728.md Section 4.

LEAKAGE FIX: the saved model is fit on ALL cases in whatever CSV trained
it. If the "final" evaluation run scores the SAME 103 cases that trained
the classifier, using the pre-fit model directly is data leakage -- it has
already seen that exact case's ground-truth label. Fixed by retraining a
fresh model EXCLUDING the case being scored, every time that case is found
in the saved training set (true leave-one-out at deployment, not just at
offline LOOCV validation). Discovered when a run using the un-fixed
version scored fault_identification=1.0 (exact match) on 4/5 spot-checked
cases -- suspiciously higher than the 0.64-0.68 LOOCV accuracy the same
classifier had validated at.
"""

import os
from typing import Dict, Optional, Tuple

import config

_MODEL_CACHE = None  # {"model": ..., "feature_columns": [...], "training_X": ..., ...}
_LOO_MODEL_CACHE: Dict[str, object] = {}  # case_id -> freshly-retrained model, cached per run


def _model_path() -> str:
    return os.path.join(config.WORK_DIR, "fault_group_classifier.pkl")


_MISSING_FILE_WARNED = False


def load_classifier():
    """Returns None (not an error) if no trained classifier exists yet --
    callers should fall back to the full, unnarrowed taxonomy in that case.
    A missing classifier is an expected state (e.g. before the first
    training run on a fresh dataset), not a bug."""
    global _MODEL_CACHE, _MISSING_FILE_WARNED
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    path = _model_path()
    if not os.path.exists(path):
        if not _MISSING_FILE_WARNED:
            # Previously silent -- printed nothing at all when the file
            # simply wasn't found (as opposed to found-but-corrupt, which
            # DID print). This left no diagnostic trail for exactly the
            # failure mode that happened in practice: files believed to be
            # uploaded to Kaggle, but not actually present at the path
            # config.WORK_DIR resolves to.
            print(f"[fault_group_classifier] no classifier found at {path} -- "
                  f"running without group-level narrowing for this run. "
                  f"If you expected a trained classifier to be active, check "
                  f"that config.WORK_DIR points to where the .pkl was placed.")
            _MISSING_FILE_WARNED = True
        return None
    try:
        import joblib
        _MODEL_CACHE = joblib.load(path)
        return _MODEL_CACHE
    except Exception as e:
        print(f"[fault_group_classifier] failed to load {path}: {e} -- "
              f"falling back to no classifier for this run")
        return None


def _get_model_for_case(bundle: Dict, case_id: Optional[str]):
    """Returns the model to use for THIS case: a freshly-retrained,
    leave-one-out model if case_id was part of the training set (cached
    per case_id so repeated calls within one run don't re-retrain), or the
    pre-fit all-data model otherwise (safe for a genuinely new/unseen
    case -- no leakage risk since it was never in the training labels)."""
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


def predict_fault_group(stats: Dict, case_id: Optional[str] = None) -> Optional[Tuple[str, float]]:
    """Returns (predicted_group, confidence) or None if no classifier is
    available. `stats` is the pipeline's in-progress stats dict -- must
    contain at least the numeric FEATURE_COLUMNS from
    scripts/train_fault_group_classifier.py; missing ones default to 0
    (same fillna(0) behavior used during training).

    case_id: pass the current case's ID so leave-one-out retraining can
    kick in if this case was part of the classifier's training set (see
    module docstring). Always pass this in production use -- omitting it
    silently reintroduces the leakage this fix exists to prevent."""
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
    # One-hot encode categoricals the same way pd.get_dummies did at
    # training time -- a category not seen during training (or a value not
    # matching any trained dummy column) simply gets no column lit up,
    # which is the correct behavior (equivalent to "unknown" implicitly).
    for c, v in categorical_raw.items():
        dummy_col = f"{c}_{v}"
        if dummy_col in trained_columns:
            row[dummy_col] = 1

    # Build the row in EXACTLY the training column order, filling any
    # trained column not touched above (numeric gaps, unmatched dummies)
    # with 0.
    X = pd.DataFrame([{col: row.get(col, 0) for col in trained_columns}])

    proba = model.predict_proba(X)[0]
    classes = model.classes_
    best_idx = proba.argmax()
    return classes[best_idx], float(proba[best_idx])