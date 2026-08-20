"""
train_fault_group_classifier.py
==================================
Trains a lightweight, non-LLM classifier (RandomForest via scikit-learn) to
predict fault_group (6-way: Application logic, JVM runtime, Cloud resource,
Middleware&DB, K8s lifecycle, Resource&perf.) from STRUCTURED features
already computed by the pipeline -- no text understanding needed, which is
exactly where the LLM has been failing (mode-collapsing to "Application
logic" regardless of evidence).

Only uses features available at real inference time (no ground-truth
leakage): graph/retrieval statistics, evidence volume, propagation path
length, and the LLM's own (unreliable) guess as one input among many --
the classifier can learn to override it. Explicitly excludes
entity_localization/retrieval_precision/retrieval_recall/fault_identification/
reasoning_process, which are only computable WITH ground truth and would
leak the answer.

Validation: leave-one-out cross-validation (LOOCV), not a single train/test
split -- RCA100 case counts per group are small enough (10-38 per group in
the full 103, likely far fewer in any partial run) that a single split
would have huge variance. LOOCV gives a more honest (if still wide-CI)
estimate.

Usage:
    python scripts/train_fault_group_classifier.py graphrag_rca_work/results/results.csv
    # Concatenate multiple results.csv runs first if you have several:
    python scripts/train_fault_group_classifier.py run1.csv run2.csv run3.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

MIN_EXAMPLES_PER_CLASS = 5   # below this, LOOCV for that class is close to
                              # meaningless (a single held-out example either
                              # is or isn't correctly classified -- no
                              # statistical signal). Warn, don't silently train.

# Features usable at real inference time, specifically BEFORE the
# Coordinator LLM call -- this classifier's live purpose is to narrow the
# Coordinator's own taxonomy prompt (see pipeline.py), so any feature only
# known AFTER the Coordinator runs (multi_agent_time_s, total_tokens) or
# that IS the Coordinator's own output (predicted_fault_type) cannot be a
# live input, even though they were fine to include in earlier offline-only
# LOOCV analysis.
FEATURE_COLUMNS = [
    "candidate_subgraph_size",
    "propagation_path_hops",
    "graph_anchor_time_s",
    "vector_index_build_time_s",
    "hybrid_retrieval_time_s",
    # NOTE (kept, after a brief removal-and-revert): these 3 timing features
    # are NOT fully reproducible run-to-run for the identical case (measured:
    # hybrid_retrieval_time_s ranged 0.17s-1.02s across 3 runs of the same
    # case, purely from system load/network jitter, not case-specific
    # signal) -- removing them was tested and measurably hurt accuracy on a
    # 10-case sample, so they were reinstated. This is a real, acknowledged
    # limitation: classifier_predicted_group/confidence for a given case is
    # not guaranteed identical across separate runs. Report this explicitly
    # in the paper's Limitations rather than implying full determinism.
    "evidence_items_retrieved",
    "evidence_items_from_vector",
    "evidence_items_from_graph_direct",
    "indexed_observations",
    "observations_before_graph_filter",
    # Semantic features (added after the first classifier run showed
    # feature importance dominated by pure volume/timing proxies) -- these
    # target actual category-discriminating signal instead. In particular,
    # evidence_count_events was added specifically because the 30-case
    # modality audit (scripts/audit_fault_groups.py) found Events
    # correlates strongly with K8s lifecycle faults (5/5 usable there,
    # near-zero elsewhere) -- exactly the kind of signal a classifier can
    # exploit that raw volume counts can't.
    "evidence_count_metrics",
    "evidence_count_logs",
    "evidence_count_traces",
    "evidence_count_events",
    "evidence_count_alerts",
    "log_error_pattern_count",
    "trend_increase_count",
    "trend_decrease_count",
    "trend_new_nonzero_count",
    "trend_dropped_zero_count",
]
CATEGORICAL_FEATURE_COLUMNS = ["graph_anchor_fault_type"]

TARGET_COLUMN = "gt_fault_group"


def load_and_merge(paths):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        n_before = len(df)
        df = df[df["error"].isna()].copy()
        if len(df) < n_before:
            print(f"Dropped {n_before - len(df)} error rows (failed pipeline runs)")
    return df


def build_features(df: pd.DataFrame):
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"\n*** NOTE: {len(missing)} feature columns not found in this CSV (likely an "
              f"older run, before these stats were added to pipeline.py): {missing} ***")
        print("Proceeding with the columns that ARE available. Re-run main.py with the "
              "current pipeline.py to get the full feature set.")
    X_numeric = df[available].fillna(0)
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURE_COLUMNS].fillna("unknown"), dummy_na=False)
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    case_ids = df["case_id"].reset_index(drop=True) if "case_id" in df.columns else pd.Series([None] * len(df))
    y = df[TARGET_COLUMN].reset_index(drop=True)
    return X, y, case_ids


def check_feasibility(y: pd.Series) -> bool:
    counts = y.value_counts()
    print("\n=== Class distribution ===")
    print(counts)
    missing_groups = set([
        "Application logic", "JVM runtime", "Cloud resource",
        "Middleware&DB", "K8s lifecycle", "Resource&perf.",
    ]) - set(counts.index)
    if missing_groups:
        print(f"\n*** WARNING: zero examples for: {sorted(missing_groups)} ***")
        print("The classifier will NEVER predict these groups no matter what --")
        print("it can only learn from classes it has seen at least once.")
    sparse = counts[counts < MIN_EXAMPLES_PER_CLASS]
    if not sparse.empty:
        print(f"\n*** WARNING: fewer than {MIN_EXAMPLES_PER_CLASS} examples for: "
              f"{dict(sparse)} ***")
        print("LOOCV results for these classes will have very wide, close-to-"
              "meaningless confidence intervals. Treat any per-class accuracy "
              "here as illustrative, not a real capability claim.")
    total_ok = len(y) >= 30 and not missing_groups
    print(f"\n{'PROCEEDING' if total_ok else 'PROCEEDING WITH CAUTION'} -- "
          f"{len(y)} total examples across {y.nunique()} classes.")
    return total_ok


def run_loocv(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score, classification_report

    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])[0]
        y_true.append(y.iloc[test_idx].values[0])
        y_pred.append(pred)

    print("\n=== Leave-one-out cross-validation results ===")
    print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.3f}  ({sum(a==b for a,b in zip(y_true,y_pred))}/{len(y_true)} correct)")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Compare against baselines: majority-class guess, and the LLM's own guess
    majority_class = y.value_counts().idxmax()
    majority_acc = (y == majority_class).mean()
    print(f"Baseline (always predict majority class '{majority_class}'): {majority_acc:.3f}")

    return y_true, y_pred


def train_final_model(X, y):
    """Fits on ALL available data (not held out) -- this is the model you'd
    actually deploy, once LOOCV above shows it's worth deploying at all."""
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Feature importances (final model, fit on all data) ===")
    print(importances.head(15))
    return clf


TARGET_COLUMN_TYPE = "gt_fault_type"


def build_features_type(df: pd.DataFrame):
    """Same feature construction as build_features(), but targeting the
    fine-grained 28-type column instead of the 6-group one."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_numeric = df[available].fillna(0)
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURE_COLUMNS].fillna("unknown"), dummy_na=False)
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    case_ids = df["case_id"].reset_index(drop=True) if "case_id" in df.columns else pd.Series([None] * len(df))
    y = df[TARGET_COLUMN_TYPE].reset_index(drop=True)
    return X, y, case_ids


def check_feasibility_type(y: pd.Series) -> None:
    """Purely informational for the 28-type target -- unlike the 6-group
    check_feasibility(), we don't gate on this, because the intended use
    (masked to the already-predicted group at inference time) makes even
    very sparse per-type classes partially useful: the model only ever has
    to discriminate among the ~3-7 types actually in a group, not all 28.
    Still worth printing so you know exactly how sparse each type is."""
    counts = y.value_counts()
    print("\n=== Fine-grained (28-type) class distribution ===")
    print(counts)
    single_example = counts[counts == 1]
    if not single_example.empty:
        print(f"\n*** NOTE: {len(single_example)} types have exactly 1 example: "
              f"{list(single_example.index)} ***")
        print("LOOCV for these types is close to meaningless in isolation (a single held-out "
              "example either is or isn't classified correctly, no real signal) -- but they still "
              "contribute at inference time via group-masking, since ruling out the OTHER "
              "types in their group is itself informative.")


def run_loocv_type(X, y):
    """Same LOOCV procedure as run_loocv(), for the 28-type target. Reports
    raw (unmasked) accuracy -- the real deployed accuracy will be higher
    than this, since live inference additionally masks to the
    already-predicted group (see pipeline/fault_type_classifier.py), which
    this raw flat-28-way LOOCV does not simulate."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score

    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = clf.predict(X.iloc[test_idx])[0]
        y_true.append(y.iloc[test_idx].values[0])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Fine-grained (28-type) LOOCV: raw flat accuracy = {acc:.3f} "
          f"({sum(a==b for a,b in zip(y_true,y_pred))}/{len(y_true)}) ===")
    print("(NOTE: this is UNMASKED accuracy across all 28 types at once -- the actual")
    print(" deployed accuracy, masked to the group already predicted by the group")
    print(" classifier, will be higher. This number is a conservative lower bound.)")
    return y_true, y_pred


def train_final_model_type(X, y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    return clf


def save_model(clf, feature_columns, X_train=None, y_train=None, case_ids_train=None, out_path=None):
    """Saves the trained model AND the exact feature column list (including
    one-hot dummy columns actually produced from training data) -- needed
    so a live prediction can build an identically-shaped feature vector.

    ALSO saves the full training set (X_train, y_train, case_ids_train) if
    given -- required for true leave-one-out at live inference time (see
    pipeline/fault_group_classifier.py's predict_fault_group()). Without
    this, the saved `clf` is fit on ALL cases in the training CSV; using it
    to score any of THOSE SAME cases (exactly what happens when the "final"
    103-case evaluation run uses a classifier trained on those same 103
    cases) is data leakage -- the model has already seen that case's
    ground-truth label during training, so predicting it back out isn't a
    real prediction. Saving the training set lets inference retrain a
    fresh model excluding the specific case being scored, each time."""
    import joblib
    import os
    import config
    out_path = out_path or os.path.join(config.WORK_DIR, "fault_group_classifier.pkl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    bundle = {"model": clf, "feature_columns": list(feature_columns)}
    if X_train is not None and y_train is not None:
        bundle["training_X"] = X_train
        bundle["training_y"] = y_train
        bundle["training_case_ids"] = case_ids_train
    joblib.dump(bundle, out_path)
    print(f"\nSaved trained classifier + feature schema to {out_path}")
    if X_train is not None:
        print(f"  (+ full training set for leave-one-out at inference time: {len(X_train)} rows)")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["graphrag_rca_work/results/results.csv"]
    df = load_and_merge(paths)
    print(f"Loaded {len(df)} rows from {len(paths)} file(s)")

    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"No {TARGET_COLUMN} column found -- re-run main.py with the "
                          f"current evaluation/scoring.py to get this column.")

    X, y, case_ids = build_features(df)
    ok = check_feasibility(y)

    if len(y) < 10:
        print("\nToo few examples to run LOOCV meaningfully (<10). Collect more "
              "cases first (aim for the full 103, or at least ~40-50 with all "
              "6 groups represented) before training.")
        sys.exit(0)

    run_loocv(X, y)
    clf = train_final_model(X, y)
    save_model(clf, X.columns, X_train=X, y_train=y, case_ids_train=case_ids)

    # --- Fine-grained (28-type) classifier, saved separately -----------------
    print(f"\n{'#' * 80}")
    print("Now training the fine-grained (28-type) classifier")
    print(f"{'#' * 80}")
    if TARGET_COLUMN_TYPE not in df.columns:
        print(f"\nNo {TARGET_COLUMN_TYPE} column found -- skipping type-level classifier "
              f"(re-run main.py with the current evaluation/scoring.py to get this column).")
    else:
        X_type, y_type, case_ids_type = build_features_type(df)
        check_feasibility_type(y_type)
        run_loocv_type(X_type, y_type)
        clf_type = train_final_model_type(X_type, y_type)
        import config
        type_out_path = os.path.join(config.WORK_DIR, "fault_type_classifier.pkl")
        save_model(clf_type, X_type.columns, X_train=X_type, y_train=y_type,
                   case_ids_train=case_ids_type, out_path=type_out_path)