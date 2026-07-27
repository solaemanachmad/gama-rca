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

# Features usable at real inference time -- no ground-truth-derived columns.
FEATURE_COLUMNS = [
    "candidate_subgraph_size",
    "propagation_path_hops",
    "graph_anchor_time_s",
    "vector_index_build_time_s",
    "hybrid_retrieval_time_s",
    "multi_agent_time_s",
    "evidence_items_retrieved",
    "evidence_items_from_vector",
    "evidence_items_from_graph_direct",
    "indexed_observations",
    "observations_before_graph_filter",
    "total_tokens",
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
# Categorical features handled separately (one-hot): the LLM's own guesses,
# which the classifier can learn to trust or override per pattern.
CATEGORICAL_FEATURE_COLUMNS = ["graph_anchor_fault_type", "predicted_fault_type"]

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
    y = df[TARGET_COLUMN].reset_index(drop=True)
    return X, y


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


if __name__ == "__main__":
    paths = sys.argv[1:] or ["graphrag_rca_work/results/results.csv"]
    df = load_and_merge(paths)
    print(f"Loaded {len(df)} rows from {len(paths)} file(s)")

    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"No {TARGET_COLUMN} column found -- re-run main.py with the "
                          f"current evaluation/scoring.py to get this column.")

    X, y = build_features(df)
    ok = check_feasibility(y)

    if len(y) < 10:
        print("\nToo few examples to run LOOCV meaningfully (<10). Collect more "
              "cases first (aim for the full 103, or at least ~40-50 with all "
              "6 groups represented) before training.")
        sys.exit(0)

    run_loocv(X, y)
    train_final_model(X, y)