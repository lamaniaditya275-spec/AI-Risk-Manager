"""
Model training.

Uses XGBoost (falls back to LightGBM if xgboost isn't installed — see
requirements.txt) with:
  - class weighting for imbalance (fraud/abuse is rare)
  - a strict TIME-BASED split (never random split — that leaks future
    patterns into training for a fraud problem)
  - probability calibration (isotonic) so the score is a genuine probability,
    not just a ranking — this matters for the cost model in evaluate.py
  - an ablation ladder: rules-only vs ML-only vs ML+graph features, so you
    can show judges *why* each component earns its place
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

try:
    import xgboost as xgb
    BACKEND = "xgboost"
except ImportError:
    import lightgbm as lgb
    BACKEND = "lightgbm"

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

try:
    # sklearn >= 1.6: 'prefit' was removed from CalibratedClassifierCV(cv=...);
    # wrap the already-fitted estimator in FrozenEstimator instead.
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

from config import RANDOM_SEED, MODEL_VERSION
from features import get_model_feature_columns


@dataclass
class TrainedModel:
    model: object
    calibrated_model: object
    anomaly_model: object
    feature_columns: list
    model_version: str
    backend: str


def time_based_split(df: pd.DataFrame, train_frac=0.66, val_frac=0.17):
    """
    Split by order_timestamp, NOT randomly. Returns (train_df, val_df, test_df).
    train_frac/val_frac define the split points; remainder goes to test.
    """
    df = df.sort_values("order_timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def _make_base_model(scale_pos_weight: float):
    if BACKEND == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_SEED,
        )
    else:
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Optional[list] = None,
    label_col: str = "label_fraud",
) -> TrainedModel:
    """
    Train the supervised model + calibration + anomaly detector.
    train_df/val_df must already have build_features() applied.
    """
    if feature_columns is None:
        feature_columns = get_model_feature_columns(train_df)

    X_train = train_df[feature_columns].fillna(-1)
    y_train = train_df[label_col].astype(int)
    X_val = val_df[feature_columns].fillna(-1)
    y_val = val_df[label_col].astype(int)

    n_pos = max(y_train.sum(), 1)
    n_neg = max(len(y_train) - y_train.sum(), 1)
    scale_pos_weight = n_neg / n_pos

    base_model = _make_base_model(scale_pos_weight)
    base_model.fit(X_train, y_train)

    val_scores = base_model.predict_proba(X_val)[:, 1]
    val_ap = average_precision_score(y_val, val_scores) if y_val.sum() > 0 else float("nan")
    print(f"[model] backend={BACKEND} validation PR-AUC={val_ap:.4f}")

    # Calibrate on validation set so scores are genuine probabilities
    if _HAS_FROZEN_ESTIMATOR:
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="isotonic")
        calibrated_model.fit(X_val, y_val)
    else:
        calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
        calibrated_model.fit(X_val, y_val)

    # Anomaly detector trained on the (mostly legitimate) training set —
    # catches patterns the supervised model has never seen labeled.
    anomaly_model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=RANDOM_SEED
    )
    anomaly_model.fit(X_train)

    return TrainedModel(
        model=base_model,
        calibrated_model=calibrated_model,
        anomaly_model=anomaly_model,
        feature_columns=feature_columns,
        model_version=MODEL_VERSION,
        backend=BACKEND,
    )


def score_orders(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a feature-engineered dataframe. Returns df with added columns:
    risk_score (calibrated probability), anomaly_score (higher = more anomalous).
    """
    X = df[trained.feature_columns].fillna(-1)
    df = df.copy()
    df["risk_score"] = trained.calibrated_model.predict_proba(X)[:, 1]

    raw_anomaly = trained.anomaly_model.decision_function(X)  # higher = more normal
    # flip and min-max normalize to [0,1] so higher = more anomalous
    anomaly = -raw_anomaly
    df["anomaly_score"] = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)

    return df


def run_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame, label_col: str = "label_fraud"):
    """
    Compares: (a) rules-only recall/precision, (b) ML without graph-lite
    features, (c) full ML with graph-lite features. Prints a small table —
    this is the evidence you show judges that each component earns its place.
    """
    from rules import evaluate_rules

    print("\n=== Ablation: Rules-only ===")
    val_df = val_df.copy()
    val_df["rules_flag"] = val_df.apply(lambda r: len(evaluate_rules(r)) > 0, axis=1)
    y_true = val_df[label_col].astype(int)
    y_pred = val_df["rules_flag"].astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"Rules-only: precision={precision:.3f} recall={recall:.3f} (tp={tp}, fp={fp}, fn={fn})")

    all_features = get_model_feature_columns(train_df)
    graph_features = [c for c in all_features if "distinct_" in c or "prior" in c]
    non_graph_features = [c for c in all_features if c not in graph_features]

    print("\n=== Ablation: ML without graph-lite features ===")
    m1 = train_model(train_df, val_df, feature_columns=non_graph_features)
    scored1 = score_orders(m1, val_df)
    ap1 = average_precision_score(val_df[label_col], scored1["risk_score"])
    print(f"PR-AUC (no graph features) = {ap1:.4f}")

    print("\n=== Ablation: ML + graph-lite features (full model) ===")
    m2 = train_model(train_df, val_df, feature_columns=all_features)
    scored2 = score_orders(m2, val_df)
    ap2 = average_precision_score(val_df[label_col], scored2["risk_score"])
    print(f"PR-AUC (with graph features) = {ap2:.4f}")

    print(f"\nGraph features changed PR-AUC by {ap2 - ap1:+.4f}")
    return {"rules_precision": precision, "rules_recall": recall, "ap_no_graph": ap1, "ap_with_graph": ap2}


if __name__ == "__main__":
    print("This module is a library. See train.py for the end-to-end training script.")
