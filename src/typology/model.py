"""Training and evaluation. The one thing worth being direct about upfront: with ~1% of
transactions labeled laundering, a model that predicts "legitimate" for everything scores
~99% accuracy while catching zero laundering -- accuracy is actively misleading here, not
just uninformative. Everything below reports precision/recall/PR-AUC instead, and states
the naive-baseline accuracy explicitly so nobody mistakes a high accuracy number for a
good model later.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "amount",
    "amount_to_threshold_ratio",
    "src_out_degree",
    "src_in_degree",
    "dst_out_degree",
    "dst_in_degree",
    "src_pagerank",
    "dst_pagerank",
    "src_total_sent",
    "dst_total_received",
    "src_tx_count",
    "amount_zscore_vs_src_history",
    "src_velocity_1day",
    "is_first_time_pair",
    "near_threshold",
]


@dataclass
class EvalResult:
    naive_baseline_accuracy: float  # accuracy of an always-predict-legitimate classifier
    precision_at_default_threshold: float
    recall_at_default_threshold: float
    pr_auc: float
    threshold_table: pd.DataFrame  # precision/recall at several thresholds -- the real, honest picture
    n_test: int
    n_positive_test: int


def train_and_evaluate(
    features: pd.DataFrame, labels: pd.Series, test_size: float = 0.3, seed: int = 42
) -> tuple[XGBClassifier, EvalResult]:
    X = features[FEATURE_COLUMNS]
    y = labels.loc[features.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    positive_rate = y_train.mean()
    scale_pos_weight = (1 - positive_rate) / positive_rate if positive_rate > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds_default = (probs >= 0.5).astype(int)

    naive_baseline_accuracy = 1 - y_test.mean()  # always predicting the majority class

    # Fixed, evenly-spaced threshold values (0.1 .. 0.9), not an index-based downsample of
    # sklearn's precision_recall_curve output. Caught by actually running this: a tree
    # ensemble outputs polarized probabilities (mostly near 0 or 1), so the curve's own
    # points cluster near threshold 0 and an index-based sample of them showed almost
    # nothing between threshold 0.00 and 1.00 -- not a useful report of the real trade-off.
    fixed_thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_table = pd.DataFrame(
        {
            "threshold": fixed_thresholds,
            "precision": [
                precision_score(y_test, (probs >= t).astype(int), zero_division=0) for t in fixed_thresholds
            ],
            "recall": [
                recall_score(y_test, (probs >= t).astype(int), zero_division=0) for t in fixed_thresholds
            ],
        }
    )

    result = EvalResult(
        naive_baseline_accuracy=float(naive_baseline_accuracy),
        precision_at_default_threshold=float(precision_score(y_test, preds_default, zero_division=0)),
        recall_at_default_threshold=float(recall_score(y_test, preds_default, zero_division=0)),
        pr_auc=float(average_precision_score(y_test, probs)),
        threshold_table=threshold_table,
        n_test=len(y_test),
        n_positive_test=int(y_test.sum()),
    )
    return model, result
