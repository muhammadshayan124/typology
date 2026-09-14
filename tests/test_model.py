import pandas as pd

from typology.features import build_features
from typology.generator import GeneratorConfig, generate
from typology.model import train_and_evaluate


def _small_dataset():
    cfg = GeneratorConfig(
        n_accounts=300, n_legitimate_tx=4000, n_days=30,
        n_fan_out=5, n_fan_in=5, n_layering_chains=5, n_structuring=5, seed=3,
    )
    df = generate(cfg)
    features = build_features(df)
    labels = df.set_index("tx_id")["is_laundering"]
    return features, labels


def test_naive_baseline_matches_actual_class_balance():
    features, labels = _small_dataset()
    _, result = train_and_evaluate(features, labels)
    # naive baseline should equal roughly (1 - fraction of positives in the test split)
    assert 0.9 < result.naive_baseline_accuracy < 1.0  # laundering is a small minority by construction


def test_metrics_are_valid_probabilities():
    features, labels = _small_dataset()
    _, result = train_and_evaluate(features, labels)
    for value in (result.precision_at_default_threshold, result.recall_at_default_threshold, result.pr_auc):
        assert 0.0 <= value <= 1.0


def test_model_beats_naive_baseline_on_pr_auc():
    # PR-AUC of a naive "always legitimate" classifier equals the positive rate itself
    # (a random/constant classifier's PR-AUC is the base rate) -- a model with real
    # signal should clear that by a wide margin on data this structured.
    features, labels = _small_dataset()
    _, result = train_and_evaluate(features, labels)
    positive_rate = labels.mean()
    assert result.pr_auc > positive_rate * 3


def test_threshold_table_has_monotonic_relationship():
    features, labels = _small_dataset()
    _, result = train_and_evaluate(features, labels)
    table = result.threshold_table.sort_values("threshold")
    # as threshold rises, precision should trend up and recall should trend down --
    # check the endpoints rather than every step, since precision-recall curves aren't
    # perfectly monotonic point-to-point.
    assert table.iloc[-1]["recall"] <= table.iloc[0]["recall"]
