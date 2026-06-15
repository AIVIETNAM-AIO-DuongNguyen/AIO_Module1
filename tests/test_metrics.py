import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.metrics import compute_metrics, expected_calibration_error
from src.utils.constants import Metric

_ALL_METRICS = [Metric.AUROC, Metric.ACCURACY, Metric.MACRO_F1, Metric.ECE]


def test_perfect_binary_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    results = compute_metrics(y_true, y_score, "binary-class", _ALL_METRICS, ece_bins=15)
    assert results["auroc"] == pytest.approx(1.0)
    assert results["accuracy"] == pytest.approx(1.0)
    assert results["macro_f1"] == pytest.approx(1.0)
    assert results["ece"] == pytest.approx(0.0, abs=1e-9)


def test_perfect_multiclass_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_score = np.eye(3)[y_true]
    results = compute_metrics(y_true, y_score, "multi-class", _ALL_METRICS, ece_bins=15)
    assert results["auroc"] == pytest.approx(1.0)
    assert results["accuracy"] == pytest.approx(1.0)
    assert results["macro_f1"] == pytest.approx(1.0)
    assert results["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_penalizes_confident_mistakes():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95]])
    ece = expected_calibration_error(y_true, y_score, n_bins=15)
    # Confidence is 0.95 but accuracy is 0.5, so the gap should be large.
    assert ece > 0.3


def test_macro_f1_matches_sklearn():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_score = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.3, 0.4, 0.3],
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )
    results = compute_metrics(y_true, y_score, "multi-class", [Metric.MACRO_F1])
    expected = f1_score(y_true, y_score.argmax(axis=1), average="macro")
    assert results["macro_f1"] == pytest.approx(expected)


def test_ece_is_bounded():
    rng = np.random.default_rng(0)
    logits = rng.random((100, 4))
    y_score = logits / logits.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, 4, size=100)
    ece = expected_calibration_error(y_true, y_score, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_unsupported_metric_is_rejected():
    y_true = np.array([0, 1])
    y_score = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_score, "binary-class", ["not_a_metric"])
