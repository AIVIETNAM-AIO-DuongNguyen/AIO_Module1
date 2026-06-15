"""Metric computation for a run's predictions.

AUROC and accuracy reuse MedMNIST's task-aware functions so results match the
official benchmark; macro-F1 uses scikit-learn; ECE is implemented here. All
functions take per-class softmax scores ``y_score`` of shape
``(n_samples, n_classes)`` and labels ``y_true`` of shape ``(n_samples,)`` or
``(n_samples, 1)``.
"""

import numpy as np
from medmnist.evaluator import getACC, getAUC
from sklearn.metrics import f1_score

from ..utils.constants import Metric


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int
) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    Follows Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017):
    the gap between confidence (max softmax probability) and accuracy is averaged
    across bins, weighted by the number of samples in each bin.
    """
    labels = np.asarray(y_true).squeeze()
    scores = np.asarray(y_score)
    confidences = scores.max(axis=1)
    predictions = scores.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    total = len(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        bin_size = int(in_bin.sum())
        if bin_size == 0:
            continue
        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: str,
    metrics: list[Metric],
    ece_bins: int = 15,
) -> dict[str, float]:
    """Compute the requested metrics and return them keyed by metric name."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    results: dict[str, float] = {}
    for metric in metrics:
        if metric is Metric.AUROC:
            results[metric.value] = float(getAUC(labels, scores, task))
        elif metric is Metric.ACCURACY:
            results[metric.value] = float(getACC(labels, scores, task))
        elif metric is Metric.MACRO_F1:
            predictions = scores.argmax(axis=1)
            results[metric.value] = float(
                f1_score(labels.squeeze(), predictions, average="macro")
            )
        elif metric is Metric.ECE:
            results[metric.value] = expected_calibration_error(labels, scores, ece_bins)
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")
    return results
