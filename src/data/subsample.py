"""Stratified subsampling to emulate scarce-label data regimes."""

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_subsample(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return sorted indices of a class-stratified subset of ``labels``.

    A fraction of ``1.0`` returns every index. Smaller fractions keep the
    per-class proportions of the full set, so a 5% regime is not skewed toward
    the majority class. The seed makes the selection reproducible.
    """
    flat = np.asarray(labels).ravel()
    total = len(flat)
    if fraction >= 1.0:
        return np.arange(total)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")

    indices = np.arange(total)
    subset, _ = train_test_split(
        indices, train_size=fraction, stratify=flat, random_state=seed
    )
    return np.sort(subset)
