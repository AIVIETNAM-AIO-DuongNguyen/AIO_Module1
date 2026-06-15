import numpy as np

from src.data.subsample import stratified_subsample


def test_full_fraction_returns_all_indices():
    labels = np.array([0, 1, 0, 1, 0, 1])
    indices = stratified_subsample(labels, 1.0, seed=0)
    assert np.array_equal(indices, np.arange(6))


def test_subset_preserves_class_balance():
    labels = np.array([0] * 50 + [1] * 50)
    indices = stratified_subsample(labels, 0.2, seed=0)
    selected = labels[indices]
    assert len(indices) == 20
    assert int((selected == 0).sum()) == 10
    assert int((selected == 1).sum()) == 10


def test_subset_is_reproducible():
    labels = np.array([0] * 50 + [1] * 50)
    first = stratified_subsample(labels, 0.2, seed=42)
    second = stratified_subsample(labels, 0.2, seed=42)
    assert np.array_equal(first, second)


def test_accepts_two_dimensional_labels():
    labels = np.array([[0], [1], [0], [1]])
    indices = stratified_subsample(labels, 0.5, seed=0)
    assert len(indices) == 2
