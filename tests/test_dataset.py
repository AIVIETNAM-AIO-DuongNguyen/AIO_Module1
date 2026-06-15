import urllib.error

import pytest

from src.data.dataset import build_dataset, get_dataset_info
from src.utils.config import DataConfig
from src.utils.constants import Preprocessing, Split


def test_get_dataset_info_pneumonia():
    info = get_dataset_info("pneumoniamnist")
    assert info.n_channels == 1
    assert info.n_classes == 2
    assert info.task == "binary-class"


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError):
        get_dataset_info("not_a_dataset")


def test_build_dataset_smoke():
    config = DataConfig(
        dataset="pneumoniamnist",
        resolution=64,
        regime=0.05,
        preprocessing=Preprocessing.P_LOCAL,
    )
    try:
        dataset = build_dataset(config, Split.TRAIN, seed=0)
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        pytest.skip(f"MedMNIST download unavailable: {error}")

    image, _ = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert len(dataset) > 0
