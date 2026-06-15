import copy

import pytest
import yaml

from src.utils.config import load_config
from src.utils.constants import Preprocessing, Split, TransferStrategy

_BASE_CONFIG = {
    "run": {"name": "t", "seed": 0},
    "data": {
        "dataset": "pneumoniamnist",
        "resolution": 64,
        "regime": 0.1,
        "preprocessing": "p_local",
    },
    "model": {"backbone": "resnet50", "transfer": "lora"},
    "train": {
        "epochs": 1,
        "batch_size": 8,
        "optimizer": {"name": "adamw", "lr": 0.001, "kwargs": {"weight_decay": 0.0001}},
    },
    "eval": {"metrics": ["auroc"]},
}


def _write_config(tmp_path, config):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_load_example_config():
    config = load_config("configs/example_run.yaml")
    assert config.run.seed == 0
    assert config.data.preprocessing is Preprocessing.P_LOCAL
    assert config.model.transfer is TransferStrategy.LORA
    assert config.eval.split is Split.TEST
    assert config.data.augmentation.horizontal_flip.enabled is True
    assert config.data.augmentation.rotation.degrees == 10


def test_invalid_preprocessing_is_rejected(tmp_path):
    config = copy.deepcopy(_BASE_CONFIG)
    config["data"]["preprocessing"] = "not_an_arm"
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, config))


def test_out_of_range_regime_is_rejected(tmp_path):
    config = copy.deepcopy(_BASE_CONFIG)
    config["data"]["regime"] = 1.5
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, config))


def test_missing_required_key_is_rejected(tmp_path):
    config = copy.deepcopy(_BASE_CONFIG)
    del config["train"]["optimizer"]["lr"]
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, config))
