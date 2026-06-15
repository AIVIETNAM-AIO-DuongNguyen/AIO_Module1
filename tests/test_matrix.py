import copy
from collections import Counter

import pytest

from src.experiments.matrix import assign_slices, generate_configs, write_configs
from src.utils.config import load_config

_BASE = {
    "run": {"output_dir": "runs"},
    "data": {"resolution": 64},
    "model": {"backbone": "resnet50", "pretrained": True},
    "train": {
        "epochs": 1,
        "batch_size": 8,
        "optimizer": {"name": "adamw", "lr": 0.001, "kwargs": {"weight_decay": 0.0001}},
    },
    "eval": {"metrics": ["auroc"]},
}


def _spec(axes):
    return {"base": copy.deepcopy(_BASE), "axes": axes}


def test_generate_count_and_unique_names():
    axes = {
        "dataset": ["pneumoniamnist", "dermamnist"],
        "preprocessing": ["p0", "p_local"],
        "transfer": ["lp", "lora"],
        "regime": [1.0, 0.05],
        "seed": [0, 1],
    }
    configs = generate_configs(_spec(axes))
    names = [config["run"]["name"] for config in configs]
    assert len(configs) == 2 * 2 * 2 * 2 * 2
    assert len(set(names)) == len(names)
    assert "pneumoniamnist_p0_lp_r100_s0" in names
    assert "dermamnist_p_local_lora_r5_s1" in names


def test_generated_configs_are_valid(tmp_path):
    axes = {
        "dataset": ["pneumoniamnist"],
        "preprocessing": ["p0", "p_global", "p_local"],
        "transfer": ["lp", "ft", "lora"],
        "regime": [0.1],
        "seed": [0],
    }
    configs = generate_configs(_spec(axes))
    paths = write_configs(configs, str(tmp_path / "core"))
    assert len(paths) == 9
    for path in paths:
        load_config(path)


def test_backbone_axis_supported():
    axes = {
        "backbone": ["resnet50", "vit_small_patch16_224"],
        "dataset": ["pneumoniamnist"],
        "preprocessing": ["p0"],
        "transfer": ["lp"],
        "regime": [1.0],
        "seed": [0],
    }
    configs = generate_configs(_spec(axes))
    assert {config["model"]["backbone"] for config in configs} == {
        "resnet50",
        "vit_small_patch16_224",
    }


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError):
        generate_configs(_spec({"dataset": ["pneumoniamnist"], "bogus": [1]}))


def test_empty_axis_is_rejected():
    with pytest.raises(ValueError):
        generate_configs(_spec({"dataset": []}))


def test_assign_slices_round_robin_is_balanced():
    assignments = assign_slices([f"run{i}" for i in range(162)], 5)
    counts = Counter(item["account"] for item in assignments)
    assert set(counts) == {0, 1, 2, 3, 4}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_assign_slices_rejects_zero_accounts():
    with pytest.raises(ValueError):
        assign_slices(["a"], 0)
