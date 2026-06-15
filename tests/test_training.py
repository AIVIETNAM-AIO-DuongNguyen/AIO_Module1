import urllib.error

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.loop import collect_predictions
from src.training.runner import build_optimizer, run_experiment
from src.utils.config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    RunMeta,
    SelectionConfig,
    TrainConfig,
)
from src.utils.constants import Metric, Preprocessing, Split, TransferStrategy


def _frozen_then_trainable_model():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
    for param in model[0].parameters():
        param.requires_grad = False
    return model


def test_build_optimizer_uses_only_trainable_params():
    model = _frozen_then_trainable_model()
    optimizer = build_optimizer(
        model, OptimizerConfig(name="adamw", lr=1e-3, kwargs={"weight_decay": 1e-4})
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    in_optimizer = sum(len(group["params"]) for group in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert in_optimizer == trainable


def test_build_optimizer_supports_sgd_with_kwargs():
    model = _frozen_then_trainable_model()
    optimizer = build_optimizer(
        model, OptimizerConfig(name="sgd", lr=0.1, kwargs={"momentum": 0.9})
    )
    assert isinstance(optimizer, torch.optim.SGD)


def test_build_optimizer_rejects_unknown_name():
    model = _frozen_then_trainable_model()
    with pytest.raises(ValueError):
        build_optimizer(model, OptimizerConfig(name="not_an_optimizer", lr=1e-3))


def test_collect_predictions_shapes_and_softmax():
    images = torch.randn(10, 3, 8, 8)
    labels = torch.randint(0, 3, (10, 1))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    y_true, y_score = collect_predictions(model, loader, torch.device("cpu"))
    assert y_true.shape == (10, 1)
    assert y_score.shape == (10, 3)
    np.testing.assert_allclose(y_score.sum(axis=1), np.ones(10), atol=1e-5)


def _smoke_config(output_dir):
    return RunConfig(
        run=RunMeta(name="smoke", seed=0, output_dir=output_dir),
        data=DataConfig(
            dataset="pneumoniamnist",
            resolution=64,
            regime=0.05,
            preprocessing=Preprocessing.P0,
        ),
        model=ModelConfig(
            backbone="resnet10t", transfer=TransferStrategy.LP, pretrained=False
        ),
        train=TrainConfig(
            epochs=1,
            batch_size=32,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            num_workers=0,
            selection=SelectionConfig(),
        ),
        eval=EvalConfig(
            metrics=[Metric.AUROC, Metric.ACCURACY, Metric.MACRO_F1, Metric.ECE],
            split=Split.TEST,
            ece_bins=15,
        ),
    )


def test_run_experiment_smoke(tmp_path):
    config = _smoke_config(str(tmp_path))
    try:
        results = run_experiment(config)
    except (RuntimeError, OSError, urllib.error.URLError) as error:
        pytest.skip(f"MedMNIST download unavailable: {error}")

    for key in ("auroc", "accuracy", "macro_f1", "ece", "trainable_params", "wall_clock_s"):
        assert key in results
    assert 0.0 <= results["auroc"] <= 1.0
    assert results["trainable_params"] > 0
    assert (tmp_path / "smoke" / "checkpoint.pt").exists()
    assert (tmp_path / "smoke" / "metrics.json").exists()
    assert (tmp_path / "results.csv").exists()
