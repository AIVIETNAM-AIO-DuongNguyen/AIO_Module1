"""Run orchestration: build everything from a config and execute one run.

A run sets the seed, builds the data loaders, model, optimizer, and loss,
trains for the configured epochs while keeping the best checkpoint by a
validation metric, then evaluates the best model on the eval split and writes
the results.
"""

import copy
import csv
import json
import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import build_dataset, get_dataset_info
from ..evaluation.metrics import compute_metrics
from ..models.transfer import build_transfer_model, count_trainable_params
from ..utils.config import OptimizerConfig, RunConfig
from ..utils.constants import Metric, Split
from ..utils.logging import get_logger
from ..utils.seed import make_generator, seed_worker, set_seed
from .loop import collect_predictions, train_one_epoch

_LOWER_IS_BETTER = {Metric.ECE}


def _optimizer_table() -> dict[str, type]:
    table = {}
    for attr in dir(torch.optim):
        candidate = getattr(torch.optim, attr)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, torch.optim.Optimizer)
            and candidate is not torch.optim.Optimizer
        ):
            table[attr.lower()] = candidate
    return table


def build_optimizer(model: nn.Module, optimizer: OptimizerConfig) -> torch.optim.Optimizer:
    """Resolve a torch.optim optimizer by name over the model's trainable params."""
    table = _optimizer_table()
    key = optimizer.name.lower()
    if key not in table:
        raise ValueError(
            f"Unknown optimizer '{optimizer.name}'. Available: {sorted(table)}."
        )
    trainable = [param for param in model.parameters() if param.requires_grad]
    try:
        return table[key](trainable, lr=optimizer.lr, **optimizer.kwargs)
    except TypeError as error:
        raise ValueError(f"Invalid kwargs for optimizer '{optimizer.name}': {error}")


def build_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int
) -> DataLoader:
    """Build a reproducible DataLoader seeded from the run seed."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_improvement(current: float, best: float | None, monitor: Metric) -> bool:
    if best is None:
        return True
    if monitor in _LOWER_IS_BETTER:
        return current < best
    return current > best


def _append_results_csv(path: str, results: dict[str, Any]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def _save_outputs(config: RunConfig, model: nn.Module, results: dict[str, Any]) -> None:
    run_dir = os.path.join(config.run.output_dir, config.run.name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pt"))
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    _append_results_csv(os.path.join(config.run.output_dir, "results.csv"), results)


def run_experiment(config: RunConfig) -> dict[str, Any]:
    """Execute one run end to end and return its results."""
    logger = get_logger("training")
    set_seed(config.run.seed)
    device = _select_device()

    info = get_dataset_info(config.data.dataset)
    train_loader = build_dataloader(
        build_dataset(config.data, Split.TRAIN, config.run.seed),
        config.train.batch_size, True, config.train.num_workers, config.run.seed,
    )
    val_loader = build_dataloader(
        build_dataset(config.data, Split.VAL, config.run.seed),
        config.train.batch_size, False, config.train.num_workers, config.run.seed,
    )
    test_loader = build_dataloader(
        build_dataset(config.data, config.eval.split, config.run.seed),
        config.train.batch_size, False, config.train.num_workers, config.run.seed,
    )

    model = build_transfer_model(config.model, info.n_classes).to(device)
    optimizer = build_optimizer(model, config.train.optimizer)
    criterion = nn.CrossEntropyLoss()

    monitor = config.train.selection.monitor
    patience = config.train.selection.early_stopping_patience
    best_score: float | None = None
    best_state = None
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(config.train.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_true, val_score = collect_predictions(model, val_loader, device)
        val_metrics = compute_metrics(
            val_true, val_score, info.task, [monitor], config.eval.ece_bins
        )
        score = val_metrics[monitor.value]
        logger.info(
            "epoch %d/%d loss %.4f val_%s %.4f",
            epoch + 1, config.train.epochs, loss, monitor.value, score,
        )
        if _is_improvement(score, best_score, monitor):
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info("early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    wall_clock = time.time() - start

    test_true, test_score = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(
        test_true, test_score, info.task, config.eval.metrics, config.eval.ece_bins
    )

    results: dict[str, Any] = {
        "run": config.run.name,
        "dataset": config.data.dataset,
        "preprocessing": config.data.preprocessing.value,
        "backbone": config.model.backbone,
        "transfer": config.model.transfer.value,
        "regime": config.data.regime,
        "seed": config.run.seed,
        "trainable_params": count_trainable_params(model),
        "wall_clock_s": round(wall_clock, 2),
        **metrics,
    }
    _save_outputs(config, model, results)
    return results
