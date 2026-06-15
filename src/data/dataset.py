"""MedMNIST dataset construction with preprocessing and regime subsampling."""

import os
from dataclasses import dataclass

import medmnist
from torch.utils.data import Dataset, Subset

from ..utils.config import DataConfig
from ..utils.constants import Split
from .preprocessing import build_transform
from .subsample import stratified_subsample


@dataclass
class DatasetInfo:
    n_channels: int
    n_classes: int
    task: str


def get_dataset_info(dataset: str) -> DatasetInfo:
    """Look up channel count, class count, and task type for a MedMNIST flag."""
    flag = dataset.lower()
    if flag not in medmnist.INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{dataset}'.")
    info = medmnist.INFO[flag]
    return DatasetInfo(
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        task=info["task"],
    )


def build_dataset(data: DataConfig, split: Split, seed: int) -> Dataset:
    """Build a MedMNIST dataset for one split with the configured preprocessing.

    Augmentation is enabled only for the training split. The scarce-label
    regime is applied by stratified subsampling of the training split only;
    validation and test splits always use their full official data.
    """
    flag = data.dataset.lower()
    if flag not in medmnist.INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{data.dataset}'.")
    info = medmnist.INFO[flag]
    dataset_class = getattr(medmnist, info["python_class"])

    is_train = split is Split.TRAIN
    transform = build_transform(
        arm=data.preprocessing,
        resolution=data.resolution,
        n_channels=info["n_channels"],
        as_rgb=data.as_rgb,
        clahe=data.clahe,
        augmentation=data.augmentation,
        train=is_train,
    )

    os.makedirs(data.root, exist_ok=True)
    dataset = dataset_class(
        split=split.value,
        transform=transform,
        download=True,
        as_rgb=data.as_rgb,
        root=data.root,
        size=data.resolution,
    )

    if is_train and data.regime < 1.0:
        indices = stratified_subsample(dataset.labels, data.regime, seed)
        return Subset(dataset, indices)
    return dataset
