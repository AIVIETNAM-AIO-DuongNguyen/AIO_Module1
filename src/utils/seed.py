"""Deterministic seeding so every run is reproducible from its config seed."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all sources of randomness used by the harness.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU and CUDA), and enables
    deterministic cuDNN behaviour so results match across repeated runs of the
    same config on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for DataLoader shuffling reproducibility."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker so multi-process loading stays deterministic.

    Intended to be passed as ``DataLoader(worker_init_fn=seed_worker)``. The
    base seed comes from PyTorch's per-worker initial seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
