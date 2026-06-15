import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import build_transform
from src.utils.config import AugmentationConfig, ClaheConfig, FlipConfig, RotationConfig
from src.utils.constants import Preprocessing


def _rgb_image(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))


def test_every_arm_outputs_normalized_tensor():
    for arm in Preprocessing:
        transform = build_transform(
            arm=arm,
            resolution=64,
            n_channels=3,
            as_rgb=True,
            clahe=ClaheConfig(),
            augmentation=AugmentationConfig(),
            train=False,
        )
        output = transform(_rgb_image())
        assert tuple(output.shape) == (3, 64, 64)


def test_augmentation_is_identical_across_arms():
    augmentation = AugmentationConfig(
        horizontal_flip=FlipConfig(enabled=True, p=0.5),
        rotation=RotationConfig(enabled=True, degrees=10),
    )
    shared = {}
    for arm in (Preprocessing.P0, Preprocessing.P_LOCAL):
        transform = build_transform(
            arm=arm,
            resolution=64,
            n_channels=3,
            as_rgb=True,
            clahe=ClaheConfig(),
            augmentation=augmentation,
            train=True,
        )
        # Drop the first step (the arm-specific contrast op); everything after
        # it must be identical, proving augmentation does not depend on the arm.
        shared[arm] = [type(step).__name__ for step in transform.transforms[1:]]
    assert shared[Preprocessing.P0] == shared[Preprocessing.P_LOCAL]


def test_eval_transform_has_no_augmentation():
    augmentation = AugmentationConfig(
        horizontal_flip=FlipConfig(enabled=True, p=0.5),
        rotation=RotationConfig(enabled=True, degrees=10),
    )
    transform = build_transform(
        arm=Preprocessing.P0,
        resolution=64,
        n_channels=3,
        as_rgb=True,
        clahe=ClaheConfig(),
        augmentation=augmentation,
        train=False,
    )
    names = [type(step).__name__ for step in transform.transforms]
    assert "RandomHorizontalFlip" not in names
    assert "RandomRotation" not in names


def test_single_channel_without_as_rgb_is_rejected():
    with pytest.raises(ValueError):
        build_transform(
            arm=Preprocessing.P0,
            resolution=64,
            n_channels=1,
            as_rgb=False,
            clahe=ClaheConfig(),
            augmentation=AugmentationConfig(),
            train=False,
        )
