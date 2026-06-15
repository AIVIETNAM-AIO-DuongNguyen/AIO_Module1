"""Preprocessing arms and the shared transform pipeline.

The transform is built in two tiers:

1. An arm-specific contrast operation (``p0`` / ``p_global`` / ``p_local``)
   applied to every image, train and eval alike.
2. A shared pipeline (resize, optional augmentation, tensor conversion,
   ImageNet normalization) that is identical for all arms.

Augmentation lives only in tier 2 and is identical across arms, so the study
measures the preprocessing arm rather than a confounded mix of preprocessing
and augmentation.
"""

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

from ..utils.config import AugmentationConfig, ClaheConfig
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD, Preprocessing


class ContrastArm:
    """Apply the arm-specific contrast operation to a PIL image.

    For colour images the operation runs on the L channel of LAB space so only
    luminance contrast changes; for single-channel images it runs directly.
    """

    def __init__(self, arm: Preprocessing, clahe: ClaheConfig) -> None:
        self._arm = arm
        if arm is Preprocessing.P_LOCAL:
            self._clahe = cv2.createCLAHE(
                clipLimit=clahe.clip_limit,
                tileGridSize=(clahe.tile_grid_size, clahe.tile_grid_size),
            )
        else:
            self._clahe = None

    def _enhance_channel(self, channel: np.ndarray) -> np.ndarray:
        if self._arm is Preprocessing.P_GLOBAL:
            return cv2.equalizeHist(channel)
        if self._arm is Preprocessing.P_LOCAL:
            return self._clahe.apply(channel)
        return channel

    def __call__(self, image: Image.Image) -> Image.Image:
        if self._arm is Preprocessing.P0:
            return image
        array = np.asarray(image)
        if array.ndim == 3:
            lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self._enhance_channel(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = self._enhance_channel(array)
        return Image.fromarray(enhanced)


def _augmentation_transforms(augmentation: AugmentationConfig) -> list:
    """Return transforms for the enabled augmentations, in a fixed order."""
    transforms = []
    if augmentation.horizontal_flip.enabled:
        transforms.append(T.RandomHorizontalFlip(p=augmentation.horizontal_flip.p))
    if augmentation.vertical_flip.enabled:
        transforms.append(T.RandomVerticalFlip(p=augmentation.vertical_flip.p))
    if augmentation.rotation.enabled:
        transforms.append(T.RandomRotation(degrees=augmentation.rotation.degrees))
    return transforms


def build_transform(
    arm: Preprocessing,
    resolution: int,
    n_channels: int,
    as_rgb: bool,
    clahe: ClaheConfig,
    augmentation: AugmentationConfig,
    train: bool,
) -> T.Compose:
    """Compose the full transform for one preprocessing arm.

    Augmentation is added only when ``train`` is True. ImageNet normalization
    requires three input channels, which ``as_rgb=True`` guarantees for the
    pretrained backbones used in this study.
    """
    effective_channels = 3 if as_rgb else n_channels
    if effective_channels != 3:
        raise ValueError(
            "ImageNet normalization expects 3 channels; set data.as_rgb to true "
            f"for this dataset (got {effective_channels} channels)."
        )

    steps: list = [ContrastArm(arm, clahe), T.Resize((resolution, resolution))]
    if train:
        steps.extend(_augmentation_transforms(augmentation))
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)
