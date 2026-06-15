"""Named constants and enumerations shared across the harness.

Centralising these prevents dataset names, arm names, and normalization values
from being hardcoded as bare strings or magic numbers elsewhere in the code.
"""

from enum import Enum


class Preprocessing(str, Enum):
    """Input-space lever: how the image is transformed before the model."""

    P0 = "p0"
    P_GLOBAL = "p_global"
    P_LOCAL = "p_local"


class TransferStrategy(str, Enum):
    """Parameter-space lever: how the pretrained backbone is adapted."""

    LP = "lp"
    FT = "ft"
    LORA = "lora"


class Split(str, Enum):
    """MedMNIST data splits, matching the strings the dataset classes expect."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Metric(str, Enum):
    """Evaluation metrics reported for each run."""

    AUROC = "auroc"
    ACCURACY = "accuracy"
    MACRO_F1 = "macro_f1"
    ECE = "ece"


# ImageNet channel statistics used to normalize inputs for ImageNet-pretrained
# backbones. Source: torchvision pretrained-model preprocessing documentation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
