"""Typed run configuration loaded from a single YAML file.

One YAML file describes exactly one run. ``load_config`` parses it into nested
dataclasses and validates enum values and numeric ranges, so a malformed config
fails fast with a clear message instead of part-way through a run.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml

from .constants import Metric, Preprocessing, Split, TransferStrategy

_CONV_TARGETS = ("all", "kernel3")


@dataclass
class ClaheConfig:
    clip_limit: float = 2.0
    tile_grid_size: int = 8


@dataclass
class FlipConfig:
    enabled: bool = False
    p: float = 0.5


@dataclass
class RotationConfig:
    enabled: bool = False
    degrees: float = 10.0


@dataclass
class AugmentationConfig:
    """Augmentation applied identically to every preprocessing arm.

    Each entry can be toggled and parametrised from config; only enabled
    augmentations are added to the training transform. Keeping this shared and
    explicit is what stops augmentation from being confounded with the
    preprocessing arm under study.
    """

    horizontal_flip: FlipConfig = field(default_factory=FlipConfig)
    vertical_flip: FlipConfig = field(default_factory=FlipConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)


@dataclass
class DataConfig:
    dataset: str
    resolution: int
    regime: float
    preprocessing: Preprocessing
    root: str = "data"
    as_rgb: bool = True
    clahe: ClaheConfig = field(default_factory=ClaheConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    # Which Conv2d layers to adapt for CNN backbones: "all" or "kernel3"
    # (3x3 convs only). Ignored for transformer backbones, which target qkv.
    conv_target: str = "kernel3"
    # Optional explicit list of target module names. When set it overrides the
    # automatic per-family selection.
    target_modules: list[str] | None = None


@dataclass
class ModelConfig:
    backbone: str
    transfer: TransferStrategy
    pretrained: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    # Extra optimizer-specific keyword arguments (e.g. weight_decay, momentum,
    # betas). Passed straight to the resolved torch.optim class.
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    monitor: Metric = Metric.AUROC
    early_stopping_patience: int = 0


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    num_workers: int = 0
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass
class EvalConfig:
    metrics: list[Metric]
    split: Split = Split.TEST
    ece_bins: int = 15


@dataclass
class RunMeta:
    name: str
    seed: int
    output_dir: str = "runs"


@dataclass
class RunConfig:
    run: RunMeta
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig


def _require(section: dict[str, Any], key: str, context: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key '{key}' in '{context}' config.")
    return section[key]


def _as_enum(enum_cls: type, value: Any, context: str) -> Any:
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum_cls)
        raise ValueError(
            f"Invalid value '{value}' for '{context}'. Allowed: {allowed}."
        )


def _build_augmentation(section: dict[str, Any]) -> AugmentationConfig:
    hflip = section.get("horizontal_flip", {})
    vflip = section.get("vertical_flip", {})
    rotation = section.get("rotation", {})
    return AugmentationConfig(
        horizontal_flip=FlipConfig(
            enabled=bool(hflip.get("enabled", False)),
            p=float(hflip.get("p", 0.5)),
        ),
        vertical_flip=FlipConfig(
            enabled=bool(vflip.get("enabled", False)),
            p=float(vflip.get("p", 0.5)),
        ),
        rotation=RotationConfig(
            enabled=bool(rotation.get("enabled", False)),
            degrees=float(rotation.get("degrees", 10.0)),
        ),
    )


def _build_data(section: dict[str, Any]) -> DataConfig:
    resolution = int(_require(section, "resolution", "data"))
    if resolution <= 0:
        raise ValueError(f"data.resolution must be positive, got {resolution}.")
    regime = float(_require(section, "regime", "data"))
    if not 0.0 < regime <= 1.0:
        raise ValueError(f"data.regime must be in (0, 1], got {regime}.")
    clahe_section = section.get("clahe", {})
    return DataConfig(
        dataset=str(_require(section, "dataset", "data")),
        resolution=resolution,
        regime=regime,
        preprocessing=_as_enum(
            Preprocessing, _require(section, "preprocessing", "data"), "data.preprocessing"
        ),
        root=str(section.get("root", "data")),
        as_rgb=bool(section.get("as_rgb", True)),
        clahe=ClaheConfig(
            clip_limit=float(clahe_section.get("clip_limit", 2.0)),
            tile_grid_size=int(clahe_section.get("tile_grid_size", 8)),
        ),
        augmentation=_build_augmentation(section.get("augmentation", {})),
    )


def _build_lora(section: dict[str, Any]) -> LoraConfig:
    conv_target = str(section.get("conv_target", "kernel3"))
    if conv_target not in _CONV_TARGETS:
        allowed = ", ".join(_CONV_TARGETS)
        raise ValueError(
            f"Invalid value '{conv_target}' for 'model.lora.conv_target'. "
            f"Allowed: {allowed}."
        )
    target_modules = section.get("target_modules")
    if target_modules is not None:
        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError("model.lora.target_modules must be a non-empty list when set.")
        target_modules = [str(name) for name in target_modules]
    return LoraConfig(
        rank=int(section.get("rank", 8)),
        alpha=int(section.get("alpha", 16)),
        conv_target=conv_target,
        target_modules=target_modules,
    )


def _build_model(section: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        backbone=str(_require(section, "backbone", "model")),
        transfer=_as_enum(
            TransferStrategy, _require(section, "transfer", "model"), "model.transfer"
        ),
        pretrained=bool(section.get("pretrained", True)),
        lora=_build_lora(section.get("lora", {})),
    )


def _build_optimizer(section: dict[str, Any]) -> OptimizerConfig:
    kwargs = section.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("train.optimizer.kwargs must be a mapping.")
    return OptimizerConfig(
        name=str(_require(section, "name", "train.optimizer")),
        lr=float(_require(section, "lr", "train.optimizer")),
        kwargs=dict(kwargs),
    )


def _build_selection(section: dict[str, Any]) -> SelectionConfig:
    patience = int(section.get("early_stopping_patience", 0))
    if patience < 0:
        raise ValueError(
            f"train.selection.early_stopping_patience must be >= 0, got {patience}."
        )
    return SelectionConfig(
        monitor=_as_enum(
            Metric, section.get("monitor", Metric.AUROC.value), "train.selection.monitor"
        ),
        early_stopping_patience=patience,
    )


def _build_train(section: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        epochs=int(_require(section, "epochs", "train")),
        batch_size=int(_require(section, "batch_size", "train")),
        optimizer=_build_optimizer(_require(section, "optimizer", "train")),
        num_workers=int(section.get("num_workers", 0)),
        selection=_build_selection(section.get("selection", {})),
    )


def _build_eval(section: dict[str, Any]) -> EvalConfig:
    metrics = _require(section, "metrics", "eval")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("eval.metrics must be a non-empty list.")
    ece_bins = int(section.get("ece_bins", 15))
    if ece_bins <= 0:
        raise ValueError(f"eval.ece_bins must be positive, got {ece_bins}.")
    return EvalConfig(
        metrics=[_as_enum(Metric, metric, "eval.metrics") for metric in metrics],
        split=_as_enum(Split, section.get("split", Split.TEST.value), "eval.split"),
        ece_bins=ece_bins,
    )


def _build_run_meta(section: dict[str, Any]) -> RunMeta:
    return RunMeta(
        name=str(_require(section, "name", "run")),
        seed=int(_require(section, "seed", "run")),
        output_dir=str(section.get("output_dir", "runs")),
    )


def build_run_config(raw: dict[str, Any]) -> RunConfig:
    """Build and validate a run config from an already-parsed mapping."""
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return RunConfig(
        run=_build_run_meta(_require(raw, "run", "root")),
        data=_build_data(_require(raw, "data", "root")),
        model=_build_model(_require(raw, "model", "root")),
        train=_build_train(_require(raw, "train", "root")),
        eval=_build_eval(_require(raw, "eval", "root")),
    )


def load_config(path: str) -> RunConfig:
    """Load and validate a run config from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return build_run_config(raw)
