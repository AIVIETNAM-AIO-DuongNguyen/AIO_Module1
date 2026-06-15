"""Transfer-learning arms: linear probe, full fine-tune, and LoRA.

All three start from the same timm backbone with a fresh head, so differences in
results are attributable to the adaptation strategy rather than the architecture.
LoRA is applied with peft, targeting the qkv projection for transformer
backbones and Conv2d layers for CNN backbones.
"""

import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from ..utils.config import LoraConfig, ModelConfig
from ..utils.constants import TransferStrategy
from .backbone import TRANSFORMER, build_backbone, detect_family, get_head_name

_KERNEL_3X3 = (3, 3)


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def apply_linear_probe(model: nn.Module) -> nn.Module:
    """Freeze the backbone and train only the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model


def apply_full_finetune(model: nn.Module) -> nn.Module:
    """Train every parameter."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def _conv_target_names(model: nn.Module, conv_target: str) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if conv_target == "kernel3" and tuple(module.kernel_size) != _KERNEL_3X3:
            continue
        names.append(name)
    return names


def _resolve_targets(model: nn.Module, lora: LoraConfig, family: str) -> list[str]:
    if lora.target_modules is not None:
        return lora.target_modules
    if family == TRANSFORMER:
        return ["qkv"]
    targets = _conv_target_names(model, lora.conv_target)
    if not targets:
        raise ValueError(
            f"No Conv2d layers matched conv_target '{lora.conv_target}' for LoRA."
        )
    return targets


def apply_lora(model: nn.Module, lora: LoraConfig, head_name: str, family: str) -> nn.Module:
    """Wrap the backbone with LoRA adapters and keep the head trainable."""
    targets = _resolve_targets(model, lora, family)
    peft_config = PeftLoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        target_modules=targets,
        modules_to_save=[head_name],
    )
    return get_peft_model(model, peft_config)


def build_transfer_model(model_cfg: ModelConfig, n_classes: int) -> nn.Module:
    """Build a backbone and apply the configured transfer strategy."""
    model = build_backbone(model_cfg, n_classes)
    strategy = model_cfg.transfer
    if strategy is TransferStrategy.LP:
        return apply_linear_probe(model)
    if strategy is TransferStrategy.FT:
        return apply_full_finetune(model)
    if strategy is TransferStrategy.LORA:
        head_name = get_head_name(model)
        family = detect_family(model)
        return apply_lora(model, model_cfg.lora, head_name, family)
    raise ValueError(f"Unsupported transfer strategy '{strategy}'.")
