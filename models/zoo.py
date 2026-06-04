from __future__ import annotations

from typing import List

from torch import nn
from torchvision import models

from config import ProbeConfig

MODEL_ZOO = {
    "resnet50": "torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)",
    "resnet101": "torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)",
    "vit_b16": "torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)",
}


def _load_torchvision_model(model_name: str) -> nn.Module:
    if model_name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if model_name == "resnet101":
        return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    if model_name == "vit_b16":
        return models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of {list(MODEL_ZOO)}")


def select_actmad_layers(model: nn.Module, model_name: str) -> List[str]:
    if model_name.startswith("resnet"):
        stage_prefixes = ["layer1", "layer2", "layer3", "layer4"]
        selected: List[str] = []
        for prefix in stage_prefixes:
            names = [name for name, module in model.named_modules() if name.startswith(prefix) and isinstance(module, nn.BatchNorm2d)]
            if names:
                selected.append(names[-1])
        return selected
    if model_name == "vit_b16":
        return [name for name, module in model.named_modules() if name.endswith("ln_2") and isinstance(module, nn.LayerNorm)]
    raise ValueError(f"Unsupported model_name '{model_name}'")


def load_frozen_model(config: ProbeConfig) -> nn.Module:
    """Load a pretrained torchvision architecture and freeze all parameters."""

    model = _load_torchvision_model(config.model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    config.actmad_layers = select_actmad_layers(model, config.model_name)
    return model
