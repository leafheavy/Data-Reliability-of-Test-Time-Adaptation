from __future__ import annotations

from typing import List, TYPE_CHECKING # 导入 TYPE_CHECKING

import os
os.environ["TORCH_HOME"] = "/home/yezhong/baseline"

from torch import nn
from torchvision import models

# 1. 只有在静态类型检查时才导入 ProbeConfig，打破运行时的循环导入
if TYPE_CHECKING:
    from config import ProbeConfig

MODEL_ZOO = {
    "resnet50": "torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)",
    "resnet101": "torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)",
    "vit_b16": "torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)",
}


def build_model(model_name: str, num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        if num_classes != model.fc.out_features:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        if num_classes != model.fc.out_features:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "vit_b16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        if num_classes != model.heads.head.out_features:
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of {list(MODEL_ZOO)}")

def _load_torchvision_model(model_name: str) -> nn.Module:
    return build_model(model_name, pretrained=True)

def select_actmad_layers(model: nn.Module, model_name: str) -> List[str]:
    # 1. 规范化模型名称（转小写并去除所有下划线，兼容 resnet50, ResNet50, vit_b16, vit_b_16 等）
    norm_name = model_name.lower().replace("_", "")
    
    # 2. 定义辅助函数：自动剥离多卡训练（DP/DDP）自动生成的 "module." 前缀
    def get_clean_name(name: str) -> str:
        return name[7:] if name.startswith("module.") else name

    # 3. 处理 ResNet 系列
    if norm_name.startswith("resnet"):
        stage_prefixes = ["layer1", "layer2", "layer3", "layer4"]
        selected: List[str] = []
        for prefix in stage_prefixes:
            names = []
            for name, module in model.named_modules():
                clean_name = get_clean_name(name)
                # 针对剥离前缀后的层名进行匹配判断
                if clean_name.startswith(prefix) and isinstance(module, nn.BatchNorm2d):
                    names.append(name) # 注册时依然使用含 "module." 的原始完整名字
            
            if names:
                selected.append(names[-1]) # 选取该 stage 的最后一个 BN 层
        return selected

    # 4. 处理 ViT 系列（支持 vit_b16 / vitb16 / vit_b_16）
    if norm_name == "vitb16":
        selected: List[str] = []
        for name, module in model.named_modules():
            clean_name = get_clean_name(name)
            # 同时兼容 torchvision 的 "ln_2" 和 timm 等库的 "norm2" 命名规则
            if (clean_name.endswith("ln_2") or clean_name.endswith("norm2")) and isinstance(module, nn.LayerNorm):
                selected.append(name)
        return selected

    raise ValueError(f"Unsupported model_name '{model_name}' (normalized to '{norm_name}')")


def load_frozen_model(config: ProbeConfig) -> nn.Module:
    """Load a pretrained torchvision architecture and freeze all parameters."""

    model = _load_torchvision_model(config.model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    config.actmad_layers = select_actmad_layers(model, config.model_name)
    return model