from __future__ import annotations

from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from config import ProbeConfig
from data.common import CIFAR_TRANSFORM, IMAGE_TRANSFORM
from data.splits import resolve_imagenet_split_root
from models.zoo import build_model, select_actmad_layers


def _clean_train_loader(config: ProbeConfig) -> DataLoader:
    if config.dataset == "cifar10_c":
        dataset = datasets.CIFAR10(
            root=str(config.data_root),
            train=True,
            download=False,
            transform=CIFAR_TRANSFORM,
        )
    elif config.dataset == "imagenet_c":
        dataset = datasets.ImageFolder(
            str(resolve_imagenet_split_root(config.data_root, config.source_split)),
            transform=IMAGE_TRANSFORM,
        )
    else:
        raise ValueError(f"Unsupported dataset '{config.dataset}'")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def default_checkpoint_path(config: ProbeConfig) -> Path:
    return Path(config.output_dir) / "checkpoints" / f"{config.dataset}_{config.model_name}_clean_{config.source_split}.pt"


def train_clean_source_model(config: ProbeConfig, checkpoint_path: Optional[str | Path] = None) -> Path:
    """Train a classifier on the clean source-training split and save it.

    This is mainly needed for CIFAR-10, where torchvision's default ResNet/ViT
    weights are ImageNet-trained and therefore do not have the correct 10-class
    output head. ImageNet experiments normally use torchvision ImageNet weights,
    which are already trained on the ImageNet training split.
    """

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    num_classes = 10 if config.dataset == "cifar10_c" else 1000
    model = build_model(config.model_name, num_classes=num_classes, pretrained=(config.dataset == "cifar10_c"))
    model.to(device)
    model.train()

    loader = _clean_train_loader(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.train_lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for _epoch in tqdm(range(config.train_epochs), desc="Training clean source model"):
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    out_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return out_path


def load_clean_source_model(config: ProbeConfig) -> nn.Module:
    """Load a model trained on the clean source train split and freeze it."""

    if config.dataset == "cifar10_c":
        num_classes = 10
        checkpoint = Path(config.model_checkpoint) if config.model_checkpoint else default_checkpoint_path(config)
        if not checkpoint.exists():
            if not config.train_if_missing:
                raise FileNotFoundError(
                    f"CIFAR-10 clean-train checkpoint not found: {checkpoint}. "
                    "Run training first or set train_if_missing=True."
                )
            checkpoint = train_clean_source_model(config, checkpoint)
        model = build_model(config.model_name, num_classes=num_classes, pretrained=False)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    elif config.dataset == "imagenet_c":
        # torchvision DEFAULT weights are trained on clean ImageNet train.
        model = build_model(config.model_name, num_classes=1000, pretrained=True)
    else:
        raise ValueError(f"Unsupported dataset '{config.dataset}'")

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    config.actmad_layers = select_actmad_layers(model, config.model_name)
    return model
