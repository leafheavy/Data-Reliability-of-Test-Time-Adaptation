from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import ProbeConfig
from data.common import IMAGE_TRANSFORM, corrupt_pil_image


class OnlineImageNetCDataset(Dataset):
    """ImageNet validation set with online ImageNet-C-style corruptions.

    Each sample returns ``(x_corrupted, x_clean, y)``. No corrupted copies are
    cached to disk; the requested corruption is generated at access time.
    """

    def __init__(self, clean_root: str | Path, corruption: str, severity: int):
        self.clean_root = Path(clean_root)
        if not self.clean_root.exists():
            raise FileNotFoundError(
                f"ImageNet clean validation directory not found: {self.clean_root}. "
                "ImageNet is currently optional/skippable; use dataset='cifar10_c' for debug."
            )
        self.base = datasets.ImageFolder(str(self.clean_root))
        self.corruption = corruption
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, y = self.base.samples[index]
        clean_img = Image.open(path).convert("RGB")
        corr_img = corrupt_pil_image(clean_img, self.corruption, self.severity)
        return IMAGE_TRANSFORM(corr_img), IMAGE_TRANSFORM(clean_img), y


def _resolve_imagenet_val_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = [
        root / "imagenet" / "val",
        root / "ImageNet" / "val",
        root / "ILSVRC2012" / "val",
        root / "val",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_imagenet_c_loader(config: ProbeConfig, corruption: str, severity: int) -> DataLoader:
    dataset = OnlineImageNetCDataset(_resolve_imagenet_val_root(config.data_root), corruption, severity)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
