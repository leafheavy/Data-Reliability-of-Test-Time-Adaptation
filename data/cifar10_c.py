from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import ProbeConfig
from data.common import CIFAR_TRANSFORM, corrupt_pil_image


class OnlineCIFAR10CDataset(Dataset):
    """CIFAR-10 debug dataset with online CIFAR-10-C-style corruptions."""

    def __init__(self, data_root: str | Path, corruption: str, severity: int, train: bool = False):
        self.data_root = Path(data_root)
        self.base = datasets.CIFAR10(root=str(self.data_root), train=train, download=False)
        self.corruption = corruption
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clean_img, y = self.base[index]
        clean_img = clean_img.convert("RGB")
        corr_img = corrupt_pil_image(clean_img, self.corruption, self.severity)
        return CIFAR_TRANSFORM(corr_img), CIFAR_TRANSFORM(clean_img), y


def get_cifar10_c_loader(config: ProbeConfig, corruption: str, severity: int) -> DataLoader:
    dataset = OnlineCIFAR10CDataset(config.data_root, corruption, severity, train=False)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
