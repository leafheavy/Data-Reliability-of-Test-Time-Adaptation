from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import ProbeConfig
from data.common import CIFAR_TRANSFORM, SYNTHETIC_CORRUPTIONS, corrupt_pil_image


def _cifar10_c_candidates(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    return [
        root / "CIFAR-10-C",
        root / "cifar10-c",
        root / "CIFAR10-C",
        root / "cifar-10-c",
    ]


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class AlignedCIFAR10CDataset(Dataset):
    """Aligned Q_c/P_0 pairs for offline CIFAR-10-C structure analysis."""

    def __init__(
        self,
        data_root: str | Path,
        corruption: str,
        severity: int,
        split: str = "test",
        corruption_source: str = "auto",
        synthetic_seed: int = 1337,
    ):
        self.data_root = Path(data_root)
        normalized = split.lower()
        if normalized not in {"train", "test"}:
            raise ValueError(f"Unsupported CIFAR-10 split {split!r}. Expected 'train' or 'test'.")
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError(f"CIFAR-10-C severity must be in [1, 5], got {severity}.")
        self.split = normalized
        self.base = datasets.CIFAR10(root=str(self.data_root), train=(normalized == "train"), download=True)
        self.corruption = corruption
        self.severity = severity
        self.synthetic_seed = synthetic_seed
        self.official_images = None
        self.official_labels = None

        if corruption_source not in {"auto", "official", "synthetic"}:
            raise ValueError("corruption_source must be one of: auto, official, synthetic.")

        c_root = _first_existing(_cifar10_c_candidates(self.data_root))
        c_path = c_root / f"{corruption}.npy" if c_root else None
        if corruption_source in {"auto", "official"} and c_path and c_path.exists():
            self.official_images = np.load(c_path, mmap_mode="r")
            labels_path = c_root / "labels.npy"
            if labels_path.exists():
                self.official_labels = np.load(labels_path, mmap_mode="r")
            expected = len(self.base) * 5
            if len(self.official_images) < expected:
                raise ValueError(f"{c_path} has {len(self.official_images)} images; expected at least {expected}.")
        elif corruption_source == "official":
            searched = "\n".join(f"  - {candidate / (corruption + '.npy')}" for candidate in _cifar10_c_candidates(self.data_root))
            raise FileNotFoundError(f"CIFAR-10-C official corruption array not found. Looked for:\n{searched}")
        elif corruption not in SYNTHETIC_CORRUPTIONS:
            raise ValueError(
                f"No official CIFAR-10-C file was found for '{corruption}', "
                "and the synthetic fallback does not implement it."
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        clean_img, y = self.base[index]
        clean_img = clean_img.convert("RGB")
        if self.official_images is not None:
            offset = (self.severity - 1) * len(self.base) + index
            corr_img = Image.fromarray(np.asarray(self.official_images[offset]).astype(np.uint8)).convert("RGB")
            if self.official_labels is not None:
                y = int(self.official_labels[offset])
        else:
            seed = self.synthetic_seed + index + 10000 * self.severity
            corr_img = corrupt_pil_image(clean_img, self.corruption, self.severity, seed=seed)
        return CIFAR_TRANSFORM(corr_img), CIFAR_TRANSFORM(clean_img), y


def get_cifar10_c_loader(config: ProbeConfig, corruption: str, severity: int) -> DataLoader:
    dataset = AlignedCIFAR10CDataset(
        config.data_root,
        corruption,
        severity,
        split=config.target_split,
        corruption_source=config.corruption_source,
        synthetic_seed=config.synthetic_seed,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
