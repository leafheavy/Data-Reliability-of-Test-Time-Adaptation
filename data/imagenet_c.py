from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from config import ProbeConfig
from data.common import IMAGE_TRANSFORM, SYNTHETIC_CORRUPTIONS, corrupt_pil_image
from data.splits import resolve_imagenet_split_root


def imagenet_c_root_candidates(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    return [
        root / "ImageNet-C",
        root / "imagenet-c",
        root / "imagenet_c",
        root / "ImageNet_C",
    ]


def resolve_imagenet_c_root(data_root: str | Path, corruption: str, severity: int) -> Path | None:
    for root in imagenet_c_root_candidates(data_root):
        for sev in (str(severity), severity):
            candidate = root / corruption / str(sev)
            if candidate.exists():
                return candidate
    return None


class AlignedImageNetCDataset(Dataset):
    """Aligned Q_c/P_0 ImageNet-C pairs for offline diagnosis."""

    def __init__(
        self,
        clean_root: str | Path,
        data_root: str | Path,
        corruption: str,
        severity: int,
        corruption_source: str = "auto",
        synthetic_seed: int = 1337,
    ):
        self.clean_root = Path(clean_root)
        if not self.clean_root.exists():
            raise FileNotFoundError(
                f"ImageNet clean split directory not found: {self.clean_root}. "
                "ImageNet is currently optional/skippable; use dataset='cifar10_c' for debug."
            )
        self.base = datasets.ImageFolder(str(self.clean_root))
        self.corruption = corruption
        self.severity = severity
        self.synthetic_seed = synthetic_seed
        self.corrupted = None

        if corruption_source not in {"auto", "official", "synthetic"}:
            raise ValueError("corruption_source must be one of: auto, official, synthetic.")

        c_root = resolve_imagenet_c_root(data_root, corruption, severity)
        if corruption_source in {"auto", "official"} and c_root is not None:
            self.corrupted = datasets.ImageFolder(str(c_root))
            if len(self.corrupted) != len(self.base):
                raise ValueError(
                    f"ImageNet-C split {c_root} has {len(self.corrupted)} samples, "
                    f"but clean reference {self.clean_root} has {len(self.base)} samples."
                )
        elif corruption_source == "official":
            searched = "\n".join(
                f"  - {root / corruption / str(severity)}"
                for root in imagenet_c_root_candidates(data_root)
            )
            raise FileNotFoundError(f"ImageNet-C official corruption directory not found. Looked for:\n{searched}")
        elif corruption not in SYNTHETIC_CORRUPTIONS:
            raise ValueError(
                f"No official ImageNet-C directory was found for '{corruption}', "
                "and the synthetic fallback does not implement it."
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, y = self.base.samples[index]
        clean_img = Image.open(path).convert("RGB")
        if self.corrupted is not None:
            corr_path, corr_y = self.corrupted.samples[index]
            corr_img = Image.open(corr_path).convert("RGB")
            y = corr_y
        else:
            seed = self.synthetic_seed + index + 100000 * self.severity
            corr_img = corrupt_pil_image(clean_img, self.corruption, self.severity, seed=seed)
        return IMAGE_TRANSFORM(corr_img), IMAGE_TRANSFORM(clean_img), y


def _resolve_imagenet_val_root(data_root: str | Path) -> Path:
    """Backward-compatible resolver for ImageNet labeled validation data."""

    return resolve_imagenet_split_root(data_root, "val")


def get_imagenet_c_loader(config: ProbeConfig, corruption: str, severity: int) -> DataLoader:
    dataset = AlignedImageNetCDataset(
        resolve_imagenet_split_root(config.data_root, config.target_split),
        config.data_root,
        corruption,
        severity,
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
