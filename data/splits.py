from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _first_existing(candidates: Iterable[Path]) -> Path:
    candidates = list(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def imagenet_split_candidates(data_root: str | Path, split: str) -> list[Path]:
    root = Path(data_root)
    return [
        root / "imagenet" / split,
        root / "ImageNet" / split,
        root / "ILSVRC2012" / split,
        root / split,
    ]


def resolve_imagenet_split_root(data_root: str | Path, split: str) -> Path:
    """Resolve an ImageNet split directory under common local layouts."""

    normalized = split.lower()
    if normalized == "test":
        normalized = "val"
    if normalized not in {"train", "val"}:
        raise ValueError(f"Unsupported ImageNet split '{split}'. Expected 'train', 'val', or labeled proxy 'test'.")

    return _first_existing(imagenet_split_candidates(data_root, normalized))
