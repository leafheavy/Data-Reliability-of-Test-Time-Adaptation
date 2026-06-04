from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _first_existing(candidates: Iterable[Path]) -> Path:
    candidates = list(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_imagenet_split_root(data_root: str | Path, split: str) -> Path:
    """Resolve an ImageNet split directory under common local layouts.

    ``split='test'`` intentionally falls back to ``val`` because the official
    ImageNet test split has no public labels. This codebase needs labels for the
    corruption/domain-shift diagnostics, so the validation split is the labeled
    test-domain proxy used by ImageNet-C-style evaluation.
    """

    root = Path(data_root)
    normalized = split.lower()
    if normalized == "test":
        normalized = "val"
    if normalized not in {"train", "val"}:
        raise ValueError(f"Unsupported ImageNet split '{split}'. Expected 'train', 'val', or labeled proxy 'test'.")

    return _first_existing(
        [
            root / "imagenet" / normalized,
            root / "ImageNet" / normalized,
            root / "ILSVRC2012" / normalized,
            root / normalized,
        ]
    )
