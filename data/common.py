from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data import corruption_method


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

CIFAR_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


IMAGENET_C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

VALID_CORRUPTIONS = set(IMAGENET_C_CORRUPTIONS)


def get_corruption_fn(name: str) -> Callable:
    if name not in VALID_CORRUPTIONS or not hasattr(corruption_method, name):
        raise ValueError(
            f"Synthetic corruption '{name}' is unavailable. "
            f"Available corruptions are: {sorted(SYNTHETIC_CORRUPTIONS)}"
        )
    return getattr(corruption_method, name)


SYNTHETIC_CORRUPTIONS = {
    name for name in VALID_CORRUPTIONS
    if hasattr(corruption_method, name)
}


def corrupt_pil_image(image: Image.Image, corruption: str, severity: int, seed: int | None = None) -> Image.Image:
    fn = get_corruption_fn(corruption)
    image = image.convert("RGB")
    if seed is None:
        arr = fn(image, severity=severity)
    else:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            arr = fn(image, severity=severity)
        finally:
            np.random.set_state(state)
    arr = np.asarray(arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def as_image_batch(batch_or_list) -> torch.Tensor:
    if isinstance(batch_or_list, torch.Tensor):
        return batch_or_list
    if len(batch_or_list) == 0:
        return torch.empty(0)
    return torch.stack([x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in batch_or_list])
