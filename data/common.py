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


# 明确列出所有合法的 corruption 方法
VALID_CORRUPTIONS = {
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "brightness",
    "contrast",
}

def get_corruption_fn(name: str) -> Callable:
    if name not in VALID_CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{name}'. "
            f"Valid corruptions are: {sorted(VALID_CORRUPTIONS)}"
        )
    return getattr(corruption_method, name)


def corrupt_pil_image(image: Image.Image, corruption: str, severity: int) -> Image.Image:
    fn = get_corruption_fn(corruption)
    image = image.convert("RGB")
    arr = fn(image, severity=severity)
    arr = np.asarray(arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def as_image_batch(batch_or_list) -> torch.Tensor:
    if isinstance(batch_or_list, torch.Tensor):
        return batch_or_list
    if len(batch_or_list) == 0:
        return torch.empty(0)
    return torch.stack([x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in batch_or_list])
