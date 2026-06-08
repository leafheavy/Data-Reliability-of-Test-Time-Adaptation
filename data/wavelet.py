from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WaveletFilters:
    low: tuple[float, ...]
    high: tuple[float, ...]


_SQRT2_INV = 1.0 / math.sqrt(2.0)

_FILTERS = {
    "haar": WaveletFilters(
        low=(_SQRT2_INV, _SQRT2_INV),
        high=(_SQRT2_INV, -_SQRT2_INV),
    ),
    # Orthonormal Daubechies-4 analysis filters. They are embedded here so the
    # offline pipeline does not depend on PyWavelets being installed.
    "db4": WaveletFilters(
        low=(
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909308,
            -0.027983769416859854,
            0.6308807679298589,
            0.7148465705529154,
            0.2303778133088964,
        ),
        high=(
            -0.2303778133088964,
            0.7148465705529154,
            -0.6308807679298589,
            -0.027983769416859854,
            0.18703481171909308,
            0.030841381835560764,
            -0.0328830116668852,
            -0.010597401785069032,
        ),
    ),
}


def available_wavelets() -> tuple[str, ...]:
    return tuple(sorted(_FILTERS))


@lru_cache(maxsize=16)
def _base_filter_bank(wavelet: str, dtype_name: str) -> torch.Tensor:
    if wavelet not in _FILTERS:
        raise ValueError(f"Unsupported wavelet '{wavelet}'. Expected one of {available_wavelets()}.")
    dtype = getattr(torch, dtype_name)
    filters = _FILTERS[wavelet]
    low = torch.tensor(filters.low, dtype=dtype)
    high = torch.tensor(filters.high, dtype=dtype)
    kernels = [
        torch.outer(low, low),
        torch.outer(low, high),
        torch.outer(high, low),
        torch.outer(high, high),
    ]
    return torch.stack(kernels, dim=0).unsqueeze(1)


def _filter_bank(wavelet: str, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weight = _base_filter_bank(wavelet, str(dtype).split(".")[-1]).to(device=device)
    return weight.repeat(channels, 1, 1, 1)


def _pad_for(wavelet: str) -> int:
    kernel_size = len(_FILTERS[wavelet].low)
    return max(kernel_size // 2 - 1, 0)


def dwt2(x: torch.Tensor, wavelet: str = "db4") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of grouped 2D DWT for NCHW tensors."""

    if x.ndim != 4:
        raise ValueError(f"Expected an NCHW tensor, got shape {tuple(x.shape)}.")
    if wavelet not in _FILTERS:
        raise ValueError(f"Unsupported wavelet '{wavelet}'. Expected one of {available_wavelets()}.")
    n, channels, _h, _w = x.shape
    pad = _pad_for(wavelet)
    padded = F.pad(x, (pad, pad, pad, pad), mode="circular") if pad else x
    weight = _filter_bank(wavelet, channels, x.device, x.dtype)
    coeffs = F.conv2d(padded, weight, stride=2, groups=channels)
    coeffs = coeffs.view(n, channels, 4, coeffs.shape[-2], coeffs.shape[-1])
    return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]


def idwt2(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    output_size: tuple[int, int],
    wavelet: str = "db4",
) -> torch.Tensor:
    """Inverse of :func:`dwt2` under the same periodic boundary convention."""

    if ll.ndim != 4:
        raise ValueError(f"Expected NCHW wavelet bands, got shape {tuple(ll.shape)}.")
    channels = ll.shape[1]
    target_h, target_w = output_size
    base = _base_filter_bank(wavelet, str(ll.dtype).split(".")[-1]).to(device=ll.device)
    kernel_size = base.shape[-1]
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    pad = (pad_left, pad_right, pad_left, pad_right)
    recon = torch.zeros(ll.shape[0], channels, target_h, target_w, device=ll.device, dtype=ll.dtype)
    for band_index, band in enumerate((ll, lh, hl, hh)):
        upsampled = torch.zeros_like(recon)
        upsampled[..., : band.shape[-2] * 2:2, : band.shape[-1] * 2:2] = band
        weight = base[band_index].flip(-1, -2).unsqueeze(0).repeat(channels, 1, 1, 1)
        recon = recon + F.conv2d(F.pad(upsampled, pad, mode="circular"), weight, groups=channels)
    return recon[..., :target_h, :target_w]


def coarse_grain_batch(x: torch.Tensor, k: int, levels: int = 3, wavelet: str = "db4") -> torch.Tensor:
    """Apply R_k: DWT coarse-graining by zeroing detail bands finer than scale k."""

    if k < 0 or k > levels:
        raise ValueError(f"k must be in [0, {levels}], got {k}.")
    if k == 0:
        return x.clone()

    current = x
    details: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    shapes: list[tuple[int, int]] = []
    for _level in range(levels):
        shapes.append((current.shape[-2], current.shape[-1]))
        ll, lh, hl, hh = dwt2(current, wavelet=wavelet)
        details.append((lh, hl, hh))
        current = ll

    for level in reversed(range(levels)):
        lh, hl, hh = details[level]
        if level < k:
            lh = torch.zeros_like(lh)
            hl = torch.zeros_like(hl)
            hh = torch.zeros_like(hh)
        current = idwt2(current, lh, hl, hh, output_size=shapes[level], wavelet=wavelet)
    return current.clamp(0.0, 1.0)


def dwt_energy_profile(x: torch.Tensor, levels: int = 3, wavelet: str = "db4") -> torch.Tensor:
    """Per-sample LL/detail energy profile used by sample-level descriptors."""

    current = x
    features = []
    for _level in range(levels):
        ll, lh, hl, hh = dwt2(current, wavelet=wavelet)
        for band in (lh, hl, hh):
            features.append(band.square().mean(dim=(1, 2, 3), keepdim=False))
        current = ll
    features.append(current.square().mean(dim=(1, 2, 3), keepdim=False))
    return torch.stack(features, dim=1)
