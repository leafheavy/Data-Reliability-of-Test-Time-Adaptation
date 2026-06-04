from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import logging

import numpy as np
import torch
from torch import nn

from config import ProbeConfig
from models.hooks import clear_activations, register_activation_hooks, remove_activation_hooks

LOGGER = logging.getLogger(__name__)
_EPS = 1e-12


@dataclass
class BatchMetrics:
    H_err: float
    H_star: float
    delta_H: float
    G_err: float
    G_star: float
    delta_G: float
    A_err: float
    A_star: float
    delta_A: float
    R_shift: np.ndarray
    R_delta: np.ndarray
    low_freq_ratio: float
    cosine_sim_delta: float
    l2_ratio_delta: float
    corruption: str
    severity: int
    model_name: str
    lambda2: float


def _stack(batch: List[torch.Tensor] | torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    tensor = batch if isinstance(batch, torch.Tensor) else torch.stack(batch)
    return tensor.to(device) if device is not None else tensor


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(_EPS))).sum(dim=1)


def compute_H_bar(model: nn.Module, batch: List[torch.Tensor] | torch.Tensor) -> float:
    device = next(model.parameters()).device
    x = _stack(batch, device)
    if x.numel() == 0:
        return float("nan")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        value = _entropy_from_logits(model(x)).mean().item()
    model.train(was_training)
    return float(value)


def compute_G_bar(model: nn.Module, batch: List[torch.Tensor] | torch.Tensor) -> float:
    """L2 norm of gradient of batch mean entropy w.r.t. model parameters."""

    device = next(model.parameters()).device
    x = _stack(batch, device)
    if x.numel() == 0:
        return float("nan")
    was_training = model.training
    requires_grad_state = [p.requires_grad for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    logits = model(x)
    h_bar = _entropy_from_logits(logits).mean()
    h_bar.backward()
    sq_sum = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            sq_sum = sq_sum + param.grad.detach().pow(2).sum()
    value = torch.sqrt(sq_sum).item()
    model.zero_grad(set_to_none=True)
    for param, state in zip(model.parameters(), requires_grad_state):
        param.requires_grad_(state)
    model.train(was_training)
    return float(value)


def _activation_moments(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.float()
    if tensor.ndim == 4:  # NCHW: channel-wise moments over batch/spatial axes
        dims = (0, 2, 3)
    elif tensor.ndim == 3:  # ViT: N tokens C
        dims = (0, 1)
    elif tensor.ndim == 2:
        dims = (0,)
    else:
        dims = tuple(range(max(tensor.ndim - 1, 1)))
    return tensor.mean(dim=dims), tensor.var(dim=dims, unbiased=False)


def compute_A_bar(
    model: nn.Module,
    batch: List[torch.Tensor] | torch.Tensor,
    source_stats: Dict[str, Dict[str, torch.Tensor]],
    layers: List[str],
    min_batch_size: int = 8,
) -> float:
    if not layers or not source_stats:
        return float("nan")
    device = next(model.parameters()).device
    x = _stack(batch, device)
    if x.shape[0] < min_batch_size:
        LOGGER.warning("ActMAD variance is unreliable for batch size %s < %s", x.shape[0], min_batch_size)
    activations = register_activation_hooks(model, layers)
    was_training = model.training
    model.eval()
    try:
        clear_activations(activations)
        with torch.no_grad():
            _ = model(x)
        total = 0.0
        for layer in layers:
            if layer not in source_stats or not activations[layer]:
                continue
            act = torch.cat(activations[layer], dim=0)
            mu, var = _activation_moments(act)
            src_mu = source_stats[layer]["mean"].to(mu.device, dtype=mu.dtype)
            src_var = source_stats[layer]["var"].to(var.device, dtype=var.dtype)
            total += (mu - src_mu).abs().sum().item() + (var - src_var).abs().sum().item()
        return float(total)
    finally:
        remove_activation_hooks(activations)
        model.train(was_training)


def compute_rapsd(images: List[torch.Tensor] | torch.Tensor, freq_bins: int = 64) -> np.ndarray:
    x = _stack(images).detach().cpu().float()
    if x.numel() == 0:
        return np.zeros(freq_bins, dtype=np.float64)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    curves = []
    for image in x:
        gray = image.mean(dim=0).numpy()
        fft = np.fft.fftshift(np.fft.fft2(gray))
        power = np.abs(fft) ** 2
        h, w = power.shape
        yy, xx = np.indices((h, w))
        radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
        bins = np.linspace(0, radius.max() + _EPS, freq_bins + 1)
        which = np.digitize(radius.ravel(), bins) - 1
        curve = np.zeros(freq_bins, dtype=np.float64)
        counts = np.bincount(which.clip(0, freq_bins - 1), minlength=freq_bins)
        sums = np.bincount(which.clip(0, freq_bins - 1), weights=power.ravel(), minlength=freq_bins)
        valid = counts > 0
        curve[valid] = sums[valid] / counts[valid]
        curves.append(curve)
    return np.mean(curves, axis=0)


def compute_R_shift(batch: List[torch.Tensor] | torch.Tensor, rapsd_src: np.ndarray, freq_bins: int = 64) -> np.ndarray:
    return compute_rapsd(batch, freq_bins=freq_bins) - np.asarray(rapsd_src)


def compute_R_delta(deltas: List[torch.Tensor] | torch.Tensor, freq_bins: int = 64) -> np.ndarray:
    return compute_rapsd(deltas, freq_bins=freq_bins)


def low_freq_ratio(rapsd: np.ndarray, r0_bin: int) -> float:
    values = np.asarray(rapsd, dtype=np.float64)
    denom = values.sum()
    if abs(denom) < _EPS:
        return 0.0
    cutoff = min(max(int(r0_bin), 0), len(values) - 1)
    return float(values[: cutoff + 1].sum() / denom)


def compute_delta_alignment(delta_model: List[torch.Tensor] | torch.Tensor, delta_data: List[torch.Tensor] | torch.Tensor) -> Dict[str, float | bool]:
    dm = _stack(delta_model).detach().cpu().float().flatten(start_dim=1)
    dd = _stack(delta_data).detach().cpu().float().flatten(start_dim=1)
    if dm.numel() == 0 or dd.numel() == 0:
        return {"cosine_sim": float("nan"), "l2_ratio": float("nan"), "freq_peak_match": False}
    cos = torch.nn.functional.cosine_similarity(dm, dd, dim=1, eps=_EPS).mean().item()
    dm_norm = dm.norm(dim=1)
    dd_norm = dd.norm(dim=1).clamp_min(_EPS)
    ratio = (dm_norm / dd_norm).mean().item()
    r_model = compute_rapsd(delta_model)
    r_data = compute_rapsd(delta_data)
    peak_match = bool(abs(int(np.argmax(r_model)) - int(np.argmax(r_data))) <= 2)
    return {"cosine_sim": float(cos), "l2_ratio": float(ratio), "freq_peak_match": peak_match}


def compute_batch_metrics(
    model: nn.Module,
    result,
    source_stats: Dict,
    config: ProbeConfig,
    corruption: str,
    severity: int,
) -> BatchMetrics:
    x_err = result.x_orig
    x_star = result.x_star if result.x_star else [x + d for x, d in zip(result.x_orig, result.delta_model)]
    H_err = compute_H_bar(model, x_err)
    H_star = compute_H_bar(model, x_star)
    G_err = compute_G_bar(model, x_err)
    G_star = compute_G_bar(model, x_star)
    activation_stats = source_stats.get("activation_stats", source_stats)
    A_err = compute_A_bar(model, x_err, activation_stats, config.actmad_layers, config.min_batch_size_for_actmad)
    A_star = compute_A_bar(model, x_star, activation_stats, config.actmad_layers, config.min_batch_size_for_actmad)
    rapsd_src = np.asarray(source_stats.get("rapsd_src", np.zeros(config.freq_bins)))
    R_shift = compute_R_shift(x_err, rapsd_src, config.freq_bins)
    R_delta = compute_R_delta(result.delta_model, config.freq_bins)
    align = compute_delta_alignment(result.delta_model, result.delta_data)
    return BatchMetrics(
        H_err=H_err,
        H_star=H_star,
        delta_H=H_star - H_err,
        G_err=G_err,
        G_star=G_star,
        delta_G=G_star - G_err,
        A_err=A_err,
        A_star=A_star,
        delta_A=A_star - A_err,
        R_shift=R_shift,
        R_delta=R_delta,
        low_freq_ratio=low_freq_ratio(R_delta, config.low_freq_cutoff_bin),
        cosine_sim_delta=float(align["cosine_sim"]),
        l2_ratio_delta=float(align["l2_ratio"]),
        corruption=corruption,
        severity=severity,
        model_name=config.model_name,
        lambda2=config.lambda2,
    )
