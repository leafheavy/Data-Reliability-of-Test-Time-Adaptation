from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import ProbeConfig
from probe.metrics import compute_A_bar, compute_G_bar, compute_H_bar, compute_R_delta, low_freq_ratio


@dataclass
class OptimizationResult:
    x_star: List[Tensor]
    delta_model: List[Tensor]
    delta_data: List[Tensor]
    traj_log: List[List[Dict]]
    batch_id: str
    x_orig: List[Tensor]
    x_clean: List[Tensor]
    labels: List[int]


def _unpack_batch(batch) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3 and isinstance(batch[0], Tensor):
        return batch[0], batch[1], batch[2]
    x_corr, x_clean, y = zip(*batch)
    return torch.stack(list(x_corr)), torch.stack(list(x_clean)), torch.as_tensor(y)


def _set_param_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def run_probe(model: nn.Module, batch, config: ProbeConfig, source_stats: Dict) -> OptimizationResult:
    """Run offline diagnostic input optimization on misclassified samples only."""

    if config.lambda1 <= 0:
        raise ValueError("config.lambda1 must be > 0 to avoid unconstrained adversarial optimization")
    if config.lambda2 < 0:
        raise ValueError("config.lambda2 must be >= 0")

    device = next(model.parameters()).device
    x_corr, x_clean, y = _unpack_batch(batch)
    x_corr = x_corr.to(device).float()
    x_clean = x_clean.to(device).float()
    y = y.to(device).long()

    was_training = model.training
    model.eval()
    _set_param_grad(model, False)

    with torch.no_grad():
        pred = model(x_corr).argmax(dim=1)
    err_mask = pred.ne(y)
    if not err_mask.any():
        empty = []
        return OptimizationResult(empty, empty, empty, [], getattr(config, "current_batch_id", "empty"), empty, empty, [])

    x_orig = x_corr[err_mask].detach()
    x_clean_err = x_clean[err_mask].detach()
    y_err = y[err_mask].detach()
    x = x_orig.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=config.opt_lr)
    traj_log: List[List[Dict]] = [[] for _ in range(x.shape[0])]
    activation_stats = source_stats.get("activation_stats", source_stats)

    for step in range(config.opt_steps):
        optimizer.zero_grad(set_to_none=True)
        _set_param_grad(model, False)
        logits = model(x)
        ce = F.cross_entropy(logits, y_err)
        l2_orig = (x - x_orig).pow(2).flatten(start_dim=1).sum(dim=1).mean()
        l2_clean = (x - x_clean_err).pow(2).flatten(start_dim=1).sum(dim=1).mean()
        loss = ce + config.lambda1 * l2_orig + config.lambda2 * l2_clean
        loss.backward()

        delta_t = x.detach() - x_orig
        h_t = compute_H_bar(model, x.detach())
        g_t = compute_G_bar(model, x.detach())
        a_t = compute_A_bar(
            model,
            x.detach(),
            activation_stats,
            config.actmad_layers,
            config.min_batch_size_for_actmad,
        )
        lf_t = low_freq_ratio(compute_R_delta(delta_t, config.freq_bins), config.low_freq_cutoff_bin)
        for sample_log in traj_log:
            sample_log.append({"step": step, "H": h_t, "G": g_t, "A": a_t, "low_freq_ratio": lf_t})

        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

    x_final = x.detach().cpu()
    x_orig_cpu = x_orig.detach().cpu()
    x_clean_cpu = x_clean_err.detach().cpu()
    delta_model = [d for d in (x_final - x_orig_cpu)]
    delta_data = [d for d in (x_clean_cpu - x_orig_cpu)]
    x_star = [img for img in x_final] if config.save_xstar else []
    result = OptimizationResult(
        x_star=x_star,
        delta_model=delta_model,
        delta_data=delta_data,
        traj_log=traj_log,
        batch_id=getattr(config, "current_batch_id", "batch"),
        x_orig=[img for img in x_orig_cpu],
        x_clean=[img for img in x_clean_cpu],
        labels=[int(v) for v in y_err.detach().cpu()],
    )
    model.train(was_training)
    _set_param_grad(model, False)
    return result
