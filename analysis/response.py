from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from analysis.gamma import distribution_distance, sample_descriptors, unpack_aligned_batch
from config import ProbeConfig
from data.wavelet import coarse_grain_batch
from probe.metrics import compute_A_bar, compute_G_bar, compute_H_bar


@dataclass
class ResponseCell:
    model_name: str
    corruption: str
    severity: int
    k: int
    Acc: float
    H_bar: float
    G: float
    A: float
    R_DWT: float


@dataclass
class ResponseSummary:
    model_name: str
    corruption: str
    severity: int
    k_best_model: int
    Gain_coarse: float


def _empty_accumulator(config: ProbeConfig) -> dict[int, Dict[str, float]]:
    return {
        k: {
            "correct": 0.0,
            "total": 0.0,
            "entropy_sum": 0.0,
            "g_sum": 0.0,
            "a_sum": 0.0,
            "r_sum": 0.0,
            "batches": 0.0,
        }
        for k in range(config.dwt_levels + 1)
    }


def collect_response_cells(
    model: torch.nn.Module,
    loader,
    config: ProbeConfig,
    source_stats: Dict | None,
    corruption: str,
    severity: int,
) -> tuple[list[ResponseCell], ResponseSummary]:
    device = next(model.parameters()).device
    activation_stats = (source_stats or {}).get("activation_stats", {})
    acc = _empty_accumulator(config)
    was_training = model.training
    model.eval()

    progress = tqdm(enumerate(loader), total=len(loader), desc="Frozen response", unit="batch")
    for batch_idx, batch in progress:
        if config.max_batches and batch_idx >= config.max_batches:
            break
        x_corr, x_clean, labels = unpack_aligned_batch(batch)
        x_corr = x_corr.to(device).float()
        x_clean = x_clean.to(device).float()
        labels = labels.to(device).long()
        for k in range(config.dwt_levels + 1):
            x_k = coarse_grain_batch(x_corr, k, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
            clean_k = coarse_grain_batch(x_clean, k, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
            with torch.no_grad():
                logits = model(x_k)
                pred = logits.argmax(dim=1)
            acc[k]["correct"] += float(pred.eq(labels).sum().item())
            acc[k]["total"] += float(labels.numel())
            acc[k]["entropy_sum"] += compute_H_bar(model, x_k.detach().cpu()) * float(labels.numel())
            acc[k]["g_sum"] += compute_G_bar(model, x_k.detach().cpu())
            acc[k]["a_sum"] += compute_A_bar(
                model,
                x_k.detach().cpu(),
                activation_stats,
                config.actmad_layers,
                config.min_batch_size_for_actmad,
            )
            r_value = distribution_distance(
                sample_descriptors(x_k.detach().cpu(), config),
                sample_descriptors(clean_k.detach().cpu(), config),
                config,
            )
            acc[k]["r_sum"] += r_value
            acc[k]["batches"] += 1.0

    model.train(was_training)
    cells = []
    for k, values in acc.items():
        total = max(values["total"], 1.0)
        batches = max(values["batches"], 1.0)
        cells.append(
            ResponseCell(
                model_name=config.model_name,
                corruption=corruption,
                severity=severity,
                k=k,
                Acc=100.0 * values["correct"] / total,
                H_bar=values["entropy_sum"] / total,
                G=values["g_sum"] / batches,
                A=values["a_sum"] / batches if not np.isnan(values["a_sum"]) else float("nan"),
                R_DWT=values["r_sum"] / batches,
            )
        )

    best = max(cells, key=lambda cell: cell.Acc)
    acc0 = next(cell.Acc for cell in cells if cell.k == 0)
    summary = ResponseSummary(
        model_name=config.model_name,
        corruption=corruption,
        severity=severity,
        k_best_model=best.k,
        Gain_coarse=best.Acc - acc0,
    )
    return cells, summary


def save_response_outputs(cells: list[ResponseCell], summaries: list[ResponseSummary], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(cell) for cell in cells]).to_csv(out / "response_curves.csv", index=False)
    pd.DataFrame([asdict(summary) for summary in summaries]).to_csv(out / "response_summary.csv", index=False)
