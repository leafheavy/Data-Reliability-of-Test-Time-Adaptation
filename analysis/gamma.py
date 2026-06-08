from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ProbeConfig
from data.wavelet import coarse_grain_batch, dwt_energy_profile

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from viz.plots import save_figure, setup_style

    PLOTTING_AVAILABLE = True
except ModuleNotFoundError:
    PLOTTING_AVAILABLE = False


LEVELS = ("pixel", "patch", "sample", "label")
EPS = 1e-12


@dataclass
class GammaCell:
    corruption: str
    severity: int
    k: int
    level: str
    gamma: float
    epsilon: float
    in_basin: bool
    distance: str


@dataclass
class GammaSummary:
    corruption: str
    severity: int
    basin_group: str
    basin_area: int
    invariant_pairs: str
    k_star_pixel: int
    k_star_patch: int
    k_star_sample: int
    k_star_label: int
    not_in_basin_pixel: bool
    not_in_basin_patch: bool
    not_in_basin_sample: bool
    not_in_basin_label: bool


def unpack_aligned_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3 and isinstance(batch[0], torch.Tensor):
        return batch[0], batch[1], batch[2]
    x_corr, x_clean, y = zip(*batch)
    return torch.stack(list(x_corr)), torch.stack(list(x_clean)), torch.as_tensor(y)


def _limit_rows(x: torch.Tensor, max_items: int, seed: int = 0) -> torch.Tensor:
    if max_items <= 0 or x.shape[0] <= max_items:
        return x
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=generator, device=x.device)[:max_items]
    return x.index_select(0, indices)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x[:, None]
    return x.flatten(start_dim=1) if x.ndim > 2 else x


def pixel_descriptors(x: torch.Tensor, config: ProbeConfig, seed: int = 0) -> torch.Tensor:
    rows = x.detach().float().permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    return _limit_rows(rows, config.max_descriptor_items, seed=seed).cpu()


def patch_descriptors(x: torch.Tensor, config: ProbeConfig, seed: int = 0) -> torch.Tensor:
    patch_size = min(config.patch_size, x.shape[-2], x.shape[-1])
    stride = min(config.patch_stride, patch_size)
    patches = F.unfold(x.detach().float(), kernel_size=patch_size, stride=stride)
    patches = patches.transpose(1, 2).reshape(-1, x.shape[1], patch_size, patch_size)
    patches = _limit_rows(patches, config.max_descriptor_items, seed=seed)
    mean = patches.mean(dim=(2, 3))
    std = patches.std(dim=(2, 3), unbiased=False)
    dx = patches[..., :, 1:] - patches[..., :, :-1]
    dy = patches[..., 1:, :] - patches[..., :-1, :]
    grad = torch.stack([
        dx.abs().mean(dim=(1, 2, 3)),
        dy.abs().mean(dim=(1, 2, 3)),
    ], dim=1)
    pooled = F.avg_pool2d(patches, kernel_size=3, stride=1, padding=1)
    high_energy = (patches - pooled).square().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    return torch.cat([mean, std, grad, high_energy], dim=1).cpu()


def sample_descriptors(x: torch.Tensor, config: ProbeConfig) -> torch.Tensor:
    x = x.detach().float()
    mean = x.mean(dim=(2, 3))
    std = x.std(dim=(2, 3), unbiased=False)
    gray = x.mean(dim=1, keepdim=True)
    brightness = gray.mean(dim=(2, 3))
    contrast = gray.std(dim=(2, 3), unbiased=False)
    low_layout = F.adaptive_avg_pool2d(gray, output_size=(4, 4)).flatten(start_dim=1)
    coarse = coarse_grain_batch(x, config.dwt_levels, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
    high_energy = (x - coarse).square().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    dwt_profile = dwt_energy_profile(x, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
    return torch.cat([mean, std, brightness, contrast, high_energy, dwt_profile, low_layout], dim=1).cpu()


def descriptors_for_level(x: torch.Tensor, level: str, config: ProbeConfig, seed: int = 0) -> torch.Tensor:
    if level == "pixel":
        return pixel_descriptors(x, config, seed=seed)
    if level == "patch":
        return patch_descriptors(x, config, seed=seed)
    if level == "sample":
        return sample_descriptors(x, config)
    raise ValueError(f"Descriptor level '{level}' is handled separately or unsupported.")


def _rbf_gamma(x: torch.Tensor, y: torch.Tensor, configured_gamma: float) -> float:
    if configured_gamma > 0:
        return configured_gamma
    sample = torch.cat([_limit_rows(x, 1024, seed=17), _limit_rows(y, 1024, seed=29)], dim=0)
    if sample.shape[0] < 2:
        return 1.0
    distances = torch.pdist(sample.float(), p=2).square()
    median = torch.median(distances[distances > 0]) if torch.any(distances > 0) else torch.tensor(1.0)
    return float(1.0 / (2.0 * median.clamp_min(EPS).item()))


def mmd_distance(x: torch.Tensor, y: torch.Tensor, config: ProbeConfig) -> float:
    x = _ensure_2d(_limit_rows(x.float(), config.max_descriptor_items, seed=101))
    y = _ensure_2d(_limit_rows(y.float(), config.max_descriptor_items, seed=103))
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    gamma = _rbf_gamma(x, y, config.mmd_gamma)
    k_xx = torch.exp(-gamma * torch.cdist(x, x).square()).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y).square()).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y).square()).mean()
    mmd2 = (k_xx + k_yy - 2.0 * k_xy).clamp_min(0.0)
    return float(torch.sqrt(mmd2).item())


def energy_distance(x: torch.Tensor, y: torch.Tensor, config: ProbeConfig) -> float:
    x = _ensure_2d(_limit_rows(x.float(), config.max_descriptor_items, seed=107))
    y = _ensure_2d(_limit_rows(y.float(), config.max_descriptor_items, seed=109))
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return float((2.0 * d_xy - d_xx - d_yy).clamp_min(0.0).item())


def sliced_wasserstein_distance(x: torch.Tensor, y: torch.Tensor, config: ProbeConfig, projections: int = 64) -> float:
    x = _ensure_2d(_limit_rows(x.float(), config.max_descriptor_items, seed=113))
    y = _ensure_2d(_limit_rows(y.float(), config.max_descriptor_items, seed=127))
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    dim = x.shape[1]
    generator = torch.Generator(device=x.device)
    generator.manual_seed(131)
    directions = torch.randn(dim, projections, generator=generator, device=x.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(EPS)
    px = torch.sort(x @ directions, dim=0).values
    py = torch.sort(y @ directions, dim=0).values
    n = min(px.shape[0], py.shape[0])
    return float((px[:n] - py[:n]).abs().mean().item())


def distribution_distance(x: torch.Tensor, y: torch.Tensor, config: ProbeConfig) -> float:
    if config.distance == "mmd":
        return mmd_distance(x, y, config)
    if config.distance == "sliced_wasserstein":
        return sliced_wasserstein_distance(x, y, config)
    if config.distance == "energy":
        return energy_distance(x, y, config)
    raise ValueError("config.distance must be one of: mmd, sliced_wasserstein, energy.")


def class_conditional_distance(
    corr: torch.Tensor,
    clean: torch.Tensor,
    corr_labels: torch.Tensor,
    config: ProbeConfig,
    clean_labels: torch.Tensor | None = None,
) -> float:
    corr = _ensure_2d(corr.float())
    clean = _ensure_2d(clean.float())
    corr_labels = corr_labels.detach().cpu().long()
    clean_labels = corr_labels if clean_labels is None else clean_labels.detach().cpu().long()
    values = []
    weights = []
    common_labels = sorted(set(corr_labels.tolist()).intersection(set(clean_labels.tolist())))
    for label in common_labels:
        corr_mask = corr_labels.eq(label)
        clean_mask = clean_labels.eq(label)
        count = min(int(corr_mask.sum().item()), int(clean_mask.sum().item()))
        if count < config.min_label_count:
            continue
        value = distribution_distance(corr[corr_mask], clean[clean_mask], config)
        if not math.isnan(value):
            values.append(value)
            weights.append(count)
    if not values:
        return float("nan")
    return float(np.average(values, weights=weights))


def _empty_descriptor_store(config: ProbeConfig) -> Dict:
    return {
        "corr": {k: {level: [] for level in LEVELS if level != "label"} for k in range(config.dwt_levels + 1)},
        "clean": {k: {level: [] for level in LEVELS if level != "label"} for k in range(config.dwt_levels + 1)},
        "labels": [],
    }


def collect_gamma_descriptors(loader, config: ProbeConfig, device: torch.device | str) -> Dict:
    store = _empty_descriptor_store(config)
    progress = tqdm(enumerate(loader), total=len(loader), desc="Collecting Gamma descriptors", unit="batch")
    for batch_idx, batch in progress:
        if config.max_batches and batch_idx >= config.max_batches:
            break
        x_corr, x_clean, labels = unpack_aligned_batch(batch)
        x_corr = x_corr.to(device).float()
        x_clean = x_clean.to(device).float()
        store["labels"].append(labels.detach().cpu().long())
        for k in range(config.dwt_levels + 1):
            corr_k = coarse_grain_batch(x_corr, k, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
            clean_k = coarse_grain_batch(x_clean, k, levels=config.dwt_levels, wavelet=config.dwt_wavelet)
            for level in ("pixel", "patch", "sample"):
                if level not in config.aggregation_levels and level != "sample":
                    continue
                seed = 1000 * batch_idx + 37 * k + len(level)
                store["corr"][k][level].append(descriptors_for_level(corr_k, level, config, seed=seed))
                store["clean"][k][level].append(descriptors_for_level(clean_k, level, config, seed=seed + 1))
    labels = torch.cat(store["labels"], dim=0) if store["labels"] else torch.empty(0, dtype=torch.long)
    store["labels"] = labels
    for domain in ("corr", "clean"):
        for k in range(config.dwt_levels + 1):
            for level in ("pixel", "patch", "sample"):
                parts = store[domain][k][level]
                store[domain][k][level] = torch.cat(parts, dim=0) if parts else torch.empty(0, 1)
    return store


def clean_clean_epsilon(store: Dict, config: ProbeConfig) -> dict[str, float]:
    epsilons: dict[str, float] = {}
    rng = np.random.default_rng(config.synthetic_seed)
    labels = store["labels"]
    for level in config.aggregation_levels:
        distances = []
        for _ in range(config.epsilon_bootstrap):
            for k in range(config.dwt_levels + 1):
                if level == "label":
                    sample_desc = store["clean"][k]["sample"]
                    n = sample_desc.shape[0]
                    if n < 4:
                        continue
                    perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
                    half = n // 2
                    a_idx, b_idx = perm[:half], perm[half:]
                    if len(a_idx) == 0 or len(b_idx) == 0:
                        continue
                    common = min(len(a_idx), len(b_idx))
                    value = class_conditional_distance(
                        sample_desc[a_idx[:common]],
                        sample_desc[b_idx[:common]],
                        labels[a_idx[:common]],
                        config,
                        clean_labels=labels[b_idx[:common]],
                    )
                else:
                    clean_desc = store["clean"][k][level]
                    n = clean_desc.shape[0]
                    if n < 4:
                        continue
                    perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
                    half = n // 2
                    value = distribution_distance(clean_desc[perm[:half]], clean_desc[perm[half:]], config)
                if not math.isnan(value):
                    distances.append(value)
        epsilons[level] = float(np.quantile(distances, config.epsilon_quantile)) if distances else float("nan")
    return epsilons


def _k_star(cells: Iterable[GammaCell], level: str, max_k: int) -> tuple[int, bool]:
    valid = [cell.k for cell in cells if cell.level == level and cell.in_basin]
    if not valid:
        return max_k + 1, True
    return min(valid), False


def _basin_group(cells: list[GammaCell]) -> str:
    levels_in_basin = {cell.level for cell in cells if cell.in_basin}
    if "label" in levels_in_basin:
        return "C_label"
    if "sample" in levels_in_basin:
        return "B_sample"
    if "pixel" in levels_in_basin or "patch" in levels_in_basin:
        return "A_pixel_patch"
    return "D_none"


def compute_gamma_cells(corruption: str, severity: int, store: Dict, config: ProbeConfig) -> tuple[list[GammaCell], GammaSummary]:
    epsilons = clean_clean_epsilon(store, config)
    labels = store["labels"]
    cells: list[GammaCell] = []
    for k in range(config.dwt_levels + 1):
        for level in config.aggregation_levels:
            if level == "label":
                gamma = class_conditional_distance(store["corr"][k]["sample"], store["clean"][k]["sample"], labels, config)
            else:
                gamma = distribution_distance(store["corr"][k][level], store["clean"][k][level], config)
            epsilon = epsilons.get(level, float("nan"))
            in_basin = bool(not math.isnan(gamma) and not math.isnan(epsilon) and gamma <= epsilon)
            cells.append(GammaCell(corruption, severity, k, level, gamma, epsilon, in_basin, config.distance))

    k_values = {}
    not_in = {}
    for level in LEVELS:
        k_values[level], not_in[level] = _k_star(cells, level, config.dwt_levels)
    invariant_pairs = [f"({cell.k},{cell.level})" for cell in cells if cell.in_basin]
    summary = GammaSummary(
        corruption=corruption,
        severity=severity,
        basin_group=_basin_group(cells),
        basin_area=len(invariant_pairs),
        invariant_pairs=";".join(invariant_pairs),
        k_star_pixel=k_values["pixel"],
        k_star_patch=k_values["patch"],
        k_star_sample=k_values["sample"],
        k_star_label=k_values["label"],
        not_in_basin_pixel=not_in["pixel"],
        not_in_basin_patch=not_in["patch"],
        not_in_basin_sample=not_in["sample"],
        not_in_basin_label=not_in["label"],
    )
    return cells, summary


def save_gamma_outputs(cells: list[GammaCell], summaries: list[GammaSummary], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(cell) for cell in cells]).to_csv(out / "gamma.csv", index=False)
    pd.DataFrame([asdict(summary) for summary in summaries]).to_csv(out / "gamma_summary.csv", index=False)
    (out / "gamma_summary.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2),
        encoding="utf-8",
    )
    plot_gamma_heatmaps(pd.DataFrame([asdict(cell) for cell in cells]), out / "heatmaps")
    plot_basin_summary(pd.DataFrame([asdict(summary) for summary in summaries]), out)


def plot_gamma_heatmaps(gamma_df: pd.DataFrame, output_dir: str | Path) -> None:
    if gamma_df.empty or not PLOTTING_AVAILABLE:
        return
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (corruption, severity), group in gamma_df.groupby(["corruption", "severity"]):
        pivot = group.pivot_table(index="level", columns="k", values="gamma", aggfunc="mean")
        ordered = [level for level in LEVELS if level in pivot.index]
        pivot = pivot.loc[ordered]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(f"Gamma(k,l): {corruption} severity {severity}")
        ax.set_xlabel("DWT coarse-graining scale k")
        ax.set_ylabel("aggregation level l")
        save_figure(fig, out / f"gamma_{corruption}_s{severity}.png")


def plot_basin_summary(summary_df: pd.DataFrame, output_dir: str | Path) -> None:
    if summary_df.empty or not PLOTTING_AVAILABLE:
        return
    setup_style()
    out = Path(output_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = summary_df.pivot_table(index="corruption", columns="severity", values="basin_area", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="mako", ax=ax)
    ax.set_title("Invariant basin area")
    save_figure(fig, out / "basin_area_heatmap.png")


def gamma_vector_frame(gamma_df: pd.DataFrame) -> pd.DataFrame:
    if gamma_df.empty:
        return pd.DataFrame()
    frame = gamma_df.copy()
    frame["feature"] = frame["level"] + "_k" + frame["k"].astype(str)
    return frame.pivot_table(index=["corruption", "severity"], columns="feature", values="gamma", aggfunc="mean")
