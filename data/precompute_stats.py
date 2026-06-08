from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from config import ProbeConfig
from data.common import CIFAR_TRANSFORM, IMAGE_TRANSFORM
from data.splits import imagenet_split_candidates, resolve_imagenet_split_root
from models.hooks import clear_activations, register_activation_hooks, remove_activation_hooks
from models.training import load_clean_source_model
from probe.metrics import _activation_moments, compute_rapsd


def _clean_loader(config: ProbeConfig):
    if config.dataset in ["cifar10_c", "cifar10"]:
        is_train = config.source_split == "train"
        cifar_transform_with_rgb = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            CIFAR_TRANSFORM,
        ])
        ds = datasets.CIFAR10(
            root=config.data_root,
            train=is_train,
            download=True,
            transform=cifar_transform_with_rgb,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    from torchvision import datasets as tv_datasets

    clean_root = resolve_imagenet_split_root(config.data_root, config.source_split)
    if not clean_root.exists():
        candidates = "\n".join(
            f"  - {candidate}" for candidate in imagenet_split_candidates(config.data_root, config.source_split)
        )
        raise FileNotFoundError(
            f"ImageNet clean source split directory not found: {clean_root}\n"
            f"Looked for --source-split {config.source_split!r} under:\n{candidates}\n"
            "Pass --data-root to the directory that contains ImageNet, or pass "
            "--source-split val if only the labeled validation split is available."
        )
    ds = tv_datasets.ImageFolder(str(clean_root), transform=IMAGE_TRANSFORM)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def _actmad_observation_count(activation: torch.Tensor, moment: torch.Tensor) -> int:
    """Return the number of scalar samples represented by each moment entry."""

    return int(activation.numel() // max(moment.numel(), 1))


def precompute_source_stats(config: ProbeConfig, max_batches: int = 0, model: torch.nn.Module | None = None) -> Dict:
    if model is None:
        device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        model = load_clean_source_model(config).to(device)
    else:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if not config.actmad_layers:
        raise ValueError(
            "config.actmad_layers is empty after loading the clean source model; "
            "ActMAD source statistics cannot be computed without activation layers."
        )

    loader = _clean_loader(config)
    activations = register_activation_hooks(model, config.actmad_layers)
    sums = {layer: None for layer in config.actmad_layers}
    sq_sums = {layer: None for layer in config.actmad_layers}
    counts = {layer: 0 for layer in config.actmad_layers}
    rapsd_sum = np.zeros(config.freq_bins, dtype=np.float64)
    n_batches = 0

    try:
        with torch.no_grad():
            progress = tqdm(
                enumerate(loader),
                total=len(loader),
                desc="Calculating source stats",
                unit="batch",
            )
            for batch_idx, (x, _y) in progress:
                if max_batches and batch_idx >= max_batches:
                    break
                x = x.to(device).float()
                clear_activations(activations)
                _ = model(x)
                for layer in config.actmad_layers:
                    if not activations[layer]:
                        raise RuntimeError(f"No activation was captured for ActMAD layer '{layer}'.")
                    act = torch.cat(activations[layer], dim=0)
                    mu, var = _activation_moments(act)
                    count = _actmad_observation_count(act, mu)
                    mu_cpu = mu.detach().cpu()
                    var_cpu = var.detach().cpu()
                    ex2_cpu = var_cpu + mu_cpu.pow(2)
                    if sums[layer] is None:
                        sums[layer] = mu_cpu * count
                        sq_sums[layer] = ex2_cpu * count
                    else:
                        sums[layer] += mu_cpu * count
                        sq_sums[layer] += ex2_cpu * count
                    counts[layer] += count
                rapsd_sum += compute_rapsd(x.detach().cpu(), config.freq_bins)
                n_batches += 1
    finally:
        remove_activation_hooks(activations)
        model.train(was_training)

    if n_batches == 0:
        raise RuntimeError(
            "No source batches were processed while precomputing statistics. "
            "Check --data-root, --source-split, and --max-batches."
        )

    activation_stats = {}
    for layer in config.actmad_layers:
        if sums[layer] is None or sq_sums[layer] is None or counts[layer] == 0:
            raise RuntimeError(f"ActMAD layer '{layer}' did not produce any source statistics.")
        mean = sums[layer] / counts[layer]
        ex2 = sq_sums[layer] / counts[layer]
        activation_stats[layer] = {"mean": mean, "var": (ex2 - mean.pow(2)).clamp_min(0)}
    rapsd_src = rapsd_sum / n_batches
    out_dir = Path(config.source_stats_path) / config.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "activation_stats.pkl", "wb") as f:
        pickle.dump(activation_stats, f)
    np.save(out_dir / "rapsd_src.npy", rapsd_src)
    return {"activation_stats": activation_stats, "rapsd_src": rapsd_src}


def main() -> None:
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=defaults.dataset, choices=["imagenet_c", "cifar10_c"])
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--model-name", default=defaults.model_name, choices=["resnet50", "resnet101", "vit_b16"])
    parser.add_argument("--output", default=defaults.source_stats_path)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-batches", type=int, default=defaults.max_batches)
    parser.add_argument("--source-split", default=defaults.source_split)
    parser.add_argument("--model-checkpoint", default=defaults.model_checkpoint)
    parser.set_defaults(train_if_missing=defaults.train_if_missing)
    parser.add_argument("--no-train-if-missing", dest="train_if_missing", action="store_false")
    parser.add_argument("--train-epochs", type=int, default=defaults.train_epochs)
    parser.add_argument("--train-lr", type=float, default=defaults.train_lr)
    args = parser.parse_args()
    config = ProbeConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model_name,
        source_stats_path=args.output,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        source_split=args.source_split,
        model_checkpoint=args.model_checkpoint,
        train_if_missing=args.train_if_missing,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
    )
    precompute_source_stats(config, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
