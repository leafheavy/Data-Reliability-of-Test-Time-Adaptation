from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from config import ProbeConfig
from data.cifar10_c import OnlineCIFAR10CDataset
from data.imagenet_c import _resolve_imagenet_val_root
from data.common import IMAGE_TRANSFORM
from models.hooks import clear_activations, register_activation_hooks, remove_activation_hooks
from models.zoo import load_frozen_model
from probe.metrics import _activation_moments, compute_rapsd


def _clean_loader(config: ProbeConfig):
    if config.dataset == "cifar10_c":
        ds = OnlineCIFAR10CDataset(config.data_root, "gaussian_noise", 1, train=False).base
        def collate(samples):
            xs, ys = zip(*samples)
            return torch.stack([IMAGE_TRANSFORM(x.convert("RGB")) for x in xs]), torch.as_tensor(ys)
        return torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate)
    from torchvision import datasets
    ds = datasets.ImageFolder(str(_resolve_imagenet_val_root(config.data_root)), transform=IMAGE_TRANSFORM)
    return torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


def precompute_source_stats(config: ProbeConfig, max_batches: int = 0) -> Dict:
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    model = load_frozen_model(config).to(device).eval()
    loader = _clean_loader(config)
    activations = register_activation_hooks(model, config.actmad_layers)
    sums = {layer: None for layer in config.actmad_layers}
    sq_sums = {layer: None for layer in config.actmad_layers}
    counts = {layer: 0 for layer in config.actmad_layers}
    rapsd_sum = np.zeros(config.freq_bins, dtype=np.float64)
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device).float()
            clear_activations(activations)
            _ = model(x)
            for layer in config.actmad_layers:
                act = torch.cat(activations[layer], dim=0)
                mu, var = _activation_moments(act)
                n = x.shape[0]
                if sums[layer] is None:
                    sums[layer] = mu.detach().cpu() * n
                    sq_sums[layer] = (var.detach().cpu() + mu.detach().cpu().pow(2)) * n
                else:
                    sums[layer] += mu.detach().cpu() * n
                    sq_sums[layer] += (var.detach().cpu() + mu.detach().cpu().pow(2)) * n
                counts[layer] += n
            rapsd_sum += compute_rapsd(x.detach().cpu(), config.freq_bins)
            n_batches += 1
    remove_activation_hooks(activations)
    activation_stats = {}
    for layer in config.actmad_layers:
        mean = sums[layer] / max(counts[layer], 1)
        ex2 = sq_sums[layer] / max(counts[layer], 1)
        activation_stats[layer] = {"mean": mean, "var": (ex2 - mean.pow(2)).clamp_min(0)}
    rapsd_src = rapsd_sum / max(n_batches, 1)
    out_dir = Path(config.source_stats_path) / config.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "activation_stats.pkl", "wb") as f:
        pickle.dump(activation_stats, f)
    np.save(out_dir / "rapsd_src.npy", rapsd_src)
    return {"activation_stats": activation_stats, "rapsd_src": rapsd_src}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10_c", choices=["imagenet_c", "cifar10_c"])
    parser.add_argument("--data-root", default="/Dataset/yezhong")
    parser.add_argument("--model-name", default="resnet50", choices=["resnet50", "resnet101", "vit_b16"])
    parser.add_argument("--output", default="/data/source_stats")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    config = ProbeConfig(dataset=args.dataset, data_root=args.data_root, model_name=args.model_name, source_stats_path=args.output, batch_size=args.batch_size)
    precompute_source_stats(config, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
