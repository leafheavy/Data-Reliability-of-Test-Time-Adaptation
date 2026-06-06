from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from config import ProbeConfig
from data.cifar10_c import OnlineCIFAR10CDataset
from data.splits import resolve_imagenet_split_root
from data.common import CIFAR_TRANSFORM, IMAGE_TRANSFORM
from models.hooks import clear_activations, register_activation_hooks, remove_activation_hooks
from models.training import load_clean_source_model
from probe.metrics import _activation_moments, compute_rapsd


from torchvision import datasets, transforms

def _clean_loader(config: ProbeConfig):
    # 支持 "cifar10_c" 或标准的 "cifar10"
    if config.dataset in ["cifar10_c", "cifar10"]:
        # 1. 确定加载的是训练集还是测试集
        is_train = (config.source_split == "train")
        
        # 2. 为了保留您原代码中 x.convert("RGB") 的安全转换，我们将它与原 transform 结合
        # 这确保了即使有单通道灰度图，也会被正确转为 3 通道 RGB
        cifar_transform_with_rgb = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            CIFAR_TRANSFORM
        ])
        
        # 3. 直接加载官方标准的干净 CIFAR-10 数据集
        ds = datasets.CIFAR10(
            root=config.data_root,
            train=is_train,
            download=True,             # 自动下载（如果本地路径不存在）
            transform=cifar_transform_with_rgb
        )
        
        # 4. 返回 DataLoader 
        # 注意：这里去掉了 collate_fn，默认 collate 支持多进程，完全兼容 num_workers > 0
        return torch.utils.data.DataLoader(
            ds, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
        )
        
    # ImageNet 加载逻辑保持不变
    from torchvision import datasets as tv_datasets
    ds = tv_datasets.ImageFolder(
        str(resolve_imagenet_split_root(config.data_root, config.source_split)), 
        transform=IMAGE_TRANSFORM
    )
    return torch.utils.data.DataLoader(
        ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )

def precompute_source_stats(config: ProbeConfig, max_batches: int = 0) -> Dict:
    load_clean_source_model(config)  # 为获取 ActMAD Layers 的 side effect 而调用
    # --- 增加安全检查: 避免用户因未配置层名字而生成空文件 ---
    if not config.actmad_layers:
        raise ValueError(
            "config.actmad_layers 为空！请确保在 ProbeConfig 实例化时传入了非空的 actmad_layers 参数，"
            "否则无法计算和保存激活统计数据。"
        )
    
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    model = load_clean_source_model(config).to(device).eval()
    loader = _clean_loader(config)
    activations = register_activation_hooks(model, config.actmad_layers)
    sums = {layer: None for layer in config.actmad_layers}
    sq_sums = {layer: None for layer in config.actmad_layers}
    counts = {layer: 0 for layer in config.actmad_layers}
    rapsd_sum = np.zeros(config.freq_bins, dtype=np.float64)
    n_batches = 0
    with torch.no_grad():
        for batch_idx, (x, _y) in tqdm(enumerate(loader), desc="Calculating source stats", unit="batch"):
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
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--no-train-if-missing", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-lr", type=float, default=0.01)
    args = parser.parse_args()
    config = ProbeConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model_name,
        source_stats_path=args.output,
        batch_size=args.batch_size,
        source_split=args.source_split,
        model_checkpoint=args.model_checkpoint,
        train_if_missing=not args.no_train_if_missing,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
    )
    precompute_source_stats(config, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
