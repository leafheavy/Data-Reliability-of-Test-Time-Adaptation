from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import torch

from analysis.causal import analyze_causal
from analysis.conditional import analyze_conditional
from analysis.correlation import analyze_correlation
from config import ProbeConfig
from data.cifar10_c import get_cifar10_c_loader
from data.imagenet_c import get_imagenet_c_loader
from models.training import load_clean_source_model
from probe.metrics import BatchMetrics, compute_batch_metrics
from probe.optimize import run_probe

CSV_COLUMNS = [
    "batch_id", "model_name", "corruption", "severity", "lambda2",
    "H_err", "H_star", "delta_H", "G_err", "G_star", "delta_G",
    "A_err", "A_star", "delta_A", "low_freq_ratio", "cosine_sim_delta", "l2_ratio_delta",
]


def _load_source_stats(config: ProbeConfig) -> Dict:
    root = Path(config.source_stats_path) / config.model_name
    act_path = root / "activation_stats.pkl"
    rapsd_path = root / "rapsd_src.npy"
    if not act_path.exists() or not rapsd_path.exists():
        return {"activation_stats": {}, "rapsd_src": np.zeros(config.freq_bins, dtype=np.float64)}
    with open(act_path, "rb") as f:
        activation_stats = pickle.load(f)
    return {"activation_stats": activation_stats, "rapsd_src": np.load(rapsd_path)}


def _existing_batch_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["batch_id"])["batch_id"].astype(str))


def _loader_for(config: ProbeConfig, corruption: str, severity: int):
    if config.dataset == "cifar10_c":
        return get_cifar10_c_loader(config, corruption, severity)
    if config.dataset == "imagenet_c":
        return get_imagenet_c_loader(config, corruption, severity)
    raise ValueError(f"Unsupported dataset '{config.dataset}'")


def _metric_row(batch_id: str, metrics: BatchMetrics) -> Dict:
    row = asdict(metrics)
    row["batch_id"] = batch_id
    row.pop("R_shift", None)
    row.pop("R_delta", None)
    return {col: row.get(col) for col in CSV_COLUMNS}


def run_full_pipeline(config: ProbeConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "traj_logs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    model = load_clean_source_model(config).to(device)
    source_stats = _load_source_stats(config)
    completed = _existing_batch_ids(metrics_path)
    new_file = not metrics_path.exists()

    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        for corruption in config.corruption_families:
            for severity in config.severities:
                loader = _loader_for(config, corruption, severity)
                for batch_idx, batch in enumerate(loader):
                    if config.max_batches and batch_idx >= config.max_batches:
                        break
                    batch_id = f"{corruption}_{severity}_{batch_idx}"
                    if batch_id in completed:
                        continue
                    setattr(config, "current_batch_id", batch_id)
                    result = run_probe(model, batch, config, source_stats)
                    if not result.delta_model:
                        completed.add(batch_id)
                        continue
                    metrics = compute_batch_metrics(model, result, source_stats, config, corruption, severity)
                    writer.writerow(_metric_row(batch_id, metrics))
                    f.flush()
                    with open(traj_dir / f"{batch_id}.pkl", "wb") as tf:
                        pickle.dump(result.traj_log, tf)
                    completed.add(batch_id)

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        analysis_dir = output_dir / "analysis"
        if not df.empty:
            analyze_correlation(df, str(analysis_dir))
            analyze_conditional(df, str(analysis_dir))
            analyze_causal(df, str(analysis_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10_c", choices=["imagenet_c", "cifar10_c"])
    parser.add_argument("--data-root", default="/Dataset/yezhong")
    parser.add_argument("--model-name", default="resnet50", choices=["resnet50", "resnet101", "vit_b16"])
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--opt-steps", type=int, default=100)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--source-stats-path", default="/data/source_stats")
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--no-train-if-missing", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-lr", type=float, default=0.01)
    args = parser.parse_args()
    config = ProbeConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        opt_steps=args.opt_steps,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_batches=args.max_batches,
        source_split=args.source_split,
        target_split=args.target_split,
        source_stats_path=args.source_stats_path,
        model_checkpoint=args.model_checkpoint,
        train_if_missing=not args.no_train_if_missing,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
    )
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
