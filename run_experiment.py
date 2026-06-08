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

from analysis.gamma import LEVELS, collect_gamma_descriptors, compute_gamma_cells, save_gamma_outputs
from analysis.response import collect_response_cells, save_response_outputs
from config import ProbeConfig
from probe.metrics import BatchMetrics, compute_batch_metrics
from probe.optimize import run_probe

CSV_COLUMNS = [
    "batch_id", "model_name", "corruption", "severity", "lambda2",
    "H_err", "H_star", "delta_H", "G_err", "G_star", "delta_G",
    "A_err", "A_star", "delta_A", "low_freq_ratio", "cosine_sim_delta", "l2_ratio_delta",
]


def _source_stats_files(config: ProbeConfig) -> tuple[Path, Path]:
    root = Path(config.source_stats_path) / config.model_name
    return root / "activation_stats.pkl", root / "rapsd_src.npy"


def _load_source_stats(config: ProbeConfig) -> Dict:
    act_path, rapsd_path = _source_stats_files(config)
    if not act_path.exists() or not rapsd_path.exists():
        raise FileNotFoundError(
            "ActMAD source statistics are missing. Expected both files:\n"
            f"  - {act_path}\n"
            f"  - {rapsd_path}\n"
            "Run scripts/run_precompute.sh first, or allow the main pipeline to precompute them."
        )
    with open(act_path, "rb") as f:
        activation_stats = pickle.load(f)
    return {"activation_stats": activation_stats, "rapsd_src": np.load(rapsd_path)}


def _load_or_precompute_source_stats(config: ProbeConfig, model: torch.nn.Module) -> Dict:
    act_path, rapsd_path = _source_stats_files(config)
    needs_precompute = not act_path.exists() or not rapsd_path.exists()

    if not needs_precompute:
        source_stats = _load_source_stats(config)
        missing_layers = [
            layer for layer in config.actmad_layers
            if layer not in source_stats.get("activation_stats", {})
        ]
        needs_precompute = bool(missing_layers)

    if needs_precompute:
        from data.precompute_stats import precompute_source_stats

        print(
            "ActMAD source statistics are missing or incomplete; "
            f"precomputing them under {Path(config.source_stats_path) / config.model_name}..."
        )
        precompute_source_stats(config, max_batches=config.max_batches, model=model)

    source_stats = _load_source_stats(config)
    missing_layers = [
        layer for layer in config.actmad_layers
        if layer not in source_stats.get("activation_stats", {})
    ]
    if missing_layers:
        raise RuntimeError(
            "ActMAD source statistics do not contain all required layers: "
            f"{missing_layers}. Delete stale stats under "
            f"{Path(config.source_stats_path) / config.model_name} and rerun precompute."
        )
    return source_stats


def _existing_batch_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    actmad_columns = ["A_err", "A_star", "delta_A"]
    if all(column in df.columns for column in actmad_columns):
        df = df.dropna(subset=actmad_columns)
    return set(df["batch_id"].astype(str))


def _drop_incomplete_actmad_rows(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    actmad_columns = ["A_err", "A_star", "delta_A"]
    if not all(column in df.columns for column in actmad_columns):
        return
    valid = ~df[actmad_columns].isna().any(axis=1)
    if bool(valid.all()):
        return
    df.loc[valid].to_csv(path, index=False)


def _loader_for(config: ProbeConfig, corruption: str, severity: int):
    if config.dataset == "cifar10_c":
        from data.cifar10_c import get_cifar10_c_loader

        return get_cifar10_c_loader(config, corruption, severity)
    if config.dataset == "imagenet_c":
        from data.imagenet_c import get_imagenet_c_loader

        return get_imagenet_c_loader(config, corruption, severity)
    raise ValueError(f"Unsupported dataset '{config.dataset}'")


def _parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _metric_row(batch_id: str, metrics: BatchMetrics) -> Dict:
    row = asdict(metrics)
    row["batch_id"] = batch_id
    row.pop("R_shift", None)
    row.pop("R_delta", None)
    return {col: row.get(col) for col in CSV_COLUMNS}


def run_structure_pipeline(config: ProbeConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    output_dir = Path(config.output_dir) / "gamma"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_cells = []
    summaries = []

    for corruption in config.corruption_families:
        for severity in config.severities:
            loader = _loader_for(config, corruption, severity)
            store = collect_gamma_descriptors(loader, config, device)
            cells, summary = compute_gamma_cells(corruption, severity, store, config)
            all_cells.extend(cells)
            summaries.append(summary)
            save_gamma_outputs(all_cells, summaries, output_dir)


def run_response_pipeline(config: ProbeConfig) -> None:
    from models.training import load_clean_source_model

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    output_dir = Path(config.output_dir) / "response"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_clean_source_model(config).to(device)
    source_stats = _load_or_precompute_source_stats(config, model)
    all_cells = []
    summaries = []

    for corruption in config.corruption_families:
        for severity in config.severities:
            loader = _loader_for(config, corruption, severity)
            cells, summary = collect_response_cells(model, loader, config, source_stats, corruption, severity)
            all_cells.extend(cells)
            summaries.append(summary)
            save_response_outputs(all_cells, summaries, output_dir)


def run_xstar_supplement(config: ProbeConfig) -> None:
    from models.training import load_clean_source_model

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    output_dir = Path(config.output_dir) / "xstar_supplement"
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "traj_logs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    model = load_clean_source_model(config).to(device)
    source_stats = _load_or_precompute_source_stats(config, model)
    _drop_incomplete_actmad_rows(metrics_path)
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
            try:
                from analysis.causal import analyze_causal
                from analysis.conditional import analyze_conditional
                from analysis.correlation import analyze_correlation

                analyze_correlation(df, str(analysis_dir))
                analyze_conditional(df, str(analysis_dir))
                analyze_causal(df, str(analysis_dir))
            except ModuleNotFoundError as exc:
                print(f"Skipping x* supplement plots because an optional plotting dependency is missing: {exc}")


def main() -> None:
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="structure", choices=["structure", "response", "xstar", "all"])
    parser.add_argument("--dataset", default=defaults.dataset, choices=["imagenet_c", "cifar10_c"])
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--corruption-source", default=defaults.corruption_source, choices=["auto", "official", "synthetic"])
    parser.add_argument("--corruptions", default=None, help="Comma-separated corruption names. Defaults to ImageNet-C 15 corruptions.")
    parser.add_argument("--severities", default=None, help="Comma-separated severities. Defaults to 1,2,3,4,5.")
    parser.add_argument("--model-name", default=defaults.model_name, choices=["resnet50", "resnet101", "vit_b16"])
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--opt-steps", type=int, default=defaults.opt_steps)
    parser.add_argument("--lambda1", type=float, default=defaults.lambda1)
    parser.add_argument("--lambda2", type=float, default=defaults.lambda2)
    parser.add_argument("--max-batches", type=int, default=defaults.max_batches)
    parser.add_argument("--source-split", default=defaults.source_split)
    parser.add_argument("--target-split", default=defaults.target_split)
    parser.add_argument("--eval-split", default=defaults.eval_split)
    parser.add_argument("--source-stats-path", default=defaults.source_stats_path)
    parser.add_argument("--model-checkpoint", default=defaults.model_checkpoint)
    parser.set_defaults(train_if_missing=defaults.train_if_missing)
    parser.add_argument("--no-train-if-missing", dest="train_if_missing", action="store_false")
    parser.add_argument("--train-epochs", type=int, default=defaults.train_epochs)
    parser.add_argument("--train-lr", type=float, default=defaults.train_lr)
    parser.add_argument("--dwt-wavelet", default=defaults.dwt_wavelet, choices=["db4", "haar"])
    parser.add_argument("--dwt-levels", type=int, default=defaults.dwt_levels)
    parser.add_argument("--aggregation-levels", default=",".join(defaults.aggregation_levels))
    parser.add_argument("--distance", default=defaults.distance, choices=["mmd", "sliced_wasserstein", "energy"])
    parser.add_argument("--mmd-gamma", type=float, default=defaults.mmd_gamma)
    parser.add_argument("--patch-size", type=int, default=defaults.patch_size)
    parser.add_argument("--patch-stride", type=int, default=defaults.patch_stride)
    parser.add_argument("--max-descriptor-items", type=int, default=defaults.max_descriptor_items)
    parser.add_argument("--epsilon-bootstrap", type=int, default=defaults.epsilon_bootstrap)
    parser.add_argument("--epsilon-quantile", type=float, default=defaults.epsilon_quantile)
    parser.add_argument("--min-label-count", type=int, default=defaults.min_label_count)
    args = parser.parse_args()
    config = ProbeConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        corruption_source=args.corruption_source,
        corruption_families=_parse_csv_list(args.corruptions, defaults.corruption_families),
        severities=_parse_int_csv(args.severities, defaults.severities),
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        opt_steps=args.opt_steps,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_batches=args.max_batches,
        source_split=args.source_split,
        target_split=args.target_split,
        eval_split=args.eval_split,
        source_stats_path=args.source_stats_path,
        model_checkpoint=args.model_checkpoint,
        train_if_missing=args.train_if_missing,
        train_epochs=args.train_epochs,
        train_lr=args.train_lr,
        dwt_wavelet=args.dwt_wavelet,
        dwt_levels=args.dwt_levels,
        aggregation_levels=_parse_csv_list(args.aggregation_levels, defaults.aggregation_levels),
        distance=args.distance,
        mmd_gamma=args.mmd_gamma,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        max_descriptor_items=args.max_descriptor_items,
        epsilon_bootstrap=args.epsilon_bootstrap,
        epsilon_quantile=args.epsilon_quantile,
        min_label_count=args.min_label_count,
    )
    unknown_levels = sorted(set(config.aggregation_levels) - set(LEVELS))
    if unknown_levels:
        raise ValueError(f"Unknown aggregation levels: {unknown_levels}. Expected a subset of {list(LEVELS)}.")
    if args.phase == "structure":
        run_structure_pipeline(config)
    elif args.phase == "response":
        run_response_pipeline(config)
    elif args.phase == "xstar":
        run_xstar_supplement(config)
    elif args.phase == "all":
        run_structure_pipeline(config)
        run_response_pipeline(config)


if __name__ == "__main__":
    main()
