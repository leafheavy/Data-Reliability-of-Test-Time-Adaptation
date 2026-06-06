from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.correlation import METRIC_COLS
from viz.plots import save_figure, setup_style


def analyze_conditional(metrics_df: pd.DataFrame, output_path: str) -> None:
    setup_style()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    
    # --- 1. 箱线图（2x2 独立子图，加入空值防护与强制数值转换） ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, metric in enumerate(METRIC_COLS):
        ax = axes[i]
        
        # 强制转换为数值类型并提取非空值，防止 dtype=object 或全 NaN 导致 Seaborn 崩溃
        metric_series = pd.to_numeric(metrics_df[metric], errors="coerce")
        valid_df = metrics_df[metric_series.notna()].copy()
        valid_df[metric] = metric_series[metric_series.notna()]
        
        # 如果该指标无有效数据，显示文字占位符，不进行箱线图绘制
        if valid_df.empty or valid_df["corruption"].nunique() == 0:
            ax.text(0.5, 0.5, f"No valid data for {metric}\n(Check if the metric is completely NaN\nor has zero variance)", 
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(f"Distribution of {metric}")
            ax.set_ylabel(metric)
            ax.set_xlabel("Corruption")
            continue
            
        sns.boxplot(data=valid_df, x="corruption", y=metric, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"Distribution of {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Corruption")
    
    fig.tight_layout()
    save_figure(fig, out / "boxplots_by_family.png")

    # --- 2. 严重度折线图（2x2 独立子图，加入空值防护） ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, metric in enumerate(METRIC_COLS):
        ax = axes[i]
        
        # 强制转换为数值类型并过滤非空值
        metric_series = pd.to_numeric(metrics_df[metric], errors="coerce")
        valid_df = metrics_df[metric_series.notna()].copy()
        valid_df[metric] = metric_series[metric_series.notna()]
        
        if valid_df.empty or valid_df["severity"].nunique() == 0:
            ax.text(0.5, 0.5, f"No valid data for {metric}\n(Check if the metric is completely NaN)", 
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_title(f"Mean {metric} vs Severity")
            ax.set_ylabel(metric)
            ax.set_xlabel("Severity")
            continue
            
        severity_means = valid_df.groupby("severity", as_index=False)[metric].mean()
        sns.lineplot(data=severity_means, x="severity", y=metric, marker="o", ax=ax)
        ax.set_title(f"Mean {metric} vs Severity")
        ax.set_ylabel(metric)
        ax.set_xlabel("Severity")
        
    fig.tight_layout()
    save_figure(fig, out / "severity_metric_means.png")

    # --- 3. 频域比例热力图（加入空值防护） ---
    lfr_series = pd.to_numeric(metrics_df["low_freq_ratio"], errors="coerce")
    valid_lfr = metrics_df[lfr_series.notna()].copy()
    valid_lfr["low_freq_ratio"] = lfr_series[lfr_series.notna()]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if not valid_lfr.empty:
        pivot = valid_lfr.pivot_table(index="corruption", columns="severity", values="low_freq_ratio", aggfunc="mean")
        sns.heatmap(pivot, annot=True, cmap="viridis", ax=ax)
    else:
        ax.text(0.5, 0.5, "No valid data for low_freq_ratio", ha="center", va="center", color="gray", transform=ax.transAxes)
    ax.set_title("Mean low-frequency ratio by corruption/severity")
    save_figure(fig, out / "rapsd_shift_mode_heatmap.png")