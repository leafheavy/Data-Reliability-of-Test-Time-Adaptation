from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from viz.plots import save_figure, setup_style

METRIC_COLS = ["delta_H", "delta_G", "delta_A", "low_freq_ratio"]


def analyze_correlation(metrics_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    setup_style()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    corr = metrics_df[METRIC_COLS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Global Spearman correlation")
    save_figure(fig, out / "corr_global.png")
    for family, group in metrics_df.groupby("corruption"):
        if len(group) < 2:
            continue
        family_corr = group[METRIC_COLS].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(family_corr, vmin=-1, vmax=1, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title(f"Spearman correlation: {family}")
        save_figure(fig, out / f"corr_{family}.png")
    return corr
