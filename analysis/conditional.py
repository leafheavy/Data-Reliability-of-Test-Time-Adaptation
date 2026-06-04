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
    long_df = metrics_df.melt(id_vars=["corruption", "severity"], value_vars=METRIC_COLS, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=long_df, x="corruption", y="value", hue="metric", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Metric distributions by corruption family")
    save_figure(fig, out / "boxplots_by_family.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    severity_means = long_df.groupby(["severity", "metric"], as_index=False)["value"].mean()
    sns.lineplot(data=severity_means, x="severity", y="value", hue="metric", marker="o", ax=ax)
    ax.set_title("Severity × metric mean curves")
    save_figure(fig, out / "severity_metric_means.png")

    pivot = metrics_df.pivot_table(index="corruption", columns="severity", values="low_freq_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", ax=ax)
    ax.set_title("Mean low-frequency ratio by corruption/severity")
    save_figure(fig, out / "rapsd_shift_mode_heatmap.png")
