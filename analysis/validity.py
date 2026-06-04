from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.correlation import METRIC_COLS
from viz.plots import save_figure, setup_style


def analyze_lambda2_effect(metrics_varA: pd.DataFrame, metrics_varB: pd.DataFrame, output_path: str) -> None:
    setup_style()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    diff = metrics_varB[METRIC_COLS].corr(method="spearman") - metrics_varA[METRIC_COLS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(diff, vmin=-1, vmax=1, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("λ2=0.5 minus λ2=0 correlation")
    save_figure(fig, out / "validity_lambda2_corr_diff.png")
    merged = metrics_varA.merge(metrics_varB, on=["batch_id", "corruption", "severity"], suffixes=("_A", "_B")) if "batch_id" in metrics_varA else pd.DataFrame()
    if not merged.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.scatterplot(data=merged, x="delta_H_A", y="delta_H_B", ax=axes[0])
        sns.scatterplot(data=merged, x="delta_A_A", y="delta_A_B", ax=axes[1])
        axes[0].set_title("ΔH: Variant A vs B")
        axes[1].set_title("ΔA: Variant A vs B")
        save_figure(fig, out / "validity_lambda2_comparison.png")
