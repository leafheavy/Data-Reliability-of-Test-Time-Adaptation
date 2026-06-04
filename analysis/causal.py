from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from viz.plots import save_figure, setup_style


def _summary(group: pd.DataFrame) -> dict:
    return {
        "count": int(len(group)),
        "corruption_counts": group["corruption"].value_counts().to_dict() if "corruption" in group else {},
        "severity_counts": group["severity"].value_counts().to_dict() if "severity" in group else {},
        "metric_means": group[["delta_H", "delta_A", "low_freq_ratio"]].mean(numeric_only=True).to_dict(),
    }


def analyze_causal(metrics_df: pd.DataFrame, output_path: str) -> None:
    setup_style()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    a_hi = metrics_df["delta_A"].quantile(0.75)
    a_lo = metrics_df["delta_A"].quantile(0.25)
    h_hi = metrics_df["delta_H"].quantile(0.75)
    h_lo = metrics_df["delta_H"].quantile(0.25)
    group_a = metrics_df[(metrics_df["delta_A"] >= a_hi) & (metrics_df["delta_H"] <= h_lo)]
    group_b = metrics_df[(metrics_df["delta_H"] >= h_hi) & (metrics_df["delta_A"] <= a_lo)]
    (out / "causal_groupA_stats.json").write_text(json.dumps(_summary(group_a), indent=2))
    (out / "causal_groupB_stats.json").write_text(json.dumps(_summary(group_b), indent=2))
    plot_df = pd.concat([group_a.assign(group="A: high ΔA low ΔH"), group_b.assign(group="B: high ΔH low ΔA")])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=plot_df, x="delta_A", y="delta_H", hue="group", size="low_freq_ratio", ax=ax)
        ax.set_title("Causal contrast groups")
        save_figure(fig, out / "causal_comparison.png")
