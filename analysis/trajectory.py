from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from viz.plots import save_figure, setup_style


def analyze_trajectory(traj_logs: List[List[Dict]], output_path: str) -> None:
    setup_style()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if not traj_logs:
        return
    metrics = ["H", "G", "A", "low_freq_ratio"]
    max_steps = max(len(sample) for sample in traj_logs)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, metrics):
        values = np.full((len(traj_logs), max_steps), np.nan)
        for i, sample in enumerate(traj_logs):
            for t, row in enumerate(sample):
                values[i, t] = row.get(metric, np.nan)
        ax.plot(np.nanmean(values, axis=0))
        ax.set_title(f"{metric}(x_t) vs t")
        ax.set_xlabel("step")
    save_figure(fig, out / "trajectory.png")
