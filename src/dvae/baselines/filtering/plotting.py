"""Figures for filtering baselines, using the shared aggregate plot style."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dvae.eval.aggregate_plot_style import (
    apply_paper_ready_line_style,
    get_metric_display_name,
    save_figure,
    setup_plot_y_axis,
)

# Names that are not in aggregate_plot_style. Local only: that module is on
# an open pull request's edit list and is not modified here.
_EXTRA_DISPLAY = {
    "mse_impute_predicted": "MSE on missing (one-step)",
    "mse_impute_filtered": "MSE on missing (filtered)",
    "mse_impute_smoothed": "MSE on missing (smoothed)",
    "mse_smoothed": "MSE smoothed",
    "kld_smoothed": "KLD smoothed",
    "spectrum_error_smoothed": "Spectrum error smoothed",
    "mse_impute_gibbs": "MSE on missing (Gibbs)",
    "mse_impute_ml": "MSE on missing (ML substitution)",
}


def _display(metric: str) -> str:
    if metric in _EXTRA_DISPLAY:
        return _EXTRA_DISPLAY[metric]
    return get_metric_display_name(metric)


def plot_trajectory_window(
    gt: np.ndarray,
    series: Mapping[str, np.ndarray],
    missing: Optional[np.ndarray],
    path: Path,
    *,
    channel: int = 0,
    flip_point: Optional[int] = None,
    title: str = "",
) -> None:
    """Overlay GT and baseline traces for one channel of one window."""
    apply_paper_ready_line_style()
    gt = np.asarray(gt, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(gt.shape[0])
    ax.plot(time, gt[:, channel], color="black", linewidth=1.5, label="GT")
    colors = {
        "predicted": "green",
        "filtered": "C0",
        "smoothed": "C1",
        "autonomous": "red",
        "gibbs": "C4",
        "ml": "C5",
        "particle": "C6",
    }
    for name, values in series.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        ax.plot(
            time[: values.shape[0]],
            values[:, channel],
            color=colors.get(name, None),
            linewidth=1.2,
            label=name,
            alpha=0.9,
        )
    if missing is not None:
        missing = np.asarray(missing, dtype=bool)
        if missing.ndim == 2:
            miss_t = missing[:, channel]
        else:
            miss_t = missing
        if miss_t.any():
            ax.scatter(
                time[miss_t],
                gt[miss_t, channel],
                s=12,
                color="black",
                marker="x",
                label="missing",
                zorder=5,
            )
    if flip_point is not None:
        ax.axvline(flip_point, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel(f"channel {channel}")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)
    plt.close(fig)


def plot_metric_bars(
    metrics: Mapping[str, float],
    keys: Sequence[str],
    path: Path,
) -> None:
    """Bar chart of a few scalar metrics, styled like the aggregate plots."""
    apply_paper_ready_line_style()
    values = []
    labels = []
    for key in keys:
        if key not in metrics:
            continue
        val = metrics[key]
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val_f):
            continue
        values.append(val_f)
        labels.append(_display(key))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(values)), 4))
    ax.bar(np.arange(len(values)), values, color="C0")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    setup_plot_y_axis(ax, values)
    ax.set_ylabel("score")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)
    plt.close(fig)
