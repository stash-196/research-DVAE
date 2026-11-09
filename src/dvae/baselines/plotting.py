"""Figures for the GRU-D and SINDy baselines.

Every axes object goes through ``aggregate_plot_style`` so these plots use
the same fonts, scales, and save padding as the OTF aggregate figures.
"""

from __future__ import annotations

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


def _as_2d(traj):
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def plot_trajectories(
    gt,
    pred,
    path,
    *,
    flip_index=None,
    channel_names=None,
    pred_label="model",
):
    """Overlay ground truth and a prediction, one panel per channel."""
    apply_paper_ready_line_style()
    gt = _as_2d(gt)
    pred = _as_2d(pred)
    length = min(len(gt), len(pred))
    gt = gt[:length]
    pred = pred[:length]
    n_state = gt.shape[1]
    if channel_names is None:
        channel_names = [f"x{d}" for d in range(n_state)]
    fig, axes = plt.subplots(
        n_state, 1, figsize=(8, 2.4 * n_state), sharex=True, squeeze=False
    )
    time = np.arange(length)
    for d in range(n_state):
        ax = axes[d, 0]
        ax.plot(time, gt[:, d], color="black", label="ground truth")
        ax.plot(time, pred[:, d], color="C0", linestyle="--", label=pred_label)
        if flip_index is not None:
            ax.axvline(int(flip_index), color="0.45", linewidth=1.0)
        ax.set_ylabel(channel_names[d])
        finite = np.concatenate(
            [
                gt[np.isfinite(gt[:, d]), d],
                pred[np.isfinite(pred[:, d]), d],
            ]
        )
        if finite.size:
            setup_plot_y_axis(ax, finite)
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("time index")
    save_figure(fig, path)
    plt.close(fig)


def plot_metric_bars(metrics, keys, path):
    """Bar chart of a few scalar scores. Non-finite values are omitted."""
    apply_paper_ready_line_style()
    labels = []
    values = []
    for key in keys:
        value = metrics.get(key, float("nan"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        labels.append(get_metric_display_name(key))
        values.append(value)
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(values)), values, color="0.25")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    setup_plot_y_axis(ax, values)
    save_figure(fig, path)
    plt.close(fig)


def plot_coefficients(names, coefficients, path):
    """Stem/bar chart of SINDy coefficients, one panel per state derivative."""
    apply_paper_ready_line_style()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    n_state = coefficients.shape[1]
    fig, axes = plt.subplots(
        n_state, 1, figsize=(10, 2.4 * n_state), squeeze=False, sharex=True
    )
    positions = np.arange(len(names))
    for k in range(n_state):
        ax = axes[k, 0]
        ax.bar(positions, coefficients[:, k], color="0.25")
        ax.set_ylabel(f"dx{k}/dt")
        column = coefficients[:, k]
        if np.any(np.isfinite(column)):
            setup_plot_y_axis(ax, column[np.isfinite(column)])
    axes[-1, 0].set_xticks(positions)
    axes[-1, 0].set_xticklabels(list(names), rotation=60, ha="right")
    save_figure(fig, path)
    plt.close(fig)


def plot_loss_curves(train_loss, val_loss, path):
    apply_paper_ready_line_style()
    train_loss = np.asarray(train_loss, dtype=np.float64)
    val_loss = np.asarray(val_loss, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color="C0", label="train")
    if len(val_loss):
        ax.plot(np.arange(1, len(val_loss) + 1), val_loss, color="C1", label="validation")
    finite = np.concatenate([train_loss[np.isfinite(train_loss)], val_loss[np.isfinite(val_loss)]])
    if finite.size:
        setup_plot_y_axis(ax, finite)
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE")
    ax.legend(frameon=False)
    save_figure(fig, path)
    plt.close(fig)
