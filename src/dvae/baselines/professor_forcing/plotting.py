"""Training curves drawn with the shared aggregate plot style."""

from __future__ import annotations

from pathlib import Path

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

PLOT_METRICS = (
    "nll",
    "mse_tf",
    "mse_auto",
    "spectrum_error_auto",
    "kld_auto",
    "disc_accuracy",
    "hidden_match",
)


def plot_metric_history(history: list[dict], save_path: str | Path) -> Path | None:
    """Save one figure of the finite scalar series in ``history``. Returns the path."""
    present = []
    for metric in PLOT_METRICS:
        values = [_finite(row.get(metric)) for row in history]
        if any(value is not None for value in values):
            present.append((metric, values))
    if not present:
        return None

    apply_paper_ready_line_style()
    epochs = [int(row.get("epoch", index)) for index, row in enumerate(history)]
    columns = 2
    rows = int(np.ceil(len(present) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 4 * rows), squeeze=False)
    for index, (metric, values) in enumerate(present):
        ax = axes[index // columns][index % columns]
        finite_epochs = [epoch for epoch, value in zip(epochs, values) if value is not None]
        finite_values = [value for value in values if value is not None]
        ax.plot(finite_epochs, finite_values, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel(get_metric_display_name(metric))
        setup_plot_y_axis(ax, finite_values)
    for index in range(len(present), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)
    plt.close(fig)
    return path


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
