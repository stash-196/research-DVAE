"""Shared display names and y-axis styling for aggregate / compare plots.

Kept free of torch / visualizers so compare jobs and unit tests can import
this without the full eval stack. Logic is the same as the former helpers
in ``aggregate_evaluation_results.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
    SymmetricalLogLocator,
)

SYMLOG_LINTHRESH = 0.01

DISPLAY_NAMES = {
    "mask_label": "Missing Ratio",
    "sampling_ratio": "Auto Ratio",
    "observation_process": "Channel",
}

METRIC_DISPLAY_NAMES = {
    "kld_tf": "KLD of Teacher Forced",
    "kld_auto": "KLD of Autonomous",
    "kld_tf_mean": "KLD TF (channel mean)",
    "kld_auto_mean": "KLD Auto (channel mean)",
    "mse_tf_mean": "MSE TF (channel mean)",
    "mse_auto_mean": "MSE Auto (channel mean)",
    "spectrum_error_gt": "Spectrum Error Ground Truth",
    "spectrum_error_tf": "Spectrum Error Teacher Forced",
    "spectrum_error_auto": "Spectrum Error Autonomous",
    "spectrum_error_tf_mean": "Spectrum Error TF (channel mean)",
    "spectrum_error_auto_mean": "Spectrum Error Auto (channel mean)",
}


def is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def sort_key(x):
    """Sort key for numerical values, fallback to string."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return str(x)


def get_display_name(param):
    """Get display name for parameter, default to param if not found."""
    return DISPLAY_NAMES.get(param, param)


def get_metric_display_name(metric):
    """Get display name for metric, default to metric if not found."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def apply_paper_ready_line_style():
    """Matplotlib rcParams matching ``get_plot_config(paper_ready=True)``.

    Duplicated here so compare overlays do not import ``dvae.visualizers``
    (torch / umap / sklearn). Returns the same config dict as that helper.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )
    return {"show_title": False}


def _symlog_vmin_vmax(values, linthresh=SYMLOG_LINTHRESH):
    """Pad finite values for axis/colorbar limits.

    If every finite value is non-negative (or non-positive), do not pad across
    zero — that unused half is what made all-positive KLD-auto symlog plots
    look like they ranged down to -1 / -10.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -linthresh, linthresh
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    all_nonneg = vmin >= 0.0
    all_nonpos = vmax <= 0.0
    if vmin == vmax:
        magnitude = max(abs(vmin), linthresh, 1e-6)
        if all_nonneg:
            return max(0.0, vmin - 0.15 * magnitude), vmax + 0.15 * magnitude
        if all_nonpos:
            return vmin - 0.15 * magnitude, min(0.0, vmax + 0.15 * magnitude)
        return -magnitude, magnitude
    span = vmax - vmin
    pad = max(span * 0.15, linthresh)
    y_min, y_max = vmin - pad, vmax + pad
    if all_nonneg:
        y_min = max(0.0, y_min)
    elif all_nonpos:
        y_max = min(0.0, y_max)
    return y_min, y_max


def _log_vmin_vmax(values):
    """Pad strictly positive values for a log-scaled axis (never crosses 0)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return SYMLOG_LINTHRESH, 1.0
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    # ~0.15 decade of padding on each side
    factor = 10 ** 0.15
    if vmin == vmax:
        return max(vmin / factor, np.nextafter(0, 1)), vmax * factor
    return vmin / factor, vmax * factor


def _configure_symlog_axis(axis, linthresh=SYMLOG_LINTHRESH):
    """Apply consistent power-of-ten tick labels to a symlog axis."""
    axis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh))
    axis.set_major_formatter(LogFormatterSciNotation())
    axis.set_minor_formatter(NullFormatter())


def _configure_log_axis(axis):
    """Power-of-ten ticks for a strictly positive log axis."""
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(LogFormatterSciNotation())
    axis.set_minor_formatter(NullFormatter())


def _needs_wide_scale(vmin, vmax):
    """True when linear ticks would be a bad fit (large span / many decades)."""
    span = vmax - vmin
    peak = max(abs(vmin), abs(vmax))
    if peak > 50 or span > 50:
        return True
    if vmin > 0 and vmax > 0:
        return vmax / max(vmin, 1e-12) > 100
    if vmax <= 0 and vmin < 0:
        return False
    if vmin < 0 < vmax:
        return peak > 20 and span > 10
    return False


# Back-compat alias used by older call sites / heatmaps.
_needs_symlog_scale = _needs_wide_scale


def _round_tick(value, decimals=4):
    """Round tick positions so labels stay readable."""
    return float(np.round(value, decimals))


def _nice_linear_ticks(vmin, vmax, n=6):
    """Evenly spaced ticks for narrow linear axes (positive or negative)."""
    span = max(vmax - vmin, 1e-12)
    raw_step = span / max(n - 1, 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = magnitude
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= raw_step:
            step = candidate
            break
    tick = np.floor(vmin / step) * step
    ticks = []
    while tick <= vmax + step * 0.51:
        ticks.append(_round_tick(tick))
        tick += step
    return ticks


def _linear_ticks_in_range(vmin, vmax):
    """Pick linear ticks that stay inside padded axis limits."""
    ticks = _nice_linear_ticks(vmin, vmax)
    ticks = [t for t in ticks if vmin - 1e-9 <= t <= vmax + 1e-9]
    if len(ticks) >= 2:
        return ticks
    locator = MaxNLocator(nbins=6, min_n_ticks=4)
    return [float(t) for t in locator.tick_values(vmin, vmax) if vmin <= t <= vmax]


def setup_plot_y_axis(ax, values):
    """Pick linear / log / symlog so tick labels always render.

    All-positive wide ranges use log (not symlog) so the unused negative
    decade branch does not appear. Symlog is reserved for signed data.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    y_min, y_max = _symlog_vmin_vmax(values)

    if _needs_wide_scale(vmin, vmax):
        if vmin > 0:
            y_min, y_max = _log_vmin_vmax(arr)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            _configure_log_axis(ax.yaxis)
            ax.set_autoscaley_on(False)
            return
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
        ax.set_ylim(y_min, y_max)
        _configure_symlog_axis(ax.yaxis)
        ax.set_autoscaley_on(False)
        return

    ax.set_yscale("linear")
    ticks = _linear_ticks_in_range(y_min, y_max)
    if ticks:
        ax.set_yticks(ticks)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_autoscaley_on(False)


# Private name kept so aggregate_evaluation_results can re-export unchanged.
_setup_plot_y_axis = setup_plot_y_axis


def save_figure(fig, path, left_margin=None):
    """Save with enough padding that axis labels are not clipped."""
    if left_margin is not None:
        fig.subplots_adjust(left=left_margin)
    else:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)


_save_figure = save_figure
