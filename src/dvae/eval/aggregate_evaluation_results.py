#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate and visualize evaluation results from multiple training runs.

This script finds evaluation_summary.yaml files in subdirectories of the given
experiment directory, aggregates the data, and generates visualizations based
on specified parameters and metrics.
"""

import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob
import csv
import shutil
from PIL import Image
from dvae.visualizers.visualizers import get_plot_config
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import (
    LogFormatterSciNotation,
    MaxNLocator,
    NullFormatter,
    ScalarFormatter,
    SymmetricalLogLocator,
)

SYMLOG_LINTHRESH = 0.01

# Fixed heatmap color limits so colorbars are comparable across images.
# Spectrum: Hellinger distance is in [0, 1]. KLD: shared symlog scale for tf + auto.
HEATMAP_COLOR_LIMITS = {
    "spectrum": {"vmin": 0.0, "vmax": 1.0, "scale": "linear"},
    "kld": {"vmin": -10.0, "vmax": 1e3, "scale": "symlog"},
}
SPECTRUM_HEATMAP_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Dictionary for display names
DISPLAY_NAMES = {
    "mask_label": "Missing Ratio",
    "sampling_ratio": "Auto Ratio",
    "observation_process": "Channel",
}

# Dictionary for metric display names
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

# Dictionary for value display names (map raw labels to nicer labels)
VALUE_DISPLAY_NAMES = {
    "raw_ch1": "Ch1",
    "raw_ch2": "Ch2",
    "raw_ch3": "Ch3",
    "raw_ch4": "Ch4",
}


def is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def get_value_display_name(val):
    """Map raw parameter values to nicer display strings if available."""
    if isinstance(val, (list, tuple)):
        return [get_value_display_name(v) for v in val]
    if val in VALUE_DISPLAY_NAMES:
        return VALUE_DISPLAY_NAMES[val]
    return val


def find_delay_embedding_gif(yaml_dir, mode):
    """Find a delay embedding GIF for the requested mode.

    The TF output name has varied across runs, so we try a small set of glob
    patterns before giving up.
    """
    post_dir = os.path.join(yaml_dir, "post_training_figs")
    if mode == "tf":
        patterns = [
            os.path.join(post_dir, "vis_delay_embedding_of_*teacher-forced*.gif"),
            os.path.join(post_dir, "vis_delay_embedding_of_*teacher_forced*.gif"),
            os.path.join(post_dir, "vis_delay_embedding_of_*teacher-_forced*.gif"),
            os.path.join(post_dir, "vis_delay_embedding_of_*teacher*forced*.gif"),
        ]
    else:
        patterns = [
            os.path.join(post_dir, "vis_delay_embedding_of_*autonomous*.gif"),
        ]

    for pattern in patterns:
        gif_files = glob.glob(pattern)
        if gif_files:
            return gif_files[0]
    return None


def sort_key(x):
    """Sort key for numerical values, fallback to string."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return str(x)


def resolve_heatmap_limits(metric):
    """Return fixed (vmin, vmax, scale) for known metrics, else None."""
    if metric.startswith("spectrum_error_"):
        cfg = HEATMAP_COLOR_LIMITS["spectrum"]
        return cfg["vmin"], cfg["vmax"], cfg["scale"]
    if metric.startswith("kld_"):
        cfg = HEATMAP_COLOR_LIMITS["kld"]
        return cfg["vmin"], cfg["vmax"], cfg["scale"]
    return None


def _symlog_vmin_vmax(values, linthresh=SYMLOG_LINTHRESH):
    """Pad finite values so symlog axes/colorbars always have a usable range."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -linthresh, linthresh
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmin == vmax:
        magnitude = max(abs(vmin), linthresh, 1e-6)
        return -magnitude, magnitude
    span = vmax - vmin
    pad = max(span * 0.15, linthresh)
    return vmin - pad, vmax + pad


def _configure_symlog_axis(axis, linthresh=SYMLOG_LINTHRESH):
    """Apply consistent power-of-ten tick labels to a symlog axis."""
    axis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh))
    axis.set_major_formatter(LogFormatterSciNotation())
    axis.set_minor_formatter(NullFormatter())


def _needs_symlog_scale(vmin, vmax):
    """Use symlog only when values span large or multi-decade ranges."""
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


def _setup_plot_y_axis(ax, values):
    """Pick linear vs symlog y-scale so tick labels always render."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    y_min, y_max = _symlog_vmin_vmax(values)

    if _needs_symlog_scale(vmin, vmax):
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


def _nice_positive_ticks(vmin, vmax, n=6):
    """Linear tick spacing for narrow all-positive ranges."""
    span = max(vmax - vmin, 1e-12)
    raw_step = span / max(n - 1, 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = magnitude
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= raw_step:
            step = candidate
            break
    start = np.floor(vmin / step) * step
    ticks = []
    tick = start
    while tick <= vmax + step * 0.51:
        if tick > 0:
            ticks.append(float(tick))
        tick += step
    return ticks


def _add_symlog_colorbar(mappable, label, vmin, vmax, linthresh=SYMLOG_LINTHRESH):
    """Colorbar with explicit ticks; decade labels when span is wide."""
    cbar = plt.colorbar(mappable, format=LogFormatterSciNotation())
    if vmin > 0 and vmax > 0 and vmax / vmin < 100:
        ticks = _nice_positive_ticks(vmin, vmax)
        if ticks:
            cbar.set_ticks(ticks)
    else:
        cbar.locator = SymmetricalLogLocator(base=10, linthresh=linthresh)
        cbar.update_ticks()
    cbar.set_label(label)
    return cbar


def _add_colorbar(mappable, label, vmin, vmax, scale):
    """Attach a colorbar with ticks appropriate for the resolved scale."""
    if scale == "symlog":
        cbar = plt.colorbar(mappable, format=LogFormatterSciNotation())
        cbar.locator = SymmetricalLogLocator(base=10, linthresh=SYMLOG_LINTHRESH)
        cbar.update_ticks()
    else:
        cbar = plt.colorbar(mappable)
        if vmin == 0.0 and vmax == 1.0:
            ticks = SPECTRUM_HEATMAP_TICKS
        else:
            ticks = _linear_ticks_in_range(vmin, vmax)
        if ticks:
            cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    cbar.set_label(label)
    return cbar


def _should_annotate_heatmap(n_rows, n_cols):
    """Annotate cell values when the grid is large enough to read."""
    return n_rows * n_cols <= 35 and max(n_rows, n_cols) <= 12


def _save_figure(fig, path, left_margin=None):
    """Save with enough padding that axis labels are not clipped."""
    if left_margin is not None:
        fig.subplots_adjust(left=left_margin)
    else:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3)


TILE_SIZE_INCHES = 3.5
SINGLE_PARAM_ROW_HEIGHT_INCHES = 5.5


def montage_figsize(rows, cols, layout_mode):
    """Figure size in inches; single-param rows get extra height for readability."""
    width = TILE_SIZE_INCHES * cols
    if layout_mode == "single_param_row":
        height = SINGLE_PARAM_ROW_HEIGHT_INCHES
    else:
        height = TILE_SIZE_INCHES * rows
    return width, height


def resolve_montage_layout(num_param1_values, has_param2):
    """
    Choose subplot grid shape.

    Two parameters: rows = param1 values, cols = param2 values (heatmap style).
    One parameter: a single horizontal row (1 x N) with full tile height.
    """
    if num_param1_values <= 0:
        return 0, 0, "empty"
    if has_param2:
        return num_param1_values, None, "two_param"
    return 1, num_param1_values, "single_param_row"


def montage_cell_for_param1(param1_index, layout_mode):
    """Map a param1 index to subplot (row, col)."""
    if layout_mode == "single_param_row":
        return 0, param1_index
    return param1_index, None


def get_display_name(param):
    """Get display name for parameter, default to param if not found."""
    return DISPLAY_NAMES.get(param, param)


def get_metric_display_name(metric):
    """Get display name for metric, default to metric if not found."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def resolve_metric(row, metric_name):
    """
    Resolve a metric from a YAML row with backward-compatible fallbacks.

    Lookup order:
      1. exact key (e.g. mse_auto_ch3)
      2. *_mean variant (e.g. mse_auto_mean for mse_auto)
      3. legacy key without suffix (e.g. mse_auto)
    """
    if metric_name in row and row[metric_name] not in (None, ""):
        return row[metric_name]

    if not metric_name.endswith("_mean"):
        mean_key = f"{metric_name}_mean"
        if mean_key in row and row[mean_key] not in (None, ""):
            return row[mean_key]

    return None


def find_yaml_files(experiment_dir):
    """Find all evaluation_summary.yaml files in subdirectories."""
    yaml_files = []
    for root, dirs, files in os.walk(experiment_dir):
        if "evaluation_summary.yaml" in files:
            yaml_files.append(os.path.join(root, "evaluation_summary.yaml"))
    return yaml_files


def load_yaml_data(yaml_files):
    """Load data from YAML files."""
    data = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            content = yaml.safe_load(f)
            content["yaml_file"] = yaml_file
            data.append(content)
    return data


def flatten_scalar_fields(prefix, value, out):
    """Flatten nested scalar data into a single dictionary for CSV export."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            flatten_scalar_fields(nested_prefix, nested_value, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = yaml.safe_dump(value, default_flow_style=True).strip()
    elif value is None:
        out[prefix] = ""
    else:
        out[prefix] = value


def build_aggregated_values_table(data, parameters):
    """Build a flat table of run-level parameters and scalar metrics."""
    rows = []
    for d in data:
        row = {"yaml_file": d.get("yaml_file", "")}

        for param in parameters:
            row[param] = get_param_value(d, param)

        # Preserve scalar top-level metrics and flatten any nested summary blocks.
        for key, value in d.items():
            if key in {"params", "config", "yaml_file"}:
                continue
            flatten_scalar_fields(key, value, row)

        rows.append(row)

    return rows


def save_aggregated_values_csv(data, output_file):
    """Save aggregated run-level values to CSV for later analysis."""
    if not data:
        return

    fieldnames = []
    for row in data:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def get_best_run(data, metric_name):
    """Return the run with the lowest numeric value for the requested metric."""
    best_row = None
    best_value = None

    for row in data:
        value = resolve_metric(row, metric_name)
        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if best_value is None or numeric_value < best_value:
            best_value = numeric_value
            best_row = row

    return best_row, best_value


def get_combined_best_run(data, metric_names):
    """Return the run with the lowest sum of ranks across the requested metrics."""
    scored_rows = []
    metric_values = {metric_name: [] for metric_name in metric_names}

    for row in data:
        row_values = {}
        valid = True
        for metric_name in metric_names:
            value = resolve_metric(row, metric_name)
            if value is None:
                valid = False
                break
            try:
                row_values[metric_name] = float(value)
            except (TypeError, ValueError):
                valid = False
                break
        if valid:
            scored_rows.append((row, row_values))
            for metric_name in metric_names:
                metric_values[metric_name].append(row_values[metric_name])

    if not scored_rows:
        return None, None, None

    metric_ranks = {}
    for metric_name, values in metric_values.items():
        sorted_unique = sorted(set(values))
        metric_ranks[metric_name] = {
            value: rank for rank, value in enumerate(sorted_unique, start=1)
        }

    best_row = None
    best_score = None
    best_values = None

    for row, row_values in scored_rows:
        score = sum(
            metric_ranks[metric_name][row_values[metric_name]]
            for metric_name in metric_names
        )
        if best_score is None or score < best_score:
            best_score = score
            best_row = row
            best_values = row_values

    return best_row, best_score, best_values


def copy_post_training_figs(source_dir, destination_dir):
    """Copy only figure files from post_training_figs into a destination directory."""
    if not os.path.isdir(source_dir):
        print(f"Warning: post_training_figs not found at {source_dir}")
        return False

    os.makedirs(destination_dir, exist_ok=True)

    figure_extensions = {".png", ".pdf", ".svg", ".gif", ".jpg", ".jpeg", ".webp"}
    copied_files = 0

    for root, _, files in os.walk(source_dir):
        rel_root = os.path.relpath(root, source_dir)
        target_root = (
            destination_dir
            if rel_root == "."
            else os.path.join(destination_dir, rel_root)
        )
        os.makedirs(target_root, exist_ok=True)

        for file_name in files:
            _, ext = os.path.splitext(file_name)
            if ext.lower() not in figure_extensions:
                continue
            shutil.copy2(
                os.path.join(root, file_name), os.path.join(target_root, file_name)
            )
            copied_files += 1

    if copied_files == 0:
        print(f"Warning: no figure files found under {source_dir}")
        return False

    return True


def is_mtrnn_run(data_row):
    """Heuristically detect MT_RNN / MT_VRNN runs."""
    model_name = get_param_value(data_row, "model_name")
    type_rnn = get_param_value(data_row, "type_rnn")
    yaml_file = data_row.get("yaml_file", "")
    haystack = " ".join(str(value) for value in [model_name, type_rnn, yaml_file])
    return any(token in haystack for token in ["MT_RNN", "MT_VRNN", "MTRNN"])


def find_matching_figure_path(yaml_dir, source_relative_dir, filename_patterns):
    """Find a figure under a run directory using a list of glob patterns."""
    source_dir = os.path.join(yaml_dir, source_relative_dir)
    for pattern in filename_patterns:
        matches = glob.glob(os.path.join(source_dir, pattern))
        if matches:
            return matches[0]
    return None


def aggregate_image_tiles(
    data,
    args,
    output_dir,
    source_relative_dir,
    filename_patterns,
    output_basename,
    title_prefix,
    model_filter=None,
):
    """Aggregate per-run image files into a tiled grid using the selected parameters."""
    param1 = args.parameters[0]
    param2 = args.parameters[1] if len(args.parameters) > 1 else None

    filtered_rows = [d for d in data if model_filter is None or model_filter(d)]
    if not filtered_rows:
        print(f"No matching runs for {output_basename}. Skipping.")
        return

    unique_param1 = sorted(
        set(
            get_param_value(d, param1)
            for d in filtered_rows
            if get_param_value(d, param1) is not None
        ),
        key=sort_key,
    )
    if param2:
        unique_param2 = sorted(
            set(
                get_param_value(d, param2)
                for d in filtered_rows
                if get_param_value(d, param2) is not None
            ),
            key=sort_key,
        )
    else:
        unique_param2 = ["-"]

    rows, cols, layout_mode = resolve_montage_layout(len(unique_param1), param2 is not None)
    if layout_mode == "two_param":
        cols = len(unique_param2)

    if rows == 0 or cols == 0:
        print(f"Not enough varying data for {output_basename}. Skipping.")
        return

    get_plot_config(paper_ready=True)

    fig, axs = plt.subplots(rows, cols, figsize=montage_figsize(rows, cols, layout_mode))
    axs = np.atleast_2d(axs).reshape(rows, cols)

    filled = {}

    for d in filtered_rows:
        val1 = get_param_value(d, param1)
        if val1 is None:
            continue

        if param2:
            val2 = get_param_value(d, param2)
            if val2 is None:
                continue
        else:
            val2 = "-"

        try:
            param1_index = unique_param1.index(val1)
            if param2:
                i = param1_index
                j = unique_param2.index(val2)
            else:
                i, j = montage_cell_for_param1(param1_index, layout_mode)
        except ValueError:
            continue

        if (i, j) in filled:
            continue

        yaml_dir = os.path.dirname(d["yaml_file"])
        figure_path = find_matching_figure_path(
            yaml_dir, source_relative_dir, filename_patterns
        )
        if not figure_path:
            axs[i, j].text(
                0.5,
                0.5,
                "No Figure",
                ha="center",
                va="center",
                transform=axs[i, j].transAxes,
            )
            filled[(i, j)] = True
            continue

        try:
            with Image.open(figure_path) as im:
                img = np.array(im.convert("RGBA"))

            non_white = np.where((img[:, :, :3] < 245).any(axis=2))
            if len(non_white[0]) > 0:
                y1, y2 = np.min(non_white[0]), np.max(non_white[0])
                x1, x2 = np.min(non_white[1]), np.max(non_white[1])
                pad = 10
                y1 = max(0, y1 - pad)
                y2 = min(img.shape[0], y2 + pad)
                x1 = max(0, x1 - pad)
                x2 = min(img.shape[1], x2 + pad)
                img = img[y1:y2, x1:x2]

            axs[i, j].imshow(img)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)
            filled[(i, j)] = True
        except Exception as e:
            print(f"Error loading figure {figure_path}: {e}")
            axs[i, j].text(
                0.5,
                0.5,
                "Error",
                ha="center",
                va="center",
                transform=axs[i, j].transAxes,
            )
            filled[(i, j)] = True

    for i in range(rows):
        for j in range(cols):
            if (i, j) not in filled:
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                for spine in axs[i, j].spines.values():
                    spine.set_visible(False)

            if layout_mode == "two_param":
                if i == 0:
                    axs[i, j].set_title(
                        str(get_value_display_name(unique_param2[j])),
                        fontsize=45,
                        pad=15,
                    )
                if j == 0:
                    axs[i, j].set_ylabel(
                        str(get_value_display_name(unique_param1[i])),
                        fontsize=45,
                        labelpad=15,
                    )
            elif layout_mode == "single_param_row":
                axs[i, j].set_title(
                    str(get_value_display_name(unique_param1[j])),
                    fontsize=45,
                    pad=15,
                )

    if layout_mode == "two_param":
        fig.suptitle(get_display_name(param2), fontsize=55, y=1.02)
        fig.supylabel(get_display_name(param1), fontsize=55)
    elif layout_mode == "single_param_row":
        fig.suptitle(get_display_name(param1), fontsize=55, y=1.05)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    output_path = os.path.join(output_dir, f"{output_basename}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"{title_prefix} saved to {output_path}")


def write_best_model_log(
    destination_dir, metric_name, metric_value, best_row, parameter_names
):
    """Write a short text log describing which model was selected."""
    log_file = os.path.join(destination_dir, "best_model.txt")
    with open(log_file, "w") as f:
        f.write(f"metric={metric_name}\n")
        f.write(f"metric_value={metric_value}\n")
        for param_name in parameter_names:
            f.write(f"{param_name}={get_param_value(best_row, param_name)}\n")
        f.write(f"yaml_file={best_row.get('yaml_file', '')}\n")
        f.write(f"model_dir={os.path.dirname(best_row.get('yaml_file', ''))}\n")
    return log_file


def export_best_model_plots(
    data, output_dir, metric_name, subdir_name, parameter_names
):
    """Copy the best run's post_training_figs into a dedicated subdirectory."""
    best_row, best_value = get_best_run(data, metric_name)
    if best_row is None:
        print(f"No valid values found for {metric_name}; skipping {subdir_name}.")
        return

    yaml_dir = os.path.dirname(best_row["yaml_file"])
    source_fig_dir = os.path.join(yaml_dir, "post_training_figs")
    destination_dir = os.path.join(output_dir, subdir_name)

    if copy_post_training_figs(source_fig_dir, destination_dir):
        log_file = write_best_model_log(
            destination_dir, metric_name, best_value, best_row, parameter_names
        )
        print(
            f"Copied best {metric_name} plots to {destination_dir} and wrote {log_file}"
        )


def export_best_combined_model_plots(
    data, output_dir, metric_names, subdir_name, parameter_names
):
    """Copy the combined best run's post_training_figs into a dedicated subdirectory."""
    best_row, best_score, best_values = get_combined_best_run(data, metric_names)
    if best_row is None:
        print(f"No valid values found for combined score; skipping {subdir_name}.")
        return

    yaml_dir = os.path.dirname(best_row["yaml_file"])
    source_fig_dir = os.path.join(yaml_dir, "post_training_figs")
    destination_dir = os.path.join(output_dir, subdir_name)

    if copy_post_training_figs(source_fig_dir, destination_dir):
        log_file = os.path.join(destination_dir, "best_model.txt")
        with open(log_file, "w") as f:
            f.write(f"metric=combined_rank_sum\n")
            f.write(f"combined_score={best_score}\n")
            for param_name in parameter_names:
                f.write(f"{param_name}={get_param_value(best_row, param_name)}\n")
            for metric_name in metric_names:
                f.write(f"{metric_name}={best_values.get(metric_name, '')}\n")
            f.write(f"yaml_file={best_row.get('yaml_file', '')}\n")
            f.write(f"model_dir={os.path.dirname(best_row.get('yaml_file', ''))}\n")
        print(f"Copied best combined plots to {destination_dir} and wrote {log_file}")


def save_yaml_list(yaml_files, output_file):
    """Save the list of YAML files to a text file."""
    with open(output_file, "w") as f:
        for yf in yaml_files:
            f.write(yf + "\n")


def get_param_value(d, param):
    """Get parameter value from data dict."""
    if "params" in d and param in d["params"]:
        val = d["params"][param]
    elif "config" in d:
        for section in d["config"].values():
            if isinstance(section, dict) and param in section:
                val = section[param]
                break
        else:
            return None
    else:
        return None

    # Special handling for mask_label
    if param == "mask_label":
        if val == "None" or val is None:
            return 0.0
        elif isinstance(val, str):
            # Extract the last number, e.g., "Markov_AvgLen15_0.5" -> 0.5
            parts = val.split("_")
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue
            return 0.0  # If no number found
        else:
            return float(val) if val else 0.0

    return val


def extract_param_values(data, param):
    """Extract unique values for a parameter."""
    values = set()
    for d in data:
        val = get_param_value(d, param)
        if val is not None:
            values.add(val)
    return sorted(list(values), key=sort_key)


def plot_1d(data, param, metric, output_dir):
    """Plot 1D graph for a metric vs one parameter."""
    param_values = []
    metric_values = []
    for d in data:
        # Get param value
        p_val = get_param_value(d, param)
        if p_val is None:
            continue
        # Get metric value
        metric_value = resolve_metric(d, metric)
        if metric_value is not None:
            param_values.append(p_val)
            metric_values.append(metric_value)

    if len(param_values) == 0:
        print(
            f"No varying data for param {param} and metric {metric}. Skipping 1D plot."
        )
        return

    # Sort values and use a scatter plot if there are overlapping X-axis entries
    # to prevent a messy zig-zag line graph
    pairs = sorted(zip(param_values, metric_values), key=lambda x: sort_key(x[0]))
    sorted_p, sorted_m = zip(*pairs)

    has_dups = len(set(sorted_p)) < len(sorted_p)

    config = get_plot_config(paper_ready=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # If x-values are categorical (non-numeric), plot against indices and set tick labels
    if not all(is_numeric(v) for v in sorted_p):
        x_vals = list(range(len(sorted_p)))
        if has_dups:
            ax.plot(x_vals, list(sorted_m), "o", alpha=0.7, markersize=8)
        else:
            ax.plot(x_vals, list(sorted_m), "o-")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(
            [get_value_display_name(p) for p in sorted_p], rotation=45
        )
    else:
        # numeric x-axis
        num_x = [float(v) for v in sorted_p]
        if has_dups:
            ax.plot(num_x, sorted_m, "o", alpha=0.7, markersize=8)
        else:
            ax.plot(num_x, sorted_m, "o-")

    _setup_plot_y_axis(ax, metric_values)

    ax.set_xlabel(get_display_name(param))
    ax.set_ylabel(get_metric_display_name(metric))
    if config["show_title"]:
        ax.set_title(f"{get_metric_display_name(metric)} vs {get_display_name(param)}")
    _save_figure(
        fig,
        os.path.join(output_dir, f"{metric}_vs_{param}.png"),
        left_margin=0.20,
    )
    plt.close(fig)


def plot_2d(data, param1, metric, output_dir, param2=None):
    """Plot 2D heatmap for a metric vs two parameters (or 1 param as single row)."""
    param1_values = extract_param_values(data, param1)

    if param2:
        param2_values = extract_param_values(data, param2)
    else:
        param2_values = ["-"]

    if len(param1_values) == 0:
        print(f"No varying data for {param1}. Skipping heatmap.")
        return

    # Create a grid
    grid = np.full((len(param1_values), len(param2_values)), np.nan)

    for d in data:
        # Get param values
        p1_val = get_param_value(d, param1)
        metric_value = resolve_metric(d, metric)
        if p1_val is None or metric_value is None:
            continue

        if param2:
            p2_val = get_param_value(d, param2)
            if p2_val is None:
                continue
            j = param2_values.index(p2_val) if p2_val in param2_values else None
        else:
            j = 0

        i = param1_values.index(p1_val) if p1_val in param1_values else None
        if i is not None and j is not None:
            grid[i, j] = metric_value

    if np.all(np.isnan(grid)):
        print(f"No valid data points for 2D plot of {metric}. Skipping.")
        return

    config = get_plot_config(paper_ready=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    finite = grid.ravel()[np.isfinite(grid.ravel())]
    data_vmin, data_vmax = float(np.min(finite)), float(np.max(finite))
    fixed_limits = resolve_heatmap_limits(metric)
    if fixed_limits is not None:
        vmin, vmax, scale = fixed_limits
        if data_vmin < vmin - 1e-9 or data_vmax > vmax + 1e-9:
            print(
                f"Warning: {metric} heatmap values [{data_vmin:.4g}, {data_vmax:.4g}] "
                f"clip outside fixed range [{vmin:g}, {vmax:g}]"
            )
    else:
        vmin, vmax = _symlog_vmin_vmax(finite)
        scale = (
            "symlog"
            if _needs_symlog_scale(data_vmin, data_vmax)
            else "linear"
        )
    if scale == "symlog":
        norm = SymLogNorm(linthresh=SYMLOG_LINTHRESH, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(grid, origin="lower", aspect="auto", norm=norm)
    _add_colorbar(
        im,
        get_metric_display_name(metric),
        vmin,
        vmax,
        scale,
    )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xticks(range(len(param2_values)))
    ax.set_xticklabels([get_value_display_name(v) for v in param2_values])
    ax.set_yticks(range(len(param1_values)))
    ax.set_yticklabels([get_value_display_name(v) for v in param1_values])

    if param2:
        ax.set_xlabel(get_display_name(param2))
    else:
        ax.set_xlabel("")

    ax.set_ylabel(get_display_name(param1))

    if config["show_title"]:
        title_str = (
            f"{get_metric_display_name(metric)} heatmap: {get_display_name(param1)}"
        )
        if param2:
            title_str += f" vs {get_display_name(param2)}"
        ax.set_title(title_str, pad=20)

    if _should_annotate_heatmap(len(param1_values), len(param2_values)):
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="white",
                    )

    out_name = f"{metric}_heatmap_{param1}"
    if param2:
        out_name += f"_vs_{param2}"
    _save_figure(fig, os.path.join(output_dir, f"{out_name}.png"))
    plt.close(fig)


def aggregate_delay_embeddings(data, args, output_dir):
    """Aggregate delay embedding GIFs into tiled figures for autonomous and teacher-forced modes."""
    param1 = args.parameters[0]
    param2 = args.parameters[1] if len(args.parameters) > 1 else None

    unique_param1 = sorted(
        set(
            get_param_value(d, param1)
            for d in data
            if get_param_value(d, param1) is not None
        ),
        key=sort_key,
    )
    if param2:
        unique_param2 = sorted(
            set(
                get_param_value(d, param2)
                for d in data
                if get_param_value(d, param2) is not None
            ),
            key=sort_key,
        )
    else:
        unique_param2 = ["-"]

    rows, cols, layout_mode = resolve_montage_layout(len(unique_param1), param2 is not None)
    if layout_mode == "two_param":
        cols = len(unique_param2)

    if rows == 0 or cols == 0:
        print("Not enough varying data for delay embeddings. Skipping.")
        return

    get_plot_config(paper_ready=True)

    for mode in ["tf", "autonomous"]:
        fig, axs = plt.subplots(
            rows, cols, figsize=montage_figsize(rows, cols, layout_mode)
        )
        axs = np.atleast_2d(axs).reshape(rows, cols)

        # Dictionary to store which cells have been filled
        filled = {}

        for d in data:
            val1 = get_param_value(d, param1)
            if val1 is None:
                continue

            if param2:
                val2 = get_param_value(d, param2)
                if val2 is None:
                    continue
            else:
                val2 = "-"

            try:
                param1_index = unique_param1.index(val1)
                if param2:
                    i = param1_index
                    j = unique_param2.index(val2)
                else:
                    i, j = montage_cell_for_param1(param1_index, layout_mode)
            except ValueError:
                continue

            if (i, j) in filled:
                continue  # Skip if already filled

            # Find the output dir
            yaml_dir = os.path.dirname(d["yaml_file"])
            gif_path = find_delay_embedding_gif(yaml_dir, mode)
            if not gif_path:
                print(f"No delay embedding GIF found for {yaml_dir}")
                axs[i, j].text(
                    0.5,
                    0.5,
                    "No GIF",
                    ha="center",
                    va="center",
                    transform=axs[i, j].transAxes,
                )
                filled[(i, j)] = True
                continue

            try:
                with Image.open(gif_path) as im:
                    im.seek(0)
                    img = np.array(im.convert("RGBA"))

                # Auto-crop the white background surrounding the GIF plot to blow up the tile
                non_white = np.where((img[:, :, :3] < 245).any(axis=2))
                if len(non_white[0]) > 0:
                    y1, y2 = np.min(non_white[0]), np.max(non_white[0])
                    x1, x2 = np.min(non_white[1]), np.max(non_white[1])
                    pad = 10
                    y1 = max(0, y1 - pad)
                    y2 = min(img.shape[0], y2 + pad)
                    x1 = max(0, x1 - pad)
                    x2 = min(img.shape[1], x2 + pad)
                    img = img[y1:y2, x1:x2]

                axs[i, j].imshow(img)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                for spine in axs[i, j].spines.values():
                    spine.set_visible(False)
                filled[(i, j)] = True
            except Exception as e:
                print(f"Error loading GIF {gif_path}: {e}")
                axs[i, j].text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center",
                    transform=axs[i, j].transAxes,
                )
                filled[(i, j)] = True

        # Hide unused subplots, leaving axis for labels
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in filled:
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    for spine in axs[i, j].spines.values():
                        spine.set_visible(False)

                if layout_mode == "two_param":
                    if i == 0:
                        axs[i, j].set_title(
                            str(get_value_display_name(unique_param2[j])),
                            fontsize=45,
                            pad=15,
                        )
                    if j == 0:
                        axs[i, j].set_ylabel(
                            str(get_value_display_name(unique_param1[i])),
                            fontsize=45,
                            labelpad=15,
                        )
                elif layout_mode == "single_param_row":
                    axs[i, j].set_title(
                        str(get_value_display_name(unique_param1[j])),
                        fontsize=45,
                        pad=15,
                    )

        if layout_mode == "two_param":
            fig.suptitle(get_display_name(param2), fontsize=55, y=1.02)
            fig.supylabel(get_display_name(param1), fontsize=55)
        elif layout_mode == "single_param_row":
            fig.suptitle(get_display_name(param1), fontsize=55, y=1.05)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.savefig(
            os.path.join(output_dir, f"aggregated_delay_embeddings_{mode}.png"),
            bbox_inches="tight",
        )
        plt.close()
        print(
            f"Aggregated delay embeddings for {mode} saved to {os.path.join(output_dir, f'aggregated_delay_embeddings_{mode}.png')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize evaluation results."
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help="Parameters to use for plotting (1 or 2).",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Filter the data to plot a specific slice by requiring exact parameter matches (e.g., --filter dim_rnn=64 type_rnn=RNN)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "kld_tf",
            "kld_auto",
            "spectrum_error_gt",
            "spectrum_error_tf",
            "spectrum_error_auto",
        ],
        help="Metrics to plot (default: common ones).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="aggregated_plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--yaml_list_file",
        type=str,
        default="yaml_files.txt",
        help="File to save list of YAML files.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    if len(args.parameters) not in [1, 2]:
        print("Provide 1 or 2 parameters.")
        return

    # Set output directory inside experiment_dir
    if len(args.parameters) == 1:
        param_dir = f"aggregate_eval_plots_{args.parameters[0]}"
    else:
        param_dir = f"aggregate_eval_plots_{args.parameters[0]}_{args.parameters[1]}"

    if args.filter:
        # Append all filters directly into the output directory name, replacing '==' with '=', and '.'
        filter_str = "_".join(args.filter).replace("==", "=").replace(".", "_")
        param_dir = f"{param_dir}_{filter_str}"

    args.output_dir = os.path.join(args.experiment_dir, param_dir)
    args.yaml_list_file = os.path.join(args.output_dir, "yaml_files.txt")

    # Find YAML files
    if args.verbose:
        print(f"Finding YAML files in {args.experiment_dir}...")
    yaml_files = find_yaml_files(args.experiment_dir)
    print(f"Found {len(yaml_files)} evaluation_summary.yaml files.")
    if args.verbose:
        for yf in yaml_files:
            print(f"  {yf}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    if args.verbose:
        print(f"Output directory: {args.output_dir}")

    # Save list
    save_yaml_list(yaml_files, args.yaml_list_file)
    if args.verbose:
        print(f"Saved YAML file list to {args.yaml_list_file}")

    # Load data
    if args.verbose:
        print("Loading YAML data...")
    data = load_yaml_data(yaml_files)
    if args.verbose:
        print(f"Loaded data from {len(data)} files.")

    # Filter data early if --filter is provided
    if args.filter:
        if args.verbose:
            print(f"Applying filters: {args.filter}")
        filtered_data = []
        for d in data:
            match = True
            for f in args.filter:
                # Support both key=value and key==value formats seamlessly
                if "==" in f:
                    k, v = f.split("==", 1)
                elif "=" in f:
                    k, v = f.split("=", 1)
                else:
                    print(f"Warning: Invalid filter format '{f}'. Ignored.")
                    continue

                # Fetch metric using our universal struct accessor
                d_val = get_param_value(d, k)

                # Coerce data_value and input_value types loosely for floats vs ints vs strings
                try:
                    if float(d_val) != float(v):
                        match = False
                except (ValueError, TypeError):
                    if str(d_val) != str(v):
                        match = False

                if not match:
                    break

            if match:
                filtered_data.append(d)
        data = filtered_data
        print(f"Data reduced to {len(data)} files after filtering.")
        if not data:
            print("No data left after filtering! Exiting.")
            return

    # Check for overlapping runs based on the selected parameters
    param_groups = defaultdict(list)
    for d in data:
        key = tuple(get_param_value(d, p) for p in args.parameters)
        param_groups[key].append(d)

    overlap_warning = False
    for p_val_tuple, files in param_groups.items():
        if len(files) > 1:
            if not overlap_warning:
                print("\n" + "=" * 60)
                print("⚠️ WARNING: MULTIPLE MODELS WITH THE SAME PLOT PARAMETERS ⚠️")
                print("=" * 60)
                overlap_warning = True

            param_str = ", ".join(
                [f"{p}={v}" for p, v in zip(args.parameters, p_val_tuple)]
            )
            print(f"\nFound {len(files)} overlapping models for [{param_str}]:")
            for d in files:
                print(f"  - {d['yaml_file']}")

            # Identify differing parameters
            all_keys = set()
            for d in files:
                if "params" in d:
                    all_keys.update(d["params"].keys())
                if "config" in d and isinstance(d["config"], dict):
                    for section in d["config"].values():
                        if isinstance(section, dict):
                            all_keys.update(section.keys())

            diff_params = {}
            for k in all_keys:
                if k in args.parameters:
                    continue  # skip the grouping parameters
                val_strs = set()
                vals_actual = []
                for d in files:
                    val = get_param_value(d, k)
                    val_str = str(val)
                    if val_str not in val_strs:
                        val_strs.add(val_str)
                        vals_actual.append(val)
                if len(vals_actual) > 1:
                    diff_params[k] = vals_actual

            if diff_params:
                print("\n  Differing configuration parameters across these models:")
                for k, vals in sorted(diff_params.items()):
                    val_formatted = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v) for v in vals
                    )
                    print(f"    - {k}: {val_formatted}")

    if overlap_warning:
        print(
            "\nNote: Plotting grids/heatmaps will arbitrarily overwrite or pick the first hit."
        )
        print("=" * 60 + "\n")

    # Save the raw run-level values for later analysis.
    aggregated_values = build_aggregated_values_table(data, args.parameters)
    aggregated_values_file = os.path.join(args.output_dir, "aggregated_values.csv")
    save_aggregated_values_csv(aggregated_values, aggregated_values_file)
    if args.verbose:
        print(f"Saved aggregated values to {aggregated_values_file}")

    # Copy the best run's post-training figures for the key metrics.
    export_best_model_plots(
        data,
        args.output_dir,
        metric_name="kld_auto",
        subdir_name="best_kl_plots",
        parameter_names=args.parameters,
    )
    export_best_model_plots(
        data,
        args.output_dir,
        metric_name="spectrum_error_auto",
        subdir_name="best_spectrum_plots",
        parameter_names=args.parameters,
    )
    export_best_combined_model_plots(
        data,
        args.output_dir,
        metric_names=["kld_auto", "spectrum_error_auto"],
        subdir_name="best_combined_plots",
        parameter_names=args.parameters,
    )

    # Aggregate selected figures into montage-style summary plots.
    aggregate_image_tiles(
        data,
        args,
        args.output_dir,
        source_relative_dir="post_training_figs",
        filename_patterns=[
            "vis_pred_true_series_final_short_inference_mode_half_half_short.png"
        ],
        output_basename="aggregated_vis_pred_true_series_final_short_inference_mode_half_half_short",
        title_prefix="Aggregated short reconstruction montage",
    )
    aggregate_image_tiles(
        data,
        args,
        args.output_dir,
        source_relative_dir="vis_during_training",
        filename_patterns=["vis_training_history_of_alpha_MT_RNN.png"],
        output_basename="aggregated_vis_training_history_of_alpha_MT_RNN",
        title_prefix="Aggregated alpha-history montage",
        model_filter=is_mtrnn_run,
    )
    aggregate_image_tiles(
        data,
        args,
        args.output_dir,
        source_relative_dir="vis_during_training",
        filename_patterns=["vis_training_history_of_sigma_MT_RNN.png"],
        output_basename="aggregated_vis_training_history_of_sigma_MT_RNN",
        title_prefix="Aggregated sigma-history montage",
        model_filter=is_mtrnn_run,
    )

    # Aggregate delay embeddings
    aggregate_delay_embeddings(data, args, args.output_dir)

    # Collect parameter values for all runs
    param_table = []
    for d in data:
        row = {}
        for param in args.parameters:
            row[param] = get_param_value(d, param)
        param_table.append(row)

    # Create a display version of the table using the mapping for nicer labels
    display_table = []
    for row in param_table:
        display_row = {}
        for param in args.parameters:
            display_row[param] = get_value_display_name(row.get(param))
        display_table.append(display_row)

    # Print table (display names)
    if display_table:
        print("\nParameter values across runs:")
        headers = args.parameters
        print("\t".join(headers))
        for row in display_table:
            print("\t".join(str(row.get(h, "N/A")) for h in headers))

    # Save to CSV (display names)
    csv_file = os.path.join(args.output_dir, "parameter_values.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.parameters)
        writer.writeheader()
        for row in display_table:
            writer.writerow({k: v for k, v in row.items() if k in args.parameters})

    if args.verbose:
        print(f"Saved parameter values to {csv_file}")

    # Plot
    if args.verbose:
        print(
            f"Generating plots for parameters: {args.parameters}, metrics: {args.metrics}"
        )
    for metric in args.metrics:
        # Plot 1D for ALL parameters, individually
        for param in args.parameters:
            if args.verbose:
                print(f"Plotting 1D: {metric} vs {param}")
            plot_1d(data, param, metric, args.output_dir)

        # Plot 2D (heatmap style coverage)
        if len(args.parameters) == 1:
            if args.verbose:
                print(
                    f"Plotting 2D: {metric} single-row heatmap for {args.parameters[0]}"
                )
            plot_2d(data, args.parameters[0], metric, args.output_dir, param2=None)
        else:
            if args.verbose:
                print(
                    f"Plotting 2D: {metric} heatmap for {args.parameters[0]} vs {args.parameters[1]}"
                )
            plot_2d(
                data,
                args.parameters[0],
                metric,
                args.output_dir,
                param2=args.parameters[1],
            )

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
