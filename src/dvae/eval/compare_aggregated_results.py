#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay metrics from several finished per-sweep aggregate CSVs.

This is a sibling of ``aggregate_evaluation_results.py``: it does **not**
re-read eval YAMLs. Each input is an ``aggregated_values.csv`` (or a
directory that contains one under ``aggregate_eval_plots_*``).

Default mode collapses extra rows by ``sampling_ratio`` only (OTF vs
interpolate vs indicate style overlays).

``--channel-pair`` instead joins 1d-trained long-form rows
(``observation_process=raw_chK``, score in the base metric column) with
joint-4d wide-form columns (``{metric}_chK``) on ``(sampling_ratio, channel)``.

Example (default overlays)::

    python src/dvae/eval/compare_aggregated_results.py \\
        --experiments \\
            "OTF|/saved_model/2026-07-01/.../20260701-XHRO_..." \\
            "interpolate|/saved_model/2026-09-04/..._interpolate" \\
            "indicate|/saved_model/2026-09-04/..._indicate_x8_..." \\
        --metrics kld_auto spectrum_error_auto kld_tf spectrum_error_tf \\
        --x-parameter sampling_ratio \\
        --output_dir /saved_model/compare_aggregates/otf_vs_interp_vs_indicate

Example (1d-sep vs joint-4d per channel)::

    python src/dvae/eval/compare_aggregated_results.py \\
        --experiments \\
            "1d-sep|/saved_model/2026-09-11/..._raw_ch1-4_1d_sep_..." \\
            "OTF-4d|/saved_model/2026-07-01/..._chAll_4d_..." \\
        --metrics kld_auto spectrum_error_auto \\
        --x-parameter sampling_ratio \\
        --channel-pair \\
        --output_dir /saved_model/compare_aggregates/1d_vs_4d_otf
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dvae.eval.aggregate_plot_style import (
    apply_paper_ready_line_style,
    get_display_name,
    get_metric_display_name,
    is_numeric,
    save_figure,
    setup_plot_y_axis,
    sort_key,
)

AGGREGATED_CSV_NAME = "aggregated_values.csv"
STD_COLUMN_SUFFIXES = ("_std", "_std_across_batches")
LONG_FORM_CHANNEL_COLUMNS = ("channel", "observation_process")
DEFAULT_CHANNELS = ("ch1", "ch2", "ch3", "ch4")
_OBS_PROCESS_SUFFIXES = ("_interpolate", "_indicate")
_CHANNEL_LABEL_RE = re.compile(r"^(?:raw[_-])?ch(\d+)$", re.IGNORECASE)


def parse_experiment_spec(spec: str) -> Tuple[str, str]:
    """Split ``label|path`` or ``label=path`` into ``(label, path)``."""
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty experiment spec")
    if "|" in spec:
        label, path = spec.split("|", 1)
    elif "=" in spec:
        label, path = spec.split("=", 1)
    else:
        raise ValueError(
            f"Invalid experiment spec {spec!r}: expected 'label|path' or 'label=path'"
        )
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise ValueError(
            f"Invalid experiment spec {spec!r}: label and path must be non-empty"
        )
    return label, path


def resolve_aggregated_csv(path: str, x_parameter: str = "sampling_ratio") -> str:
    """Resolve ``aggregated_values.csv`` from a file, aggregate dir, or exp root.

    Search order for a directory:
      1. ``<path>/aggregated_values.csv``
      2. ``<path>/aggregate_eval_plots_<x_parameter>/aggregated_values.csv``
      3. unique ``<path>/aggregate_eval_plots_*/aggregated_values.csv``
         (prefer a directory name that contains ``x_parameter``)
    """
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Path does not exist: {path}. Pass the experiment root, its "
            f"aggregate_eval_plots_* directory, or {AGGREGATED_CSV_NAME} itself."
        )

    direct = os.path.join(path, AGGREGATED_CSV_NAME)
    if os.path.isfile(direct):
        return direct

    preferred = os.path.join(
        path, f"aggregate_eval_plots_{x_parameter}", AGGREGATED_CSV_NAME
    )
    if os.path.isfile(preferred):
        return preferred

    matches: List[str] = []
    try:
        for name in sorted(os.listdir(path)):
            if not name.startswith("aggregate_eval_plots_"):
                continue
            candidate = os.path.join(path, name, AGGREGATED_CSV_NAME)
            if os.path.isfile(candidate):
                matches.append(candidate)
    except OSError as exc:
        raise FileNotFoundError(
            f"Could not search {path} for {AGGREGATED_CSV_NAME}: {exc}"
        ) from exc

    if not matches:
        raise FileNotFoundError(
            f"Could not find {AGGREGATED_CSV_NAME} under {path}. "
            f"Looked for the file itself, <path>/{AGGREGATED_CSV_NAME}, "
            f"<path>/aggregate_eval_plots_{x_parameter}/{AGGREGATED_CSV_NAME}, "
            f"and <path>/aggregate_eval_plots_*/{AGGREGATED_CSV_NAME}. "
            "Run the per-sweep aggregate first."
        )

    tagged = [m for m in matches if x_parameter in os.path.basename(os.path.dirname(m))]
    if len(tagged) == 1:
        return tagged[0]
    if len(matches) == 1:
        return matches[0]

    listed = "\n  ".join(matches)
    raise FileNotFoundError(
        f"Multiple {AGGREGATED_CSV_NAME} files under {path}; "
        f"pass the aggregate_eval_plots_* directory (or the CSV) explicitly.\n  {listed}"
    )


def _to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_x_value(value):
    """Canonicalize an x-axis value so 0.5 and '0.50' group together."""
    if value is None or value == "":
        return None
    if is_numeric(value):
        return float(value)
    return str(value)


def load_aggregated_csv(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader)


def _std_column_name(fieldnames: Iterable[str], metric: str) -> Optional[str]:
    names = set(fieldnames)
    for suffix in STD_COLUMN_SUFFIXES:
        candidate = f"{metric}{suffix}"
        if candidate in names:
            return candidate
    return None


def normalize_channel_label(value) -> Optional[str]:
    """Map ``raw_chK`` / ``chK`` / indicate-interpolate variants to ``chK``."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    for suffix in _OBS_PROCESS_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    match = _CHANNEL_LABEL_RE.fullmatch(text)
    if match:
        return f"ch{int(match.group(1))}"
    return None


def channel_sort_key(channel) -> Tuple[int, object]:
    match = re.fullmatch(r"ch(\d+)", str(channel))
    if match:
        return (0, int(match.group(1)))
    return (1, str(channel))


def channel_display_name(channel: str) -> str:
    match = re.fullmatch(r"ch(\d+)", str(channel))
    if match:
        return f"Ch{int(match.group(1))}"
    return str(channel)


def _is_std_column(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in STD_COLUMN_SUFFIXES)


def wide_channel_columns(
    fieldnames: Iterable[str], metric: str
) -> Dict[str, str]:
    """Return ``{chK: column}`` for wide-form ``{metric}_chK`` columns."""
    mapping: Dict[str, str] = {}
    prefix = f"{metric}_"
    for name in fieldnames:
        if not name.startswith(prefix) or _is_std_column(name):
            continue
        channel = normalize_channel_label(name[len(prefix) :])
        if channel:
            mapping[channel] = name
    return mapping


def detect_long_channel_column(
    rows: Sequence[Dict[str, str]],
) -> Optional[str]:
    """Return the column that holds ``raw_chK`` / ``chK`` labels, if any."""
    if not rows:
        return None
    fieldnames = rows[0].keys()
    for column in LONG_FORM_CHANNEL_COLUMNS:
        if column not in fieldnames:
            continue
        if any(normalize_channel_label(row.get(column)) for row in rows):
            return column
    return None


def expand_rows_to_channel_long(
    rows: Sequence[Dict[str, str]],
    x_parameter: str,
    metric: str,
) -> List[Dict[str, object]]:
    """Normalize 1d-long or joint-4d-wide rows to ``(x, channel, value)``.

    Long form (1d-sep): rows keyed by ``sampling_ratio`` +
    ``observation_process`` (``raw_chK``); score is the base metric column.
    Wide form (joint-4d): one row per ``sampling_ratio``; scores live in
    ``{metric}_chK`` columns.
    """
    if not rows:
        return []

    long_col = detect_long_channel_column(rows)
    wide_cols = wide_channel_columns(rows[0].keys(), metric)
    has_base_metric = any(
        row.get(metric) not in (None, "") and _to_float(row.get(metric)) is not None
        for row in rows
    )

    if long_col and has_base_metric:
        return _expand_long_form_rows(rows, x_parameter, metric, long_col)
    if wide_cols:
        return _expand_wide_form_rows(rows, x_parameter, metric, wide_cols)

    available = sorted(rows[0].keys())
    raise ValueError(
        f"Cannot build channel-pair series for {metric!r}: need either "
        f"long-form {LONG_FORM_CHANNEL_COLUMNS} values like raw_ch1 plus a "
        f"{metric!r} column, or wide-form {metric}_chK columns. "
        f"Available columns: {available}"
    )


def _expand_long_form_rows(
    rows: Sequence[Dict[str, str]],
    x_parameter: str,
    metric: str,
    channel_column: str,
) -> List[Dict[str, object]]:
    std_col = _std_column_name(rows[0].keys(), metric)
    points: List[Dict[str, object]] = []
    for row in rows:
        x_val = normalize_x_value(row.get(x_parameter))
        channel = normalize_channel_label(row.get(channel_column))
        y_val = _to_float(row.get(metric))
        if x_val is None or channel is None or y_val is None:
            continue
        std_val = _to_float(row.get(std_col)) if std_col else None
        points.append(
            {"x": x_val, "channel": channel, "value": y_val, "std": std_val}
        )
    return points


def _expand_wide_form_rows(
    rows: Sequence[Dict[str, str]],
    x_parameter: str,
    metric: str,
    channel_columns: Dict[str, str],
) -> List[Dict[str, object]]:
    fieldnames = rows[0].keys()
    points: List[Dict[str, object]] = []
    for row in rows:
        x_val = normalize_x_value(row.get(x_parameter))
        if x_val is None:
            continue
        for channel, column in channel_columns.items():
            y_val = _to_float(row.get(column))
            if y_val is None:
                continue
            std_col = _std_column_name(fieldnames, f"{metric}_{channel}")
            std_val = _to_float(row.get(std_col)) if std_col else None
            points.append(
                {"x": x_val, "channel": channel, "value": y_val, "std": std_val}
            )
    return points


def summarize_metric_by_x_channel(
    points: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Group expanded channel-long points by ``(x, channel)``."""
    groups: Dict[Tuple[object, str], List[float]] = defaultdict(list)
    stds_from_col: Dict[Tuple[object, str], List[float]] = defaultdict(list)
    for point in points:
        key = (point["x"], str(point["channel"]))
        groups[key].append(float(point["value"]))
        std_val = point.get("std")
        if std_val is not None:
            stds_from_col[key].append(float(std_val))

    series: List[Dict[str, object]] = []
    for x_val, channel in sorted(groups, key=lambda item: (sort_key(item[0]), channel_sort_key(item[1]))):
        values = np.asarray(groups[(x_val, channel)], dtype=float)
        mean = float(np.mean(values))
        if values.size > 1:
            std: Optional[float] = float(np.std(values, ddof=1))
        elif stds_from_col.get((x_val, channel)):
            std = float(np.mean(stds_from_col[(x_val, channel)]))
        else:
            std = None
        series.append(
            {
                "x": x_val,
                "channel": channel,
                "mean": mean,
                "std": std,
                "n": int(values.size),
            }
        )
    return series


def pivot_channel_series(
    series: Sequence[Dict[str, object]],
) -> Dict[str, List[Dict[str, object]]]:
    """Split ``{x, channel, mean, ...}`` rows into per-channel overlay series."""
    by_channel: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for point in series:
        by_channel[str(point["channel"])].append(
            {
                "x": point["x"],
                "mean": point["mean"],
                "std": point["std"],
                "n": point["n"],
            }
        )
    return dict(by_channel)


def channels_for_figure(found: Iterable[str]) -> List[str]:
    """Prefer the ch1–4 panel set used by XHRO 1d-sep vs joint-4d compares."""
    found_set = {str(ch) for ch in found}
    extras = sorted(found_set - set(DEFAULT_CHANNELS), key=channel_sort_key)
    if found_set & set(DEFAULT_CHANNELS) or not found_set:
        return list(DEFAULT_CHANNELS) + extras
    return sorted(found_set, key=channel_sort_key)


def summarize_metric_by_x(
    rows: Sequence[Dict[str, str]],
    x_parameter: str,
    metric: str,
) -> List[Dict[str, object]]:
    """Group rows by ``x_parameter`` and return mean / std / n.

    Std comes from replicate rows at the same x when ``n > 1``. If there is
    only one row, a ``{metric}_std`` or ``{metric}_std_across_batches`` column
    is used when present.
    """
    if not rows:
        return []
    std_col = _std_column_name(rows[0].keys(), metric)
    groups: Dict[object, List[float]] = defaultdict(list)
    stds_from_col: Dict[object, List[float]] = defaultdict(list)

    for row in rows:
        x_val = normalize_x_value(row.get(x_parameter))
        y_val = _to_float(row.get(metric))
        if x_val is None or y_val is None:
            continue
        groups[x_val].append(y_val)
        if std_col:
            std_val = _to_float(row.get(std_col))
            if std_val is not None:
                stds_from_col[x_val].append(std_val)

    series: List[Dict[str, object]] = []
    for x_val in sorted(groups, key=sort_key):
        values = np.asarray(groups[x_val], dtype=float)
        mean = float(np.mean(values))
        if values.size > 1:
            std: Optional[float] = float(np.std(values, ddof=1))
        elif stds_from_col.get(x_val):
            std = float(np.mean(stds_from_col[x_val]))
        else:
            std = None
        series.append({"x": x_val, "mean": mean, "std": std, "n": int(values.size)})
    return series


def intersect_x_values(
    series_by_label: Dict[str, List[Dict[str, object]]],
) -> List[object]:
    """Return sorted intersection of x values; warn about per-label extras."""
    x_sets = {
        label: {point["x"] for point in series} for label, series in series_by_label.items()
    }
    if not x_sets:
        return []
    common = set.intersection(*x_sets.values()) if x_sets else set()
    extras = {label: sorted(xs - common, key=sort_key) for label, xs in x_sets.items()}
    if any(extras.values()):
        print(
            "Warning: sampling grids differ; plotting the intersection of "
            f"{len(common)} shared x-values. Dropped points:"
        )
        for label, missing in extras.items():
            if missing:
                print(f"  {label}: {missing}")
    return sorted(common, key=sort_key)


def filter_series_to_x(
    series: Sequence[Dict[str, object]], x_values: Sequence[object]
) -> List[Dict[str, object]]:
    wanted = set(x_values)
    return [point for point in series if point["x"] in wanted]


class ExperimentAggregate:
    """One labeled aggregate CSV plus per-metric summaries."""

    def __init__(self, label: str, csv_path: str, rows: List[Dict[str, str]]):
        self.label = label
        self.csv_path = csv_path
        self.rows = rows
        self.fieldnames = list(rows[0].keys()) if rows else []

    def require_x_parameter(self, x_parameter: str) -> None:
        if x_parameter not in self.fieldnames:
            raise ValueError(
                f"CSV for {self.label!r} ({self.csv_path}) has no column "
                f"{x_parameter!r}. Columns: {self.fieldnames}"
            )


def load_experiments(
    specs: Sequence[str], x_parameter: str
) -> List[ExperimentAggregate]:
    loaded: List[ExperimentAggregate] = []
    seen_labels = set()
    for spec in specs:
        label, raw_path = parse_experiment_spec(spec)
        if label in seen_labels:
            raise ValueError(f"Duplicate experiment label {label!r}")
        seen_labels.add(label)
        csv_path = resolve_aggregated_csv(raw_path, x_parameter=x_parameter)
        rows = load_aggregated_csv(csv_path)
        if not rows:
            raise ValueError(f"CSV for {label!r} is empty: {csv_path}")
        experiment = ExperimentAggregate(label, csv_path, rows)
        experiment.require_x_parameter(x_parameter)
        loaded.append(experiment)
        print(f"Loaded {label}: {csv_path} ({len(rows)} rows)")
    return loaded


def build_combined_rows(
    experiments: Sequence[ExperimentAggregate],
    metrics: Sequence[str],
    x_parameter: str,
    summaries: Dict[str, Dict[str, List[Dict[str, object]]]],
) -> List[Dict[str, object]]:
    combined: List[Dict[str, object]] = []
    for experiment in experiments:
        per_metric = summaries[experiment.label]
        x_values = sorted(
            {point["x"] for series in per_metric.values() for point in series},
            key=sort_key,
        )
        n_by_x: Dict[object, int] = {}
        for series in per_metric.values():
            for point in series:
                n_by_x[point["x"]] = max(n_by_x.get(point["x"], 0), int(point["n"]))
        lookup = {
            metric: {point["x"]: point for point in series}
            for metric, series in per_metric.items()
        }
        for x_val in x_values:
            row: Dict[str, object] = {
                "experiment": experiment.label,
                "source_csv": experiment.csv_path,
                x_parameter: x_val,
                "n": n_by_x.get(x_val, 0),
            }
            for metric in metrics:
                point = lookup.get(metric, {}).get(x_val)
                if point is None:
                    row[metric] = ""
                    row[f"{metric}_std"] = ""
                else:
                    row[metric] = point["mean"]
                    row[f"{metric}_std"] = (
                        point["std"] if point["std"] is not None else ""
                    )
            combined.append(row)
    return combined


def build_combined_rows_channel_pair(
    experiments: Sequence[ExperimentAggregate],
    metrics: Sequence[str],
    x_parameter: str,
    summaries: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]],
) -> List[Dict[str, object]]:
    """One combined row per ``(experiment, x, channel)``."""
    combined: List[Dict[str, object]] = []
    for experiment in experiments:
        per_metric = summaries[experiment.label]
        keys: List[Tuple[object, str]] = []
        seen = set()
        n_by_key: Dict[Tuple[object, str], int] = {}
        for by_channel in per_metric.values():
            for channel, series in by_channel.items():
                for point in series:
                    key = (point["x"], str(channel))
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
                    n_by_key[key] = max(n_by_key.get(key, 0), int(point["n"]))
        keys.sort(key=lambda item: (sort_key(item[0]), channel_sort_key(item[1])))
        lookup = {
            metric: {
                (point["x"], channel): point
                for channel, series in by_channel.items()
                for point in series
            }
            for metric, by_channel in per_metric.items()
        }
        for x_val, channel in keys:
            row: Dict[str, object] = {
                "experiment": experiment.label,
                "source_csv": experiment.csv_path,
                x_parameter: x_val,
                "channel": channel,
                "n": n_by_key.get((x_val, channel), 0),
            }
            for metric in metrics:
                point = lookup.get(metric, {}).get((x_val, channel))
                if point is None:
                    row[metric] = ""
                    row[f"{metric}_std"] = ""
                else:
                    row[metric] = point["mean"]
                    row[f"{metric}_std"] = (
                        point["std"] if point["std"] is not None else ""
                    )
            combined.append(row)
    return combined


def save_combined_csv(
    rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    x_parameter: str,
    output_file: str,
    extra_fieldnames: Optional[Sequence[str]] = None,
) -> None:
    fieldnames = ["experiment", "source_csv", x_parameter]
    if extra_fieldnames:
        fieldnames.extend(extra_fieldnames)
    fieldnames.append("n")
    for metric in metrics:
        fieldnames.append(metric)
        fieldnames.append(f"{metric}_std")
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric_overlay(
    experiments: Sequence[ExperimentAggregate],
    metric: str,
    x_parameter: str,
    common_x: Sequence[object],
    summaries: Dict[str, List[Dict[str, object]]],
    output_path: str,
) -> bool:
    """Draw one overlay curve per experiment. Return False if nothing to plot."""
    plotted_y: List[float] = []
    config = apply_paper_ready_line_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    drew_any = False

    for experiment in experiments:
        series = filter_series_to_x(summaries.get(experiment.label, []), common_x)
        if not series:
            print(
                f"Warning: {experiment.label!r} has no values for {metric} "
                f"on the shared {x_parameter} grid; skipping that curve."
            )
            continue
        xs = [point["x"] for point in series]
        means = [float(point["mean"]) for point in series]
        stds = [point["std"] for point in series]
        if all(is_numeric(x) for x in xs):
            plot_x = [float(x) for x in xs]
        else:
            plot_x = list(range(len(xs)))
            ax.set_xticks(plot_x)
            ax.set_xticklabels([str(x) for x in xs], rotation=45)
        (line,) = ax.plot(plot_x, means, "o-", label=experiment.label)
        plotted_y.extend(means)
        if any(std is not None for std in stds):
            lower = []
            upper = []
            for mean, std in zip(means, stds):
                if std is None:
                    lower.append(mean)
                    upper.append(mean)
                else:
                    lo = mean - float(std)
                    hi = mean + float(std)
                    # Keep error bands valid if the y-axis later becomes log.
                    if mean > 0 and lo <= 0:
                        lo = max(np.nextafter(0, 1), mean * 1e-3)
                    lower.append(lo)
                    upper.append(hi)
                    plotted_y.extend([lo, hi])
            ax.fill_between(plot_x, lower, upper, color=line.get_color(), alpha=0.2)
        drew_any = True

    if not drew_any:
        plt.close(fig)
        print(f"No data to plot for {metric}. Skipping.")
        return False

    setup_plot_y_axis(ax, plotted_y)
    ax.set_xlabel(get_display_name(x_parameter))
    ax.set_ylabel(get_metric_display_name(metric))
    ax.legend(loc="best")
    if config["show_title"]:
        ax.set_title(
            f"{get_metric_display_name(metric)} vs {get_display_name(x_parameter)}"
        )
    save_figure(fig, output_path, left_margin=0.20)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return True


def _draw_overlay_curve(
    ax,
    series: Sequence[Dict[str, object]],
    label: str,
    plotted_y: List[float],
):
    """Draw one experiment curve + optional std band. Mutates ``plotted_y``."""
    xs = [point["x"] for point in series]
    means = [float(point["mean"]) for point in series]
    stds = [point["std"] for point in series]
    if all(is_numeric(x) for x in xs):
        plot_x = [float(x) for x in xs]
    else:
        plot_x = list(range(len(xs)))
        ax.set_xticks(plot_x)
        ax.set_xticklabels([str(x) for x in xs], rotation=45)
    (line,) = ax.plot(plot_x, means, "o-", label=label)
    plotted_y.extend(means)
    if any(std is not None for std in stds):
        lower = []
        upper = []
        for mean, std in zip(means, stds):
            if std is None:
                lower.append(mean)
                upper.append(mean)
            else:
                lo = mean - float(std)
                hi = mean + float(std)
                if mean > 0 and lo <= 0:
                    lo = max(np.nextafter(0, 1), mean * 1e-3)
                lower.append(lo)
                upper.append(hi)
                plotted_y.extend([lo, hi])
        ax.fill_between(plot_x, lower, upper, color=line.get_color(), alpha=0.2)
    return True


def plot_metric_channel_panels(
    experiments: Sequence[ExperimentAggregate],
    metric: str,
    x_parameter: str,
    summaries: Dict[str, Dict[str, List[Dict[str, object]]]],
    output_path: str,
) -> bool:
    """Four-panel (ch1–4) overlay of experiment curves. Return False if empty."""
    found_channels = {
        channel
        for by_channel in summaries.values()
        for channel in by_channel
        if by_channel[channel]
    }
    channels = channels_for_figure(found_channels)
    if not any(summaries.get(exp.label, {}).get(ch) for exp in experiments for ch in channels):
        print(f"No per-channel data to plot for {metric}. Skipping.")
        return False

    n_channels = len(channels)
    n_cols = 2
    n_rows = int(math.ceil(n_channels / n_cols))
    config = apply_paper_ready_line_style()
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14.0, 5.2 * n_rows), squeeze=False
    )
    plotted_y: List[float] = []
    drew_any = False
    legend_handles = None
    legend_labels = None

    for idx, channel in enumerate(channels):
        ax = axes[idx // n_cols][idx % n_cols]
        series_by_label = {
            experiment.label: summaries.get(experiment.label, {}).get(channel, [])
            for experiment in experiments
        }
        nonempty = {
            label: series for label, series in series_by_label.items() if series
        }
        if not nonempty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(channel_display_name(channel))
            continue
        common_x = intersect_x_values(nonempty)
        if not common_x:
            ax.text(
                0.5,
                0.5,
                "no shared x",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(channel_display_name(channel))
            continue
        for experiment in experiments:
            series = filter_series_to_x(
                series_by_label.get(experiment.label, []), common_x
            )
            if not series:
                continue
            _draw_overlay_curve(ax, series, experiment.label, plotted_y)
            drew_any = True
        ax.set_title(channel_display_name(channel))
        ax.set_xlabel(get_display_name(x_parameter))
        ax.set_ylabel(get_metric_display_name(metric))
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for idx in range(n_channels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if not drew_any:
        plt.close(fig)
        print(f"No data to plot for {metric}. Skipping.")
        return False

    if plotted_y:
        for idx, _channel in enumerate(channels):
            ax = axes[idx // n_cols][idx % n_cols]
            if ax.lines:
                setup_plot_y_axis(ax, plotted_y)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=max(1, len(legend_labels)),
            bbox_to_anchor=(0.5, 1.02),
        )
    if config["show_title"]:
        fig.suptitle(
            f"{get_metric_display_name(metric)} vs {get_display_name(x_parameter)}"
        )
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return True


def run_compare(
    experiment_specs: Sequence[str],
    metrics: Sequence[str],
    x_parameter: str,
    output_dir: str,
    channel_pair: bool = False,
) -> Dict[str, object]:
    """Load CSVs, write overlay PNGs + combined CSV. Returns output paths."""
    if not experiment_specs:
        raise ValueError("Provide at least one --experiments label|path pair")
    if not metrics:
        raise ValueError("Provide at least one --metrics name")

    os.makedirs(output_dir, exist_ok=True)
    experiments = load_experiments(experiment_specs, x_parameter)
    if channel_pair:
        return _run_compare_channel_pair(
            experiments, metrics, x_parameter, output_dir
        )

    summaries: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for experiment in experiments:
        summaries[experiment.label] = {}
        missing = [m for m in metrics if m not in experiment.fieldnames]
        if missing:
            print(
                f"Warning: {experiment.label!r} CSV is missing columns {missing}: "
                f"{experiment.csv_path}"
            )
        for metric in metrics:
            summaries[experiment.label][metric] = summarize_metric_by_x(
                experiment.rows, x_parameter, metric
            )

    combined_rows = build_combined_rows(
        experiments, metrics, x_parameter, summaries
    )
    combined_csv = os.path.join(output_dir, "compare_aggregated_values.csv")
    save_combined_csv(combined_rows, metrics, x_parameter, combined_csv)
    print(f"Wrote {combined_csv}")

    plot_paths: List[str] = []
    for metric in metrics:
        series_by_label = {
            experiment.label: summaries[experiment.label][metric]
            for experiment in experiments
        }
        nonempty = {
            label: series for label, series in series_by_label.items() if series
        }
        if not nonempty:
            print(f"Warning: no experiment has values for {metric}; skipping plot.")
            continue
        common_x = intersect_x_values(nonempty)
        if not common_x:
            print(
                f"Warning: no shared {x_parameter} values for {metric}; skipping plot."
            )
            continue
        plot_path = os.path.join(
            output_dir, f"compare_{metric}_vs_{x_parameter}.png"
        )
        if plot_metric_overlay(
            experiments, metric, x_parameter, common_x, series_by_label, plot_path
        ):
            plot_paths.append(plot_path)

    if not plot_paths:
        raise ValueError(
            f"Did not produce any overlay plots. Check that the CSVs contain "
            f"{x_parameter} and at least one of: {list(metrics)}"
        )

    return {"combined_csv": combined_csv, "plot_paths": plot_paths}


def _run_compare_channel_pair(
    experiments: Sequence[ExperimentAggregate],
    metrics: Sequence[str],
    x_parameter: str,
    output_dir: str,
) -> Dict[str, object]:
    """Join 1d-long and 4d-wide aggregates on ``(x, channel)`` and plot panels."""
    summaries: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}
    for experiment in experiments:
        summaries[experiment.label] = {}
        for metric in metrics:
            try:
                long_points = expand_rows_to_channel_long(
                    experiment.rows, x_parameter, metric
                )
            except ValueError as exc:
                print(f"Warning: {experiment.label!r}: {exc}")
                summaries[experiment.label][metric] = {}
                continue
            summarized = summarize_metric_by_x_channel(long_points)
            summaries[experiment.label][metric] = pivot_channel_series(summarized)
            channels = sorted(
                summaries[experiment.label][metric], key=channel_sort_key
            )
            print(
                f"{experiment.label} {metric}: "
                f"{len(summarized)} (x, channel) points "
                f"across {channels}"
            )

    combined_rows = build_combined_rows_channel_pair(
        experiments, metrics, x_parameter, summaries
    )
    combined_csv = os.path.join(output_dir, "compare_aggregated_values.csv")
    save_combined_csv(
        combined_rows,
        metrics,
        x_parameter,
        combined_csv,
        extra_fieldnames=["channel"],
    )
    print(f"Wrote {combined_csv}")

    plot_paths: List[str] = []
    for metric in metrics:
        series_by_label = {
            experiment.label: summaries[experiment.label][metric]
            for experiment in experiments
        }
        if not any(series_by_label.values()):
            print(
                f"Warning: no experiment has per-channel values for {metric}; "
                "skipping plot."
            )
            continue
        plot_path = os.path.join(
            output_dir, f"compare_{metric}_vs_{x_parameter}_by_channel.png"
        )
        if plot_metric_channel_panels(
            experiments, metric, x_parameter, series_by_label, plot_path
        ):
            plot_paths.append(plot_path)

    if not plot_paths:
        raise ValueError(
            f"Did not produce any channel-pair plots. Check that the CSVs "
            f"contain {x_parameter} plus either observation_process=raw_chK "
            f"with base metric columns or {{metric}}_chK columns for: "
            f"{list(metrics)}"
        )

    return {"combined_csv": combined_csv, "plot_paths": plot_paths}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay metrics from finished aggregate_eval_plots_*/aggregated_values.csv "
            "files (one curve per labeled experiment). Use --channel-pair to join "
            "1d-sep long-form and joint-4d wide-form per-channel scores."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help=(
            "Labeled sources as label|path or label=path. path may be the "
            "experiment root, an aggregate_eval_plots_* directory, or "
            "aggregated_values.csv itself."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["kld_auto", "spectrum_error_auto"],
        help="Metric columns to overlay (default: kld_auto spectrum_error_auto).",
    )
    parser.add_argument(
        "--x-parameter",
        default="sampling_ratio",
        help="X-axis column shared across CSVs (default: sampling_ratio).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for overlay PNGs and compare_aggregated_values.csv.",
    )
    parser.add_argument(
        "--channel-pair",
        action="store_true",
        help=(
            "Join 1d-trained long-form rows (observation_process=raw_chK, "
            "score in the base metric column) with joint-4d wide-form "
            "{metric}_chK columns on (sampling_ratio, channel). Writes a "
            "four-panel overlay per metric and a combined CSV with a "
            "channel column. Unset keeps the default mean±std collapse "
            "by sampling_ratio only."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_compare(
            experiment_specs=args.experiments,
            metrics=args.metrics,
            x_parameter=args.x_parameter,
            output_dir=args.output_dir,
            channel_pair=args.channel_pair,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
