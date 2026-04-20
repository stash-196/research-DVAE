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
from PIL import Image
from dvae.visualizers.visualizers import get_plot_config

# Dictionary for display names
DISPLAY_NAMES = {
    "mask_label": "Missing Ratio",
    "sampling_ratio": "Auto Ratio",
}

# Dictionary for metric display names
METRIC_DISPLAY_NAMES = {
    "kld_tf": "KLD of Teacher Forced",
    "kld_auto": "KLD of Autonomous",
    "spectrum_error_gt": "Spectrum Error Ground Truth",
    "spectrum_error_tf": "Spectrum Error Teacher Forced",
    "spectrum_error_auto": "Spectrum Error Autonomous",
}


def get_display_name(param):
    """Get display name for parameter, default to param if not found."""
    return DISPLAY_NAMES.get(param, param)


def get_metric_display_name(metric):
    """Get display name for metric, default to metric if not found."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


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
    return sorted(list(values))


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
        if metric in d:
            param_values.append(p_val)
            metric_values.append(d[metric])

    if len(param_values) < 2:
        print(
            f"Not enough varying data for param {param} and metric {metric} (found {len(param_values)} points). Skipping plot."
        )
        return

    config = get_plot_config(paper_ready=True)
    plt.figure(figsize=(8, 6))
    plt.plot(param_values, metric_values, "o-")
    plt.xlabel(get_display_name(param))
    plt.ylabel(get_metric_display_name(metric))
    if config["show_title"]:
        plt.title(f"{get_metric_display_name(metric)} vs {get_display_name(param)}")
    plt.savefig(os.path.join(output_dir, f"{metric}_vs_{param}.png"))
    plt.close()


def plot_2d(data, param1, param2, metric, output_dir):
    """Plot 2D heatmap for a metric vs two parameters."""
    param1_values = extract_param_values(data, param1)
    param2_values = extract_param_values(data, param2)

    if len(param1_values) < 2 or len(param2_values) < 2:
        print(
            f"Not enough varying data for parameters {param1} ({len(param1_values)} values) or {param2} ({len(param2_values)} values). Skipping heatmap."
        )
        return

    # Create a grid
    grid = np.full((len(param1_values), len(param2_values)), np.nan)

    for d in data:
        # Get param values
        p1_val = get_param_value(d, param1)
        p2_val = get_param_value(d, param2)
        if p1_val is None or p2_val is None or metric not in d:
            continue

        i = param1_values.index(p1_val) if p1_val in param1_values else None
        j = param2_values.index(p2_val) if p2_val in param2_values else None
        if i is not None and j is not None:
            grid[i, j] = d[metric]

    if np.all(np.isnan(grid)):
        print(
            f"No valid data points for 2D plot of {metric} vs {param1} and {param2}. Skipping."
        )
        return

    config = get_plot_config(paper_ready=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
    )
    plt.colorbar(label=get_metric_display_name(metric))

    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xticks(range(len(param2_values)))
    ax.set_xticklabels(param2_values)
    ax.set_yticks(range(len(param1_values)))
    ax.set_yticklabels(param1_values)

    plt.xlabel(get_display_name(param2))
    plt.ylabel(get_display_name(param1))
    if config["show_title"]:
        plt.title(
            f"{get_metric_display_name(metric)} heatmap: {get_display_name(param1)} vs {get_display_name(param2)}",
            pad=20,
        )

    # Add annotations if grid is small enough
    if len(param1_values) <= 5 and len(param2_values) <= 5:
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(grid[i, j]):
                    plt.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="white",
                    )

    plt.savefig(os.path.join(output_dir, f"{metric}_heatmap_{param1}_vs_{param2}.png"))
    plt.close()


def aggregate_delay_embeddings(data, output_dir):
    """Aggregate delay embedding GIFs into tiled figures for autonomous and teacher-forced modes."""
    # Get global unique values
    unique_sampling = sorted(set(get_param_value(d, "sampling_ratio") for d in data))
    unique_mask = sorted(set(get_param_value(d, "mask_label") for d in data))

    rows = len(unique_sampling)
    cols = len(unique_mask)

    config = get_plot_config(paper_ready=True)

    for mode in ["tf", "autonomous"]:
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axs = np.atleast_2d(axs).reshape(rows, cols)

        # Dictionary to store which cells have been filled
        filled = {}

        for d in data:
            sampling = get_param_value(d, "sampling_ratio")
            mask = get_param_value(d, "mask_label")
            i = unique_sampling.index(sampling)
            j = unique_mask.index(mask)

            if (i, j) in filled:
                continue  # Skip if already filled

            # Find the output dir
            yaml_dir = os.path.dirname(d["yaml_file"])
            # Assume the GIF is in post_training_figs/vis_delay_embedding_of_*mode*.gif
            if mode == "tf":
                gif_pattern = os.path.join(
                    yaml_dir,
                    "post_training_figs",
                    "vis_delay_embedding_of_*teacher-forced*.gif",
                )
            else:
                gif_pattern = os.path.join(
                    yaml_dir,
                    "post_training_figs",
                    "vis_delay_embedding_of_*autonomous*.gif",
                )
            gif_files = glob.glob(gif_pattern)
            if not gif_files:
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

            # Take the first one
            gif_path = gif_files[0]
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

                # Top row prints the mask label, left column prints the sampling ratio
                if i == 0:
                    axs[i, j].set_title(str(unique_mask[j]), fontsize=45, pad=15)
                if j == 0:
                    axs[i, j].set_ylabel(
                        str(unique_sampling[i]), fontsize=45, labelpad=15
                    )

        # Global axis labels
        fig.suptitle(get_display_name("mask_label"), fontsize=55, y=1.02)
        fig.supylabel(get_display_name("sampling_ratio"), fontsize=55)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
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
        "parameters",
        nargs="+",
        help="Parameters to use for plotting (1 or 2).",
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

    # Aggregate delay embeddings
    aggregate_delay_embeddings(data, args.output_dir)

    # Collect parameter values for all runs
    param_table = []
    for d in data:
        row = {}
        for param in args.parameters:
            row[param] = get_param_value(d, param)
        param_table.append(row)

    # Print table
    if param_table:
        print("\nParameter values across runs:")
        headers = args.parameters
        print("\t".join(headers))
        for row in param_table:
            print("\t".join(str(row.get(h, "N/A")) for h in headers))

    # Save to CSV
    csv_file = os.path.join(args.output_dir, "parameter_values.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.parameters)
        writer.writeheader()
        for row in param_table:
            writer.writerow({k: v for k, v in row.items() if k in args.parameters})

    if args.verbose:
        print(f"Saved parameter values to {csv_file}")

    # Plot
    if args.verbose:
        print(
            f"Generating plots for parameters: {args.parameters}, metrics: {args.metrics}"
        )
    for metric in args.metrics:
        if len(args.parameters) == 1:
            if args.verbose:
                print(f"Plotting 1D: {metric} vs {args.parameters[0]}")
            plot_1d(data, args.parameters[0], metric, args.output_dir)
        else:
            if args.verbose:
                print(
                    f"Plotting 2D: {metric} heatmap for {args.parameters[0]} vs {args.parameters[1]}"
                )
            plot_2d(
                data, args.parameters[0], args.parameters[1], metric, args.output_dir
            )

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
