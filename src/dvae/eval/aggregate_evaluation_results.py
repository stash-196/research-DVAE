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


def extract_param_values(data, param):
    """Extract unique values for a parameter."""
    values = set()
    for d in data:
        if "params" in d and param in d["params"]:
            values.add(d["params"][param])
        elif "config" in d and param in d["config"]:
            # Check in config sections
            for section in d["config"].values():
                if isinstance(section, dict) and param in section:
                    values.add(section[param])
                    break
    return sorted(list(values))


def plot_1d(data, param, metric, output_dir):
    """Plot 1D graph for a metric vs one parameter."""
    param_values = []
    metric_values = []
    for d in data:
        # Get param value
        p_val = None
        if "params" in d and param in d["params"]:
            p_val = d["params"][param]
        elif "config" in d:
            for section in d["config"].values():
                if isinstance(section, dict) and param in section:
                    p_val = section[param]
                    break
        if p_val is None:
            continue
        # Get metric value
        if metric in d:
            param_values.append(p_val)
            metric_values.append(d[metric])

    if not param_values:
        print(f"No data for param {param} and metric {metric}")
        return

    plt.figure()
    plt.plot(param_values, metric_values, "o-")
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {param}")
    plt.savefig(os.path.join(output_dir, f"{metric}_vs_{param}.png"))
    plt.close()


def plot_2d(data, param1, param2, metric, output_dir):
    """Plot 2D heatmap for a metric vs two parameters."""
    param1_values = extract_param_values(data, param1)
    param2_values = extract_param_values(data, param2)

    # Create a grid
    grid = np.full((len(param1_values), len(param2_values)), np.nan)

    for d in data:
        # Get param values
        p1_val = None
        p2_val = None
        if "params" in d:
            p1_val = d["params"].get(param1)
            p2_val = d["params"].get(param2)
        if p1_val is None or p2_val is None:
            # Check config
            for section in d.get("config", {}).values():
                if isinstance(section, dict):
                    if p1_val is None and param1 in section:
                        p1_val = section[param1]
                    if p2_val is None and param2 in section:
                        p2_val = section[param2]
        if p1_val is None or p2_val is None or metric not in d:
            continue

        i = param1_values.index(p1_val) if p1_val in param1_values else None
        j = param2_values.index(p2_val) if p2_val in param2_values else None
        if i is not None and j is not None:
            grid[i, j] = d[metric]

    plt.figure()
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[
            min(param2_values),
            max(param2_values),
            min(param1_values),
            max(param1_values),
        ],
    )
    plt.colorbar(label=metric)
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f"{metric} heatmap: {param1} vs {param2}")
    plt.savefig(os.path.join(output_dir, f"{metric}_heatmap_{param1}_vs_{param2}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize evaluation results."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        required=True,
        help="Parameters to use for plotting (1 or 2).",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["kl_tf", "kl_auto"], help="Metrics to plot."
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

    args = parser.parse_args()

    if len(args.parameters) not in [1, 2]:
        print("Provide 1 or 2 parameters.")
        return

    # Find YAML files
    yaml_files = find_yaml_files(args.experiment_dir)
    print(f"Found {len(yaml_files)} evaluation_summary.yaml files.")

    # Save list
    save_yaml_list(yaml_files, args.yaml_list_file)

    # Load data
    data = load_yaml_data(yaml_files)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot
    for metric in args.metrics:
        if len(args.parameters) == 1:
            plot_1d(data, args.parameters[0], metric, args.output_dir)
        else:
            plot_2d(
                data, args.parameters[0], args.parameters[1], metric, args.output_dir
            )

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
