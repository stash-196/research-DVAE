#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import uuid
import torch
import argparse
import json
import yaml
import sys
import time
import numpy as np
from dvae.learning_algo import LearningAlgorithm
from dvae.dataset import sinusoid_dataset, lorenz63_dataset
from dvae.utils import (
    loss_MSE,
    create_autonomous_mode_selector,
    run_parallel_visualizations,
    power_spectrum_error,
    calculate_power_spectrum_error,
    calculate_expected_accuracy,
    rmse,
    r_squared,
    expand_autonomous_mode_selector,
    load_device_paths,
)
from dvae.visualizers import (
    visualize_variable_evolution,
    visualize_sequences,
    visualize_spectral_analysis,
    visualize_teacherforcing_2_autonomous,
    visualize_embedding_space,
    visualize_accuracy_over_time,
    visualize_delay_embedding,
    visualize_alpha_history_and_spectrums,
    visualize_errors_from_lst,
)
from dvae.visualizers import get_plot_config
from dvae.eval.utils import (
    run_spectrum_analysis,
    run_mse_analysis,
    run_geometry_analysis,
    compute_delay_embedding,
    state_space_kl,
    compute_local_drift_statistics,
    run_forward_with_mode,
    get_flip_point_for_mode,
    get_channel_benchmarks,
    merge_batch_metric_dicts,
    flatten_analysis_to_batch_metrics,
)
from dvae.eval.utils.batch_all_visuals import (
    collect_batch_visual_record,
    compute_reference_channel_errors,
    compute_stitched_kld_metrics,
    render_batch_all_visuals,
    render_summary_error_bars,
)
from dvae.eval.utils.forward_modes import mode_selector_to_1d

from torch.nn.functional import mse_loss
import plotly.graph_objects as go
import plotly.express as px
import pickle
import configparser
from typing import Any, Optional
from dataclasses import replace
from dvae.dataset.dataset_builder import build_dataloader, DatasetConfig


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument("--cfg", type=str, default=None, help="config path")
        self.parser.add_argument(
            "--saved_dict", type=str, required=True, help="trained model dict"
        )
        self.parser.add_argument(
            "--save-3d",
            dest="save_3d",
            type=lambda s: str(s).lower() in ("true", "1", "yes", "y"),
            default=True,
            help="Whether to save 3D visualizations (True/False). Default: True",
        )
        self.parser.add_argument(
            "--max-eval-batches",
            type=int,
            default=20,
            help="Max test batches for quantitative metrics (default: 20).",
        )
        self.parser.add_argument(
            "--auto-eval-mode",
            type=str,
            default="half_half",
            help=(
                "Mode selector for autonomous metrics: half_half (default; TF then Auto, "
                "window length 2*block_len), alternating_blocks, even_bursts, "
                "flip_at_index, all_1."
            ),
        )
        self.parser.add_argument(
            "--auto-eval-flip-point",
            type=int,
            default=None,
            help="Flip point for flip_at_index auto eval mode (optional).",
        )
        self.parser.add_argument(
            "--auto-eval-block-len",
            type=int,
            default=1000,
            help=(
                "Block length for alternating_blocks (default: 1000, paper protocol). "
                "Also used as even_bursts flip interval when --auto-eval-mode even_bursts "
                "and ratio is not set via legacy path."
            ),
        )
        self.parser.add_argument(
            "--auto-eval-ratio",
            type=float,
            default=0.1,
            help="Autonomous ratio for even_bursts / flip_at_middle metrics (default: 0.1).",
        )
        self.parser.add_argument(
            "--metrics-seq-len",
            type=int,
            default=None,
            help=(
                "Override sequence length for quantitative metrics windows. "
                "If omitted, length is chosen from --auto-eval-mode: "
                "half_half/flip_at_index -> 2*block_len; "
                "alternating_blocks -> min(10*block_len, data); "
                "else min(20000, data)."
            ),
        )
        self.parser.add_argument(
            "--skip-metrics-viz-after-batch-0",
            dest="skip_metrics_viz_after_batch_0",
            type=lambda s: str(s).lower() in ("true", "1", "yes", "y"),
            default=True,
            help=(
                "Only save metric figures (MSE/KLD/spectrum/GIFs) for window 0; "
                "later windows compute scalars only (default: True)."
            ),
        )

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        if params["cfg"] is None:
            params["cfg"] = os.path.join(
                os.path.dirname(params["saved_dict"]), "config.ini"
            )
        return params


def _is_indicate_observation(observation_process: Optional[str]) -> bool:
    """Return True for 'only_x_indicate' or any '<base>_indicate' form."""
    if observation_process is None:
        return False
    if observation_process == "only_x_indicate":
        return True
    if isinstance(observation_process, str) and observation_process.endswith(
        "_indicate"
    ):
        return True
    return False


def visualize_training_mode_short_sequence(
    dataset_name,
    dataset_config,
    device,
    dvae,
    save_fig_dir,
    seq_len=1000,
):
    """
    Visualize a short sequence with training-mode missingness applied (half-half TF/auto).
    This shows what the model sees during training, with proper NaN overlay on mode selector.

    Args:
    - dataset_name: Name of the dataset
    - dataset_config: DatasetConfig object for building the dataloader
    - device: Device to use
    - dvae: The trained model
    - save_fig_dir: Directory to save visualizations
    - seq_len: Sequence length to use
    """
    try:
        # Build training dataloader with same config but split="train"
        train_dataloader_result = build_dataloader(
            dataset_name, dataset_config, split="train", eval_mode=False
        )
        if isinstance(train_dataloader_result, tuple):
            train_dataloader = train_dataloader_result[0]
        else:
            train_dataloader = train_dataloader_result

        # Update sequence length to the short eval length
        train_dataloader.dataset.update_sequence_length(seq_len)

        # Get first batch
        batch_data = next(iter(train_dataloader))
        batch_data = batch_data.to(device)
        batch_data = batch_data.permute(1, 0, 2)  # (seq_len, batch_size, x_dim)
        seq_len_short, batch_size_short, x_dim_short = batch_data.shape

        # Extract missing mask for training batch (for visualization only)
        missing_mask_short = None
        observation_process = getattr(dataset_config, "observation_process", None)

        if _is_indicate_observation(observation_process) and batch_data.size(2) >= 2:
            # For only_x_indicate, dimension 1 is the is_observed indicator
            is_observed = batch_data[:, :, 1]  # (seq_len, batch_size)
            missing_mask_short = (
                (is_observed < 0.5).float().unsqueeze(-1)
            )  # (seq_len, batch_size, 1)

        elif hasattr(train_dataloader.dataset, "missing_mask"):
            # Fallback: extract from dataset.missing_mask
            batch_start_idx = (
                train_dataloader.dataset.data_idx[0]
                if len(train_dataloader.dataset.data_idx) > 0
                else 0
            )
            batch_end_idx = min(
                batch_start_idx + seq_len_short,
                len(train_dataloader.dataset.missing_mask),
            )
            missing_mask_slice = train_dataloader.dataset.missing_mask[
                batch_start_idx:batch_end_idx
            ]

            # Ensure (seq_len, batch_size, 1) shape
            if isinstance(missing_mask_slice, np.ndarray):
                missing_mask_short = torch.from_numpy(missing_mask_slice).float()
            else:
                missing_mask_short = missing_mask_slice.float()

            if missing_mask_short.ndim == 1:
                missing_mask_short = missing_mask_short.unsqueeze(1).expand(
                    -1, batch_size_short, -1
                )  # (seq_len, batch_size, 1)
            elif missing_mask_short.ndim == 2:
                missing_mask_short = missing_mask_short.unsqueeze(
                    -1
                )  # (seq_len, batch_size, 1)

        # === Create base mode selector (half-half TF/auto) ===
        base_mode_selector = create_autonomous_mode_selector(
            seq_len_short,
            mode="half_half",
            batch_size=batch_size_short,
            x_dim=x_dim_short,
            device=device,
        )

        # === Overlay NaN positions with autonomous mode (just like learning_algo.py) ===
        model_mode_selector = torch.where(
            batch_data.isnan().detach(),
            torch.tensor(1.0, device=device),
            base_mode_selector,
        )

        # === Force pure TF on mask dimension if observation_process is an indicate variant ===
        if _is_indicate_observation(observation_process):
            model_mode_selector = model_mode_selector.clone()
            model_mode_selector[:, :, 1] = 0.0  # Mask channel: pure TF

        # Generate reconstruction
        batch_data_tensor = batch_data.clone().detach().to(device)
        recon_short = (
            dvae(
                batch_data_tensor,
                mode_selector=model_mode_selector,
                inference_mode=True,
            )
            .detach()
            .cpu()
            .numpy()
        )

        # Visualize with training missingness (just first sample for clarity)
        visualize_teacherforcing_2_autonomous(
            batch_data[:, :1, :],
            dvae,
            auto_mode_selector=model_mode_selector[:, :1, :],
            save_path=save_fig_dir,
            explain="training_mode_short_with_missingness",
            inference_mode=True,
            missing_mask=(
                missing_mask_short[:, :1, :] if missing_mask_short is not None else None
            ),
            hide_mask_output=_is_indicate_observation(observation_process),
        )

        print("[Eval] Training mode short visualization completed")

    except Exception as e:
        print(f"[Eval] [Warning] Could not visualize training mode short sequence: {e}")


def compute_missing_stats(dataset_name, dataset_config):
    """
    Compute missing-value statistics for the x signal for train and test datasets
    and return them so they can be stored in evaluation_summary.yaml.
    """
    try:
        stats = {}

        def summarize_missing_x(dataset):
            missing_mask = getattr(dataset, "missing_mask", None)
            if missing_mask is None:
                return 0, 0

            mask_arr = np.asarray(missing_mask)
            if mask_arr.ndim == 0:
                return int(mask_arr), 1

            if mask_arr.ndim == 1:
                missing_count = int(np.sum(mask_arr))
                total_x = int(mask_arr.shape[0])
                return missing_count, total_x

            x_mask = mask_arr[:, 0]
            missing_count = int(np.sum(x_mask))
            total_x = int(x_mask.shape[0])
            return missing_count, total_x

        # TRAIN
        train_res = build_dataloader(
            dataset_name, dataset_config, split="train", eval_mode=False
        )
        if isinstance(train_res, tuple):
            train_dataloader = train_res[0]
        else:
            train_dataloader = train_res
        train_dataset = train_dataloader.dataset

        missing_x_train, total_x_train = summarize_missing_x(train_dataset)

        stats["train"] = {
            "missing_count": int(missing_x_train),
            "total_x": int(total_x_train),
            "missing_percentage": float(missing_x_train) / max(1, total_x_train),
        }

        # TEST
        test_res = build_dataloader(
            dataset_name, dataset_config, split="test", eval_mode=True
        )
        if isinstance(test_res, tuple):
            test_dataloader = test_res[0]
        else:
            test_dataloader = test_res
        test_dataset = test_dataloader.dataset

        missing_x_test, total_x_test = summarize_missing_x(test_dataset)

        stats["test"] = {
            "missing_count": int(missing_x_test),
            "total_x": int(total_x_test),
            "missing_percentage": float(missing_x_test) / max(1, total_x_test),
        }

        print("[Eval] Computed missing-value stats")
        return stats

    except Exception as e:
        print(f"[Eval] Warning: could not compute missing stats: {e}")
        return None


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    params = Options().get_params()
    params["job_id"] = "eval_" + str(uuid.uuid4())[:8]  # Add a unique job_id for

    # Load device-specific paths
    device_config = load_device_paths(os.path.join("config", "device_paths.yaml"))
    params["device_config"] = device_config

    device = "cpu"
    learning_algo = LearningAlgorithm(params=params)
    learning_algo.build_model(device=device)
    dvae = learning_algo.model.to(device)
    dvae.device = device
    if "final" in params["saved_dict"]:
        loaded_model = torch.load(params["saved_dict"], map_location="cpu")
        dvae.load_state_dict(loaded_model)
    elif "checkpoint" in params["saved_dict"]:
        loaded_model = torch.load(
            params["saved_dict"], weights_only=False, map_location="cpu"
        )
        dvae.load_state_dict(loaded_model["model_state_dict"])
    else:
        print(
            "[Eval] Warning: Unrecognized model file name format. (final or checkpoint expected)"
        )

    dvae.eval()
    cfg = learning_algo.cfg
    print("[Eval] Total params: {:,}".format(dvae.parameter_count()))

    # Convert config to dict for YAML
    config_dict = {section: dict(cfg[section]) for section in cfg.sections()}

    # Create DatasetConfig for evaluation
    dataset_config = DatasetConfig(
        data_dir=learning_algo.data_dir,
        x_dim=cfg.getint("Network", "x_dim"),
        batch_size=cfg.getint("DataFrame", "batch_size"),
        shuffle=cfg.getboolean("DataFrame", "shuffle"),
        num_workers=cfg.getint("DataFrame", "num_workers"),
        sample_rate=cfg.getint("DataFrame", "sample_rate"),
        skip_rate=cfg.getint("DataFrame", "skip_rate"),
        val_indices=cfg.getfloat("DataFrame", "val_indices"),
        observation_process=cfg.get("DataFrame", "observation_process"),
        overlap=cfg.getboolean("DataFrame", "overlap"),
        with_nan=cfg.getboolean("DataFrame", "with_nan", fallback=False),
        seq_len=None,
        device=device,
        dataset_label=cfg.get("DataFrame", "dataset_label", fallback=None),
        mask_label=cfg.get("DataFrame", "mask_label", fallback=None),
    )

    # Build the test dataloader once
    test_dataloader = build_dataloader(
        learning_algo.dataset_name, dataset_config, split="test", eval_mode=True
    )

    overlap = cfg["DataFrame"].getboolean("overlap")

    # Check if "alpha" exists in the config.ini under the [Network] section
    alphas_per_unit = None
    if learning_algo.optimize_alphas is not None:
        alphas_per_unit = dvae.alphas_per_unit()

    ############################################################################
    # Path to the directory where the model and loss_model.pckl are saved
    save_dir = os.path.dirname(params["saved_dict"])
    loss_file = os.path.join(save_dir, "loss_model.pckl")

    # Initialize metrics
    metrics: dict[str, Any] = {"params": params}
    metrics["total_params"] = int(dvae.parameter_count())
    # Add config to metrics
    metrics["config"] = config_dict

    # Check if the loss_model.pckl file exists
    if os.path.isfile(loss_file):
        print(f"[Eval] Loading loss data from {loss_file}")
        with open(loss_file, "rb") as f:
            # Load the data
            loaded_data = pickle.load(f)

    else:
        print(f"[Eval][Warning] No loss data file found at {loss_file}")

    ############################################################################

    with torch.no_grad():

        # Visualize results
        save_fig_dir = os.path.join(
            os.path.dirname(params["saved_dict"]), "post_training_figs"
        )
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)

        # Enforce paper-ready plotting style for all evaluation visualizations
        try:
            get_plot_config(paper_ready=True)
        except Exception:
            pass

        ############################################################################
        # For shorter sequence
        ############################################################################
        if learning_algo.dataset_name == "Lorenz63":
            new_seq_len = 1000
        else:
            new_seq_len = min(1000, len(test_dataloader.dataset.seq))
        print(f"[Eval] New sequence length: {new_seq_len}")
        # set random seed
        torch.manual_seed(42)
        test_dataloader.dataset.update_sequence_length(new_seq_len)
        # Prepare the long sequence data
        for i, batch_data_long in enumerate(test_dataloader):
            # batch_data_long = next(iter(test_dataloader))  # Single batch for demonstration
            batch_data_long = batch_data_long.to(device)
            # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
            batch_data_long = batch_data_long.permute(1, 0, 2)
            seq_len_long, batch_size_long, _ = batch_data_long.shape
            half_point_long = seq_len_long // 2

            # Extract missing mask for this batch
            missing_mask_long = None
            observation_process = getattr(dataset_config, "observation_process", None)

            if (
                _is_indicate_observation(observation_process)
                and batch_data_long.size(2) >= 2
            ):
                # For only_x_indicate, dimension 1 is the is_observed indicator (1.0=observed, 0.0=missing)
                # Derive missing_mask directly from this indicator to ensure perfect alignment
                is_observed = batch_data_long[:, :, 1]  # (seq_len, batch_size)
                # missing_mask = True where is_observed == 0.0
                missing_mask_long = (
                    (is_observed < 0.5).float().unsqueeze(-1)
                )  # (seq_len, batch_size, 1)

            elif hasattr(test_dataloader.dataset, "missing_mask"):
                # Fallback for other observation processes
                batch_start_idx = (
                    test_dataloader.dataset.data_idx[i]
                    if i < len(test_dataloader.dataset.data_idx)
                    else 0
                )
                batch_end_idx = min(
                    batch_start_idx + seq_len_long,
                    len(test_dataloader.dataset.missing_mask),
                )
                missing_mask_slice = test_dataloader.dataset.missing_mask[
                    batch_start_idx:batch_end_idx
                ]

                # Convert to tensor and expand to batch size
                if isinstance(missing_mask_slice, np.ndarray):
                    missing_mask_long = (
                        torch.from_numpy(missing_mask_slice)
                        .float()
                        .unsqueeze(1)
                        .expand(-1, batch_size_long, -1)
                    )
                else:
                    missing_mask_long = missing_mask_slice.unsqueeze(1).expand(
                        -1, batch_size_long, -1
                    )

            # Plot the spectral analysis
            autonomous_mode_selector_long = create_autonomous_mode_selector(
                seq_len_long,
                mode="half_half",
                batch_size=batch_size_long,
                x_dim=dataset_config.x_dim,
            )
            # === MASKING BASELINE: Force pure teacher-forcing on mask dimension during evaluation ===
            # For observation_process == "only_x_indicate", the mask channel must always be 100% TF
            # to keep evaluation consistent with training behavior.
            if _is_indicate_observation(
                getattr(dataset_config, "observation_process", None)
            ):
                autonomous_mode_selector_long = autonomous_mode_selector_long.clone()
                autonomous_mode_selector_long[:, :, 1] = (
                    0.0  # mask dimension always 100% TF
                )
            # ====================================================================================
            # turn input into tensor and send to GPU if needed
            batch_data_long_tensor = batch_data_long.clone().detach().to(device)
            recon_data_long = (
                dvae(
                    batch_data_long_tensor,
                    mode_selector=autonomous_mode_selector_long,
                    inference_mode=True,
                )
                .detach()
                .cpu()
                .numpy()
            )

            if test_dataloader.dataset.is_segmented_1d:
                x_data_long = batch_data_long[:, 0, :].reshape(-1)
                recon_x_data_long = recon_data_long[:, 0, :].reshape(-1)
            else:
                x_data_long = batch_data_long[:, 0, 0]
                recon_x_data_long = recon_data_long[:, 0, 0]

            # Plot the reconstruction vs true sequence
            visualize_teacherforcing_2_autonomous(
                batch_data_long[:, 1:, :],
                dvae,
                auto_mode_selector=autonomous_mode_selector_long[:, 1:, :],
                save_path=save_fig_dir,
                explain=f"final_short_inference_mode_half_half_short",
                inference_mode=True,
                missing_mask=(
                    missing_mask_long[1:, :, :]
                    if missing_mask_long is not None
                    else None
                ),
                is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
                hide_mask_output=_is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ),
            )

        # Visualize training mode short sequence with missingness
        print("[Eval] Visualizing training mode short sequence with missingness...")
        visualize_training_mode_short_sequence(
            dataset_name=learning_algo.dataset_name,
            dataset_config=dataset_config,
            device=device,
            dvae=dvae,
            save_fig_dir=save_fig_dir,
            seq_len=new_seq_len,
        )

        # Compute missing-value statistics (train & test)
        missingness_stats = compute_missing_stats(
            dataset_name=learning_algo.dataset_name,
            dataset_config=dataset_config,
        )
        if missingness_stats is not None:
            metrics["missingness_statistics"] = missingness_stats
            # Flattened summary keys for easy ingestion
            train_s = missingness_stats.get("train", {})
            test_s = missingness_stats.get("test", {})
            metrics["missingness_train_count"] = int(train_s.get("missing_count", 0))
            metrics["missingness_train_total_x"] = int(train_s.get("total_x", 0))
            metrics["missingness_train_pct"] = float(
                train_s.get("missing_percentage", 0.0)
            )
            metrics["missingness_test_count"] = int(test_s.get("missing_count", 0))
            metrics["missingness_test_total_x"] = int(test_s.get("total_x", 0))
            metrics["missingness_test_pct"] = float(
                test_s.get("missing_percentage", 0.0)
            )

        # visualize the hidden states
        visualize_variable_evolution(
            dvae.h,
            batch_data=batch_data_long,
            save_dir=save_fig_dir,
            variable_name=f"hidden",
            alphas=alphas_per_unit,
            add_lines_lst=[half_point_long],
            is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
        )

        # visualize the x_features
        visualize_variable_evolution(
            dvae.feature_x,
            batch_data=batch_data_long,
            save_dir=save_fig_dir,
            variable_name=f"x_features",
            add_lines_lst=[half_point_long],
            is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
        )

        ############################################################################
        # For longer sequence
        ############################################################################

        new_seq_len = min(20000, len(test_dataloader.dataset.seq))
        print(f"[Eval] Long-sequence eval: seq_len={new_seq_len}")
        test_dataloader.dataset.update_sequence_length(new_seq_len)
        n_long_windows = len(test_dataloader.dataset.data_idx)
        print(
            f"[Eval] Test set has {n_long_windows} window(s) at seq_len={new_seq_len} "
            f"(qualitative figures use window 0 only)"
        )
        sys.stdout.flush()
        # Prepare the long sequence data
        for i, batch_data_long in enumerate(test_dataloader):
            # batch_data_long = next(iter(test_dataloader))  # Single batch for demonstration
            batch_data_long = batch_data_long.to(device)
            # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
            batch_data_long = batch_data_long.permute(1, 0, 2)
            seq_len_long, batch_size_long, _ = batch_data_long.shape
            half_point_long = seq_len_long // 2
            start_frame = (
                test_dataloader.dataset.data_idx[i]
                if i < len(test_dataloader.dataset.data_idx)
                else 0
            )
            print(
                f"[Eval] Qualitative visualizations: window 0 "
                f"(start_frame={start_frame}, seq_len={seq_len_long})"
            )
            sys.stdout.flush()

            # Extract missing mask for this batch
            missing_mask_long = None
            observation_process = getattr(dataset_config, "observation_process", None)

            if (
                _is_indicate_observation(observation_process)
                and batch_data_long.size(2) >= 2
            ):
                # For only_x_indicate, dimension 1 is the is_observed indicator (1.0=observed, 0.0=missing)
                # Derive missing_mask directly from this indicator to ensure perfect alignment
                is_observed = batch_data_long[:, :, 1]  # (seq_len, batch_size)
                # missing_mask = True where is_observed == 0.0
                missing_mask_long = (
                    (is_observed < 0.5).float().unsqueeze(-1)
                )  # (seq_len, batch_size, 1)

            elif hasattr(test_dataloader.dataset, "missing_mask"):
                # Fallback for other observation processes
                batch_start_idx = start_frame
                batch_end_idx = min(
                    batch_start_idx + seq_len_long,
                    len(test_dataloader.dataset.missing_mask),
                )
                missing_mask_slice = test_dataloader.dataset.missing_mask[
                    batch_start_idx:batch_end_idx
                ]

                # Convert to tensor and expand to batch size
                if isinstance(missing_mask_slice, np.ndarray):
                    missing_mask_long = (
                        torch.from_numpy(missing_mask_slice)
                        .float()
                        .unsqueeze(1)
                        .expand(-1, batch_size_long, -1)
                    )
                else:
                    missing_mask_long = missing_mask_slice.unsqueeze(1).expand(
                        -1, batch_size_long, -1
                    )

            # Plot the spectral analysis
            autonomous_mode_selector_long = create_autonomous_mode_selector(
                seq_len_long,
                # mode="half_half",
                mode="even_bursts",
                autonomous_ratio=0.1,
                batch_size=batch_size_long,
                x_dim=dataset_config.x_dim,
            )
            # === MASKING BASELINE: Force pure teacher-forcing on mask dimension during evaluation ===
            # For observation_process == "only_x_indicate", the mask channel must always be 100% TF
            # to keep evaluation consistent with training behavior.
            if _is_indicate_observation(
                getattr(dataset_config, "observation_process", None)
            ):
                autonomous_mode_selector_long = autonomous_mode_selector_long.clone()
                autonomous_mode_selector_long[:, :, 1] = (
                    0.0  # mask dimension always 100% TF
                )
            # ====================================================================================
            # turn input into tensor and send to GPU if needed
            batch_data_long_tensor = batch_data_long.clone().detach().to(device)
            recon_data_long = (
                dvae(
                    batch_data_long_tensor,
                    mode_selector=autonomous_mode_selector_long,
                    inference_mode=True,
                )
                .detach()
                .cpu()
                .numpy()
            )

            if test_dataloader.dataset.is_segmented_1d:
                x_data_long = batch_data_long[:, 0, :].reshape(-1)
                recon_x_data_long = recon_data_long[:, 0, :].reshape(-1)
            else:
                x_data_long = batch_data_long[:, 0, 0]
                recon_x_data_long = recon_data_long[:, 0, 0]

            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                auto_mode_selector=autonomous_mode_selector_long,
                save_path=save_fig_dir,
                explain="final_long_inference_mode_even_burst",
                inference_mode=True,
                missing_mask=missing_mask_long,
                is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
                hide_mask_output=_is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ),
            )
            # Also visualize a half-half TF/Auto schedule on the same long sequence
            autonomous_mode_selector_half = create_autonomous_mode_selector(
                seq_len_long,
                mode="half_half",
                batch_size=batch_size_long,
                x_dim=dataset_config.x_dim,
            )
            if _is_indicate_observation(
                getattr(dataset_config, "observation_process", None)
            ):
                autonomous_mode_selector_half = autonomous_mode_selector_half.clone()
                autonomous_mode_selector_half[:, :, 1] = 0.0
            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                auto_mode_selector=autonomous_mode_selector_half,
                save_path=save_fig_dir,
                explain="final_long_inference_mode_half_half",
                inference_mode=True,
                missing_mask=missing_mask_long,
                is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
                hide_mask_output=_is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ),
            )
            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                auto_mode_selector=autonomous_mode_selector_long,
                save_path=save_fig_dir,
                explain="final_long_generative_mode",
                inference_mode=False,
                missing_mask=missing_mask_long,
                is_segmented_1d=test_dataloader.dataset.is_segmented_1d,
                hide_mask_output=_is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ),
            )

            # Hidden-state 3D embedding uses warmed half-half pass (batch 0 only)
            _, half_mode_selector = run_forward_with_mode(
                dvae,
                batch_data_long,
                mode="half_half",
                observation_process=observation_process,
            )
            teacher_forced_mask = ~half_mode_selector[:, 0, 0].bool()
            autonomous_mask = half_mode_selector[:, 0, 0].bool()
            teacherforced_states = dvae.h[teacher_forced_mask, 0, :]
            autonomous_states = dvae.h[autonomous_mask, 0, :]
            embedding_states_list = [teacherforced_states, autonomous_states]
            embedding_states_conditions = ["teacher-forced", "autonomous"]
            embedding_states_colors = ["Greens", "Reds"]

            if params.get("save_3d", True):
                hidden_3d_gif_dir = os.path.join(save_fig_dir, "3d_hidden_gifs")
                if not os.path.exists(hidden_3d_gif_dir):
                    os.makedirs(hidden_3d_gif_dir)
                vis_embedding_space_params = [
                    {
                        "states_list": embedding_states_list,
                        "save_dir": hidden_3d_gif_dir,
                        "variable_name": f"hidden",
                        "condition_names": embedding_states_conditions,
                        "base_colors": embedding_states_colors,
                        "technique": "kernel_pca",
                    },
                    {
                        "states_list": embedding_states_list,
                        "save_dir": hidden_3d_gif_dir,
                        "variable_name": f"hidden",
                        "condition_names": embedding_states_conditions,
                        "base_colors": embedding_states_colors,
                        "technique": "ica",
                    },
                    {
                        "states_list": embedding_states_list,
                        "save_dir": hidden_3d_gif_dir,
                        "variable_name": f"hidden",
                        "condition_names": embedding_states_conditions,
                        "base_colors": embedding_states_colors,
                        "technique": "tsne",
                    },
                ]
                print("[Eval] [CHECKPOINT] Starting parallel visualizations...")
                sys.stdout.flush()
                run_parallel_visualizations(
                    visualize_embedding_space, vis_embedding_space_params
                )
                print("[Eval] [CHECKPOINT] Parallel visualizations completed")
                sys.stdout.flush()
            else:
                print("[Eval] Skipping 3D visualizations (save_3d=False)")
                sys.stdout.flush()

            break  # Visualizations only on first long-sequence batch

        ############################################################################
        # Quantitative metrics (multi-batch, modular mode selectors)
        ############################################################################
        auto_eval_mode = params.get("auto_eval_mode", "half_half")
        auto_eval_flip_point = params.get("auto_eval_flip_point")
        auto_eval_block_len = int(params.get("auto_eval_block_len", 1000) or 1000)
        auto_eval_ratio = params.get("auto_eval_ratio", 0.1)
        max_eval_batches = params.get("max_eval_batches", 20)
        skip_metrics_viz_after_batch_0 = params.get(
            "skip_metrics_viz_after_batch_0", True
        )
        observation_process = getattr(dataset_config, "observation_process", None)

        # Metrics window length depends on how free-run is driven (not the
        # qualitative long-seq length above). half_half: one TF block + one Auto
        # block of length block_len each -> T=2*block_len, many batches.
        data_len = len(test_dataloader.dataset.seq)
        metrics_seq_len_override = params.get("metrics_seq_len")
        if metrics_seq_len_override is not None:
            metrics_seq_len = int(metrics_seq_len_override)
        elif auto_eval_mode in ("half_half", "flip_at_index"):
            # TF warm-up of block_len, then free-run of block_len (Composer path).
            metrics_seq_len = 2 * auto_eval_block_len
            if auto_eval_mode == "flip_at_index" and auto_eval_flip_point is not None:
                # Keep enough room for flip + at least one block of free-run.
                metrics_seq_len = max(
                    metrics_seq_len, int(auto_eval_flip_point) + auto_eval_block_len
                )
        elif auto_eval_mode == "alternating_blocks":
            # Several fixed TF/Auto cycles on one stream (paper-style multi-block).
            metrics_seq_len = min(max(10 * auto_eval_block_len, 2 * auto_eval_block_len), data_len)
        else:
            # even_bursts / all_1 / stress tests: keep a long window.
            metrics_seq_len = min(20000, data_len)

        metrics_seq_len = max(2, min(int(metrics_seq_len), data_len))
        # batch_size=1 so each dataloader step is one independent window
        # (training batch_size would pack many windows into one "batch" and
        # max_eval_batches would not match multi-start half_half trials).
        metrics_dataset_config = replace(
            dataset_config, batch_size=1, shuffle=False, num_workers=0
        )
        metrics_dataloader = build_dataloader(
            learning_algo.dataset_name,
            metrics_dataset_config,
            split="test",
            eval_mode=True,
        )
        metrics_dataloader.dataset.update_sequence_length(metrics_seq_len)
        new_seq_len = metrics_seq_len
        print(
            f"[Eval] Metrics seq_len={metrics_seq_len} "
            f"(mode={auto_eval_mode}, block_len={auto_eval_block_len}, batch_size=1)"
        )

        n_test_windows = len(metrics_dataloader.dataset.data_idx)
        n_eval_windows = min(n_test_windows, max_eval_batches)
        batch_metric_dicts = []
        batch_visual_records = []
        mse_results_list = []
        geom_results_list = []
        spectrum_results_list = []
        metrics_save_fig_dir = os.path.join(save_fig_dir, "metrics_batches")

        print(
            f"[Eval] Quantitative metrics: up to {n_eval_windows} window(s) "
            f"(test_set={n_test_windows}, max_eval_batches={max_eval_batches}, "
            f"seq_len={new_seq_len})"
        )
        extra_mode_info = ""
        if auto_eval_mode == "alternating_blocks":
            extra_mode_info = f" (block_len={auto_eval_block_len})"
        elif auto_eval_mode == "even_bursts":
            extra_mode_info = f" (ratio={auto_eval_ratio})"
        elif auto_eval_flip_point is not None:
            extra_mode_info = f" (flip_point={auto_eval_flip_point})"
        print(
            f"[Eval] Forward passes per window: TF=all_0, Auto={auto_eval_mode}"
            f"{extra_mode_info}"
        )
        if skip_metrics_viz_after_batch_0:
            print(
                "[Eval] Metric figures: window 0 only "
                "(set --skip-metrics-viz-after-batch-0 false to save all windows)"
            )
        else:
            print("[Eval] Metric figures: enabled for every evaluated window")
        sys.stdout.flush()

        for i, batch_data_long in enumerate(metrics_dataloader):
            if i >= max_eval_batches:
                break

            batch_data_long = batch_data_long.to(device).permute(1, 0, 2)
            seq_len_long = batch_data_long.shape[0]
            start_frame = (
                metrics_dataloader.dataset.data_idx[i]
                if i < len(metrics_dataloader.dataset.data_idx)
                else 0
            )
            flip_point = get_flip_point_for_mode(
                seq_len_long,
                auto_eval_mode,
                flip_point=auto_eval_flip_point,
                block_len=auto_eval_block_len,
            )
            save_metric_figures = (i == 0) or (not skip_metrics_viz_after_batch_0)

            print(
                f"[Eval] Evaluating window {i + 1}/{n_eval_windows} "
                f"(start_frame={start_frame}, seq_len={seq_len_long}, "
                f"flip_point={flip_point}, metric_figures="
                f"{'yes' if save_metric_figures else 'no'})"
            )
            sys.stdout.flush()

            recon_tf, _ = run_forward_with_mode(
                dvae,
                batch_data_long,
                mode="all_0",
                observation_process=observation_process,
            )
            recon_auto_warmed, mode_selector_auto = run_forward_with_mode(
                dvae,
                batch_data_long,
                mode=auto_eval_mode,
                observation_process=observation_process,
                flip_point=auto_eval_flip_point,
                block_len=auto_eval_block_len,
                autonomous_ratio=auto_eval_ratio,
            )

            channel_benchmarks = get_channel_benchmarks(
                batch_data_long=batch_data_long,
                recon_tf=recon_tf.detach().cpu().numpy(),
                recon_auto_warmed=recon_auto_warmed.detach().cpu().numpy(),
                flip_point=flip_point,
                dataset=metrics_dataloader.dataset,
                batch_idx=i,
                observation_process=observation_process or "",
                dataset_name=learning_algo.dataset_name,
                auto_mode=auto_eval_mode,
                mode_selector=mode_selector_auto,
                block_len=auto_eval_block_len,
                autonomous_ratio=auto_eval_ratio,
            )
            print(
                f"[Eval] Auto free-run steps={channel_benchmarks.get('n_auto_steps')} "
                f"blocks={channel_benchmarks.get('n_auto_blocks')} "
                f"ranges={channel_benchmarks.get('auto_block_ranges')}"
            )

            batch_fig_dir = os.path.join(metrics_save_fig_dir, f"batch_{i}")
            if save_metric_figures:
                os.makedirs(batch_fig_dir, exist_ok=True)

                # Series plot of the *same* schedule/window that feeds geometry delay
                # embeds (TF=green, Auto=red). Filename encodes mode + length.
                drive_explain = (
                    f"metrics_drive_batch{i}"
                    f"_mode_{auto_eval_mode}"
                    f"_T{seq_len_long}"
                )
                if auto_eval_mode == "alternating_blocks":
                    drive_explain += f"_block{auto_eval_block_len}"
                elif auto_eval_mode == "even_bursts":
                    drive_explain += f"_ratio{auto_eval_ratio}"
                elif auto_eval_flip_point is not None:
                    drive_explain += f"_flip{auto_eval_flip_point}"

                # Sidecar so the schedule is readable without parsing the filename
                schedule_meta = {
                    "batch_idx": i,
                    "start_frame": start_frame,
                    "seq_len": seq_len_long,
                    "auto_eval_mode": auto_eval_mode,
                    "auto_eval_block_len": auto_eval_block_len,
                    "auto_eval_ratio": auto_eval_ratio,
                    "auto_eval_flip_point": auto_eval_flip_point,
                    "flip_point_summary": flip_point,
                    "n_auto_steps": channel_benchmarks.get("n_auto_steps"),
                    "n_auto_blocks": channel_benchmarks.get("n_auto_blocks"),
                    "auto_block_ranges": channel_benchmarks.get("auto_block_ranges"),
                    "note": (
                        "vis_pred_true_series_metrics_drive_*.png shows the forward "
                        "pass whose Auto segments are mask-gathered for delay-embed / "
                        "spectrum / MSE auto metrics in this batch folder."
                    ),
                }
                schedule_path = os.path.join(
                    batch_fig_dir, f"metrics_drive_schedule_batch{i}.yaml"
                )
                with open(schedule_path, "w") as sf:
                    yaml.dump(schedule_meta, sf, default_flow_style=False)
                print(f"[Eval] Wrote metrics drive schedule: {schedule_path}")

                # missing mask for shading (optional)
                missing_mask_metrics = None
                if hasattr(metrics_dataloader.dataset, "missing_mask"):
                    try:
                        mm = metrics_dataloader.dataset.get_missing_mask(i)
                        mm = np.asarray(mm)
                        if mm.ndim == 1:
                            missing_mask_metrics = (
                                torch.from_numpy(mm[:seq_len_long].astype(np.float32))
                                .view(seq_len_long, 1, 1)
                                .expand(-1, 1, batch_data_long.shape[2])
                            )
                        elif mm.ndim == 2:
                            missing_mask_metrics = (
                                torch.from_numpy(mm[:seq_len_long].astype(np.float32))
                                .unsqueeze(1)
                            )
                    except Exception as exc:
                        print(f"[Eval] metrics missing_mask skip: {exc}")

                visualize_teacherforcing_2_autonomous(
                    batch_data_long,
                    dvae,
                    auto_mode_selector=mode_selector_auto,
                    save_path=batch_fig_dir,
                    explain=drive_explain,
                    inference_mode=True,
                    missing_mask=missing_mask_metrics,
                    is_segmented_1d=getattr(
                        metrics_dataloader.dataset, "is_segmented_1d", False
                    ),
                    hide_mask_output=_is_indicate_observation(observation_process),
                )
                print(
                    f"[Eval] Metrics-drive series plot saved under {batch_fig_dir} "
                    f"(explain={drive_explain})"
                )

            spectrum_results = run_spectrum_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=None,
                save_fig_dir=batch_fig_dir if save_metric_figures else None,
                i=i,
                autonomous_mode_selector_long=None,
                dataset_name=learning_algo.dataset_name,
                model_name=learning_algo.model_name,
                cfg=cfg,
                dvae_model=dvae,
                batch_data_long=batch_data_long,
                channel_benchmarks=channel_benchmarks,
                save_figures=save_metric_figures,
            )
            mse_results = run_mse_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=None,
                save_fig_dir=batch_fig_dir if save_metric_figures else None,
                i=i,
                autonomous_mode_selector_long=None,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
                channel_benchmarks=channel_benchmarks,
                save_figures=save_metric_figures,
            )
            geom_results = run_geometry_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=None,
                save_fig_dir=batch_fig_dir if save_metric_figures else None,
                i=i,
                autonomous_mode_selector_long=None,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
                channel_benchmarks=channel_benchmarks,
                save_figures=save_metric_figures,
            )

            batch_metric_dicts.append(
                flatten_analysis_to_batch_metrics(
                    mse_results, geom_results, spectrum_results
                )
            )
            mse_results_list.append(mse_results)
            geom_results_list.append(geom_results)
            spectrum_results_list.append(spectrum_results)

            # Store for batch_all visuals (all windows, not only batch_0 figures)
            recon_mixed_np = recon_auto_warmed.detach().cpu().numpy()
            if recon_mixed_np.ndim == 3:
                recon_mixed_1d = recon_mixed_np[:, 0, 0]
            else:
                recon_mixed_1d = recon_mixed_np.reshape(-1)
            mode_1d = mode_selector_to_1d(mode_selector_auto)
            rec = collect_batch_visual_record(
                channel_benchmarks=channel_benchmarks,
                recon_mixed_full=recon_mixed_1d,
                mode_selector_1d=mode_1d,
                batch_idx=i,
                start_frame=int(start_frame) if start_frame is not None else i,
            )
            if rec is not None:
                batch_visual_records.append(rec)

        if batch_metric_dicts:
            merged = merge_batch_metric_dicts(batch_metric_dicts)
            metrics["eval_schema_version"] = 2
            metrics["auto_eval_mode"] = auto_eval_mode
            metrics["auto_eval_block_len"] = auto_eval_block_len
            metrics["auto_eval_ratio"] = auto_eval_ratio
            metrics["metrics_seq_len"] = new_seq_len
            metrics.update(merged)

            metrics["mse_tf"] = merged.get("mse_tf", merged.get("mse_tf_mean"))
            metrics["mse_auto"] = merged.get("mse_auto", merged.get("mse_auto_mean"))
            # Per-window mean KLD kept for debugging; primary kld_* overwritten by stitched below
            metrics["kld_tf_mean_across_windows"] = merged.get(
                "kld_tf", merged.get("kld_tf_mean")
            )
            metrics["kld_auto_mean_across_windows"] = merged.get(
                "kld_auto", merged.get("kld_auto_mean")
            )
            metrics["spectrum_error_tf"] = merged.get(
                "spectrum_error_tf", merged.get("spectrum_error_tf_mean")
            )
            metrics["spectrum_error_auto"] = merged.get(
                "spectrum_error_auto", merged.get("spectrum_error_auto_mean")
            )
            metrics["spectrum_error_gt"] = 0.0
            metrics["spectrum_distance"] = {
                "tf": metrics["spectrum_error_tf"],
                "auto": metrics["spectrum_error_auto"],
                "gt": 0.0,
            }

            n_scored = merged.get("n_eval_batches", len(batch_metric_dicts))
            print(
                f"[Eval] Aggregated metrics over {n_scored} window(s) "
                f"(mean ± std across windows in YAML)"
            )
            print(f"[Eval] MSE Teacher-forced (mean): {metrics['mse_tf']:.6f}")
            print(f"[Eval] MSE Autonomous (mean): {metrics['mse_auto']:.6f}")

            # --- batch_all: stitched visuals across all scored windows ---
            batch_all_dir = os.path.join(metrics_save_fig_dir, "batch_all")
            print(
                f"[Eval] Rendering batch_all visuals ({len(batch_visual_records)} windows) "
                f"-> {batch_all_dir}"
            )
            sys.stdout.flush()
            render_batch_all_visuals(
                batch_visual_records,
                save_dir=batch_all_dir,
                gap=1,
                explain_suffix=f"mode_{auto_eval_mode}_n{len(batch_visual_records)}",
            )

            # Stitched-cloud KLD (bar height) + per-window median/IQR (spread)
            kld_metrics = compute_stitched_kld_metrics(
                batch_visual_records, geom_results_list=geom_results_list
            )
            # Primary keys for aggregation: stitched geometry KLD
            metrics["kld_tf"] = kld_metrics.get("kld_tf_stitched")
            metrics["kld_auto"] = kld_metrics.get("kld_auto_stitched")
            metrics["kld_tf_stitched"] = kld_metrics.get("kld_tf_stitched")
            metrics["kld_auto_stitched"] = kld_metrics.get("kld_auto_stitched")
            metrics["kld_tf_median"] = kld_metrics.get("kld_tf_median")
            metrics["kld_auto_median"] = kld_metrics.get("kld_auto_median")
            metrics["kld_tf_iqr"] = kld_metrics.get("kld_tf_iqr")
            metrics["kld_auto_iqr"] = kld_metrics.get("kld_auto_iqr")
            metrics["kld_tf_q25"] = kld_metrics.get("kld_tf_q25")
            metrics["kld_tf_q75"] = kld_metrics.get("kld_tf_q75")
            metrics["kld_auto_q25"] = kld_metrics.get("kld_auto_q25")
            metrics["kld_auto_q75"] = kld_metrics.get("kld_auto_q75")
            metrics["kld_tf_std_across_windows"] = kld_metrics.get(
                "kld_tf_std_across_windows"
            )
            metrics["kld_auto_std_across_windows"] = kld_metrics.get(
                "kld_auto_std_across_windows"
            )
            # Nested block for aggregators / future analysis
            metrics["kld_geometry"] = {
                "stitched_tf": kld_metrics.get("kld_tf_stitched"),
                "stitched_auto": kld_metrics.get("kld_auto_stitched"),
                "per_window_tf": kld_metrics.get("kld_tf_per_window"),
                "per_window_auto": kld_metrics.get("kld_auto_per_window"),
                "note": kld_metrics.get("kld_metric_note"),
            }
            print(
                f"[Eval] KL stitched (batch_all geometry): "
                f"TF={metrics['kld_tf']:.4f}  Auto={metrics['kld_auto']:.4f}"
            )
            print(
                f"[Eval] KL per-window median [IQR]: "
                f"TF={metrics['kld_tf_median']:.4f} "
                f"[{metrics['kld_tf_q25']:.4f}, {metrics['kld_tf_q75']:.4f}]  "
                f"Auto={metrics['kld_auto_median']:.4f} "
                f"[{metrics['kld_auto_q25']:.4f}, {metrics['kld_auto_q75']:.4f}]"
            )

            # Physical multi-channel baselines (Lorenz y/z; XHRO ch1–ch3 vs target)
            # Old paper bars: refs | GT(0) | TF | Auto
            reference_errors = None
            if (
                hasattr(metrics_dataloader.dataset, "get_full_xyz")
                and batch_visual_records
            ):
                try:
                    auto_len = len(batch_visual_records[0]["gt_auto"])
                    td = int(batch_visual_records[0].get("time_delay", 10))
                    dd = int(batch_visual_records[0].get("delay_dims", 3))
                    full_len = len(batch_visual_records[0]["gt_full"])
                    flip = full_len // 2 if auto_eval_mode == "half_half" else None
                    if (
                        auto_eval_mode == "flip_at_index"
                        and auto_eval_flip_point is not None
                    ):
                        flip = int(auto_eval_flip_point)
                    batch_ids = [r["batch_idx"] for r in batch_visual_records]
                    primary_key = batch_visual_records[0].get("key")
                    reference_errors = compute_reference_channel_errors(
                        metrics_dataloader.dataset,
                        batch_indices=batch_ids,
                        auto_seg_len=auto_len,
                        dataset_name=learning_algo.dataset_name,
                        observation_process=observation_process,
                        primary_key=primary_key,
                        time_delay=td,
                        delay_dims=dd,
                        dt=float(channel_benchmarks.get("dt", 0.01)),
                        flip_point=flip,
                    )
                    if reference_errors:
                        print(
                            f"[Eval] Multi-channel reference bars "
                            f"({learning_algo.dataset_name}): {reference_errors}"
                        )
                    else:
                        print(
                            f"[Eval] Multi-channel reference bars "
                            f"({learning_algo.dataset_name}): empty (skipped)"
                        )
                except Exception as exc:
                    print(f"[Eval] batch_all reference channels skipped: {exc}")
                    reference_errors = None

            render_summary_error_bars(
                mse_results_list,
                geom_results_list,
                spectrum_results_list,
                save_dir=batch_all_dir,
                reference_errors=reference_errors,
                kld_metrics=kld_metrics,
            )

            log_file = os.path.join(save_dir, "evaluation_log.txt")
            with open(log_file, "w") as f:
                f.write(f"KL Teacher-forced (stitched): {metrics['kld_tf']:.4f}\n")
                f.write(f"KL Autonomous (stitched): {metrics['kld_auto']:.4f}\n")
                f.write(
                    f"KL TF median/IQR: {metrics['kld_tf_median']:.4f} "
                    f"[{metrics['kld_tf_q25']:.4f}, {metrics['kld_tf_q75']:.4f}]\n"
                )
                f.write(
                    f"KL Auto median/IQR: {metrics['kld_auto_median']:.4f} "
                    f"[{metrics['kld_auto_q25']:.4f}, {metrics['kld_auto_q75']:.4f}]\n"
                )
                f.write(f"MSE Teacher-forced: {metrics['mse_tf']:.6f}\n")
                f.write(f"MSE Autonomous: {metrics['mse_auto']:.6f}\n")
                f.write(f"Auto eval mode: {auto_eval_mode}\n")
                f.write(f"Auto eval block_len: {auto_eval_block_len}\n")
                f.write(f"Auto eval ratio: {auto_eval_ratio}\n")
                f.write(f"Eval batches: {len(batch_metric_dicts)}\n")

        ##############################################################################
        # ============================================================
        # 2. Local drift statistics (NEW - short controlled segments)
        # ============================================================
        # This section validates the implicit regularizer theory by measuring
        # how quickly the autonomous trajectory drifts from the teacher-forced (TF)
        # trajectory in the *early phase* (20-40 steps) after a fork point.
        #
        # Intuition:
        #   - Small drift + negative cross-term → Auto is self-correcting (good).
        #   - Large drift or positive cross-term → Auto amplifies errors (bad).
        #   - We compute ΔMSE = ||d||^2 + 2*(d^T e), which directly measures
        #     the change in MSE when switching from TF to Auto.
        #
        print("[Eval] Starting local drift statistics computation...")
        sys.stdout.flush()
        new_seq_len_short = 60  # Short sequences to stay in linear regime
        test_dataloader.dataset.update_sequence_length(new_seq_len_short)
        n_drift_windows = len(test_dataloader.dataset.data_idx)
        print(
            f"[Eval] Local drift: seq_len={new_seq_len_short}, "
            f"flip_point=30, {n_drift_windows} window(s)"
        )
        sys.stdout.flush()

        drift_stats_list = []
        drift_per_step_d_norm_list = []
        drift_per_step_cross_list = []
        drift_per_step_delta_mse_list = []

        for batch_idx, batch_data in enumerate(test_dataloader):
            drift_start = (
                test_dataloader.dataset.data_idx[batch_idx]
                if batch_idx < len(test_dataloader.dataset.data_idx)
                else 0
            )
            print(
                f"[Eval] Local drift window {batch_idx + 1}/{n_drift_windows} "
                f"(start_frame={drift_start}, seq_len={new_seq_len_short})"
            )
            sys.stdout.flush()
            batch_data = batch_data.to(device).permute(
                1, 0, 2
            )  # (seq_len, batch_size, x_dim)

            # Choose where to flip from TF to Auto
            # Typically around 50-60% through the short sequence
            flip_point = 30
            auto_len = new_seq_len_short - flip_point

            # Compute drift statistics for this batch
            stats = compute_local_drift_statistics(
                dvae=dvae,
                batch_data=batch_data,
                flip_point=flip_point,
                auto_len=auto_len,
                device=device,
                observation_process=getattr(
                    dataset_config, "observation_process", None
                ),
            )
            print(f"[Eval] [CHECKPOINT] Drift batch {batch_idx} computed")
            sys.stdout.flush()

            drift_stats_list.append(stats)
            # Average per-step values across samples so each batch contributes
            # a (auto_len,) vector. This avoids shape mismatches when the
            # last batch has a smaller batch_size.
            drift_per_step_d_norm_list.append(np.mean(stats["per_step_d_norm"], axis=1))
            drift_per_step_cross_list.append(np.mean(stats["per_step_cross"], axis=1))
            drift_per_step_delta_mse_list.append(
                np.mean(stats["per_step_delta_mse"], axis=1)
            )

            # Limit to first 50 batches for computational efficiency
            if batch_idx >= 49:
                break

        print("[Eval] [CHECKPOINT] Local drift statistics computation completed")
        sys.stdout.flush()

        # Aggregate drift statistics across all batches
        # Each stat in drift_stats_list is a dict with 'd_norm', 'cross_term', 'delta_mse'
        all_d_norm = np.array([s["d_norm"] for s in drift_stats_list])
        all_cross = np.array([s["cross_term"] for s in drift_stats_list])
        all_delta_mse = np.array([s["delta_mse"] for s in drift_stats_list])

        # Compute mean and std across batches
        metrics["local_drift_avg_d_norm"] = float(np.mean(all_d_norm))
        metrics["local_drift_avg_d_norm_std"] = float(np.std(all_d_norm))
        metrics["local_drift_avg_cross_term"] = float(np.mean(all_cross))
        metrics["local_drift_avg_cross_term_std"] = float(np.std(all_cross))
        metrics["local_drift_avg_delta_mse"] = float(np.mean(all_delta_mse))
        metrics["local_drift_avg_delta_mse_std"] = float(np.std(all_delta_mse))

        print(
            f"[Eval] Local Drift - Mean ||d||²: {metrics['local_drift_avg_d_norm']:.6f}"
        )
        print(
            f"[Eval] Local Drift - Mean d^T e: {metrics['local_drift_avg_cross_term']:.6f}"
        )
        print(
            f"[Eval] Local Drift - Mean ΔMSE: {metrics['local_drift_avg_delta_mse']:.6f}"
        )

        # Optionally save per-step drift growth for visualization
        # Stack all per-step arrays: (n_batches, auto_len)
        all_per_step_d_norm = np.stack(drift_per_step_d_norm_list, axis=0)
        all_per_step_cross = np.stack(drift_per_step_cross_list, axis=0)
        all_per_step_delta_mse = np.stack(drift_per_step_delta_mse_list, axis=0)

        # Compute mean and std across batches at each step
        mean_d_norm_over_time = np.mean(all_per_step_d_norm, axis=0)  # (auto_len,)
        std_d_norm_over_time = np.std(all_per_step_d_norm, axis=0)
        mean_cross_over_time = np.mean(all_per_step_cross, axis=0)
        std_cross_over_time = np.std(all_per_step_cross, axis=0)
        mean_delta_mse_over_time = np.mean(all_per_step_delta_mse, axis=0)

        # Store time series for later visualization/analysis
        metrics["local_drift_per_step_d_norm_mean"] = mean_d_norm_over_time.tolist()
        metrics["local_drift_per_step_d_norm_std"] = std_d_norm_over_time.tolist()
        metrics["local_drift_per_step_cross_mean"] = mean_cross_over_time.tolist()
        metrics["local_drift_per_step_cross_std"] = std_cross_over_time.tolist()
        metrics["local_drift_per_step_delta_mse_mean"] = (
            mean_delta_mse_over_time.tolist()
        )
        metrics["local_drift_flip_point"] = int(flip_point)
        metrics["local_drift_auto_len"] = int(auto_len)
        metrics["local_drift_n_batches"] = len(drift_stats_list)

        print(f"[Eval] Local Drift computed from {len(drift_stats_list)} batches")

    # Wait for parallel visualizations to complete (optional - can be removed if not needed)
    # The YAML is already saved above, so we can exit early if desired

    print("[Eval] [CHECKPOINT] Saving final evaluation summary YAML...")
    sys.stdout.flush()
    # Save the metrics as YAML (human-readable JSON-like format)
    metrics_file = os.path.join(save_dir, "evaluation_summary.yaml")
    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"[Eval] Metrics saved to: {metrics_file}")
    print("[Eval] [CHECKPOINT] Evaluation completed successfully!")
    sys.stdout.flush()

    # Save the metrics as YAML (human-readable JSON-like format)
    metrics_file = os.path.join(save_dir, "evaluation_summary.yaml")
    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"[Eval] Metrics saved to: {metrics_file}")
