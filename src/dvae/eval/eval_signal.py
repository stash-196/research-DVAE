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
)

from torch.nn.functional import mse_loss
import plotly.graph_objects as go
import plotly.express as px
import pickle
import configparser
from typing import Any, Optional
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
        print(f"[Eval] New sequence length: {new_seq_len}")
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
                hide_mask_output=_is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ),
            )

            # Run spectrum analysis and visualization
            print("[Eval] [CHECKPOINT] Starting spectrum analysis...")
            sys.stdout.flush()
            spectrum_results = run_spectrum_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=recon_data_long,
                save_fig_dir=save_fig_dir,
                i=i,
                autonomous_mode_selector_long=autonomous_mode_selector_long,
                dataset_name=learning_algo.dataset_name,
                model_name=learning_algo.model_name,
                cfg=cfg,
                dvae_model=dvae,
            )
            print("[Eval] [CHECKPOINT] Spectrum analysis completed")
            sys.stdout.flush()

            # Add spectrum errors to metrics
            for key, error in zip(
                spectrum_results["signal_keys"],
                spectrum_results["power_spectrum_errors"],
            ):
                metrics[f"spectrum_error_{key}"] = float(error)

            print("[Eval] [CHECKPOINT] Starting MSE analysis...")
            sys.stdout.flush()
            mse_results = run_mse_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=recon_data_long,
                save_fig_dir=save_fig_dir,
                i=i,
                autonomous_mode_selector_long=autonomous_mode_selector_long,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
            )
            print("[Eval] [CHECKPOINT] MSE analysis completed")
            sys.stdout.flush()

            for key, error in zip(
                mse_results["signal_keys"], mse_results["mse_errors"]
            ):
                metrics[f"mse_{key}"] = float(error)

            print("[Eval] [CHECKPOINT] Starting geometry analysis...")
            sys.stdout.flush()
            geom_results = run_geometry_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=recon_data_long,
                save_fig_dir=save_fig_dir,
                i=i,
                autonomous_mode_selector_long=autonomous_mode_selector_long,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
            )
            print("[Eval] [CHECKPOINT] Geometry analysis completed")
            sys.stdout.flush()

            for key, error in zip(
                geom_results["signal_keys"], geom_results["kld_scores"]
            ):
                metrics[f"kld_{key}"] = float(error)

            # Extract TF and Auto metrics from the analysis results
            tf_index = geom_results["signal_keys"].index("tf")
            auto_index = geom_results["signal_keys"].index("auto")
            mse_tf = mse_results["mse_errors"][tf_index]
            mse_auto = mse_results["mse_errors"][auto_index]
            kld_tf = geom_results["kld_scores"][tf_index]
            kld_auto = geom_results["kld_scores"][auto_index]

            print(f"[Eval] MSE Teacher-forced: {mse_tf:.6f}")
            print(f"[Eval] MSE Autonomous: {mse_auto:.6f}")
            print(f"[Eval] KL Teacher-forced: {kld_tf:.4f}")
            print(f"[Eval] KL Autonomous: {kld_auto:.4f}")

            # Add to metrics
            metrics["mse_tf"] = float(mse_tf)
            metrics["mse_auto"] = float(mse_auto)
            metrics["kld_tf"] = float(kld_tf)
            metrics["kld_auto"] = float(kld_auto)

            # Add spectrum distance metrics
            if spectrum_results:
                spectrum_errors = spectrum_results["power_spectrum_errors"]
                signal_names = spectrum_results["signal_keys"]
                metrics["spectrum_distance"] = {
                    name.replace("\n", " "): float(error)
                    for name, error in zip(signal_names, spectrum_errors)
                }

            # Logging
            # Log the metrics to a file in the same directory as the .pt file
            log_file = os.path.join(save_dir, "evaluation_log.txt")
            with open(log_file, "w") as f:
                f.write(f"KL Teacher-forced: {kld_tf:.4f}\n")
                f.write(f"KL Autonomous: {kld_auto:.4f}\n")
                f.write(f"MSE Teacher-forced: {mse_tf:.6f}\n")
                f.write(f"MSE Autonomous: {mse_auto:.6f}\n")

            teacher_forced_mask = ~autonomous_mode_selector_long[:, 0, 0].bool()
            autonomous_mask = autonomous_mode_selector_long[:, 0, 0].bool()
            teacherforced_states = dvae.h[teacher_forced_mask, 0, :]
            autonomous_states = dvae.h[autonomous_mask, 0, :]
            embedding_states_list = [teacherforced_states, autonomous_states]
            embedding_states_conditions = ["teacher-forced", "autonomous"]
            embedding_states_colors = ["Greens", "Reds"]

            # visualize the hidden states 3d (created later if enabled)

            # Save the metrics as YAML before starting parallel visualizations
            metrics_file = os.path.join(save_dir, "evaluation_summary.yaml")
            with open(metrics_file, "w") as f:
                yaml.dump(metrics, f, default_flow_style=False)
            print(f"[Eval] Metrics saved to: {metrics_file}")
            sys.stdout.flush()
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

            # break after the first batch
            break

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
        print("[Eval] [CHECKPOINT] Starting local drift statistics computation...")
        sys.stdout.flush()
        new_seq_len_short = 60  # Short sequences to stay in linear regime
        test_dataloader.dataset.update_sequence_length(new_seq_len_short)

        drift_stats_list = []
        drift_per_step_d_norm_list = []
        drift_per_step_cross_list = []
        drift_per_step_delta_mse_list = []

        for batch_idx, batch_data in enumerate(test_dataloader):
            print(f"[Eval] [CHECKPOINT] Processing drift batch {batch_idx}...")
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
