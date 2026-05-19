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
from typing import Any
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

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        if params["cfg"] is None:
            params["cfg"] = os.path.join(
                os.path.dirname(params["saved_dict"]), "config.ini"
            )
        return params


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
            # Plot the spectral analysis
            autonomous_mode_selector_long = create_autonomous_mode_selector(
                seq_len_long,
                mode="half_half",
                batch_size=batch_size_long,
                x_dim=dataset_config.x_dim,
            )
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
            )

        # visualize the hidden states
        visualize_variable_evolution(
            dvae.h,
            batch_data=batch_data_long,
            save_dir=save_fig_dir,
            variable_name=f"hidden",
            alphas=alphas_per_unit,
            add_lines_lst=[half_point_long],
        )

        # visualize the x_features
        visualize_variable_evolution(
            dvae.feature_x,
            batch_data=batch_data_long,
            save_dir=save_fig_dir,
            variable_name=f"x_features",
            add_lines_lst=[half_point_long],
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
            # Plot the spectral analysis
            autonomous_mode_selector_long = create_autonomous_mode_selector(
                seq_len_long,
                # mode="half_half",
                mode="even_bursts",
                autonomous_ratio=0.1,
                batch_size=batch_size_long,
                x_dim=dataset_config.x_dim,
            )
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
            )
            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                auto_mode_selector=autonomous_mode_selector_long,
                save_path=save_fig_dir,
                explain="final_long_generative_mode",
                inference_mode=False,
            )

            # Run spectrum analysis and visualization
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

            # Add spectrum errors to metrics
            for key, error in zip(
                spectrum_results["signal_keys"],
                spectrum_results["power_spectrum_errors"],
            ):
                metrics[f"spectrum_error_{key}"] = float(error)

            mse_results = run_mse_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=recon_data_long,
                save_fig_dir=save_fig_dir,
                i=i,
                autonomous_mode_selector_long=autonomous_mode_selector_long,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
            )
            for key, error in zip(
                mse_results["signal_keys"], mse_results["mse_errors"]
            ):
                metrics[f"mse_{key}"] = float(error)

            geom_results = run_geometry_analysis(
                test_dataloader=test_dataloader,
                recon_data_long=recon_data_long,
                save_fig_dir=save_fig_dir,
                i=i,
                autonomous_mode_selector_long=autonomous_mode_selector_long,
                dataset_name=learning_algo.dataset_name,
                batch_data_long=batch_data_long,
            )
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

            # visualize the hidden states 3d
            hidden_3d_gif_dir = os.path.join(save_fig_dir, "3d_hidden_gifs")
            if not os.path.exists(hidden_3d_gif_dir):
                os.makedirs(hidden_3d_gif_dir)
            vis_embedding_space_params = [
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'nmf'},
                {
                    "states_list": embedding_states_list,
                    "save_dir": hidden_3d_gif_dir,
                    "variable_name": f"hidden",
                    "condition_names": embedding_states_conditions,
                    "base_colors": embedding_states_colors,
                    "technique": "kernel_pca",
                },
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'isomap'},
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'lle'},
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'umap'},
                {
                    "states_list": embedding_states_list,
                    "save_dir": hidden_3d_gif_dir,
                    "variable_name": f"hidden",
                    "condition_names": embedding_states_conditions,
                    "base_colors": embedding_states_colors,
                    "technique": "ica",
                },
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'mds'},
                # {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors},
                {
                    "states_list": embedding_states_list,
                    "save_dir": hidden_3d_gif_dir,
                    "variable_name": f"hidden",
                    "condition_names": embedding_states_conditions,
                    "base_colors": embedding_states_colors,
                    "technique": "tsne",
                },
            ]

            # Save the metrics as YAML before starting parallel visualizations
            metrics_file = os.path.join(save_dir, "evaluation_summary.yaml")
            with open(metrics_file, "w") as f:
                yaml.dump(metrics, f, default_flow_style=False)
            print(f"[Eval] Metrics saved to: {metrics_file}")

            run_parallel_visualizations(
                visualize_embedding_space, vis_embedding_space_params
            )

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
        new_seq_len_short = 60  # Short sequences to stay in linear regime
        test_dataloader.dataset.update_sequence_length(new_seq_len_short)

        drift_stats_list = []
        drift_per_step_d_norm_list = []
        drift_per_step_cross_list = []
        drift_per_step_delta_mse_list = []

        for batch_idx, batch_data in enumerate(test_dataloader):
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

    #     ############################################################################
    #     # Prepare shorter sequence data
    #     # Single batch for demonstration
    #     new_seq_len = 2000
    #     test_dataloader.dataset.update_sequence_length(new_seq_len)
    #     batch_data = next(iter(test_dataloader))
    #     batch_data = batch_data.to(device)
    #     # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
    #     batch_data = batch_data.permute(1, 0, 2)
    #     seq_len, batch_size, x_dim = batch_data.shape
    #     half_point = seq_len // 2
    #     num_iterations = 100
    #     # iterated batch data of single series To calculate the accuracy measure for the same time series
    #     batch_data_repeated = batch_data.repeat(1, num_iterations, 1)

    #     autonomous_mode_selector = create_autonomous_mode_selector(
    #         seq_len,
    #         "even_bursts",
    #         autonomous_ratio=0.1,
    #     ).astype(bool)
    #     expanded_autonomous_mode_selector = expand_autonomous_mode_selector(
    #         autonomous_mode_selector, x_dim
    #     )

    #     # turn input into tensor and send to GPU if needed
    #     batch_data_repeated_tensor = torch.tensor(
    #         batch_data_repeated, device=dvae.device
    #     )
    #     recon_data_repeated = (
    #         dvae(batch_data_repeated_tensor, mode_selector=autonomous_mode_selector)
    #         .cpu()
    #         .numpy()
    #     )

    #     batch_data_repeated = batch_data_repeated.reshape(
    #         seq_len, batch_size, num_iterations, x_dim
    #     )
    #     recon_data_repeated = recon_data_repeated.reshape(
    #         seq_len, batch_size, num_iterations, x_dim
    #     )

    #     # Calculate expected RMSE
    #     expected_rmse, expected_rmse_variance = calculate_expected_accuracy(
    #         batch_data_repeated, recon_data_repeated, rmse
    #     )

    #     # Calculate expected R^2
    #     expected_r2, expected_r2_variance = calculate_expected_accuracy(
    #         batch_data_repeated, recon_data_repeated, r_squared
    #     )

    #     # Visualize results
    #     save_dir = os.path.dirname(params["saved_dict"])

    #     visualize_accuracy_over_time(
    #         expected_rmse,
    #         expected_rmse_variance,
    #         save_dir,
    #         measure="rsme",
    #         num_batches=batch_size,
    #         num_iter=num_iterations,
    #         explain="over multiple series",
    #         autonomous_mode_selector=expanded_autonomous_mode_selector,
    #     )
    #     visualize_accuracy_over_time(
    #         expected_r2,
    #         expected_r2_variance,
    #         save_dir,
    #         measure="r2",
    #         num_batches=batch_size,
    #         num_iter=num_iterations,
    #         explain="over multiple series",
    #         autonomous_mode_selector=expanded_autonomous_mode_selector,
    #     )

    #     # Check if the model has a z variable
    #     if hasattr(dvae, "z_mean"):
    #         # visualize the latent states
    #         visualize_variable_evolution(
    #             dvae.z_mean,
    #             batch_data=batch_data,
    #             save_dir=save_fig_dir,
    #             variable_name=f"z_mean_posterior",
    #             add_lines_lst=[half_point],
    #         )
    #         visualize_variable_evolution(
    #             dvae.z_logvar,
    #             batch_data=batch_data,
    #             save_dir=save_fig_dir,
    #             variable_name=f"z_logvar_posterior",
    #             add_lines_lst=[half_point],
    #         )
    #         visualize_variable_evolution(
    #             dvae.z_mean_p,
    #             batch_data=batch_data,
    #             save_dir=save_fig_dir,
    #             variable_name=f"z_mean_prior",
    #             add_lines_lst=[half_point],
    #         )
    #         visualize_variable_evolution(
    #             dvae.z_logvar_p,
    #             batch_data=batch_data,
    #             save_dir=save_fig_dir,
    #             variable_name=f"z_logvar_prior",
    #             add_lines_lst=[half_point],
    #         )

    #     # Plot the reconstruction vs true sequence
    #     visualize_teacherforcing_2_autonomous(
    #         batch_data,
    #         dvae,
    #         mode_selector=autonomous_mode_selector,
    #         save_path=save_fig_dir,
    #         explain="final_generative_mode",
    #         inference_mode=False,
    #     )
    #     visualize_teacherforcing_2_autonomous(
    #         batch_data,
    #         dvae,
    #         mode_selector=autonomous_mode_selector,
    #         save_path=save_fig_dir,
    #         explain="final_inference_mode",
    #         inference_mode=True,
    #     )

    # Save the metrics as YAML (human-readable JSON-like format)
    metrics_file = os.path.join(save_dir, "evaluation_summary.yaml")
    with open(metrics_file, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"[Eval] Metrics saved to: {metrics_file}")
