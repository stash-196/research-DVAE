#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import uuid
from sklearn import metrics
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
    compute_delay_embedding,
    state_space_kl,
)

from torch.nn.functional import mse_loss
import plotly.graph_objects as go
import plotly.express as px
import pickle
import configparser
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
    print(
        "[Eval] Total params: %.2fM"
        % (sum(p.numel() for p in dvae.parameters()) / 1000000.0)
    )

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
    metrics = {"params": params}
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

        if learning_algo.dataset_name == "Lorenz63":
            new_seq_len = 10000
        else:
            new_seq_len = min(10000, len(test_dataloader.dataset.seq))
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

            if learning_algo.dataset_name == "Lorenz63":
                time_delay = 10
                delay_embedding_dimensions = 3
            elif learning_algo.dataset_name in ["Xhro", "SHO", "DampedSHO"]:
                time_delay = 5
                delay_embedding_dimensions = 3

            embedded_true_x = compute_delay_embedding(
                observation=batch_data_long[:, 0, :].reshape(-1).numpy(),
                delay=time_delay,
                dimensions=delay_embedding_dimensions,
            )

            embedded_recon_teacher = compute_delay_embedding(
                observation=recon_data_long[
                    ~autonomous_mode_selector_long.bool()  # , 0, :
                ].reshape(-1),
                delay=time_delay,
                dimensions=delay_embedding_dimensions,
            )

            embedded_recon_auto = compute_delay_embedding(
                observation=recon_data_long[
                    autonomous_mode_selector_long.bool(),  # , 0, :
                ].reshape(-1),
                delay=time_delay,
                dimensions=delay_embedding_dimensions,
            )

            kl_tf_error = state_space_kl(
                true_traj=embedded_true_x,
                gen_traj=embedded_recon_teacher,
                use_gmm=True,
            )
            print(f"[Eval] KL Teacher-forced: {kl_tf_error:.4f}")
            kl_auto_error = state_space_kl(
                true_traj=embedded_true_x,
                gen_traj=embedded_recon_auto,
                use_gmm=True,
            )
            print(f"[Eval] KL Autonomous: {kl_auto_error:.4f}")

            # Add to metrics
            metrics["kld_tf"] = float(kl_tf_error)
            metrics["kld_auto"] = float(kl_auto_error)

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
                f.write(f"KL Teacher-forced: {kl_tf_error:.4f}\n")
                f.write(f"KL Autonomous: {kl_auto_error:.4f}\n")

            if True:
                visualize_delay_embedding(
                    embedded=embedded_true_x,
                    save_dir=save_fig_dir,
                    variable_name=f"true_signal_inference_mode_τ{time_delay}_d{delay_embedding_dimensions}",
                    base_color="Blues",
                )
                visualize_delay_embedding(
                    embedded=embedded_recon_teacher,
                    save_dir=save_fig_dir,
                    variable_name=f"teacher-forced_reconstruction_inference_mode_τ{time_delay}_d{delay_embedding_dimensions}",
                    base_color="Greens",
                )
                visualize_delay_embedding(
                    embedded=embedded_recon_auto,
                    save_dir=save_fig_dir,
                    variable_name=f"autonomous_reconstruction_inference_mode_τ{time_delay}_d{delay_embedding_dimensions}",
                    base_color="Reds",
                )

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
