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
    dvae.load_state_dict(torch.load(params["saved_dict"], map_location="cpu"))

    dvae.eval()
    cfg = learning_algo.cfg
    print(
        "[Eval] Total params: %.2fM"
        % (sum(p.numel() for p in dvae.parameters()) / 1000000.0)
    )

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
        learning_algo.dataset_name, dataset_config, split="test"
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

    # Check if the loss_model.pckl file exists
    if os.path.isfile(loss_file):
        print(f"[Eval] Loading loss data from {loss_file}")
        with open(loss_file, "rb") as f:
            # Load the data
            loaded_data = pickle.load(f)

    else:
        print(f"[Eval] No loss data file found at {loss_file}")

    ############################################################################

    VISUALIZE_3D = True

    with torch.no_grad():

        # Visualize results
        save_fig_dir = os.path.join(
            os.path.dirname(params["saved_dict"]), "post_training_figs"
        )
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)

        ############################################################################

        new_seq_len = min(10000, len(test_dataloader.dataset.seq))
        print(f"[Eval] New sequence length: {new_seq_len}")
        test_dataloader.dataset.update_sequence_length(new_seq_len)
        # Prepare the long sequence data
        for i, batch_data_long in enumerate(test_dataloader):
            full_xyz_data = test_dataloader.dataset.get_full_xyz(
                i
            )  # Get full xyz data for the same index

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

            # Visualize the spectral analysis
            long_data_lst = [
                full_xyz_data[:, 1],
                full_xyz_data[:, 2],
                # batch_data_long[:, 0, :].reshape(-1), update to interpolate linearly later
                full_xyz_data[:, 0],
                recon_data_long[~autonomous_mode_selector_long.bool()],
                recon_data_long[autonomous_mode_selector_long.bool()],
            ]
            name_lst = ["y", "z", "observed_x", "teacher-forced", "autonomous"]
            true_signal_index = 2
            colors_lst = ["orange", "magenta", "blue", "green", "red"]

            dt = 1e-2
            sampling_rate = 1 / dt
            power_spectrum_lst, frequencies, periods = visualize_spectral_analysis(
                data_lst=long_data_lst,
                name_lst=name_lst,
                colors_lst=colors_lst,
                save_dir=save_fig_dir,
                sampling_rate=sampling_rate,
                max_sequences=50000,
            )

            power_spectrum_error_lst = calculate_power_spectrum_error(
                power_spectrum_lst, true_signal_index, filter_std=3
            )
            visualize_errors_from_lst(
                power_spectrum_error_lst,
                name_lst=name_lst,
                save_dir=save_fig_dir,
                explain="power_spectrum_error",
                error_unit="dB",
                colors=colors_lst,
            )

            # Visualize the alphas against the power spectral density
            if (
                learning_algo.model_name == "MT_RNN"
                or learning_algo.model_name == "MT_VRNN"
            ):
                sigmas_history = loaded_data["sigmas_history"]
                kl_warm_epochs = loaded_data["kl_warm_epochs"]
                # Visualize the alphas
                # if lorenz63, true_alphas = [0.00490695, 0.02916397, 0.01453569]
                if cfg["DataFrame"]["dataset_name"] == "Lorenz63":
                    true_alphas = [0.00490695, 0.02916397, 0.01453569]
                elif cfg["DataFrame"]["dataset_name"] == "Sinusoid":
                    if s_dim == 1:
                        true_alphas = [dt * 2 * np.pi]
                    elif s_dim == 2:
                        true_alphas = [dt * 2 * np.pi, dt * 2 * np.pi / 100]
                    elif s_dim == 3:
                        true_alphas = [
                            dt * 2 * np.pi,
                            dt * 2 * np.pi / 100,
                            2 * np.pi / 1000,
                        ]
                else:
                    raise ValueError("Unsupported dataset_name in configuration file.")
                visualize_alpha_history_and_spectrums(
                    sigmas_history=sigmas_history,
                    power_spectrum_lst=power_spectrum_lst[:3],
                    spectrum_color_lst=colors_lst[:3],
                    spectrum_name_lst=name_lst,
                    frequencies=frequencies,
                    dt=dt,
                    save_dir=save_fig_dir,
                    kl_warm_epochs=kl_warm_epochs,
                    true_alphas=true_alphas,
                )

            # Plot the reconstruction vs true sequence
            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                mode_selector=autonomous_mode_selector_long,
                save_path=save_fig_dir,
                explain="final_long_inference_mode",
                inference_mode=True,
            )
            visualize_teacherforcing_2_autonomous(
                batch_data_long,
                dvae,
                mode_selector=autonomous_mode_selector_long,
                save_path=save_fig_dir,
                explain="final_long_generative_mode",
                inference_mode=False,
            )

            time_delay = 10
            delay_emedding_dimensions = 3
            if VISUALIZE_3D:
                embedded_true_x = visualize_delay_embedding(
                    observation=batch_data_long[:, 0, :].reshape(-1),
                    delay=time_delay,
                    dimensions=delay_emedding_dimensions,
                    save_dir=save_fig_dir,
                    variable_name="true_signal_inference_mode",
                    base_color="Blues",
                )
                embedded_ = visualize_delay_embedding(
                    observation=recon_data_long[
                        ~autonomous_mode_selector_long, 0, :
                    ].reshape(-1),
                    delay=time_delay,
                    dimensions=delay_emedding_dimensions,
                    save_dir=save_fig_dir,
                    variable_name="teacher-forced_reconstruction_inference_mode",
                    base_color="Greens",
                )
                visualize_delay_embedding(
                    observation=recon_data_long[
                        autonomous_mode_selector_long, 0, :
                    ].reshape(-1),
                    delay=time_delay,
                    dimensions=delay_emedding_dimensions,
                    save_dir=save_fig_dir,
                    variable_name="autonomous_reconstruction_inference_mode",
                    base_color="Reds",
                )

            teacherforced_states = dvae.h[~autonomous_mode_selector_long, 0, :]
            autonomous_states = dvae.h[autonomous_mode_selector_long, 0, :]
            embedding_states_list = [teacherforced_states, autonomous_states]
            embedding_states_conditions = ["teacher-forced", "autonomous"]
            embedding_states_colors = ["Greens", "Reds"]

            # # visualize the hidden states 3d
            # # vis_embedding_space_params = [
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'nmf'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'kernel_pca'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'isomap'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'lle'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'umap'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'ica'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'mds'},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors},
            # #     {'states_list': embedding_states_list, 'save_dir': save_dir, 'variable_name': f'hidden', 'condition_names': embedding_states_conditions, 'base_colors': embedding_states_colors, 'technique': 'tsne'},
            # # ]
            # # run_parallel_visualizations(visualize_embedding_space, vis_embedding_space_params)
            #

            # visualize the hidden states 3d in different techniques
            # if VISUALIZE_3D:
            if False:
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="nmf",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="kernel_pca",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="isomap",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="lle",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="umap",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="ica",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="mds",
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                )
                visualize_embedding_space(
                    [teacherforced_states, autonomous_states],
                    save_dir=save_fig_dir,
                    variable_name="hidden",
                    condition_names=[f"teacher-forced", f"autonomous"],
                    base_colors=["Greens", "Reds"],
                    technique="tsne",
                )

            # break after the first batch
            break

        ############################################################################
        # Prepare shorter sequence data
        # Single batch for demonstration
        new_seq_len = min(1000, learning_algo.sequence_len)
        test_dataloader.dataset.update_sequence_length(new_seq_len)
        batch_data = next(iter(test_dataloader))
        batch_data = batch_data.to(device)
        # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
        batch_data = batch_data.permute(1, 0, 2)
        seq_len, batch_size, x_dim = batch_data.shape
        half_point = seq_len // 2
        num_iterations = 100
        # iterated batch data of single series To calculate the accuracy measure for the same time series
        batch_data_repeated = batch_data.repeat(1, num_iterations, 1)

        autonomous_mode_selector = create_autonomous_mode_selector(
            seq_len,
            "even_bursts",
            autonomous_ratio=0.1,
        ).astype(bool)
        expanded_autonomous_mode_selector = expand_autonomous_mode_selector(
            autonomous_mode_selector, x_dim
        )

        # turn input into tensor and send to GPU if needed
        batch_data_repeated_tensor = torch.tensor(
            batch_data_repeated, device=dvae.device
        )
        recon_data_repeated = (
            dvae(batch_data_repeated_tensor, mode_selector=autonomous_mode_selector)
            .cpu()
            .numpy()
        )

        batch_data_repeated = batch_data_repeated.reshape(
            seq_len, batch_size, num_iterations, x_dim
        )
        recon_data_repeated = recon_data_repeated.reshape(
            seq_len, batch_size, num_iterations, x_dim
        )

        # Calculate expected RMSE
        expected_rmse, expected_rmse_variance = calculate_expected_accuracy(
            batch_data_repeated, recon_data_repeated, rmse
        )

        # Calculate expected R^2
        expected_r2, expected_r2_variance = calculate_expected_accuracy(
            batch_data_repeated, recon_data_repeated, r_squared
        )

        # Visualize results
        save_dir = os.path.dirname(params["saved_dict"])

        visualize_accuracy_over_time(
            expected_rmse,
            expected_rmse_variance,
            save_dir,
            measure="rsme",
            num_batches=batch_size,
            num_iter=num_iterations,
            explain="over multiple series",
            autonomous_mode_selector=expanded_autonomous_mode_selector,
        )
        visualize_accuracy_over_time(
            expected_r2,
            expected_r2_variance,
            save_dir,
            measure="r2",
            num_batches=batch_size,
            num_iter=num_iterations,
            explain="over multiple series",
            autonomous_mode_selector=expanded_autonomous_mode_selector,
        )

        # visualize the hidden states
        visualize_variable_evolution(
            dvae.h,
            batch_data=batch_data,
            save_dir=save_fig_dir,
            variable_name=f"hidden",
            alphas=alphas_per_unit,
            add_lines_lst=[half_point],
        )

        # visualize the x_features
        visualize_variable_evolution(
            dvae.feature_x,
            batch_data=batch_data,
            save_dir=save_fig_dir,
            variable_name=f"x_features",
            add_lines_lst=[half_point],
        )

        # Check if the model has a z variable
        if hasattr(dvae, "z_mean"):
            # visualize the latent states
            visualize_variable_evolution(
                dvae.z_mean,
                batch_data=batch_data,
                save_dir=save_fig_dir,
                variable_name=f"z_mean_posterior",
                add_lines_lst=[half_point],
            )
            visualize_variable_evolution(
                dvae.z_logvar,
                batch_data=batch_data,
                save_dir=save_fig_dir,
                variable_name=f"z_logvar_posterior",
                add_lines_lst=[half_point],
            )
            visualize_variable_evolution(
                dvae.z_mean_p,
                batch_data=batch_data,
                save_dir=save_fig_dir,
                variable_name=f"z_mean_prior",
                add_lines_lst=[half_point],
            )
            visualize_variable_evolution(
                dvae.z_logvar_p,
                batch_data=batch_data,
                save_dir=save_fig_dir,
                variable_name=f"z_logvar_prior",
                add_lines_lst=[half_point],
            )

        # Plot the reconstruction vs true sequence
        visualize_teacherforcing_2_autonomous(
            batch_data,
            dvae,
            mode_selector=autonomous_mode_selector,
            save_path=save_fig_dir,
            explain="final_generative_mode",
            inference_mode=False,
        )
        visualize_teacherforcing_2_autonomous(
            batch_data,
            dvae,
            mode_selector=autonomous_mode_selector,
            save_path=save_fig_dir,
            explain="final_inference_mode",
            inference_mode=True,
        )

    metrics = {
        "params": params,
        "power_spectrum_error": power_spectrum_error_lst,
    }

    # Create directory to save metrics if it does not exist
    eval_save_path = os.path.join(save_dir, "evaluation")
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    # Save the metrics
    metrics_file = os.path.join(eval_save_path, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
