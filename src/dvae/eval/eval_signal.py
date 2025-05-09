#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt

https://github.com/stash-196/research-DVAE

# Evaluation on speech data
python eval_wsj.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT
python eval_wsj.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT # schedule sampling

# Evaluation on human motion data
python eval_h36m.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT
python eval_h36m.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT # schedule sampling
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.dataset import sinusoid_dataset, lorenz63_dataset
from dvae.utils import (
    loss_MSE,
    create_autonomous_mode_selector,
    run_parallel_visualizations,
    power_spectrum_error,
)
from visualizers import (
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
from dvae.utils import merge_configs
import configparser


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument("--ss", action="store_true", help="schedule sampling")
        self.parser.add_argument("--cfg", type=str, default=None, help="config path")
        self.parser.add_argument(
            "--device_cfg",
            type=str,
            default="config/cfg_device.ini",
            help="device config path",
        )
        self.parser.add_argument(
            "--saved_dict", type=str, default=None, help="trained model dict"
        )

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)

        # If cfg is not provided, set it based on the directory of saved_dict
        if params["cfg"] is None:
            params["cfg"] = os.path.join(
                os.path.dirname(params["saved_dict"]), "config.ini"
            )

        return params


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    params = Options().get_params()
    merged_config = merge_configs(params["device_cfg"], params["cfg"])

    # Update paths in the merged configuration
    merged_config["User"]["data_dir"] = merged_config["Paths"]["data_dir"]
    merged_config["User"]["saved_root"] = merged_config["Paths"]["saved_root"]

    # Save the merged configuration temporarily
    merged_config_path = os.path.join(
        os.path.dirname(params["saved_dict"]), "merged_config.ini"
    )
    with open(merged_config_path, "w") as configfile:
        config = configparser.ConfigParser()
        for section, section_values in merged_config.items():
            config[section] = section_values
        config.write(configfile)

    # Update params to use the merged configuration
    params["cfg"] = merged_config_path

    device = "cpu"

    if params["ss"]:
        learning_algo = LearningAlgorithm_ss(params=params)
        learning_algo.build_model()
        dvae = learning_algo.model
        dvae.device = device
        dvae.out_mean = True
    else:
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

    data_dir = cfg.get("Paths", "data_dir")
    x_dim = cfg.getint("Network", "x_dim")
    dataset_label = cfg.get("DataFrame", "dataset_label", fallback=None)
    num_workers = cfg.getint("DataFrame", "num_workers")
    sample_rate = cfg.getint("DataFrame", "sample_rate")
    skip_rate = cfg.getint("DataFrame", "skip_rate")
    observation_process = cfg.get("DataFrame", "observation_process")
    overlap = cfg.getboolean("DataFrame", "overlap")
    seq_len = cfg.getint("DataFrame", "sequence_len")
    s_dim = cfg.getint("DataFrame", "s_dim")
    with_nan = cfg.getboolean("DataFrame", "with_nan", fallback=False)

    # specify seq_len for the visualization
    seq_len = min(1000, seq_len)

    if cfg["DataFrame"]["dataset_name"] == "Sinusoid":
        # Load test dataset
        test_dataset = sinusoid_dataset.Sinusoid(
            path_to_data=data_dir,
            dataset_label=dataset_label,
            split="test",
            seq_len=seq_len,
            x_dim=x_dim,
            sample_rate=sample_rate,
            observation_process=observation_process,
            device=device,
            overlap=overlap,
            skip_rate=skip_rate,
            val_indices=1,
            shuffle=False,
            s_dim=s_dim,
        )
        # Build test dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            num_workers=num_workers,
        )
        # Load test dataset for long sequence
        test_dataset_long = sinusoid_dataset.Sinusoid(
            path_to_data=data_dir,
            dataset_label=dataset_label,
            split="test",
            seq_len=None,
            x_dim=x_dim,
            sample_rate=sample_rate,
            observation_process=observation_process,
            device=device,
            overlap=overlap,
            skip_rate=skip_rate,
            val_indices=1,
            shuffle=False,
            s_dim=s_dim,
        )
        # Build test dataloader
        test_dataloader_long = torch.utils.data.DataLoader(
            test_dataset_long, batch_size=1, shuffle=False, num_workers=num_workers
        )
    elif cfg["DataFrame"]["dataset_name"] == "Lorenz63":
        # Load test dataset
        test_dataset = lorenz63_dataset.Lorenz63(
            dataset_label=dataset_label,
            path_to_data=data_dir,
            split="test",
            seq_len=seq_len,
            x_dim=x_dim,
            sample_rate=sample_rate,
            observation_process=observation_process,
            device=device,
            overlap=overlap,
            skip_rate=skip_rate,
            val_indices=1,
            shuffle=False,
            with_nan=with_nan,
        )
        # Build test dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            shuffle=False,
            num_workers=num_workers,
        )

        # Load test dataset for long sequence
        test_dataset_long = lorenz63_dataset.Lorenz63(
            path_to_data=data_dir,
            dataset_label=dataset_label,
            split="test",
            seq_len=None,
            x_dim=x_dim,
            sample_rate=sample_rate,
            observation_process=observation_process,
            device=device,
            overlap=overlap,
            skip_rate=skip_rate,
            val_indices=1,
            shuffle=False,
            with_nan=with_nan,
        )
        # Build test dataloader
        test_dataloader_long = torch.utils.data.DataLoader(
            test_dataset_long, batch_size=1, shuffle=False, num_workers=num_workers
        )
    else:
        raise ValueError("Unsupported dataset_name in configuration file.")

    overlap = cfg["DataFrame"].getboolean("overlap")

    test_num = len(test_dataloader.dataset)
    test_num_long = len(test_dataloader_long.dataset)

    print("[Eval] Test samples: {}, {}".format(test_num, test_num_long))

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

    def calculate_expected_accuracy(true_data, reconstructed_data, accuracy_measure):
        """
        Calculate the expected accuracy measure over batch data.

        Args:
        - true_data: The true data in batch format (seq_len, batch_size, x_dim).
        - reconstructed_data: The reconstructed data in the same format as true_data.
        - accuracy_measure: Function to compute the accuracy measure (e.g., RMSE).

        Returns:
        - A tensor of accuracy values averaged over batches.
        """
        seq_len, batch_size, num_iterations, x_dim = true_data.shape

        reshaped_true = true_data.reshape(seq_len * x_dim, batch_size, num_iterations)
        reshaped_recon = reconstructed_data.reshape(
            seq_len * x_dim, batch_size, num_iterations
        )

        accuracy_values = torch.zeros(seq_len * x_dim, device=true_data.device)
        variance_values = torch.zeros(seq_len * x_dim, device=true_data.device)

        for t in range(seq_len * x_dim):
            accuracy_t = accuracy_measure(reshaped_true[t], reshaped_recon[t])
            accuracy_values[t] = accuracy_t.mean()
            variance_values[t] = accuracy_t.var()

        return accuracy_values, variance_values

    def rmse(true_data, pred_data):
        # RMSE per batch item
        return torch.sqrt(torch.mean((true_data - pred_data) ** 2, dim=-1))

    def r_squared(true_data, pred_data, epsilon=1e-8):
        """
        Coefficient of Determination (R^2) calculation with stability check.

        Args:
        - true_data: True data tensor.
        - pred_data: Predicted data tensor.
        - epsilon: A small value to ensure numerical stability.

        Returns:
        - R^2 value.
        """
        ss_total = torch.sum(
            (true_data - true_data.mean(dim=-1, keepdim=True)) ** 2, dim=-1
        )
        ss_res = torch.sum((true_data - pred_data) ** 2, dim=-1)

        # Avoid division by very small or zero values
        stable_ss_total = torch.where(
            ss_total > epsilon, ss_total, torch.ones_like(ss_total) * epsilon
        )
        r2 = 1 - ss_res / stable_ss_total
        return r2  # R^2 per batch item

    def expand_autonomous_mode_selector(selector, x_dim):
        return np.repeat(selector, x_dim)

    def calculate_power_spectrum_error(
        power_spectrum_lst, true_signal_index, filter_std
    ):
        """
        Calculate the power spectrum error for each signal in the list.

        Args:
        - power_spectrum_lst: A list of power spectra.
        - true_signal_index: The index of the true signal in the list.

        Returns:
        - A list of power spectrum errors.
        """
        true_power_spectrum = power_spectrum_lst[true_signal_index]
        power_spectrum_errors = []
        for power_spectrum in power_spectrum_lst:
            power_spectrum_errors.append(
                power_spectrum_error(
                    power_spectrum, true_power_spectrum, filter_std=filter_std
                )
            )

        return power_spectrum_errors

    VISUALIZE_3D = True

    with torch.no_grad():

        # Visualize results
        save_fig_dir = os.path.join(
            os.path.dirname(params["saved_dict"]), "post_training_figs"
        )
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)

        ############################################################################

        # Prepare the long sequence data
        for i, batch_data_long in enumerate(test_dataloader_long):
            full_xyz_data = test_dataloader_long.dataset.get_full_xyz(
                i
            )  # Get full xyz data for the same index

            # batch_data_long = next(iter(test_dataloader_long))  # Single batch for demonstration
            batch_data_long = batch_data_long.to(device)
            # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
            batch_data_long = batch_data_long.permute(1, 0, 2)
            seq_len_long, batch_size_long, _ = batch_data_long.shape
            half_point_long = seq_len_long // 2
            # Plot the spectral analysis
            autonomous_mode_selector_long = create_autonomous_mode_selector(
                seq_len_long, mode="even_bursts", autonomous_ratio=0.1
            ).astype(bool)
            # autonomous_mode_selector_long = create_autonomous_mode_selector(
            #     seq_len_long, 'half_half').astype(bool)

            # turn input into tensor and send to GPU if needed
            batch_data_long_tensor = batch_data_long.clone().detach().to(device)
            recon_data_long = (
                dvae(
                    batch_data_long_tensor,
                    mode_selector=autonomous_mode_selector_long,
                    inference_mode=True,
                )
                .cpu()
                .numpy()
            )

            # Visualize the spectral analysis
            long_data_lst = [
                full_xyz_data[:, 1],
                full_xyz_data[:, 2],
                batch_data_long[:, 0, :].reshape(-1),
                recon_data_long[~autonomous_mode_selector_long, 0, :].reshape(-1),
                recon_data_long[autonomous_mode_selector_long, 0, :].reshape(-1),
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
            if learning_algo.optimize_alphas is not None:
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
        "config": merged_config,
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
