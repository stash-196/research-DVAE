from dvae.utils import (
    calculate_power_spectrum_error,
)

from dvae.visualizers.visualizers import (
    visualize_spectral_analysis,
    visualize_errors_from_lst,
    visualize_alpha_history_and_spectrums,
)


def run_spectrum_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
    model_name,
    loaded_data,
    cfg,
):
    if dataset_name == "Lorenz63":
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)
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

    elif dataset_name == "DampedSHO":
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)
        # Visualize the spectral analysis
        long_data_lst = [
            full_xyz_data[:1000, 0],
            recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        name_lst = ["Ground Truth", "teacher-forced", "autonomous"]
        true_signal_index = 2
        colors_lst = ["blue", "green", "red"]
        dt = 1e-2

    elif dataset_name == "SHO":
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)
        # Visualize the spectral analysis
        long_data_lst = [
            full_xyz_data[:1000, 0],
            recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        name_lst = ["Ground Truth", "teacher-forced", "autonomous"]
        true_signal_index = 2
        colors_lst = ["blue", "green", "red"]
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
    if model_name == "MT_RNN" or model_name == "MT_VRNN":
        sigmas_history = loaded_data["sigmas_history"]
        kl_warm_epochs = loaded_data["kl_warm_epochs"]
        # Visualize the alphas
        # if lorenz63, true_alphas = [0.00490695, 0.02916397, 0.01453569]
        if test_dataloader.dataset.true_alphas is not None:
            true_alphas = test_dataloader.dataset.true_alphas
        else:
            true_alphas = []

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
