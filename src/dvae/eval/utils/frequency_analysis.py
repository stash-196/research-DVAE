from dvae.utils import calculate_power_spectrum_error
from dvae.visualizers.visualizers import (
    visualize_spectral_analysis,
    visualize_errors_from_lst,
    visualize_alpha_history_and_spectrums,
)
import numpy as np
import pandas as pd
from dvae.eval.utils.benchmark_signals import get_benchmark_signals


def run_spectrum_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
    model_name,
    cfg,
    dvae_model,
    batch_data_long=None,
):
    sig_info = get_benchmark_signals(
        dataset_name,
        test_dataloader,
        i,
        recon_data_long,
        autonomous_mode_selector_long,
        batch_data_long,
    )
    long_data_lst = sig_info["long_data_lst"]
    name_lst = sig_info["name_lst"]
    key_lst = sig_info["key_lst"]
    true_signal_index = sig_info["true_signal_index"]
    colors_lst = sig_info["colors_lst"]
    dt = sig_info["dt"]

    sampling_rate = 1 / dt
    power_spectrum_lst, frequencies, periods = visualize_spectral_analysis(
        data_lst=long_data_lst,
        name_lst=name_lst,
        colors_lst=colors_lst,
        save_dir=save_fig_dir,
        sampling_rate=sampling_rate,
        max_sequences=50000,
    )
    visualize_spectral_analysis(
        data_lst=long_data_lst,
        name_lst=name_lst,
        colors_lst=colors_lst,
        save_dir=save_fig_dir,
        sampling_rate=sampling_rate,
        max_sequences=50000,
        use_log_scale=True,
    )
    power_spectrum_error_lst = calculate_power_spectrum_error(
        power_spectrum_lst, true_signal_index, filter_std=3
    )
    print("[Eval] Power Spectrum Errors (in dB):")
    for name, error in zip(name_lst, power_spectrum_error_lst):
        print(f"  {name}: {error:.4f} dB")
    visualize_errors_from_lst(
        power_spectrum_error_lst,
        name_lst=name_lst,
        save_dir=save_fig_dir,
        explain="power_spectrum_error",
        error_unit="dB",
        colors=colors_lst,
    )
    return {
        "power_spectrum_errors": power_spectrum_error_lst,
        "signal_keys": key_lst,
    }
