import numpy as np
from dvae.eval.utils.durstewitz_eval_metrics import power_spectrum_error
from dvae.visualizers.visualizers import (
    visualize_spectral_analysis,
    visualize_errors_from_lst,
)


def _compute_power_spectrum(data, sampling_rate):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) < 4:
        raise ValueError("Sequence too short for spectrum analysis.")
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1 / sampling_rate)
    nonzero = freqs > 0
    ps = np.abs(fft[nonzero]) ** 2
    if not np.any(ps > 0):
        raise ValueError("Power spectrum has no positive values.")
    return ps


def _spectrum_pair_errors(gt, pred, sampling_rate):
    gt_ps = _compute_power_spectrum(gt, sampling_rate)
    pred_ps = _compute_power_spectrum(pred, sampling_rate)
    min_len = min(len(gt_ps), len(pred_ps))
    # hellinger_distance in power_spectrum_error expects 2D (n_series, n_freq)
    gt_2d = gt_ps[:min_len].reshape(1, -1)
    pred_2d = pred_ps[:min_len].reshape(1, -1)
    err = power_spectrum_error(pred_2d, gt_2d, filter_std=3)
    return float(err)


def run_spectrum_analysis_from_benchmarks(
    channel_benchmarks, save_fig_dir, batch_idx=0, save_figures=True
):
    channels = channel_benchmarks["channels"]
    sampling_rate = 1.0 / channel_benchmarks["dt"]

    per_channel = {}
    plot_data = []
    plot_names = []
    plot_colors = []
    tf_errors = []
    auto_errors = []
    tf_keys = []
    auto_keys = []
    tf_names = []
    auto_names = []

    print("[Eval] Power Spectrum Errors (per channel, auto segment, dB):")
    for ch in channels:
        key = ch["key"]
        gt = ch["gt_auto"]
        tf = ch["tf_auto"]
        auto = ch["auto_seg"]

        plot_data.extend([gt, tf, auto])
        plot_names.extend(
            [f"{key} GT", f"{key} TF", f"{key} Auto"]
        )
        plot_colors.extend([ch["color"], "green", "red"])

        try:
            err_tf = _spectrum_pair_errors(gt, tf, sampling_rate)
            err_auto = _spectrum_pair_errors(gt, auto, sampling_rate)
        except Exception as exc:
            print(f"  [Eval] Spectrum error skipped for {key}: {exc}")
            err_tf = float("nan")
            err_auto = float("nan")

        per_channel[key] = {
            "spectrum_error_tf": err_tf,
            "spectrum_error_auto": err_auto,
        }
        print(f"  {key} TF: {err_tf:.4f} dB  Auto: {err_auto:.4f} dB")

        tf_errors.append(err_tf)
        auto_errors.append(err_auto)
        tf_keys.append(f"{key}_tf")
        auto_keys.append(f"{key}_auto")
        tf_names.append(f"{key}\nTF")
        auto_names.append(f"{key}\nAuto")

    if save_figures:
        if plot_data:
            try:
                visualize_spectral_analysis(
                    data_lst=plot_data,
                    name_lst=plot_names,
                    colors_lst=plot_colors,
                    save_dir=save_fig_dir,
                    sampling_rate=sampling_rate,
                    max_sequences=50000,
                    explain=f"batch{batch_idx}",
                )
                visualize_spectral_analysis(
                    data_lst=plot_data,
                    name_lst=plot_names,
                    colors_lst=plot_colors,
                    save_dir=save_fig_dir,
                    sampling_rate=sampling_rate,
                    max_sequences=50000,
                    use_log_scale=True,
                    explain=f"batch{batch_idx}_log",
                )
            except Exception as exc:
                print(f"[Eval] Warning: spectral visualization skipped: {exc}")

        visualize_errors_from_lst(
            tf_errors,
            name_lst=tf_names,
            save_dir=save_fig_dir,
            explain=f"spectrum_tf_per_channel_batch{batch_idx}",
            error_unit="dB",
            colors=["green"] * len(tf_errors),
        )
        visualize_errors_from_lst(
            auto_errors,
            name_lst=auto_names,
            save_dir=save_fig_dir,
            explain=f"spectrum_auto_per_channel_batch{batch_idx}",
            error_unit="dB",
            colors=["red"] * len(auto_errors),
        )

    tf_values = [
        v["spectrum_error_tf"]
        for v in per_channel.values()
        if np.isfinite(v["spectrum_error_tf"])
    ]
    auto_values = [
        v["spectrum_error_auto"]
        for v in per_channel.values()
        if np.isfinite(v["spectrum_error_auto"])
    ]
    spectrum_error_tf_mean = float(np.mean(tf_values)) if tf_values else float("nan")
    spectrum_error_auto_mean = (
        float(np.mean(auto_values)) if auto_values else float("nan")
    )

    return {
        "per_channel": per_channel,
        "spectrum_error_tf_mean": spectrum_error_tf_mean,
        "spectrum_error_auto_mean": spectrum_error_auto_mean,
        "spectrum_error_tf": spectrum_error_tf_mean,
        "spectrum_error_auto": spectrum_error_auto_mean,
        "spectrum_error_gt": 0.0,
        "power_spectrum_errors": tf_errors + auto_errors,
        "signal_keys": tf_keys + auto_keys,
        "tf_keys": tf_keys,
        "auto_keys": auto_keys,
    }


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
    channel_benchmarks=None,
    save_figures=True,
):
    if channel_benchmarks is not None:
        return run_spectrum_analysis_from_benchmarks(
            channel_benchmarks, save_fig_dir, i, save_figures=save_figures
        )

    from dvae.eval.utils.benchmark_signals import get_benchmark_signals

    sig_info = get_benchmark_signals(
        dataset_name,
        test_dataloader,
        i,
        recon_data_long,
        autonomous_mode_selector_long,
        batch_data_long,
    )
    return run_spectrum_analysis_from_benchmarks(
        sig_info["channel_benchmarks"], save_fig_dir, i, save_figures=save_figures
    )