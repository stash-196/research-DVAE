import numpy as np
from dvae.visualizers.visualizers import visualize_errors_from_lst


def calc_masked_mse(gt, pred, mask=None):
    """MSE ignoring NaNs and optional boolean missing mask (True = missing)."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    min_len = min(gt.shape[0], pred.shape[0])
    gt = gt[:min_len]
    pred = pred[:min_len]

    valid = np.isfinite(gt) & np.isfinite(pred)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)[:min_len]
        valid &= ~mask

    if not np.any(valid):
        return float("nan")

    return float(np.mean((gt[valid] - pred[valid]) ** 2))


def run_mse_analysis_from_benchmarks(
    channel_benchmarks, save_fig_dir, batch_idx=0, save_figures=True
):
    channels = channel_benchmarks["channels"]

    per_channel = {}
    tf_errors = []
    auto_errors = []
    tf_keys = []
    auto_keys = []
    tf_names = []
    auto_names = []
    tf_colors = []
    auto_colors = []

    print("[Eval] Mean Squared Error vs GT (per channel):")
    for ch in channels:
        key = ch["key"]
        mse_tf = calc_masked_mse(ch["gt_full"], ch["tf_full"], ch.get("mask_full"))
        mse_auto = calc_masked_mse(ch["gt_auto"], ch["auto_seg"], ch.get("mask_auto"))
        per_channel[key] = {"mse_tf": mse_tf, "mse_auto": mse_auto}
        print(f"  {key} TF: {mse_tf:.6f}  Auto: {mse_auto:.6f}")

        tf_errors.append(mse_tf)
        auto_errors.append(mse_auto)
        tf_keys.append(f"{key}_tf")
        auto_keys.append(f"{key}_auto")
        tf_names.append(f"{key}\nTF")
        auto_names.append(f"{key}\nAuto")
        tf_colors.append("green")
        auto_colors.append("red")

    tf_values = [v["mse_tf"] for v in per_channel.values() if np.isfinite(v["mse_tf"])]
    auto_values = [
        v["mse_auto"] for v in per_channel.values() if np.isfinite(v["mse_auto"])
    ]
    mse_tf_mean = float(np.mean(tf_values)) if tf_values else float("nan")
    mse_auto_mean = float(np.mean(auto_values)) if auto_values else float("nan")

    if save_figures:
        # One multi-bar chart (GT=0, then TF/Auto per channel) — not single-bar panels
        comb_errs, comb_names, comb_colors = [0.0], ["Ground\nTruth"], ["blue"]
        for ch_key, vals in per_channel.items():
            comb_names.append(f"{ch_key}\nTF")
            comb_errs.append(vals["mse_tf"])
            comb_colors.append("green")
            comb_names.append(f"{ch_key}\nAuto")
            comb_errs.append(vals["mse_auto"])
            comb_colors.append("red")
        visualize_errors_from_lst(
            comb_errs,
            name_lst=comb_names,
            save_dir=save_fig_dir,
            explain=f"mse_error_per_signal_batch{batch_idx}",
            error_unit="MSE",
            colors=comb_colors,
        )

    return {
        "per_channel": per_channel,
        "mse_tf_mean": mse_tf_mean,
        "mse_auto_mean": mse_auto_mean,
        "mse_tf": mse_tf_mean,
        "mse_auto": mse_auto_mean,
        "mse_errors": tf_errors + auto_errors,
        "signal_keys": tf_keys + auto_keys,
        "tf_keys": tf_keys,
        "auto_keys": auto_keys,
    }


def run_mse_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
    batch_data_long=None,
    channel_benchmarks=None,
    save_figures=True,
):
    if channel_benchmarks is not None:
        return run_mse_analysis_from_benchmarks(
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
    return run_mse_analysis_from_benchmarks(
        sig_info["channel_benchmarks"], save_fig_dir, i, save_figures=save_figures
    )