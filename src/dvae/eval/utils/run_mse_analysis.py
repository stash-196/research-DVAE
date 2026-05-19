import numpy as np
from dvae.eval.utils.benchmark_signals import get_benchmark_signals
from dvae.visualizers.visualizers import visualize_errors_from_lst


def calc_mse(signal1, signal2):
    # Ensure they have the same shape up to the length of the shorter one
    min_len = min(signal1.shape[0], signal2.shape[0])
    return np.nanmean((signal1[:min_len] - signal2[:min_len]) ** 2)


def run_mse_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
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

    print("[Eval] Mean Squared Error vs GT:")
    mse_errors = []

    gt_signal = long_data_lst[true_signal_index]

    for j, (name, sig) in enumerate(zip(name_lst, long_data_lst)):
        if j == true_signal_index:
            mse_errors.append(0.0)
            continue

        err = calc_mse(gt_signal, sig)
        mse_errors.append(float(err))
        print(f"  MSE {name.replace(chr(10), " ")}: {err:.4f}")

    # Visualize MSE error bars
    visualize_errors_from_lst(
        mse_errors,
        name_lst=name_lst,
        save_dir=save_fig_dir,
        explain="mse_error_per_signal",
        error_unit="MSE",
        colors=colors_lst,
    )

    return {
        "mse_errors": mse_errors,
        "signal_keys": key_lst,
    }
