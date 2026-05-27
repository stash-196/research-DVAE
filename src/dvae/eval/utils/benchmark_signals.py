import pandas as pd
import numpy as np

def get_benchmark_signals(
    dataset_name,
    test_dataloader,
    i,
    recon_data_long,
    autonomous_mode_selector_long,
    batch_data_long
):
    if dataset_name == "Lorenz63":
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)
        long_data_lst = [
            full_xyz_data[:, 2],
            full_xyz_data[:, 1],
            full_xyz_data[:, 0],
            recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        name_lst = ["z", "y", "Ground\nTruth", "Teacher-\nForced", "Autonomous"]
        key_lst = ["z", "y", "gt", "tf", "auto"]
        true_signal_index = 2
        colors_lst = ["orange", "magenta", "blue", "green", "red"]
        dt = 1e-2
        time_delay = 10
        delay_dims = 3

    elif dataset_name in ["DampedSHO", "SHO"]:
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)
        long_data_lst = [
            full_xyz_data[:1000, 0],
            recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        if dataset_name == "DampedSHO":
            name_lst = ["Ground Truth", "Teacher-Forced", "Autonomous"]
        else:
            name_lst = ["Ground\nTruth", "Teacher-\nForced", "Autonomous"]

        key_lst = ["gt", "tf", "auto"]
        true_signal_index = 0
        colors_lst = ["blue", "green", "red"]
        dt = 1e-2
        time_delay = 5
        delay_dims = 3

    elif dataset_name == "Xhro":
        full_xyz_data = test_dataloader.dataset.get_full_xyz(i)[
            ["ch1", "ch2", "ch3", "ch4"]
        ].to_numpy()
        data_slice = full_xyz_data[:10000]
        interpolated_data = (
            pd.DataFrame(data_slice, columns=["ch1", "ch2", "ch3", "ch4"])
            .interpolate(method="linear")
            .values
        )
        long_data_lst = [
            interpolated_data[:, 0],
            interpolated_data[:, 1],
            interpolated_data[:, 2],
            interpolated_data[:, 3],
            recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        name_lst = [
            "ch1", "ch2", "ch3", "Ground\nTruth", "Teacher-\nForced", "Autonomous"
        ]
        key_lst = ["ch1", "ch2", "ch3", "gt", "tf", "auto"]
        true_signal_index = 3
        colors_lst = ["cyan", "orange", "magenta", "blue", "green", "red"]
        dt = 1 / 250.0
        time_delay = 5
        delay_dims = 3

    else:
        # Fallback
        gt_signal = batch_data_long[:, 0, 0].cpu().numpy()
        long_data_lst = [
            gt_signal,
            recon_data_long[~autonomous_mode_selector_long.bool(), 0, 0] if recon_data_long.ndim > 2 else recon_data_long[~autonomous_mode_selector_long.bool()],
            recon_data_long[autonomous_mode_selector_long.bool(), 0, 0] if recon_data_long.ndim > 2 else recon_data_long[autonomous_mode_selector_long.bool()],
        ]
        name_lst = ["Ground\nTruth", "Teacher-\nForced", "Autonomous"]
        key_lst = ["gt", "tf", "auto"]
        true_signal_index = 0
        colors_lst = ["blue", "green", "red"]
        dt = 1e-2
        time_delay = 5
        delay_dims = 3

    return {
        "long_data_lst": long_data_lst,
        "name_lst": name_lst,
        "key_lst": key_lst,
        "true_signal_index": true_signal_index,
        "colors_lst": colors_lst,
        "dt": dt,
        "time_delay": time_delay,
        "delay_dims": delay_dims
    }
