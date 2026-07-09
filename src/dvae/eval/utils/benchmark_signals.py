"""Build per-channel ground-truth and reconstruction signals for evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dvae.dataset.xhro_dataset import select_columns_for_obs_conditions

CHANNEL_COLORS = {
    "ch1": "cyan",
    "ch2": "orange",
    "ch3": "magenta",
    "ch4": "yellow",
    "x": "blue",
    "y": "magenta",
    "z": "orange",
    "gt": "blue",
}


def is_indicate_observation(observation_process: Optional[str]) -> bool:
    if observation_process is None:
        return False
    if observation_process == "only_x_indicate":
        return True
    return isinstance(observation_process, str) and observation_process.endswith(
        "_indicate"
    )


def resolve_channel_keys(
    observation_process: str,
    x_dim: int,
    dataset_name: str,
) -> List[Tuple[str, int]]:
    """
    Return (key, dim_index) pairs for signal channels to score (excludes mask channel).
    """
    if is_indicate_observation(observation_process):
        return [("signal", 0)]

    if observation_process in select_columns_for_obs_conditions.get("original", {}):
        cols = select_columns_for_obs_conditions["original"][observation_process]
        return [(col, i) for i, col in enumerate(cols)]

    if dataset_name == "Lorenz63":
        if x_dim == 1:
            return [("x", 0)]
        return [(f"dim{d}", d) for d in range(x_dim)]

    if dataset_name in ("Xhro", "XhroPacketLoss") and x_dim == 4:
        return [(f"ch{i + 1}", i) for i in range(4)]

    return [(f"dim{d}", d) for d in range(x_dim)]


def _get_dt(dataset, dataset_name: str) -> float:
    if hasattr(dataset, "sampling_freq") and dataset.sampling_freq:
        return 1.0 / float(dataset.sampling_freq)
    if dataset_name in ("Xhro", "XhroPacketLoss"):
        return 1.0 / 250.0
    if dataset_name == "Lorenz63":
        return 1e-2
    return 1e-2


def _get_delay_params(dataset_name: str) -> Tuple[int, int]:
    if dataset_name == "Lorenz63":
        return 10, 3
    return 5, 3


def _align_missing_mask(
    dataset, batch_idx: int, seq_len: int, n_channels: int
) -> Optional[np.ndarray]:
    if not hasattr(dataset, "get_missing_mask"):
        return None
    try:
        mask = dataset.get_missing_mask(batch_idx)
    except Exception:
        return None

    mask = np.asarray(mask)
    if mask.ndim == 1:
        if mask.shape[0] < seq_len:
            return None
        mask = mask[:seq_len]
        if n_channels == 1:
            return mask.astype(bool)
        return np.tile(mask[:, None], (1, n_channels))

    if mask.ndim == 2:
        if mask.shape[0] < seq_len:
            return None
        mask = mask[:seq_len]
        if mask.shape[1] >= n_channels:
            return mask[:, :n_channels].astype(bool)
        return None

    return None


def get_channel_benchmarks(
    batch_data_long: torch.Tensor,
    recon_tf: np.ndarray,
    recon_auto_warmed: np.ndarray,
    flip_point: int,
    dataset,
    batch_idx: int,
    observation_process: str,
    dataset_name: str,
    auto_mode: str = "half_half",
) -> Dict[str, Any]:
    """
    Build per-channel GT / TF / Auto signals for metric computation.

    Uses normalized batch data (training distribution). Auto signals are the
    contiguous tail starting at flip_point from a single warmed forward pass.
    """
    seq_len = batch_data_long.shape[0]
    flip_point = int(max(0, min(flip_point, seq_len)))
    x_dim = batch_data_long.shape[2]

    gt_np = batch_data_long[:, 0, :].detach().cpu().numpy()
    if recon_tf.ndim == 3:
        tf_np = recon_tf[:, 0, :]
    else:
        tf_np = recon_tf
    if recon_auto_warmed.ndim == 3:
        auto_np = recon_auto_warmed[:, 0, :]
    else:
        auto_np = recon_auto_warmed

    channel_specs = resolve_channel_keys(observation_process, x_dim, dataset_name)
    missing_mask_full = _align_missing_mask(
        dataset, batch_idx, seq_len, len(channel_specs)
    )

    channels: List[Dict[str, Any]] = []
    for col_i, (key, dim_idx) in enumerate(channel_specs):
        if dim_idx >= x_dim:
            continue

        gt_full = gt_np[:, dim_idx]
        tf_full = tf_np[:, dim_idx]
        auto_full = auto_np[:, dim_idx]

        gt_auto = gt_full[flip_point:]
        tf_auto = tf_full[flip_point:]
        auto_seg = auto_full[flip_point:]

        mask_full = None
        mask_auto = None
        if missing_mask_full is not None and col_i < missing_mask_full.shape[1]:
            mask_full = missing_mask_full[:, col_i]
            mask_auto = mask_full[flip_point:]

        channels.append(
            {
                "key": key,
                "name": key,
                "color": CHANNEL_COLORS.get(key, "blue"),
                "dim_idx": dim_idx,
                "gt_full": gt_full,
                "tf_full": tf_full,
                "auto_full": auto_full,
                "gt_auto": gt_auto,
                "tf_auto": tf_auto,
                "auto_seg": auto_seg,
                "mask_full": mask_full,
                "mask_auto": mask_auto,
            }
        )

    dt = _get_dt(dataset, dataset_name)
    time_delay, delay_dims = _get_delay_params(dataset_name)

    return {
        "channels": channels,
        "flip_point": flip_point,
        "auto_mode": auto_mode,
        "seq_len": seq_len,
        "dt": dt,
        "time_delay": time_delay,
        "delay_dims": delay_dims,
        "is_multidim": len(channels) > 1,
        "dataset_name": dataset_name,
        "observation_process": observation_process,
    }


def get_benchmark_signals(
    dataset_name,
    test_dataloader,
    i,
    recon_data_long,
    autonomous_mode_selector_long,
    batch_data_long=None,
):
    """
    Legacy wrapper for 1D-style flat signal lists.

    Prefer get_channel_benchmarks() for new evaluation code. This path remains
    for backward compatibility when only a mixed-mode reconstruction is available.
    """
    if batch_data_long is None:
        raise ValueError("batch_data_long is required for benchmark signal extraction.")

    seq_len = batch_data_long.shape[0]
    flip_point = seq_len // 2

    if isinstance(recon_data_long, np.ndarray):
        recon_arr = recon_data_long
    else:
        recon_arr = np.asarray(recon_data_long)

    if recon_arr.ndim == 3:
        recon_tf = recon_arr
        recon_auto = recon_arr
    else:
        recon_tf = recon_arr
        recon_auto = recon_arr

    observation_process = getattr(test_dataloader.dataset, "observation_process", None)
    benchmarks = get_channel_benchmarks(
        batch_data_long=batch_data_long,
        recon_tf=recon_tf,
        recon_auto_warmed=recon_auto,
        flip_point=flip_point,
        dataset=test_dataloader.dataset,
        batch_idx=i,
        observation_process=observation_process or "",
        dataset_name=dataset_name,
        auto_mode="legacy_mixed",
    )

    long_data_lst = []
    name_lst = []
    key_lst = []
    colors_lst = []

    for ch in benchmarks["channels"]:
        long_data_lst.extend([ch["gt_auto"], ch["tf_auto"], ch["auto_seg"]])
        name_lst.extend(
            [
                f"{ch['name']}\nGT",
                f"{ch['name']}\nTF",
                f"{ch['name']}\nAuto",
            ]
        )
        key_lst.extend([f"{ch['key']}_gt", f"{ch['key']}_tf", f"{ch['key']}_auto"])
        colors_lst.extend([ch["color"], "green", "red"])

    true_signal_index = 0 if long_data_lst else 0

    return {
        "long_data_lst": long_data_lst,
        "name_lst": name_lst,
        "key_lst": key_lst,
        "true_signal_index": true_signal_index,
        "colors_lst": colors_lst,
        "dt": benchmarks["dt"],
        "time_delay": benchmarks["time_delay"],
        "delay_dims": benchmarks["delay_dims"],
        "channel_benchmarks": benchmarks,
    }