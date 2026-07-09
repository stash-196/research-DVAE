import numpy as np
import torch

from dvae.eval.utils.forward_modes import run_forward_with_mode


def compute_local_drift_statistics(
    dvae,
    batch_data,
    flip_point,
    auto_len=25,
    device="cpu",
    observation_process=None,
):
    """
    Compute drift statistics by forking from TF to Auto at flip_point.

    Uses create_autonomous_mode_selector schedules (all_0 and flip_at_index)
    for modular consistency with training and evaluation.
    """
    seq_len, batch_size, x_dim = batch_data.shape
    flip_point = int(flip_point)

    if flip_point < 0 or flip_point >= seq_len:
        raise ValueError(f"flip_point {flip_point} out of range [0, {seq_len})")
    if flip_point + auto_len > seq_len:
        print(
            f"[Drift] Warning: flip_point + auto_len ({flip_point + auto_len}) "
            f"exceeds seq_len ({seq_len}); truncating to {seq_len}"
        )
        auto_len = seq_len - flip_point

    batch_on_device = batch_data.clone().to(device)

    recon_tf, _ = run_forward_with_mode(
        dvae,
        batch_on_device,
        mode="all_0",
        observation_process=observation_process,
    )
    recon_auto, _ = run_forward_with_mode(
        dvae,
        batch_on_device,
        mode="flip_at_index",
        observation_process=observation_process,
        flip_point=flip_point,
    )

    per_step_d_norm = []
    per_step_cross = []
    per_step_delta_mse = []
    per_channel_d_norm = {d: [] for d in range(x_dim)}
    per_channel_cross = {d: [] for d in range(x_dim)}

    for k in range(auto_len):
        t = flip_point + k
        pred_auto = recon_auto[t, :, :]
        pred_tf = recon_tf[t, :, :]
        true_x = batch_on_device[t, :, :]

        d_k = pred_auto - pred_tf
        e_k = pred_tf - true_x

        d_norm_sq = (d_k**2).sum(dim=1)
        cross_k = (d_k * e_k).sum(dim=1)
        delta_mse_k = d_norm_sq + 2 * cross_k

        per_step_d_norm.append(d_norm_sq.cpu().numpy())
        per_step_cross.append(cross_k.cpu().numpy())
        per_step_delta_mse.append(delta_mse_k.cpu().numpy())

        for d in range(x_dim):
            d_d = d_k[:, d]
            e_d = e_k[:, d]
            per_channel_d_norm[d].append((d_d**2).cpu().numpy())
            per_channel_cross[d].append((d_d * e_d).cpu().numpy())

    per_step_d_norm = np.stack(per_step_d_norm, axis=0)
    per_step_cross = np.stack(per_step_cross, axis=0)
    per_step_delta_mse = np.stack(per_step_delta_mse, axis=0)

    per_channel_stats = {}
    for d in range(x_dim):
        d_stack = np.stack(per_channel_d_norm[d], axis=0)
        c_stack = np.stack(per_channel_cross[d], axis=0)
        per_channel_stats[f"dim{d}"] = {
            "d_norm": float(np.mean(d_stack)),
            "cross_term": float(np.mean(c_stack)),
        }

    return {
        "d_norm": float(np.mean(per_step_d_norm)),
        "cross_term": float(np.mean(per_step_cross)),
        "delta_mse": float(np.mean(per_step_delta_mse)),
        "per_step_d_norm": per_step_d_norm,
        "per_step_cross": per_step_cross,
        "per_step_delta_mse": per_step_delta_mse,
        "per_channel": per_channel_stats,
        "flip_point": flip_point,
        "auto_len": auto_len,
    }