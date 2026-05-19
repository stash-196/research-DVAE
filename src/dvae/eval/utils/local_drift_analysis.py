import numpy as np
import torch


def compute_local_drift_statistics(
    dvae, batch_data, flip_point, auto_len=25, device="cpu"
):
    """
    Compute drift statistics by forking the hidden state at flip_point
    and running parallel teacher-forced (TF) and autonomous (Auto) rollouts.

    This validates the implicit regularizer theory: we measure how quickly
    the autonomous trajectory deviates from the TF trajectory in the *early phase*
    (first ~30 steps) when a first-order Taylor approximation still holds.

    Args:
        dvae: DVAE model (must be in eval mode).
        batch_data: Tensor of shape (seq_len, batch_size, x_dim).
        flip_point: Integer time index where to fork from TF to Auto.
        auto_len: Length of the rollout after flip_point (typically 20-40 steps).
        device: Device on which to compute (e.g., 'cpu' or 'cuda').

    Returns:
        dict with keys:
            - 'd_norm': Mean ||d_k||^2 over all steps and samples.
            - 'cross_term': Mean d_k^T e_k over all steps and samples.
            - 'delta_mse': Mean (||d_k||^2 + 2 d_k^T e_k) — direct MSE change.
            - 'per_step_d_norm': Array of per-step drift magnitudes (shape: [auto_len, batch_size]).
            - 'per_step_cross': Array of per-step cross terms.
            - 'per_step_delta_mse': Array of per-step MSE changes.

    Mathematical background:
        At fork_point, we split into two trajectories from the same hidden state h_fork:

        TF trajectory:
            h_{k+1}^TF = dvae_step(h_k^TF, x_{t0+k})
            y_k^TF = dvae_readout(h_k^TF)

        Auto trajectory (from same h_fork):
            h_{k+1}^A = dvae_step(h_k^A, y_k^A)
            y_k^A = dvae_readout(h_k^A)

        Drift: d_k = y_k^A - y_k^TF
        Base error: e_k = y_k^TF - x_{t0+k}

        Then:
            MSE_auto_k = ||y_k^A - x_{t0+k}||^2
                       = ||(d_k + e_k)||^2
                       = ||d_k||^2 + 2 d_k^T e_k + ||e_k||^2

            ΔMSE_k = MSE_auto_k - MSE_tf_k = ||d_k||^2 + 2 d_k^T e_k

        - If d_k^T e_k < 0 (large magnitude), Auto is correcting TF errors → beneficial.
        - If ||d_k||^2 is small but d_k^T e_k > 0, Auto amplifies errors → harmful.
        - Early-phase behavior (k ~ 5-15) reveals whether the model's implicit
          regularization encourages stable autonomous dynamics.
    """
    seq_len, batch_size, x_dim = batch_data.shape

    # Validate inputs
    if flip_point < 0 or flip_point >= seq_len:
        raise ValueError(f"flip_point {flip_point} out of range [0, {seq_len})")
    if flip_point + auto_len > seq_len:
        print(
            f"[Drift] Warning: flip_point + auto_len ({flip_point + auto_len}) "
            f"exceeds seq_len ({seq_len}); truncating to {seq_len}"
        )
        auto_len = seq_len - flip_point

    # ========================================================================
    # Step 1: Run forward pass with pure teacher-forced (TF) mode
    # ========================================================================
    mode_selector_tf = torch.zeros(
        (seq_len, batch_size, x_dim), device=device, dtype=torch.float32
    )

    with torch.no_grad():
        recon_tf = dvae(
            batch_data.clone().to(device),
            mode_selector=mode_selector_tf,
            inference_mode=True,
        )

    # ========================================================================
    # Step 2: Run forward pass with mixed mode: TF up to flip_point, Auto after
    # ========================================================================
    mode_selector_auto = torch.zeros(
        (seq_len, batch_size, x_dim), device=device, dtype=torch.float32
    )
    # Set Auto mode (1.0) from flip_point onwards
    mode_selector_auto[flip_point:, :, :] = 1.0

    with torch.no_grad():
        recon_auto = dvae(
            batch_data.clone().to(device),
            mode_selector=mode_selector_auto,
            inference_mode=True,
        )

    # ========================================================================
    # Step 3: Compute drift metrics for steps [flip_point, flip_point + auto_len)
    # ========================================================================
    per_step_d_norm = []
    per_step_cross = []
    per_step_delta_mse = []

    for k in range(auto_len):
        t = flip_point + k

        # Extract predictions and ground truth at this step
        pred_auto = recon_auto[t, :, :]  # (batch_size, x_dim)
        pred_tf = recon_tf[t, :, :]  # (batch_size, x_dim)
        true_x = batch_data[t, :, :].to(device)  # (batch_size, x_dim)

        # Drift: d_k = pred_auto - pred_tf
        d_k = pred_auto - pred_tf  # (batch_size, x_dim)

        # Base error: e_k = pred_tf - true_x
        e_k = pred_tf - true_x  # (batch_size, x_dim)

        # Compute ||d_k||^2 per sample: sum over dimensions
        d_norm_sq = (d_k**2).sum(dim=1)  # (batch_size,)

        # Compute d_k^T e_k per sample: dot product over dimensions
        cross_k = (d_k * e_k).sum(dim=1)  # (batch_size,)

        # ΔMSE = ||d_k||^2 + 2 * d_k^T e_k
        delta_mse_k = d_norm_sq + 2 * cross_k  # (batch_size,)

        per_step_d_norm.append(d_norm_sq.cpu().numpy())
        per_step_cross.append(cross_k.cpu().numpy())
        per_step_delta_mse.append(delta_mse_k.cpu().numpy())

    # Stack: shape (auto_len, batch_size)
    per_step_d_norm = np.stack(per_step_d_norm, axis=0)
    per_step_cross = np.stack(per_step_cross, axis=0)
    per_step_delta_mse = np.stack(per_step_delta_mse, axis=0)

    # ========================================================================
    # Step 4: Aggregate statistics
    # ========================================================================
    avg_d_norm = float(np.mean(per_step_d_norm))
    avg_cross_term = float(np.mean(per_step_cross))
    avg_delta_mse = float(np.mean(per_step_delta_mse))

    return {
        "d_norm": avg_d_norm,
        "cross_term": avg_cross_term,
        "delta_mse": avg_delta_mse,
        "per_step_d_norm": per_step_d_norm,  # (auto_len, batch_size)
        "per_step_cross": per_step_cross,  # (auto_len, batch_size)
        "per_step_delta_mse": per_step_delta_mse,  # (auto_len, batch_size)
        "flip_point": flip_point,
        "auto_len": auto_len,
    }
