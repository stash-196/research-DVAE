import numpy as np
import torch

from dvae.eval.utils.forward_modes import is_indicate_observation, run_forward_with_mode

# Appendix B names. ``cross`` is the mean of 2 d e, so
# base_error + drift + cross equals the autonomous MSE on the same entries.
RAW_DRIFT_METRICS = ("base_error", "drift", "cross", "auto_total")
VARNORM_DRIFT_METRICS = tuple(name + "_varnorm" for name in RAW_DRIFT_METRICS)
# Prefix mean of ||d||^2 over the first k steps after the flip.
DRIFT_SNAPSHOTS = ("drift_k1", "drift_k5", "drift_k")
DRIFT_METRIC_NAMES = RAW_DRIFT_METRICS + VARNORM_DRIFT_METRICS + DRIFT_SNAPSHOTS
# Plotted every epoch. Snapshots are logged and stored, not extra panels.
DRIFT_PLOT_NAMES = RAW_DRIFT_METRICS + VARNORM_DRIFT_METRICS


def fork_tf_and_auto(
    dvae,
    batch_data,
    flip_point,
    device="cpu",
    observation_process=None,
):
    """Full teacher-forced pass, and the pass that flips to autonomous.

    ``x_tf`` is ``all_0``. ``x_auto`` is teacher-forced until ``flip_point``
    and autonomous after it. Both forwards go through
    ``run_forward_with_mode``, the same schedules as
    ``compute_local_drift_statistics``.
    """
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
        flip_point=int(flip_point),
    )
    return batch_on_device, recon_tf, recon_auto


def observed_entry_mask(values, observation_process=None) -> np.ndarray:
    """True on entries the validation loss would keep.

    Ground-truth NaNs are missing. An indicate observation also drops the
    mask channels (odd dimensions); the loss is computed on the signal only.
    """
    arr = np.asarray(values)
    mask = ~np.isnan(arr)
    if (
        is_indicate_observation(observation_process)
        and arr.ndim >= 1
        and arr.shape[-1] > 1
    ):
        keep = np.ones(arr.shape[-1], dtype=bool)
        keep[1::2] = False
        mask = mask & keep.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return mask


def _as_window(window) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            "Drift windows must have shape (time, channel), "
            f"got {np.asarray(window).shape}."
        )
    return arr


def enforce_drift_span(window_len: int, flip: int, horizon: int) -> None:
    """Flip and horizon are fixed at training start. A short window errors."""
    window_len = int(window_len)
    flip = int(flip)
    horizon = int(horizon)
    if horizon < 5:
        raise ValueError(
            "val_drift_horizon must be an integer >= 5 so drift is logged "
            f"at k=1, 5, and K, got {horizon}."
        )
    if flip < 0 or flip >= window_len:
        raise ValueError(
            f"drift flip {flip} is outside window length {window_len}."
        )
    if flip + horizon > window_len:
        raise ValueError(
            "val_drift_horizon "
            f"{horizon} starting at flip {flip} exceeds window length "
            f"{window_len}."
        )


def x_true_variance(windows, observation_process=None) -> np.ndarray:
    """Population variance of ``x_true`` per channel, over the val windows.

    Observed entries only. A channel with no observed samples is NaN.
    The result is a property of the fixed windows, not of the model.
    """
    columns = None
    width = None
    for window in windows:
        arr = _as_window(window)
        if width is None:
            width = int(arr.shape[1])
            columns = [[] for _ in range(width)]
        elif int(arr.shape[1]) != width:
            raise ValueError(
                "Drift windows do not share a channel count: "
                f"{width} and {arr.shape[1]}."
            )
        mask = observed_entry_mask(arr, observation_process)
        for dim in range(width):
            columns[dim].append(arr[mask[:, dim], dim])
    if width is None:
        raise ValueError("Drift variance needs at least one window.")
    variance = np.full(width, np.nan, dtype=np.float64)
    for dim in range(width):
        column = np.concatenate(columns[dim]) if columns[dim] else np.asarray([])
        if column.size == 0:
            continue
        variance[dim] = float(np.var(column))
    return variance


def _mean_masked(values, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        raise ValueError("drift validation horizon has no observed entries.")
    # np.nan * False is still NaN, so drop unobserved entries before the sum.
    return float(np.sum(np.where(mask, values, 0.0)) / count)


def drift_metrics_from_series(
    truths,
    teacher,
    autonomous,
    flip,
    horizon,
    variance,
    observation_process=None,
) -> dict:
    """Appendix B decomposition on the first ``horizon`` steps after ``flip``.

    ``e = x_tf - x_true`` and ``d = x_auto - x_tf``. Over observed entries
    in that span, ``base_error`` is mean ``e^2``, ``drift`` is mean ``d^2``,
    ``cross`` is mean ``2 d e``, and ``auto_total`` is their sum. That sum
    is the autonomous MSE on the same entries.

    The ``*_varnorm`` values divide each entry by the variance of its
    channel before the mean. ``drift_k1``, ``drift_k5``, and ``drift_k``
    are the same drift reduced on the first 1, 5, and K steps.
    """
    horizon = int(horizon)
    flip = int(flip)
    truths = [_as_window(window) for window in truths]
    teacher = [_as_window(window) for window in teacher]
    autonomous = [_as_window(window) for window in autonomous]
    if not truths or len(teacher) != len(truths) or len(autonomous) != len(truths):
        raise ValueError("Drift truths, teacher, and autonomous lists must match.")
    length = int(truths[0].shape[0])
    width = int(truths[0].shape[1])
    enforce_drift_span(length, flip, horizon)
    scale = np.asarray(variance, dtype=np.float64).reshape(-1)
    if scale.shape != (width,):
        raise ValueError(
            f"x_true variance has shape {scale.shape}, expected ({width},)."
        )
    e2_rows = []
    d2_rows = []
    cross_rows = []
    mask_rows = []
    for truth, tf, auto in zip(truths, teacher, autonomous):
        if truth.shape != (length, width) or tf.shape != truth.shape or auto.shape != truth.shape:
            raise ValueError(
                "Drift series must share the window shape "
                f"{(length, width)}; got {truth.shape}, {tf.shape}, {auto.shape}."
            )
        sl = slice(flip, flip + horizon)
        err = tf[sl] - truth[sl]
        delta = auto[sl] - tf[sl]
        mask = observed_entry_mask(truth[sl], observation_process)
        e2_rows.append(err * err)
        d2_rows.append(delta * delta)
        cross_rows.append(2.0 * delta * err)
        mask_rows.append(mask)
    e2 = np.stack(e2_rows, axis=0)
    d2 = np.stack(d2_rows, axis=0)
    cross = np.stack(cross_rows, axis=0)
    mask = np.stack(mask_rows, axis=0)
    base_error = _mean_masked(e2, mask)
    drift = _mean_masked(d2, mask)
    cross_term = _mean_masked(cross, mask)
    safe = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    usable = mask & np.isfinite(scale) & (scale > 0.0)
    if int(usable.sum()) == 0:
        base_n = drift_n = cross_n = float("nan")
    else:
        weights = np.where(usable, 1.0 / safe.reshape(1, 1, -1), 0.0)
        base_n = _mean_masked(e2 * weights, usable)
        drift_n = _mean_masked(d2 * weights, usable)
        cross_n = _mean_masked(cross * weights, usable)
    return {
        "base_error": base_error,
        "drift": drift,
        "cross": cross_term,
        "auto_total": base_error + drift + cross_term,
        "base_error_varnorm": base_n,
        "drift_varnorm": drift_n,
        "cross_varnorm": cross_n,
        "auto_total_varnorm": base_n + drift_n + cross_n,
        "drift_k1": _mean_masked(d2[:, :1], mask[:, :1]),
        "drift_k5": _mean_masked(d2[:, :5], mask[:, :5]),
        "drift_k": drift,
    }


def score_fixed_validation_windows(
    model,
    windows,
    tf_steps,
    horizon,
    variance,
    observation_process,
    device,
) -> dict:
    """Spectrum error plus the Appendix B drift metrics on one window set.

    The autonomous pass is the spectrum free-run (teacher-forced until
    ``tf_steps``). The teacher-forced pass is the extra full ``all_0``
    rollout. Both use the flip and horizon passed in; nothing here reads
    the live sequence length.
    """
    from dvae.eval.utils.spectrum_validation import free_run_spectrum_error

    if not windows:
        raise ValueError("Drift validation needs the fixed spectrum windows.")
    truths = []
    teacher = []
    autonomous = []
    for window in windows:
        arr = _as_window(window)
        enforce_drift_span(arr.shape[0], tf_steps, horizon)
        batch = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)
        _truth, recon_tf, recon_auto = fork_tf_and_auto(
            model,
            batch,
            flip_point=int(tf_steps),
            device=device,
            observation_process=observation_process,
        )
        truths.append(arr)
        teacher.append(recon_tf[:, 0, :].detach().cpu().numpy())
        autonomous.append(recon_auto[:, 0, :].detach().cpu().numpy())
    metrics = drift_metrics_from_series(
        truths,
        teacher,
        autonomous,
        flip=int(tf_steps),
        horizon=int(horizon),
        variance=variance,
        observation_process=observation_process,
    )
    metrics["spectrum"] = float(
        free_run_spectrum_error(autonomous, truths, int(tf_steps))
    )
    return metrics


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

    batch_on_device, recon_tf, recon_auto = fork_tf_and_auto(
        dvae,
        batch_data,
        flip_point=flip_point,
        device=device,
        observation_process=observation_process,
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