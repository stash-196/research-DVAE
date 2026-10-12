"""Autonomous-rollout scores aligned with the OTF eval summary.

The scalar names match what ``eval_signal`` writes into
``evaluation_summary.yaml`` and what aggregate / compare plots read:

* ``mse_auto`` / ``mse_auto_mean`` — masked MSE (``calc_masked_mse``)
* ``spectrum_error_auto`` / ``spectrum_error_auto_mean`` — Hellinger
  distance of smoothed power spectra (``power_spectrum_error``)
* ``kld_auto`` / ``kld_auto_mean`` — GMM KL on delay embeddings
  (Lorenz63 uses tau=10, m=3; other datasets tau=5, m=3)

Teacher-forced scores use the same functions with the ``_tf`` suffix.
"""

from __future__ import annotations

import warnings

import numpy as np

from dvae.eval.utils.delay_embedding import compute_delay_embedding
from dvae.eval.utils.delay_params import _get_delay_params
from dvae.eval.utils.durstewitz_eval_metrics import (
    power_spectrum_error,
    state_space_kl,
)

# Same reduction as dvae.eval.utils.run_mse_analysis.calc_masked_mse.
# Imported when that module (and its visualizer stack) is available so the
# number cannot drift; the local copy keeps this baseline importable without
# umap.


def _masked_mse_local(gt, pred, mask=None):
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


def calc_masked_mse(gt, pred, mask=None):
    try:
        from dvae.eval.utils.run_mse_analysis import calc_masked_mse as repo_mse
    except Exception:
        return _masked_mse_local(gt, pred, mask)
    return float(repo_mse(gt, pred, mask))


def _as_2d(traj):
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"trajectory must be (T,) or (T, D), got {arr.shape}")
    return arr


def spectrum_pair_error(gt, pred, sampling_rate):
    """Hellinger spectrum error for one 1-D series.

    Matches ``frequency_analysis._spectrum_pair_errors``: drop non-finite
    samples, one-sided periodogram, then ``power_spectrum_error``.
    """
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gt = gt[np.isfinite(gt)]
    pred = pred[np.isfinite(pred)]
    if len(gt) < 4 or len(pred) < 4:
        return float("nan")

    def _ps(data):
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1.0 / sampling_rate)
        ps = np.abs(fft[freqs > 0]) ** 2
        if not np.any(ps > 0):
            return None
        return ps

    gt_ps = _ps(gt)
    pred_ps = _ps(pred)
    if gt_ps is None or pred_ps is None:
        return float("nan")
    min_len = min(len(gt_ps), len(pred_ps))
    gt_ps = gt_ps[:min_len]
    pred_ps = pred_ps[:min_len]
    with warnings.catch_warnings():
        # Identical spectra make the inner product slightly larger than 1.
        warnings.simplefilter("ignore", RuntimeWarning)
        err = power_spectrum_error(
            pred_ps.reshape(1, -1),
            gt_ps.reshape(1, -1),
            filter_std=3,
        )
    err = float(err)
    # power_spectrum_error adds 1e-10 after normalizing, so two identical
    # spectra produce a Hellinger inner product slightly above 1 and a NaN.
    # That is a perfect match, not a missing score.
    if not np.isfinite(err) and np.allclose(gt_ps, pred_ps):
        return 0.0
    return err


def _delay_kld(gt_1d, pred_1d, time_delay, delay_dims):
    need = (delay_dims - 1) * time_delay + 10
    gt_1d = np.asarray(gt_1d, dtype=np.float64)
    pred_1d = np.asarray(pred_1d, dtype=np.float64)
    if len(gt_1d) < need or len(pred_1d) < need:
        return float("nan")
    try:
        gt_emb = compute_delay_embedding(
            gt_1d, delay=time_delay, dimensions=delay_dims, handle_nan="remove"
        )
        pred_emb = compute_delay_embedding(
            pred_1d, delay=time_delay, dimensions=delay_dims, handle_nan="remove"
        )
    except ValueError:
        return float("nan")
    if len(gt_emb) < 10 or len(pred_emb) < 10:
        return float("nan")
    try:
        return float(state_space_kl(gt_emb, pred_emb, use_gmm=True))
    except Exception:
        return float("nan")


def _nanmean(values):
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def score_trajectory_pair(
    gt,
    pred,
    *,
    dataset_name="Lorenz63",
    dt=0.01,
    prefix="auto",
    channel_names=None,
    short_horizon=20,
):
    """Score one predicted trajectory against ground truth.

    ``gt`` and ``pred`` are ``(T,)`` or ``(T, D)``. NaNs in ``gt`` are
    ignored by the masked MSE (missing measurements). ``prefix`` is ``auto``
    or ``tf`` and is written into the metric names.
    """
    gt = _as_2d(gt)
    pred = _as_2d(pred)
    length = min(len(gt), len(pred))
    gt = gt[:length]
    pred = pred[:length]
    n_state = gt.shape[1]
    if pred.shape[1] != n_state:
        raise ValueError(
            f"pred width {pred.shape[1]} != gt width {n_state}"
        )
    if channel_names is None:
        if n_state == 3:
            channel_names = ["x", "y", "z"]
        else:
            channel_names = [f"ch{i}" for i in range(n_state)]
    time_delay, delay_dims = _get_delay_params(dataset_name)
    sampling_rate = 1.0 / float(dt) if dt else 1.0

    per_channel = {}
    mse_values = []
    spec_values = []
    kld_values = []
    for d, name in enumerate(channel_names):
        mse = calc_masked_mse(gt[:, d], pred[:, d])
        spec = spectrum_pair_error(gt[:, d], pred[:, d], sampling_rate)
        kld = _delay_kld(gt[:, d], pred[:, d], time_delay, delay_dims)
        per_channel[name] = {
            f"mse_{prefix}": mse,
            f"spectrum_error_{prefix}": spec,
            f"kld_{prefix}": kld,
        }
        mse_values.append(mse)
        spec_values.append(spec)
        kld_values.append(kld)

    mse_mean = _nanmean(mse_values)
    spec_mean = _nanmean(spec_values)
    kld_mean = _nanmean(kld_values)
    short = min(int(short_horizon), length)
    short_mse = []
    for d in range(n_state):
        short_mse.append(calc_masked_mse(gt[:short, d], pred[:short, d]))

    out = {
        f"mse_{prefix}": mse_mean,
        f"mse_{prefix}_mean": mse_mean,
        f"spectrum_error_{prefix}": spec_mean,
        f"spectrum_error_{prefix}_mean": spec_mean,
        f"kld_{prefix}": kld_mean,
        f"kld_{prefix}_mean": kld_mean,
        f"mse_{prefix}_short": _nanmean(short_mse),
        "per_channel": per_channel,
        "time_delay": int(time_delay),
        "delay_dims": int(delay_dims),
        "dt": float(dt),
        "n_steps": int(length),
        "short_horizon": int(short),
    }
    for name, scores in per_channel.items():
        for key, value in scores.items():
            out[f"{key}_{name}"] = value
    return out
