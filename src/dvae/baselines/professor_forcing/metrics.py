"""Autonomous-rollout metrics using the repo's eval helpers.

``mse_tf`` / ``mse_auto`` use ``calc_masked_mse``. Spectrum error uses
``frequency_analysis._spectrum_pair_errors`` (Hellinger distance on smoothed
power spectra). State-space KL uses delay embedding then
``state_space_kl(..., use_gmm=True)``. The autonomous interval is the
``half_half`` schedule from ``create_autonomous_mode_selector_1d``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dvae.baselines.professor_forcing.generator import ProfessorForcingGenerator
from dvae.baselines.professor_forcing.losses import indicate_process
from dvae.eval.utils.benchmark_signals import resolve_channel_keys
from dvae.eval.utils.delay_embedding import compute_delay_embedding
from dvae.eval.utils.delay_params import _get_delay_params
from dvae.eval.utils.durstewitz_eval_metrics import state_space_kl
from dvae.eval.utils.frequency_analysis import _spectrum_pair_errors
from dvae.eval.utils.run_mse_analysis import calc_masked_mse
from dvae.utils.model_mode_selector import create_autonomous_mode_selector_1d

LORENZ_DT = 1e-2


def half_half_auto_mask(seq_len: int) -> np.ndarray:
    selector = create_autonomous_mode_selector_1d(seq_len, mode="half_half")
    return np.asarray(selector) > 0.5


def sampling_rate_for(dataset_name: str) -> float:
    if dataset_name == "Lorenz63":
        return 1.0 / LORENZ_DT
    return 1.0 / LORENZ_DT


@torch.no_grad()
def rollout_pair(
    generator: ProfessorForcingGenerator,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced emission and a half_half emission. Feedback is the mean.

    Evaluation always feeds the conditional mean, so ``mse_auto`` is a
    deterministic closed loop even if training sampled Gaussian feedback.
    """
    teacher = generator.rollout(x, mode="teacher", feedback="mean")
    selector = create_autonomous_mode_selector_1d(x.shape[0], mode="half_half")
    selector_t = torch.as_tensor(selector, device=x.device, dtype=x.dtype)
    mixed = generator.rollout(x, mode_selector=selector_t, feedback="mean")
    return teacher.emission, mixed.emission


def score_batch(
    gt: np.ndarray,
    pred_tf: np.ndarray,
    pred_half: np.ndarray,
    observation_process: str,
    dataset_name: str = "Lorenz63",
) -> dict[str, Any]:
    """Average per-sequence metrics. Arrays are ``(T, B, D)`` or ``(T, D)``."""
    gt = np.asarray(gt, dtype=np.float64)
    pred_tf = np.asarray(pred_tf, dtype=np.float64)
    pred_half = np.asarray(pred_half, dtype=np.float64)
    if gt.ndim == 2:
        gt = gt[:, None, :]
        pred_tf = pred_tf[:, None, :]
        pred_half = pred_half[:, None, :]
    rows = [
        score_sequence(
            gt[:, b],
            pred_tf[:, b],
            pred_half[:, b],
            observation_process,
            dataset_name,
        )
        for b in range(gt.shape[1])
    ]
    return _average_metric_rows(rows)


def score_sequence(
    gt: np.ndarray,
    pred_tf: np.ndarray,
    pred_half: np.ndarray,
    observation_process: str,
    dataset_name: str = "Lorenz63",
) -> dict[str, Any]:
    gt = np.asarray(gt, dtype=np.float64)
    pred_tf = np.asarray(pred_tf, dtype=np.float64)
    pred_half = np.asarray(pred_half, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt[:, None]
        pred_tf = pred_tf[:, None]
        pred_half = pred_half[:, None]
    seq_len, x_dim = gt.shape
    auto = half_half_auto_mask(seq_len)
    specs = resolve_channel_keys(observation_process or "", x_dim, dataset_name)
    tau, embed_dim = _get_delay_params(dataset_name)
    rate = sampling_rate_for(dataset_name)

    per_channel: dict[str, dict[str, float]] = {}
    for key, dim in specs:
        if dim >= x_dim:
            continue
        mask = _missing_mask(gt, dim, observation_process)
        mse_tf = calc_masked_mse(gt[:, dim], pred_tf[:, dim], mask)
        mask_auto = None if mask is None else mask[auto]
        mse_auto = calc_masked_mse(gt[auto, dim], pred_half[auto, dim], mask_auto)
        gt_auto = gt[auto, dim]
        tf_auto = pred_tf[auto, dim]
        auto_seg = pred_half[auto, dim]
        per_channel[key] = {
            "mse_tf": mse_tf,
            "mse_auto": mse_auto,
            "spectrum_error_tf": _safe_spectrum(gt_auto, tf_auto, rate),
            "spectrum_error_auto": _safe_spectrum(gt_auto, auto_seg, rate),
            "kld_tf": _safe_kld(gt_auto, tf_auto, tau, embed_dim),
            "kld_auto": _safe_kld(gt_auto, auto_seg, tau, embed_dim),
        }

    summary = _mean_channels(per_channel)
    summary["per_channel"] = per_channel
    summary["time_delay"] = int(tau)
    summary["delay_dims"] = int(embed_dim)
    summary["n_auto_steps"] = int(np.sum(auto))
    return summary


def _missing_mask(gt: np.ndarray, dim: int, observation_process: str | None) -> np.ndarray | None:
    """Boolean mask, True where the eval should ignore the step.

    Raw missing values are already non-finite, and ``calc_masked_mse`` drops
    those. Indicate series are zero-filled, so the paired indicator column is
    the missingness bit the eval mask would have recorded.
    """
    if not indicate_process(observation_process):
        return None
    mask_dim = dim + 1
    if mask_dim >= gt.shape[1]:
        return None
    return np.asarray(gt[:, mask_dim] < 0.5)


def _safe_spectrum(gt: np.ndarray, pred: np.ndarray, sampling_rate: float) -> float:
    try:
        return float(_spectrum_pair_errors(gt, pred, sampling_rate))
    except Exception:
        return float("nan")


def _safe_kld(gt: np.ndarray, pred: np.ndarray, tau: int, embed_dim: int) -> float:
    try:
        gt_emb = compute_delay_embedding(np.asarray(gt, dtype=np.float64), tau, embed_dim)
        pred_emb = compute_delay_embedding(np.asarray(pred, dtype=np.float64), tau, embed_dim)
    except Exception:
        return float("nan")
    if gt_emb is None or pred_emb is None or len(gt_emb) < 10 or len(pred_emb) < 10:
        return float("nan")
    try:
        return float(state_space_kl(gt_emb, pred_emb, use_gmm=True))
    except Exception:
        return float("nan")


def _mean_channels(per_channel: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in (
        "mse_tf",
        "mse_auto",
        "spectrum_error_tf",
        "spectrum_error_auto",
        "kld_tf",
        "kld_auto",
    ):
        values = [
            float(channel[metric])
            for channel in per_channel.values()
            if np.isfinite(channel[metric])
        ]
        mean = float(np.mean(values)) if values else float("nan")
        out[metric] = mean
        out[f"{metric}_mean"] = mean
    return out


def _average_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {
        "time_delay": rows[0]["time_delay"],
        "delay_dims": rows[0]["delay_dims"],
        "n_auto_steps": rows[0]["n_auto_steps"],
    }
    keys = [key for key in rows[0] if key not in ("per_channel", "time_delay", "delay_dims", "n_auto_steps")]
    for key in keys:
        values = [float(row[key]) for row in rows if np.isfinite(row[key])]
        summary[key] = float(np.mean(values)) if values else float("nan")
    channel_keys = rows[0]["per_channel"].keys()
    per_channel = {}
    for channel in channel_keys:
        per_channel[channel] = {}
        for metric in rows[0]["per_channel"][channel]:
            values = [
                float(row["per_channel"][channel][metric])
                for row in rows
                if np.isfinite(row["per_channel"][channel][metric])
            ]
            per_channel[channel][metric] = float(np.mean(values)) if values else float("nan")
    summary["per_channel"] = per_channel
    return summary
