"""Score baseline trajectories with the repo's imputation and free-run metrics.

The numbers come from the same helpers ``eval_signal`` uses:

* ``calc_masked_mse`` / ``run_mse_analysis_from_benchmarks``
* ``run_spectrum_analysis_from_benchmarks`` (Hellinger distance on smoothed
  power spectra)
* ``run_geometry_analysis_from_benchmarks`` (delay-embedding GMM KLD, plus the
  intrinsic-dimension fields that helper already returns)
* ``get_channel_benchmarks`` for the half-and-half split and channel names

MSE follows that code: missing positions (mask True) are excluded, and so are
non-finite ground-truth samples. That is reconstruction error on observed
samples, which is what OTF runs report as ``mse_tf`` / ``mse_auto``. Imputation
error on the missing positions is reported separately as ``mse_impute_*``
because the shared helper deliberately does not score those positions.

``predicted`` (one-step mean under causal updates) is placed in the teacher-forced
slot. ``autonomous`` (open loop after the flip) is placed in the free-run slot.
Both are scored on the free-run timesteps, matching ``eval_signal``.

EMD and Rosenstein Lyapunov live on the open free-run-metrics pull request and
are not imported here. Jacobian Lyapunov needs a torch module inside the eval
loop, which this package does not modify. Filtered and smoothed series are
written to the result archive so those hooks can be pointed at them later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray


def half_half_auto_mask(t_len: int, flip_point: Optional[int] = None):
    if flip_point is None:
        flip_point = t_len // 2
    flip_point = int(max(0, min(flip_point, t_len)))
    auto = np.zeros(t_len, dtype=bool)
    auto[flip_point:] = True
    return auto, flip_point


def mse_on_missing(gt: Array, pred: Array, missing: Array) -> float:
    """Mean squared error on positions the mask marks missing."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if gt.shape != pred.shape or gt.shape != missing.shape:
        raise ValueError("gt, pred, and missing must share a shape")
    valid = missing & np.isfinite(gt) & np.isfinite(pred)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((gt[valid] - pred[valid]) ** 2))


class _MaskDataset:
    """Just enough of a dataset for ``get_channel_benchmarks``."""

    def __init__(self, missing: Array, dt: float) -> None:
        self._missing = np.asarray(missing)
        self.sampling_freq = 1.0 / float(dt) if dt else None

    def get_missing_mask(self, index: int) -> Array:
        del index
        return self._missing


def _as_batch(series: Array) -> "np.ndarray":
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    return series


def build_channel_benchmarks(
    gt: Array,
    pred_tf: Array,
    pred_auto: Array,
    missing: Array,
    *,
    dataset_name: str,
    observation_process: str,
    dt: float,
    flip_point: Optional[int] = None,
    auto_mode: str = "half_half",
    time_delay: Optional[int] = None,
    delay_dims: Optional[int] = None,
):
    import torch

    from dvae.eval.utils.benchmark_signals import get_channel_benchmarks
    from dvae.eval.utils.delay_params import _get_delay_params

    gt = _as_batch(gt)
    pred_tf = _as_batch(pred_tf)
    pred_auto = _as_batch(pred_auto)
    missing = np.asarray(missing, dtype=bool)
    if missing.ndim == 1 and gt.shape[1] == 1:
        missing = missing.reshape(-1, 1)
    t_len = gt.shape[0]
    if auto_mode == "half_half" and flip_point is None:
        _, flip_point = half_half_auto_mask(t_len, flip_point)
    if auto_mode == "all_1":
        flip_point = 0
    if time_delay is None or delay_dims is None:
        default_tau, default_m = _get_delay_params(dataset_name)
        if time_delay is None:
            time_delay = default_tau
        if delay_dims is None:
            delay_dims = default_m
    batch = torch.tensor(gt, dtype=torch.float32).view(t_len, 1, gt.shape[1])
    return get_channel_benchmarks(
        batch_data_long=batch,
        recon_tf=pred_tf.reshape(t_len, 1, gt.shape[1]),
        recon_auto_warmed=pred_auto.reshape(t_len, 1, gt.shape[1]),
        flip_point=flip_point,
        dataset=_MaskDataset(missing, dt),
        batch_idx=0,
        observation_process=observation_process,
        dataset_name=dataset_name,
        auto_mode=auto_mode,
        time_delay=int(time_delay),
        delay_dims=int(delay_dims),
    )


def score_benchmarks(
    benchmarks: Dict[str, Any],
    *,
    compute_geometry: bool = True,
) -> Dict[str, float]:
    from dvae.eval.utils.frequency_analysis import run_spectrum_analysis_from_benchmarks
    from dvae.eval.utils.metrics_aggregate import flatten_analysis_to_batch_metrics
    from dvae.eval.utils.run_mse_analysis import run_mse_analysis_from_benchmarks

    mse = run_mse_analysis_from_benchmarks(
        benchmarks, save_fig_dir=None, batch_idx=0, save_figures=False
    )
    spectrum = run_spectrum_analysis_from_benchmarks(
        benchmarks, save_fig_dir=None, batch_idx=0, save_figures=False
    )
    if compute_geometry:
        from dvae.eval.utils.run_geometry_analysis import (
            run_geometry_analysis_from_benchmarks,
        )

        try:
            geom = run_geometry_analysis_from_benchmarks(
                benchmarks, save_fig_dir=None, batch_idx=0, save_figures=False
            )
        except Exception as exc:
            print(f"[filtering baseline] geometry metrics failed: {exc}")
            geom = {
                "kld_tf": float("nan"),
                "kld_auto": float("nan"),
                "kld_tf_mean": float("nan"),
                "kld_auto_mean": float("nan"),
            }
    else:
        geom = {
            "kld_tf": float("nan"),
            "kld_auto": float("nan"),
            "kld_tf_mean": float("nan"),
            "kld_auto_mean": float("nan"),
        }
    return flatten_analysis_to_batch_metrics(mse, geom, spectrum)


def score_otf_window(
    gt: Array,
    pred_tf: Array,
    pred_auto: Array,
    missing: Array,
    *,
    dataset_name: str,
    observation_process: str,
    dt: float,
    flip_point: Optional[int] = None,
    time_delay: Optional[int] = None,
    delay_dims: Optional[int] = None,
    compute_geometry: bool = True,
    filtered: Optional[Array] = None,
    smoothed: Optional[Array] = None,
) -> Dict[str, Any]:
    """OTF-comparable metrics plus missing-entry imputation MSE."""
    gt_b = _as_batch(gt)
    missing_b = np.asarray(missing, dtype=bool)
    if missing_b.ndim == 1:
        missing_b = np.broadcast_to(missing_b.reshape(-1, 1), gt_b.shape).copy()
    benchmarks = build_channel_benchmarks(
        gt_b,
        pred_tf,
        pred_auto,
        missing_b,
        dataset_name=dataset_name,
        observation_process=observation_process,
        dt=dt,
        flip_point=flip_point,
        auto_mode="half_half",
        time_delay=time_delay,
        delay_dims=delay_dims,
    )
    metrics = score_benchmarks(benchmarks, compute_geometry=compute_geometry)
    metrics["mse_impute_predicted"] = mse_on_missing(gt_b, _as_batch(pred_tf), missing_b)
    if filtered is not None:
        metrics["mse_impute_filtered"] = mse_on_missing(
            gt_b, _as_batch(filtered), missing_b
        )
    if smoothed is not None:
        metrics["mse_impute_smoothed"] = mse_on_missing(
            gt_b, _as_batch(smoothed), missing_b
        )
        smooth_bench = build_channel_benchmarks(
            gt_b,
            smoothed,
            smoothed,
            missing_b,
            dataset_name=dataset_name,
            observation_process=observation_process,
            dt=dt,
            auto_mode="all_1",
            time_delay=time_delay,
            delay_dims=delay_dims,
        )
        smooth_metrics = score_benchmarks(
            smooth_bench, compute_geometry=compute_geometry
        )
        # ``all_1`` puts the whole series in the free-run slot.
        for src, dst in (
            ("mse_auto", "mse_smoothed"),
            ("kld_auto", "kld_smoothed"),
            ("spectrum_error_auto", "spectrum_error_smoothed"),
        ):
            if src in smooth_metrics:
                metrics[dst] = smooth_metrics[src]
    metrics["flip_point"] = int(benchmarks["flip_point"])
    metrics["n_auto_steps"] = int(benchmarks["n_auto_steps"])
    return metrics


def aggregate_windows(window_metrics):
    from dvae.eval.utils.metrics_aggregate import merge_batch_metric_dicts

    return merge_batch_metric_dicts(window_metrics)
