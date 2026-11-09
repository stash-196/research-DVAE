#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit SINDy on Lorenz63 and score autonomous rollouts.

Uses the Lorenz63 dataset class for masks and observation slices, then the
metrics named in the OTF eval summary (mse_auto, kld_auto,
spectrum_error_auto). Does not call ``eval_signal`` or ``LearningAlgorithm``.

Missing-data strategy (see ``dvae.baselines.sindy``):

* ``observed_segments`` — fit only on contiguous fully observed rows
* ``interpolate`` — linear fill, then fit (optimistic across long gaps)

``only_x`` (and any observation that drops a Lorenz coordinate) is fit in
the remaining coordinates. That model cannot represent the 3-D attractor.
The written ``limits.txt`` states this explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dvae.baselines.gru_d_runtime import _dump_yaml, resolve_io_paths
from dvae.baselines.plotting import plot_coefficients, plot_metric_bars, plot_trajectories
from dvae.baselines.sindy import fit_sindy, score_rollouts
from dvae.dataset.lorenz63_dataset import Lorenz63, lorenz63_resolved_x_dim
from dvae.utils.path_manager import find_project_root


def _to_numpy(seq):
    if hasattr(seq, "detach"):
        seq = seq.detach().cpu().numpy()
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim == 1:
        seq = seq[:, None]
    return seq


def _load_split(data_dir, split, seq_len, observation_process, mask_label, dataset_label, x_dim):
    width = lorenz63_resolved_x_dim(observation_process, x_dim)
    return Lorenz63(
        data_dir=data_dir,
        dataset_label=dataset_label,
        mask_label=mask_label,
        split=split,
        seq_len=seq_len,
        x_dim=width,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.2,
        observation_process=observation_process,
        device="cpu",
        overlap=False,
        with_nan=False,
        shuffle=False,
        eval_mode=False,
    )


def _windows(dataset):
    series = _to_numpy(dataset.seq)
    windows = []
    for start in list(dataset.data_idx):
        stop = start + int(dataset.seq_len)
        if stop <= len(series):
            windows.append(series[start:stop])
    return windows


def _yaml_windows(rows):
    public = []
    for row in rows:
        public.append(
            {
                key: value
                for key, value in row.items()
                if key not in ("gt_auto", "pred_auto", "per_channel")
            }
        )
    return public


def main():
    parser = argparse.ArgumentParser(description="SINDy baseline on Lorenz63")
    parser.add_argument("--data-dir", default=None, help="overrides device_paths / ./data")
    parser.add_argument(
        "--dataset-label",
        default="sigma10_rho28_beta8d3_N108k_dt0.01",
    )
    parser.add_argument("--observation-process", default="all_xyz")
    parser.add_argument("--mask-label", default="None")
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--x-dim", type=int, default=1, help="used for only_x; multi-column slices override it")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--missing-strategy",
        choices=("observed_segments", "interpolate"),
        default="observed_segments",
    )
    parser.add_argument("--backend", choices=("numpy", "pysindy"), default="numpy")
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(find_project_root(__file__))
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Reuse the host lookup. The ini fallback is ./data, which matches
        # {data_dir}/lorenz63/data/<label>/...
        from dvae.utils.read_config import myconf

        cfg = myconf()
        cfg.read_dict({"User": {"data_dir": "./data", "saved_root": "./saved_model"}})
        data_dir, _, _ = resolve_io_paths(cfg, root / "config" / "device_paths.yaml")

    try:
        train_ds = _load_split(
            data_dir,
            "train",
            args.seq_len,
            args.observation_process,
            args.mask_label,
            args.dataset_label,
            args.x_dim,
        )
        test_ds = _load_split(
            data_dir,
            "test",
            args.seq_len,
            args.observation_process,
            args.mask_label,
            args.dataset_label,
            args.x_dim,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Lorenz63 data not found: {exc}") from exc

    train_series = _to_numpy(train_ds.seq)
    fit = fit_sindy(
        train_series,
        dt=args.dt,
        degree=args.degree,
        threshold=args.threshold,
        missing_strategy=args.missing_strategy,
        backend=args.backend,
        min_segment_len=args.min_segment_len,
    )
    windows = _windows(test_ds)[: args.max_windows]
    scores = score_rollouts(fit, windows, dataset_name="Lorenz63")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "equations.txt").write_text(fit.equations() + "\n")
    (output_dir / "limits.txt").write_text(fit.limits + "\n")
    summary = {
        "model_name": "SINDy",
        "dataset_name": "Lorenz63",
        "observation_process": args.observation_process,
        "mask_label": args.mask_label,
        "missing_strategy": fit.missing_strategy,
        "backend": fit.backend,
        "polynomial_degree": fit.degree,
        "threshold": fit.threshold,
        "state_dim": fit.state_dim,
        "partial_observation": fit.partial_observation,
        "n_segments": fit.n_segments,
        "n_rows": fit.n_rows,
        "n_dropped": fit.n_dropped,
        "dt": fit.dt,
        "mse_auto": scores["mse_auto"],
        "mse_auto_mean": scores["mse_auto_mean"],
        "mse_auto_short": scores["mse_auto_short"],
        "kld_auto": scores["kld_auto"],
        "kld_auto_mean": scores["kld_auto_mean"],
        "spectrum_error_auto": scores["spectrum_error_auto"],
        "spectrum_error_auto_mean": scores["spectrum_error_auto_mean"],
        "n_windows": scores["n_windows"],
        "n_diverged": scores["n_diverged"],
        "feature_names": fit.feature_names,
        "coefficients": fit.coefficients,
        "limits": fit.limits,
        "windows": _yaml_windows(scores["windows"]),
        "notes": (
            "Fit on the full training trajectory (not the RNN windows). "
            "Scored on test windows with a half-series free run, the same "
            "flip as mode half_half. mse_auto is the full autonomous tail "
            "and grows under chaos even when the vector field is right; "
            "mse_auto_short is the first 20 forecast steps. kld_auto and "
            "spectrum_error_auto match the OTF aggregate metrics."
        ),
    }
    _dump_yaml(output_dir / "evaluation_summary.yaml", summary)
    plot_metric_bars(
        summary,
        ["mse_auto", "mse_auto_short", "kld_auto", "spectrum_error_auto"],
        str(output_dir / "metrics.png"),
    )
    plot_coefficients(
        fit.feature_names,
        fit.coefficients,
        str(output_dir / "coefficients.png"),
    )
    plotted = False
    for row in scores["windows"]:
        if row.get("gt_auto") is None:
            continue
        plot_trajectories(
            row["gt_auto"],
            row["pred_auto"],
            str(output_dir / "rollout_window0.png"),
            channel_names=["x", "y", "z"][: fit.state_dim],
            pred_label="SINDy",
        )
        plotted = True
        break
    print(fit.equations())
    print(fit.limits)
    print(
        f"[SINDy] windows={scores['n_windows']} diverged={scores['n_diverged']} "
        f"mse_auto={scores['mse_auto']} mse_auto_short={scores['mse_auto_short']} "
        f"kld_auto={scores['kld_auto']} spectrum_error_auto={scores['spectrum_error_auto']} "
        f"plot={plotted}"
    )


if __name__ == "__main__":
    main()
