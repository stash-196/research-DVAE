"""Baseline scores use the same metric helpers as eval_signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dvae.baselines.filtering.metrics_adapter import (
    mse_on_missing,
    score_otf_window,
)
from dvae.eval.utils.run_mse_analysis import calc_masked_mse


def test_imputation_mse_is_only_on_missing_entries():
    gt = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 1.0, 0.0, 3.0])
    missing = np.array([False, False, True, False])
    assert mse_on_missing(gt.reshape(-1, 1), pred.reshape(-1, 1), missing.reshape(-1, 1)) == 4.0
    # The shared OTF helper excludes those positions, so this error is 0.
    assert calc_masked_mse(gt, pred, missing) == 0.0


def test_otf_window_reports_shared_metric_keys():
    t_len = 80
    time = np.arange(t_len)
    gt = np.sin(2 * np.pi * time / 20.0).reshape(-1, 1)
    # A pure delay has the same power spectrum, and the shared Hellinger
    # helper can return nan when the overlap is 1 up to roundoff. A second
    # tone keeps both spectra finite and distinct.
    pred_tf = (
        0.85 * np.sin(2 * np.pi * time / 20.0) + 0.15 * np.sin(2 * np.pi * time / 5.0)
    ).reshape(-1, 1)
    pred_auto = np.sin(2 * np.pi * time / 7.0).reshape(-1, 1)
    missing = np.zeros_like(gt, dtype=bool)
    missing[10:14] = True
    filtered = gt.copy()
    smoothed = gt.copy()
    smoothed[missing] = 0.0
    metrics = score_otf_window(
        gt,
        pred_tf,
        pred_auto,
        missing,
        dataset_name="Lorenz63",
        observation_process="only_x",
        dt=0.01,
        time_delay=2,
        delay_dims=3,
        compute_geometry=True,
        filtered=filtered,
        smoothed=smoothed,
    )
    for key in (
        "mse_tf",
        "mse_auto",
        "spectrum_error_tf",
        "spectrum_error_auto",
        "kld_tf",
        "kld_auto",
        "mse_impute_filtered",
        "mse_impute_smoothed",
    ):
        assert key in metrics, key
        assert np.isfinite(metrics[key]), key
    # Teacher-forced trace is a small bias everywhere, including the free-run
    # half. The autonomous trace is exact on the teacher-forced half and
    # biased only after the flip, so its free-run MSE is larger.
    assert metrics["mse_auto"] > metrics["mse_tf"]
    assert metrics["mse_impute_filtered"] == 0.0
    assert metrics["mse_impute_smoothed"] > 0.0


def test_demo_runner_writes_kalman_and_oracle_metrics(tmp_path: Path):
    """End-to-end on a synthetic Lorenz mask. No dataset files, geometry off."""
    from dvae.baselines.filtering.runner import load_bundle_from_cfg, run_method

    cfg = {
        "dataset": "demo",
        "observation_process": "only_x",
        "demo_steps": 120,
        "mask_rate": 0.25,
        "seed": 0,
        "metrics_seq_len": 120,
        "max_windows": 1,
        "em_iters": 5,
        "state_dim": 2,
        "compute_geometry": False,
        "save_plots": True,
        "kalman_init": "ols",
    }
    bundle = load_bundle_from_cfg(cfg)
    summary = run_method("kalman", bundle, cfg, tmp_path / "kalman")
    assert summary["role"] == "baseline"
    assert summary["method"] == "kalman"
    assert (tmp_path / "kalman" / "metrics.json").is_file()
    assert (tmp_path / "kalman" / "window0_traces.png").is_file()
    assert np.isfinite(summary["mse_tf"])
    assert np.isfinite(summary["mse_impute_smoothed"])

    oracle = run_method("oracle_ekf", bundle, cfg, tmp_path / "oracle_ekf")
    assert oracle["role"] == "oracle"
    assert "Not a fair competitor" in oracle["note"]

    tresp_cfg = dict(cfg)
    tresp_cfg.update(
        {
            "delay": 2,
            "hidden": 4,
            "epochs": 3,
            "do_gibbs": True,
            "do_ml": False,
            "gibbs_sweeps": 1,
            "gibbs_burn": 0,
            "n_sir": 4,
            "n_samples": 2,
            "tresp_train": "observed_targets",
        }
    )
    tresp = run_method("tresp", bundle, tresp_cfg, tmp_path / "tresp")
    assert tresp["role"] == "baseline"
    assert np.isfinite(tresp["mse_tf"])
    assert "Tresp" in tresp["note"]
