"""SINDy on synthetic trajectories. No Lorenz pickles and no pysindy required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dvae.baselines.plotting import plot_coefficients, plot_metric_bars, plot_trajectories
from dvae.baselines.rollout_metrics import calc_masked_mse, score_trajectory_pair, spectrum_pair_error
from dvae.baselines.sindy import (
    contiguous_observed_segments,
    fit_sindy,
    interpolate_missing,
    polynomial_library,
    score_rollouts,
)


def _lorenz_rhs(state):
    x, y, z = state
    return np.array(
        [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z],
        dtype=np.float64,
    )


def _rk4(rhs, x0, dt, n_steps):
    traj = np.empty((n_steps, len(x0)), dtype=np.float64)
    state = np.asarray(x0, dtype=np.float64)
    for i in range(n_steps):
        traj[i] = state
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return traj


def _lorenz_trajectory():
    raw = _rk4(_lorenz_rhs, np.array([1.0, 1.0, 1.0]), 0.01, 5000)
    return raw[1000:]


def _coef(fit, name, state):
    return float(fit.coefficients[fit.feature_names.index(name), state])


def test_polynomial_library_order():
    theta, names = polynomial_library(np.zeros((3, 2)), 2)
    assert theta.shape == (3, 6)
    assert names == ["1", "x0", "x1", "x0*x0", "x0*x1", "x1*x1"]


def test_clean_lorenz_recovers_sparse_field():
    traj = _lorenz_trajectory()
    fit = fit_sindy(traj, dt=0.01, degree=2, threshold=0.2, missing_strategy="observed_segments")
    assert fit.state_dim == 3
    assert fit.partial_observation is False
    assert fit.backend == "numpy"
    # dx/dt = -10 x + 10 y
    assert _coef(fit, "x0", 0) == pytest.approx(-10.0, abs=0.5)
    assert _coef(fit, "x1", 0) == pytest.approx(10.0, abs=0.5)
    # dy/dt = 28 x - y - x z
    assert _coef(fit, "x0", 1) == pytest.approx(28.0, abs=0.5)
    assert _coef(fit, "x1", 1) == pytest.approx(-1.0, abs=0.5)
    assert _coef(fit, "x0*x2", 1) == pytest.approx(-1.0, abs=0.5)
    # dz/dt = x y - beta z
    assert _coef(fit, "x0*x1", 2) == pytest.approx(1.0, abs=0.5)
    assert _coef(fit, "x2", 2) == pytest.approx(-(8.0 / 3.0), abs=0.5)
    for name in fit.feature_names:
        if name not in ("x0", "x1"):
            assert abs(_coef(fit, name, 0)) < 0.5, fit.equations()
    horizon = fit.simulate(traj[200], 40)
    assert np.mean((horizon - traj[200:240]) ** 2) < 1.0


def test_observed_segments_drop_gaps_and_still_identify_dx():
    traj = _lorenz_trajectory()
    gappy = traj.copy()
    gappy[100:140] = np.nan
    gappy[800:830, 1] = np.nan
    segments, n_dropped = contiguous_observed_segments(gappy, min_len=8)
    assert n_dropped == 70
    assert len(segments) >= 2
    fit = fit_sindy(
        gappy, dt=0.01, degree=2, threshold=0.2, missing_strategy="observed_segments"
    )
    assert fit.n_dropped == 70
    assert fit.n_segments >= 2
    assert _coef(fit, "x0", 0) == pytest.approx(-10.0, abs=0.75)
    assert _coef(fit, "x1", 0) == pytest.approx(10.0, abs=0.75)
    assert "not taken across a gap" in fit.limits


def test_interpolate_strategy_fills_and_is_labeled_optimistic():
    traj = _lorenz_trajectory()[:500]
    gappy = traj.copy()
    gappy[50:60, 0] = np.nan
    filled = interpolate_missing(gappy)
    assert np.isfinite(filled).all()
    fit = fit_sindy(gappy, dt=0.01, degree=2, threshold=0.2, missing_strategy="interpolate")
    assert fit.missing_strategy == "interpolate"
    assert "optimistic" in fit.limits
    assert fit.n_segments == 1


def test_only_x_is_partial_and_documented():
    traj = _lorenz_trajectory()
    fit = fit_sindy(traj[:, 0], dt=0.01, degree=2, threshold=0.05)
    assert fit.state_dim == 1
    assert fit.partial_observation is True
    assert "cannot represent" in fit.limits
    assert "only_x" in fit.limits
    rolled = score_rollouts(fit, [traj[:300, :1], traj[300:600, :1]], dataset_name="Lorenz63")
    for key in ("mse_auto", "mse_auto_mean", "kld_auto", "spectrum_error_auto"):
        assert key in rolled


def test_fully_observed_rollout_metric_names_match_otf():
    traj = _lorenz_trajectory()
    fit = fit_sindy(traj[:2500], dt=0.01, degree=2, threshold=0.2)
    windows = [traj[2500:2900], traj[2900:3300]]
    scores = score_rollouts(fit, windows, dataset_name="Lorenz63")
    assert scores["n_windows"] == 2
    assert scores["mse_auto_short"] < 5.0
    for key in (
        "mse_auto",
        "mse_auto_mean",
        "kld_auto",
        "kld_auto_mean",
        "spectrum_error_auto",
        "spectrum_error_auto_mean",
    ):
        assert key in scores
        assert np.isfinite(scores[key])


def test_identical_series_metrics_are_zero_mse():
    rng = np.random.default_rng(0)
    series = rng.normal(size=(400, 2))
    scores = score_trajectory_pair(
        series, series.copy(), dataset_name="Lorenz63", dt=0.01, prefix="auto"
    )
    assert scores["mse_auto"] == 0.0
    assert scores["mse_auto_mean"] == 0.0
    assert scores["spectrum_error_auto"] == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(scores["kld_auto"])
    shifted = series + 0.05
    shifted_scores = score_trajectory_pair(
        series, shifted, dataset_name="Lorenz63", dt=0.01, prefix="auto"
    )
    assert np.isfinite(shifted_scores["spectrum_error_auto"])
    assert shifted_scores["mse_auto"] > 0.0


def test_masked_mse_matches_repo_helper():
    gt = np.array([1.0, np.nan, 3.0, 4.0])
    pred = np.array([1.0, 9.0, 0.0, 4.0])
    mask = np.array([False, False, False, True])
    local = calc_masked_mse(gt, pred, mask)
    from dvae.eval.utils.run_mse_analysis import calc_masked_mse as repo_mse

    assert local == pytest.approx(repo_mse(gt, pred, mask))
    assert local == pytest.approx(np.mean([0.0, 9.0]))


def test_spectrum_pair_matches_eval_helper():
    pytest.importorskip("umap")
    from dvae.eval.utils.frequency_analysis import _spectrum_pair_errors

    rng = np.random.default_rng(1)
    gt = np.sin(np.linspace(0, 20, 256))
    pred = gt + 0.05 * rng.normal(size=gt.shape)
    assert spectrum_pair_error(gt, pred, sampling_rate=100.0) == pytest.approx(
        _spectrum_pair_errors(gt, pred, 100.0)
    )


def test_plots_use_files(tmp_path):
    gt = np.sin(np.linspace(0, 4, 80))
    pred = gt + 0.1
    plot_trajectories(gt, pred, tmp_path / "traj.png", flip_index=40, channel_names=["x"])
    plot_metric_bars(
        {"mse_auto": 0.2, "kld_auto": 1.5, "spectrum_error_auto": 0.1},
        ["mse_auto", "kld_auto", "spectrum_error_auto"],
        tmp_path / "metrics.png",
    )
    plot_coefficients(["1", "x0"], np.array([[0.0, 1.0], [-10.0, 10.0]]), tmp_path / "coef.png")
    for name in ("traj.png", "metrics.png", "coef.png"):
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_pysindy_backend_optional():
    pytest.importorskip("pysindy")
    traj = _lorenz_trajectory()[:1500]
    fit = fit_sindy(traj, dt=0.01, degree=2, threshold=0.2, backend="pysindy")
    assert fit.backend == "pysindy"
    assert fit._pysindy_model is not None
    rolled = fit.simulate(traj[100], 20)
    assert rolled.shape == (20, 3)
    assert np.isfinite(rolled).all()
