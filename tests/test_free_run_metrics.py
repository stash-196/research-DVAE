"""Earth-mover distance and Rosenstein LLE on synthetic series.

No wearable recordings. Lorenz uses the classic (sigma, rho, beta) = (10, 28, 8/3)
parameters, whose largest exponent is about 0.9 per unit time.
"""

from __future__ import annotations

import configparser

import numpy as np
import pytest

from dvae.eval.utils.distribution_distance import (
    exact_emd,
    per_dimension_w1,
    sliced_wasserstein,
    wasserstein_1d,
)
from dvae.eval.utils.free_run_metrics import (
    free_run_window_metrics,
    resolve_free_run_metric_settings,
)
from dvae.eval.utils.rosenstein_lyapunov import rosenstein_lle, rosenstein_on_benchmarks


def _lorenz_x(n: int = 12000, dt: float = 0.01, transient: int = 8000) -> np.ndarray:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def field(state):
        x, y, z = state
        return np.array(
            [sigma * (y - x), x * (rho - z) - y, x * y - beta * z],
            dtype=np.float64,
        )

    state = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(transient + n):
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if i >= transient:
            out[i - transient] = state[0]
    return out


def test_w1_of_a_shift_and_of_identical_samples():
    grid = np.linspace(0.0, 1.0, 80)
    rng = np.random.default_rng(0)
    assert wasserstein_1d(grid, grid + 0.5, max_samples=1000, rng=rng) == pytest.approx(
        0.5
    )
    assert wasserstein_1d(grid, grid, max_samples=1000, rng=rng) == pytest.approx(0.0)


def test_per_dimension_w1_and_sliced_wasserstein_on_a_translation():
    rng = np.random.default_rng(1)
    cloud = rng.normal(size=(60, 2))
    shifted = cloud + np.array([0.3, -0.4])
    w1 = per_dimension_w1(cloud, shifted, max_samples=200, rng=rng)
    assert w1[0] == pytest.approx(0.3, abs=1e-6)
    assert w1[1] == pytest.approx(0.4, abs=1e-6)
    sliced = sliced_wasserstein(
        cloud, shifted, n_projections=48, max_samples=200, rng=rng
    )
    assert sliced > 0.05
    identical = sliced_wasserstein(
        cloud, cloud, n_projections=16, max_samples=200, rng=rng
    )
    assert identical == pytest.approx(0.0, abs=1e-9)


def test_exact_emd_is_skipped_when_not_cheap_and_optional_when_small():
    rng = np.random.default_rng(2)
    cloud = rng.normal(size=(40, 2))
    shifted = cloud + np.array([0.3, 0.4])
    skipped, reason = exact_emd(cloud, shifted, max_n=10, max_dim=5)
    assert reason == "n_above_cap"
    assert np.isnan(skipped)
    high_dim, dim_reason = exact_emd(
        rng.normal(size=(8, 6)), rng.normal(size=(8, 6)), max_n=20, max_dim=5
    )
    assert dim_reason == "dim_above_cap"
    assert np.isnan(high_dim)
    value, exact_reason = exact_emd(cloud, shifted, max_n=200, max_dim=5)
    if exact_reason == "pot_unavailable":
        assert np.isnan(value)
    else:
        assert exact_reason is None
        assert value == pytest.approx(0.5, abs=0.05)


def test_subsample_keeps_a_pure_shift_near_the_true_w1():
    rng = np.random.default_rng(3)
    base = np.linspace(-1.0, 1.0, 5000)
    shifted = base + 1.25
    score = wasserstein_1d(base, shifted, max_samples=128, rng=rng)
    assert score == pytest.approx(1.25, abs=0.05)


def test_rosenstein_lorenz_is_near_0_9():
    series = _lorenz_x()
    exponent = rosenstein_lle(series, m=5, tau=10, dt=0.01)
    assert np.isfinite(exponent)
    assert exponent == pytest.approx(0.9, abs=0.2)


def test_rosenstein_reports_gt_auto_and_difference():
    series = _lorenz_x(n=8000)
    drifted = series + 0.01 * np.sin(np.linspace(0.0, 40.0, series.size))
    benchmarks = {
        "channels": [
            {
                "key": "x",
                "gt_auto": series,
                "tf_auto": series,
                "auto_seg": drifted,
            }
        ],
        "dt": 0.01,
        "time_delay": 10,
        "delay_dims": 5,
    }
    result = rosenstein_on_benchmarks(benchmarks, embed_dim=5, tau=10)
    assert result["lle_rosenstein_gt"] == pytest.approx(0.9, abs=0.2)
    assert np.isfinite(result["lle_rosenstein_auto"])
    assert result["lle_rosenstein_diff"] == pytest.approx(
        result["lle_rosenstein_auto"] - result["lle_rosenstein_gt"]
    )
    same = rosenstein_on_benchmarks(
        {
            "channels": [
                {
                    "key": "x",
                    "gt_auto": series,
                    "tf_auto": series,
                    "auto_seg": series,
                }
            ],
            "dt": 0.01,
            "time_delay": 10,
            "delay_dims": 5,
        },
        embed_dim=5,
        tau=10,
    )
    assert same["lle_rosenstein_diff"] == pytest.approx(0.0, abs=1e-9)


def test_short_series_returns_nan_without_raising():
    assert np.isnan(rosenstein_lle(np.linspace(0.0, 1.0, 30), m=5, tau=4, dt=0.01))


def test_free_run_settings_default_and_config_override():
    defaults = resolve_free_run_metric_settings(None)
    assert defaults["emd_max_samples"] == 512
    assert defaults["emd_n_projections"] == 32
    assert defaults["emd_exact_max_n"] == 200
    assert defaults["emd_exact_max_dim"] == 5
    assert defaults["rosenstein_embed_dim"] is None
    assert defaults["rosenstein_theiler"] is None
    assert defaults["rosenstein_evolution"] == 150

    cfg = configparser.ConfigParser()
    cfg.add_section("Evaluation")
    cfg.set("Evaluation", "emd_max_samples", "64")
    cfg.set("Evaluation", "rosenstein_embed_dim", "4")
    cfg.set("Evaluation", "rosenstein_theiler", "12")
    cfg.set("Evaluation", "emd_exact_max_dim", "")
    overridden = resolve_free_run_metric_settings(cfg)
    assert overridden["emd_max_samples"] == 64
    assert overridden["rosenstein_embed_dim"] == 4
    assert overridden["rosenstein_theiler"] == 12
    assert overridden["emd_exact_max_dim"] == 5


def test_window_metrics_flatten_w1_and_lle_keys():
    rng = np.random.default_rng(4)
    gt = rng.normal(size=400)
    auto = gt + 0.2
    benchmarks = {
        "channels": [
            {"key": "x", "gt_auto": gt, "tf_auto": gt, "auto_seg": auto},
            {"key": "y", "gt_auto": gt, "tf_auto": auto, "auto_seg": auto + 0.1},
        ],
        "dt": 0.05,
        "time_delay": 2,
        "delay_dims": 3,
    }
    settings = resolve_free_run_metric_settings(None)
    settings["emd_max_samples"] = 1000
    settings["rosenstein_evolution"] = 40
    settings["rosenstein_embed_dim"] = 3
    settings["rosenstein_tau"] = 2
    scalars, book = free_run_window_metrics(benchmarks, settings)
    assert scalars["emd_w1_auto_x"] == pytest.approx(0.2, abs=1e-6)
    assert scalars["emd_w1_auto_y"] == pytest.approx(0.3, abs=1e-6)
    assert scalars["emd_w1_auto"] == pytest.approx(0.25, abs=1e-6)
    assert "lle_rosenstein_gt" in scalars
    assert "lle_rosenstein_diff" in scalars
    assert book["emd_max_samples"] == 1000
    assert "emd_exact_skip_reason_auto" in book
