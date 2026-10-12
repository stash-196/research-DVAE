"""Appendix B base-error / drift terms on the fixed validation windows.

Synthetic arrays and a fake model. No dataset files.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from dvae.eval.utils.local_drift_analysis import (
    drift_metrics_from_series,
    enforce_drift_span,
    observed_entry_mask,
    score_fixed_validation_windows,
    x_true_variance,
)
from dvae.run_options import (
    parse_val_criteria,
    parse_val_drift_horizon,
    spectrum_monitoring_enabled,
)
from dvae.visualizers.visualize_training_metrics import build_val_criteria_figure


def _windows():
    truth = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 4.0],
            [0.5, 1.5, 4.0],
            [1.5, 0.5, 4.0],
            [0.2, 1.2, 4.0],
            [1.2, 0.2, 4.0],
            [0.8, 1.8, 4.0],
            [1.8, 0.7, 4.0],
            [0.3, 1.1, 4.0],
            [1.1, 0.4, 4.0],
            [0.6, 1.4, 4.0],
            [1.4, 0.9, 4.0],
        ],
        dtype=np.float64,
    )
    teacher = truth + np.array(
        [
            [0.1, -0.2, 0.2],
            [0.0, 0.1, 0.2],
            [-0.1, 0.2, 0.2],
            [0.2, -0.1, 0.2],
            [0.3, 0.1, 0.2],
            [-0.2, 0.4, 0.2],
            [0.1, -0.3, 0.2],
            [0.4, 0.2, 0.2],
            [-0.1, 0.2, 0.2],
            [0.2, -0.2, 0.2],
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.2],
        ]
    )
    delta = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.2, -0.1, 0.5],
            [0.5, 0.2, 0.5],
            [-0.4, 0.3, 0.5],
            [0.1, -0.2, 0.5],
            [0.3, 0.4, 0.5],
            [0.2, 0.1, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
        ]
    )
    autonomous = teacher + delta
    return truth, teacher, autonomous


def test_base_error_cross_and_drift_sum_to_auto_mse():
    truth, teacher, autonomous = _windows()
    flip, horizon = 4, 6
    # An unobserved entry inside the horizon must not enter the means.
    truth = truth.copy()
    teacher = teacher.copy()
    autonomous = autonomous.copy()
    truth[5, 1] = np.nan
    teacher[5, 1] = 80.0
    autonomous[5, 1] = -40.0
    variance = x_true_variance([truth])
    metrics = drift_metrics_from_series(
        [truth], [teacher], [autonomous], flip, horizon, variance
    )
    span = slice(flip, flip + horizon)
    mask = observed_entry_mask(truth[span])
    auto_err = autonomous[span] - truth[span]
    auto_mse = float(np.mean(auto_err[mask] ** 2))
    assert (
        metrics["base_error"]
        + metrics["error_drift_cross"]
        + metrics["auto_drift"]
    ) == pytest.approx(auto_mse)
    assert metrics["error_drift_total"] == pytest.approx(auto_mse)
    assert metrics["recon_k"] == pytest.approx(metrics["error_drift_total"])
    assert metrics["recon_k"] == pytest.approx(auto_mse)
    assert metrics["auto_drift_k"] == pytest.approx(metrics["auto_drift"])
    assert not mask[1, 1]

    blown = autonomous.copy()
    blown[5, 1] = 1.0e6
    again = drift_metrics_from_series(
        [truth], [teacher], [blown], flip, horizon, variance
    )
    assert again["error_drift_total"] == pytest.approx(metrics["error_drift_total"])
    assert again["recon_k"] == pytest.approx(metrics["recon_k"])

    scale = variance.reshape(1, -1)
    usable = mask & np.isfinite(scale) & (scale > 0.0)
    broadcast = np.broadcast_to(scale, auto_err.shape)
    norm_mse = float(np.mean((auto_err[usable] ** 2) / broadcast[usable]))
    assert (
        metrics["base_error_varnorm"]
        + metrics["auto_drift_varnorm"]
        + metrics["error_drift_cross_varnorm"]
    ) == pytest.approx(norm_mse)
    assert metrics["error_drift_total_varnorm"] == pytest.approx(norm_mse)
    assert variance[2] == pytest.approx(0.0)
    assert not usable[:, 2].any()
    assert mask[:, 2].all()

    step = autonomous[flip] - teacher[flip]
    step_mask = observed_entry_mask(truth[flip : flip + 1])[0]
    assert metrics["auto_drift_k1"] == pytest.approx(float(np.mean(step[step_mask] ** 2)))
    prefix = slice(flip, flip + 5)
    prefix_mask = observed_entry_mask(truth[prefix])
    prefix_d = autonomous[prefix] - teacher[prefix]
    assert metrics["auto_drift_k5"] == pytest.approx(
        float(np.mean(prefix_d[prefix_mask] ** 2))
    )
    assert metrics["auto_drift_k5"] != pytest.approx(metrics["auto_drift"])


def test_indicate_mask_ignores_the_mask_channel():
    truth, teacher, autonomous = _windows()
    signal = truth[:, :1]
    mask_channel = np.zeros((truth.shape[0], 1))
    paired = np.concatenate([signal, mask_channel], axis=1)
    tf = paired.copy()
    auto = paired.copy()
    auto[4:10, 0] += 0.5
    auto[4:10, 1] += 10.0
    variance = x_true_variance([paired], "only_x_indicate")
    metrics = drift_metrics_from_series(
        [paired],
        [tf],
        [auto],
        4,
        6,
        variance,
        observation_process="only_x_indicate",
    )
    assert metrics["base_error"] == pytest.approx(0.0)
    assert metrics["auto_drift"] == pytest.approx(0.25)
    assert metrics["error_drift_cross"] == pytest.approx(0.0)
    assert metrics["error_drift_total"] == pytest.approx(0.25)
    assert metrics["recon_k"] == pytest.approx(metrics["error_drift_total"])


class _ShiftAuto:
    device = "cpu"

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        out = batch.clone()
        if mode_selector is None:
            return out
        return torch.where(mode_selector > 0.5, out + 0.5, out)


def test_fixed_windows_score_drift_with_the_same_flip():
    length = 16
    step = np.arange(length, dtype=np.float32)
    window = np.column_stack([step, step + 3.0])
    windows = [window, window + 1.0]
    flip, horizon = 6, 5
    variance = x_true_variance(windows)
    scored = score_fixed_validation_windows(
        _ShiftAuto(), windows, flip, horizon, variance, None, "cpu"
    )
    assert scored["base_error"] == pytest.approx(0.0, abs=1e-6)
    assert scored["auto_drift"] == pytest.approx(0.25, abs=1e-6)
    assert scored["error_drift_cross"] == pytest.approx(0.0, abs=1e-6)
    assert scored["error_drift_total"] == pytest.approx(
        scored["base_error"] + scored["error_drift_cross"] + scored["auto_drift"]
    )
    assert scored["recon_k"] == pytest.approx(scored["error_drift_total"])
    assert scored["auto_drift_k1"] == pytest.approx(scored["auto_drift"])
    assert scored["auto_drift_k"] == pytest.approx(scored["auto_drift"])
    # A constant shift is removed before the periodogram, so the free-run
    # spectrum matches and Hellinger is the known NaN. Drift still sees it.
    assert np.isnan(scored["spectrum"])
    with pytest.raises(ValueError, match="exceeds window length"):
        enforce_drift_span(length, flip, 20)


def test_drift_horizon_and_criteria_defaults():
    assert parse_val_drift_horizon(None) == 60
    assert parse_val_drift_horizon("") == 60
    assert parse_val_drift_horizon("off") == 60
    assert parse_val_drift_horizon("12") == 12
    with pytest.raises(ValueError, match=">= 5"):
        parse_val_drift_horizon("4")
    assert parse_val_criteria(None) == ("recon", "spectrum")
    assert parse_val_criteria("drift") == ("auto_drift",)
    assert parse_val_criteria("cross") == ("error_drift_cross",)
    assert parse_val_criteria("auto_total") == ("error_drift_total",)
    assert parse_val_criteria("recon,base_error,auto_total") == (
        "recon",
        "base_error",
        "error_drift_total",
    )
    assert parse_val_criteria("auto_drift,error_drift_cross") == (
        "auto_drift",
        "error_drift_cross",
    )
    with pytest.raises(ValueError, match="repeats"):
        parse_val_criteria("drift,auto_drift")
    assert spectrum_monitoring_enabled(("recon",), legacy=True) is False
    assert spectrum_monitoring_enabled(("drift",), legacy=True) is True
    assert spectrum_monitoring_enabled(("base_error", "cross"), legacy=True) is True
    with pytest.raises(ValueError, match="auto_total"):
        parse_val_criteria("energy")


def _warmup_colors(ax):
    colors = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        if x.size >= 2 and np.allclose(x, x[0]):
            colors.append(line.get_color())
    return colors


def test_drift_layouts_share_range_and_keep_markers():
    from dvae.visualizers.visualize_training_metrics import build_drift_panels_figure

    drift = {
        "base_error": [1.0, 2.0],
        "auto_drift": [0.4, 0.2],
        "error_drift_cross": [0.1, -0.1],
        "error_drift_total": [1.5, 2.1],
    }
    recon_k = [1.5, 2.1]
    lyapunov = {
        "ftle": [0.02, 0.05],
        "lle_mean": [0.4, 0.3],
        "lle_std": [0.01, 0.02],
        "lle_gt": 0.69,
    }
    fig = build_val_criteria_figure(
        ("recon", "drift"),
        [0.5, 2.5],
        [0.3, 2.3],
        [2.0, 4.0],
        [0, 3],
        [0, 4],
        [0],
        [0, 6],
        "VRNN",
        "ss",
        "none",
        best_epoch=1,
        drift=drift,
        recon_k=recon_k,
        lyapunov=lyapunov,
    )
    titles = [ax.get_legend().get_title().get_text() for ax in fig.axes]
    assert titles == [
        "VRNN: recon (selected)",
        "VRNN: spectrum error",
        "VRNN: drift terms",
        "VRNN: Lyapunov",
        "VRNN: selection score",
    ]
    overlay = fig.axes[2]
    labels = [line.get_label() for line in overlay.lines]
    assert "recon_k" in labels
    assert "auto_drift (selected)" in labels
    assert "error_drift_cross" in labels
    recon_line = next(line for line in overlay.lines if line.get_label() == "recon_k")
    total_line = next(
        line for line in overlay.lines if line.get_label() == "error_drift_total"
    )
    assert recon_line.get_linewidth() > total_line.get_linewidth()
    assert recon_line.get_zorder() < total_line.get_zorder()
    assert list(overlay.lines).index(recon_line) < list(overlay.lines).index(total_line)
    cross = next(
        line for line in overlay.lines if line.get_label() == "error_drift_cross"
    )
    assert float(np.asarray(cross.get_ydata(), dtype=float)[-1]) < 0.0
    lyap = fig.axes[3]
    lyap_labels = [line.get_label() for line in lyap.lines]
    assert "finite-time exponent" in lyap_labels
    assert "Rosenstein LLE" in lyap_labels
    assert "GT LLE" in lyap_labels
    for ax in fig.axes:
        colors = _warmup_colors(ax)
        assert colors.count("c") == 2
        assert colors.count("r") == 2
        assert "g" not in colors
        assert colors.count("m") == 2
    best = [
        [
            float(np.asarray(line.get_ydata(), dtype=float).reshape(-1)[0])
            for line in ax.lines
            if line.get_label() == "best epoch"
        ]
        for ax in fig.axes
    ]
    assert best[0] == [2.5]
    assert best[2] == [2.1]
    assert best[-1] == [4.0]

    log_fig = build_val_criteria_figure(
        ("recon", "auto_drift"),
        [0.5, 2.5],
        [0.3, 2.3],
        [2.0, 4.0],
        [0, 3],
        [0, 4],
        [0],
        [0, 6],
        "VRNN",
        "ss",
        "none",
        best_epoch=1,
        drift=drift,
        recon_k=recon_k,
        lyapunov=lyapunov,
        log_scale=True,
    )
    assert all(ax.get_yscale() == "symlog" for ax in log_fig.axes)
    log_cross = next(
        line
        for line in log_fig.axes[2].lines
        if line.get_label() == "error_drift_cross"
    )
    assert float(np.asarray(log_cross.get_ydata(), dtype=float)[-1]) < 0.0

    panels = build_drift_panels_figure(
        drift,
        [0, 3],
        [0, 4],
        [0],
        [0, 6],
        "VRNN",
        "ss",
        "none",
        criteria=("drift",),
        best_epoch=1,
    )
    panel_titles = [ax.get_legend().get_title().get_text() for ax in panels.axes]
    assert panel_titles == [
        "VRNN: base_error",
        "VRNN: auto_drift (selected)",
        "VRNN: error_drift_cross",
        "VRNN: error_drift_total",
    ]
    ylims = [ax.get_ylim() for ax in panels.axes]
    assert ylims[0] == ylims[1] == ylims[2] == ylims[3]
    assert ylims[0][0] < 0.0
    log_panels = build_drift_panels_figure(
        drift,
        [0, 3],
        [0, 4],
        [0],
        [0, 6],
        "VRNN",
        "ss",
        "none",
        criteria=("auto_drift",),
        best_epoch=1,
        log_scale=True,
    )
    assert all(ax.get_yscale() == "symlog" for ax in log_panels.axes)
    assert [ax.get_ylim() for ax in log_panels.axes][0] == [
        ax.get_ylim() for ax in log_panels.axes
    ][-1]
    for ax in list(panels.axes) + list(log_panels.axes):
        colors = _warmup_colors(ax)
        assert "c" in colors and "r" in colors and "m" in colors


def test_finite_time_exponent_is_the_log_norm_slope():
    from dvae.eval.utils.local_drift_analysis import finite_time_exponent

    horizon = 8
    flip = 2
    length = flip + horizon
    truth = np.zeros((length, 1))
    teacher = np.zeros((length, 1))
    autonomous = np.zeros((length, 1))
    for step in range(horizon):
        autonomous[flip + step, 0] = np.exp(0.1 * (step + 1))
    metrics = drift_metrics_from_series(
        [truth], [teacher], [autonomous], flip, horizon, np.array([1.0])
    )
    assert metrics["ftle"] == pytest.approx(0.1)
    assert metrics["recon_k"] == pytest.approx(metrics["error_drift_total"])
    assert finite_time_exponent(
        [np.exp(0.1 * k) for k in range(1, 9)], k_start=1, k_end=8
    ) == pytest.approx(0.1)
    narrowed = drift_metrics_from_series(
        [truth],
        [teacher],
        [autonomous],
        flip,
        horizon,
        np.array([1.0]),
        ftle_k_start=3,
        ftle_k_end=6,
    )
    assert narrowed["ftle"] == pytest.approx(0.1)


def test_rosenstein_cost_on_a_synthetic_epoch(capsys):
    import time

    from dvae.eval.utils.local_drift_analysis import rosenstein_on_free_run

    # Logistic map at r=4. Ten windows, 1000 free-run steps, one channel.
    windows = []
    state = 0.3
    for index in range(10):
        series = np.empty(2000, dtype=np.float64)
        x = state + 0.01 * index
        for step in range(2000):
            x = 4.0 * x * (1.0 - x)
            series[step] = x
        windows.append(series.reshape(-1, 1))
        state = series[-1]
    started = time.perf_counter()
    mean, std = rosenstein_on_free_run(windows, tf_steps=1000, m=3, tau=5)
    elapsed = time.perf_counter() - started
    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert std >= 0.0
    gt, _gt_std = rosenstein_on_free_run(windows, tf_steps=1000, m=3, tau=5)
    assert gt == pytest.approx(mean)
    print(f"Rosenstein synthetic cost per epoch: {elapsed:.4f} s")
    captured = capsys.readouterr().out
    assert "Rosenstein synthetic cost per epoch:" in captured
