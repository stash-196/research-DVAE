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
    assert metrics["base_error"] + metrics["cross"] + metrics["drift"] == pytest.approx(
        auto_mse
    )
    assert metrics["auto_total"] == pytest.approx(auto_mse)
    assert metrics["drift_k"] == pytest.approx(metrics["drift"])
    assert not mask[1, 1]

    blown = autonomous.copy()
    blown[5, 1] = 1.0e6
    again = drift_metrics_from_series(
        [truth], [teacher], [blown], flip, horizon, variance
    )
    assert again["auto_total"] == pytest.approx(metrics["auto_total"])

    scale = variance.reshape(1, -1)
    usable = mask & np.isfinite(scale) & (scale > 0.0)
    broadcast = np.broadcast_to(scale, auto_err.shape)
    norm_mse = float(np.mean((auto_err[usable] ** 2) / broadcast[usable]))
    assert (
        metrics["base_error_varnorm"]
        + metrics["drift_varnorm"]
        + metrics["cross_varnorm"]
    ) == pytest.approx(norm_mse)
    assert metrics["auto_total_varnorm"] == pytest.approx(norm_mse)
    assert variance[2] == pytest.approx(0.0)
    assert not usable[:, 2].any()
    assert mask[:, 2].all()

    step = autonomous[flip] - teacher[flip]
    step_mask = observed_entry_mask(truth[flip : flip + 1])[0]
    assert metrics["drift_k1"] == pytest.approx(float(np.mean(step[step_mask] ** 2)))
    prefix = slice(flip, flip + 5)
    prefix_mask = observed_entry_mask(truth[prefix])
    prefix_d = autonomous[prefix] - teacher[prefix]
    assert metrics["drift_k5"] == pytest.approx(
        float(np.mean(prefix_d[prefix_mask] ** 2))
    )
    assert metrics["drift_k5"] != pytest.approx(metrics["drift"])


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
    assert metrics["drift"] == pytest.approx(0.25)
    assert metrics["cross"] == pytest.approx(0.0)
    assert metrics["auto_total"] == pytest.approx(0.25)


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
    assert scored["drift"] == pytest.approx(0.25, abs=1e-6)
    assert scored["cross"] == pytest.approx(0.0, abs=1e-6)
    assert scored["auto_total"] == pytest.approx(
        scored["base_error"] + scored["cross"] + scored["drift"]
    )
    assert scored["drift_k1"] == pytest.approx(scored["drift"])
    assert scored["drift_k"] == pytest.approx(scored["drift"])
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
    assert parse_val_criteria("drift") == ("drift",)
    assert parse_val_criteria("recon,base_error,auto_total") == (
        "recon",
        "base_error",
        "auto_total",
    )
    assert spectrum_monitoring_enabled(("recon",), legacy=True) is False
    assert spectrum_monitoring_enabled(("drift",), legacy=True) is True
    assert spectrum_monitoring_enabled(("base_error", "cross"), legacy=True) is True
    with pytest.raises(ValueError, match="auto_total"):
        parse_val_criteria("energy")


def test_drift_panels_keep_warmup_lines_and_the_best_epoch():
    drift = {
        "base_error": [1.0, 2.0],
        "drift": [0.4, 0.2],
        "cross": [0.1, -0.1],
        "auto_total": [1.5, 2.1],
        "base_error_varnorm": [0.5, 1.0],
        "drift_varnorm": [0.2, 0.1],
        "cross_varnorm": [0.05, -0.05],
        "auto_total_varnorm": [0.75, 1.05],
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
    )
    titles = [ax.get_legend().get_title().get_text() for ax in fig.axes]
    assert titles == [
        "VRNN: recon (selected)",
        "VRNN: spectrum error",
        "VRNN: base_error",
        "VRNN: drift (selected)",
        "VRNN: cross",
        "VRNN: auto_total",
        "VRNN: base_error / var",
        "VRNN: drift / var",
        "VRNN: cross / var",
        "VRNN: auto_total / var",
        "VRNN: selection score",
    ]
    best = []
    for ax in fig.axes:
        marked = [
            float(np.asarray(line.get_ydata(), dtype=float).reshape(-1)[0])
            for line in ax.lines
            if line.get_label() == "best epoch"
        ]
        best.append(marked)
        colors = []
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            if x.size >= 2 and np.allclose(x, x[0]):
                colors.append(line.get_color())
        assert colors.count("c") == 2
        assert colors.count("r") == 2
        assert "g" not in colors
        assert colors.count("m") == 2
    assert best[3] == [0.2]
    assert best[6] == [1.0]
    assert best[-1] == [4.0]
