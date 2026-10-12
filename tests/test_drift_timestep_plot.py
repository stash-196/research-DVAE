"""Per-timestep e^2, e*d, and d^2 trajectories on synthetic windows.

The traces cover the full window. The checkpoint figure is written under
vis_during_training/drift_timestep with the epoch in the filename. The
arrays themselves are not stored in the training history.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvae.eval.utils.local_drift_analysis import (
    drift_metrics_from_series,
    observed_entry_mask,
    x_true_variance,
)
from dvae.training_history import (
    DRIFT_SERIES_KEYS,
    EPOCH_SERIES_KEYS,
    build_training_history,
    load_training_history,
    save_training_history,
)
from dvae.visualizers.visualize_training_metrics import (
    build_drift_timestep_figure,
    drift_timestep_figure_path,
    save_drift_timestep_figures,
)


_STEP_KEYS = (
    "val_e2_per_step",
    "val_ed_per_step",
    "val_d2_per_step",
    "val_e2_per_step_varnorm",
    "val_ed_per_step_varnorm",
    "val_d2_per_step_varnorm",
    "e2_per_step",
    "ed_per_step",
    "d2_per_step",
    "e2_per_step_varnorm",
    "ed_per_step_varnorm",
    "d2_per_step_varnorm",
)


def _channel_mean(values, mask):
    chosen = np.asarray(mask, dtype=bool)
    count = chosen.sum(axis=-1)
    summed = np.sum(np.where(chosen, values, 0.0), axis=-1)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    ok = count > 0
    out[ok] = summed[ok] / count[ok]
    return out


def _synthetic_windows():
    """Three windows, longer than the post-flip horizon. No wearable data."""
    rng = np.random.default_rng(7)
    n_windows, length, channels = 3, 12, 2
    truth = rng.normal(loc=0.2, scale=1.5, size=(n_windows, length, channels))
    teacher = truth + rng.normal(scale=0.15, size=truth.shape)
    autonomous = teacher + rng.normal(scale=0.4, size=truth.shape)
    # One missing entry before the flip, so the channel mean drops it.
    truth[1, 2, 1] = np.nan
    teacher[1, 2, 1] = np.nan
    autonomous[1, 2, 1] = np.nan
    return truth, teacher, autonomous


def _metrics(truth, teacher, autonomous):
    windows = [truth[index] for index in range(truth.shape[0])]
    variance = x_true_variance(windows)
    return drift_metrics_from_series(
        windows,
        [teacher[index] for index in range(truth.shape[0])],
        [autonomous[index] for index in range(truth.shape[0])],
        flip=4,
        horizon=5,
        variance=variance,
    ), variance


def test_traces_span_the_full_window_and_match_channel_means():
    truth, teacher, autonomous = _synthetic_windows()
    metrics, variance = _metrics(truth, teacher, autonomous)
    length = truth.shape[1]
    err = teacher - truth
    delta = autonomous - teacher
    masks = np.stack(
        [observed_entry_mask(truth[index]) for index in range(truth.shape[0])]
    )
    expected = {
        "e2_per_step": _channel_mean(err * err, masks),
        "ed_per_step": _channel_mean(err * delta, masks),
        "d2_per_step": _channel_mean(delta * delta, masks),
    }
    usable = masks & np.isfinite(variance) & (variance > 0.0)
    inv = (1.0 / np.where(usable.any(axis=(0, 1)), variance, 1.0)).reshape(1, 1, -1)
    expected_norm = {
        "e2_per_step_varnorm": _channel_mean(err * err * inv, usable),
        "ed_per_step_varnorm": _channel_mean(err * delta * inv, usable),
        "d2_per_step_varnorm": _channel_mean(delta * delta * inv, usable),
    }
    for name, values in expected.items():
        assert metrics[name].shape == (truth.shape[0], length)
        assert metrics[name].shape[1] != 5
        np.testing.assert_allclose(metrics[name], values, equal_nan=True)
    for name, values in expected_norm.items():
        assert metrics[name].shape == (truth.shape[0], length)
        np.testing.assert_allclose(metrics[name], values, equal_nan=True)
    # The missing channel is left out of that timestep's mean.
    only_channel = (err[1, 2, 0] ** 2)
    assert metrics["e2_per_step"][1, 2] == pytest.approx(only_channel)

    bumped = teacher.copy()
    bumped[:, 0, :] += 3.0
    bumped_metrics, _variance = _metrics(truth, bumped, autonomous)
    assert bumped_metrics["base_error"] == pytest.approx(metrics["base_error"])
    assert not np.allclose(
        bumped_metrics["e2_per_step"][:, 0],
        metrics["e2_per_step"][:, 0],
        equal_nan=True,
    )


def _axes_share_x(left, right):
    group = left.get_shared_x_axes()
    if hasattr(group, "joined"):
        return bool(group.joined(left, right))
    siblings = group.get_siblings(left)
    return right in siblings


def _trajectory_lines(ax):
    lines = []
    for line in ax.lines:
        style = line.get_linestyle()
        if style in ("--", "dashed"):
            continue
        lines.append(line)
    return lines


def test_checkpoint_figure_is_saved_with_the_epoch_in_the_filename(tmp_path):
    truth, teacher, autonomous = _synthetic_windows()
    metrics, _variance = _metrics(truth, teacher, autonomous)
    terms = {
        "e2": metrics["e2_per_step"],
        "ed": metrics["ed_per_step"],
        "d2": metrics["d2_per_step"],
    }
    varnorm = {
        "e2": metrics["e2_per_step_varnorm"],
        "ed": metrics["ed_per_step_varnorm"],
        "d2": metrics["d2_per_step_varnorm"],
    }
    epoch = 4
    vis_dir = tmp_path / "vis_during_training"
    written = save_drift_timestep_figures(
        terms, vis_dir, epoch, flip=4, varnorm_terms=varnorm
    )
    raw = drift_timestep_figure_path(vis_dir, epoch, normalized=False)
    norm = drift_timestep_figure_path(vis_dir, epoch, normalized=True)
    assert written == [raw, norm]
    assert os.path.isfile(raw)
    assert os.path.isfile(norm)
    assert os.path.getsize(raw) > 0
    assert raw.endswith(
        os.path.join(
            "vis_during_training",
            "drift_timestep",
            "vis_drift_timestep_epoch_4.png",
        )
    )
    assert norm.endswith(
        os.path.join(
            "vis_during_training",
            "drift_timestep",
            "vis_drift_timestep_varnorm_epoch_4.png",
        )
    )

    fig = build_drift_timestep_figure(terms, flip=4)
    try:
        assert len(fig.axes) == 3
        assert _axes_share_x(fig.axes[0], fig.axes[1])
        assert _axes_share_x(fig.axes[1], fig.axes[2])
        for ax, key in zip(fig.axes, ("e2", "ed", "d2")):
            lines = _trajectory_lines(ax)
            faint = [line for line in lines if line.get_alpha() == pytest.approx(0.2)]
            solid = [line for line in lines if line.get_alpha() == pytest.approx(1.0)]
            assert len(faint) == truth.shape[0]
            assert len(solid) == 1
            with np.errstate(all="ignore"):
                mean = np.nanmean(terms[key], axis=0)
            np.testing.assert_allclose(solid[0].get_ydata(), mean, equal_nan=True)
        assert fig.axes[-1].get_xlabel() == "$t$"
    finally:
        plt.close(fig)


def test_history_does_not_store_the_timestep_arrays(tmp_path):
    for name in _STEP_KEYS:
        assert name not in EPOCH_SERIES_KEYS
        assert name not in DRIFT_SERIES_KEYS
    series = {key: np.zeros(2, dtype=float) for key in EPOCH_SERIES_KEYS}
    cube = np.ones((2, 3, 12), dtype=float)
    for name in _STEP_KEYS:
        series[name] = cube
    history = build_training_history(
        end=2,
        model_name="RNN",
        tag="synthetic",
        auto_sampling_method="none",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon",),
        val_reference={"recon": 1.0},
        series=series,
        dataset_stats={},
        drift_monitored=True,
        drift_horizon=5,
        drift_flip=4,
    )
    for name in _STEP_KEYS:
        assert name not in history
    path = tmp_path / "loss_model.pckl"
    save_training_history(path, history)
    loaded = load_training_history(path)
    for name in _STEP_KEYS:
        assert name not in loaded
    assert "val_base_error" not in loaded
