"""Pin the four-panel drift figure from commit c1be352.

Synthetic arrays only. The cosine is the observed-channel formula, the
panels share x and keep their own y scales, and the per-step traces stay
out of the checkpoint.
"""

from __future__ import annotations

import ast
import configparser
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dvae.eval.utils.local_drift_analysis import (
    DRIFT_METRIC_NAMES,
    drift_metrics_from_series,
    fork_tf_and_auto,
)
from dvae.model.mt_rnn import MT_RNN
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


ROOT = Path(__file__).resolve().parents[2]
ALGO_PATH = ROOT / "src" / "dvae" / "learning_algo.py"

_STEP_KEYS = (
    "e2_per_step",
    "ed_per_step",
    "d2_per_step",
    "cos_per_step",
    "e2_per_step_varnorm",
    "ed_per_step_varnorm",
    "d2_per_step_varnorm",
    "cos_per_step_varnorm",
    "val_cos_per_step",
    "val_cos_per_step_varnorm",
)


def _score(truths, teachers, autonomous, variance, flip=5, horizon=5, observation_process=None):
    return drift_metrics_from_series(
        truths,
        teachers,
        autonomous,
        flip=flip,
        horizon=horizon,
        variance=np.asarray(variance, dtype=np.float64),
        observation_process=observation_process,
    )


def _window(err, delta, truth=None):
    """Build one window whose ``e`` and ``d`` equal the given arrays."""
    err = np.asarray(err, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if truth is None:
        truth = np.zeros(err.shape, dtype=np.float64)
    else:
        truth = np.asarray(truth, dtype=np.float64)
    teacher = truth + err
    autonomous = teacher + delta
    return truth, teacher, autonomous


def _finite_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    summed = np.sum(np.where(finite, values, 0.0), axis=0)
    mean = np.full(count.shape, np.nan, dtype=float)
    ok = count > 0
    mean[ok] = summed[ok] / count[ok]
    return mean


def _trajectory_lines(ax):
    return [
        line
        for line in ax.get_lines()
        if line.get_linestyle() == "-" and line.get_alpha() is not None
    ]


def test_panels_share_x_ticks_and_keep_their_own_y_scales():
    length = 40
    e2 = np.vstack(
        [
            np.geomspace(1e-3, 1e3, length),
            np.geomspace(1e-2, 1e2, length),
        ]
    )
    ed = np.vstack(
        [
            np.linspace(-80.0, 80.0, length),
            np.linspace(40.0, -40.0, length),
        ]
    )
    d2 = np.full((2, length), 0.25)
    cos = np.vstack(
        [
            np.linspace(-1.0, 1.0, length),
            np.linspace(1.0, -1.0, length),
        ]
    )
    fig = build_drift_timestep_figure(
        {"e2": e2, "ed": ed, "d2": d2, "cos": cos}, flip=10
    )
    try:
        assert len(fig.axes) == 4
        fig.canvas.draw()
        limits = [ax.get_xlim() for ax in fig.axes]
        ticks = [np.asarray(ax.get_xticks(), dtype=float) for ax in fig.axes]
        for limit in limits:
            assert limit[0] <= 0.0
            assert limit[1] >= length - 1
        for limit in limits[1:]:
            np.testing.assert_allclose(limit, limits[0])
        for tick in ticks[1:]:
            np.testing.assert_allclose(tick, ticks[0])
        cosine = fig.axes[3]
        assert cosine.get_yscale() == "linear"
        assert cosine.get_ylim() == (-1.0, 1.0)
        assert cosine.get_ylabel() == r"$\cos_t$"
        # The squared and product panels are free to pick log or symlog.
        assert fig.axes[0].get_ylim() != cosine.get_ylim()
        assert fig.axes[0].get_yscale() != cosine.get_yscale()
        assert fig.axes[1].get_yscale() != cosine.get_yscale()
        for left, right in zip(fig.axes, fig.axes[1:]):
            assert not left.get_shared_y_axes().joined(left, right)
        label_x = [ax.yaxis.label.get_window_extent().x1 for ax in fig.axes]
        spine_x = [ax.spines["left"].get_window_extent().x0 for ax in fig.axes]
        assert max(label_x) - min(label_x) < 1.0
        assert max(spine_x) - min(spine_x) < 1.0
    finally:
        plt.close(fig)


def test_cosine_uses_observed_channels_and_is_nan_before_the_flip():
    err = np.zeros((10, 2))
    delta = np.zeros((10, 2))
    # Parallel on both channels: (3, 4) · (6, 8) / (5 * 10) = 1.
    err[0] = (3.0, 4.0)
    delta[0] = (6.0, 8.0)
    # Opposite sign.
    err[1] = (3.0, 4.0)
    delta[1] = (-6.0, -8.0)
    # Orthogonal.
    err[2] = (1.0, -1.0)
    delta[2] = (1.0, 1.0)
    # Zero error, nonzero drift.
    err[3] = (0.0, 0.0)
    delta[3] = (1.0, 2.0)
    # Nonzero error, zero drift (a teacher-forced step).
    err[4] = (1.0, 2.0)
    delta[4] = (0.0, 0.0)
    # Horizon after the flip still has observed entries.
    err[5:] = 1.0
    delta[5:] = 1.0
    truth, teacher, autonomous = _window(err, delta)
    # No channel observed at t=6.
    truth[6] = np.nan
    teacher[6] = np.nan
    autonomous[6] = np.nan
    metrics = _score([truth], [teacher], [autonomous], variance=(1.0, 4.0), flip=5)
    cos = metrics["cos_per_step"][0]
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(-1.0)
    assert cos[2] == pytest.approx(0.0)
    assert np.isnan(cos[3])
    assert np.isnan(cos[4])
    assert np.isnan(cos[6])

    # An unobserved channel must not enter the dot product. Channel 1 would
    # make this cosine 0.8; with only channel 0, d is 0 and the cosine is NaN.
    leak_err = np.zeros((10, 2))
    leak_delta = np.zeros((10, 2))
    leak_err[0] = (3.0, 4.0)
    leak_delta[0] = (0.0, 4.0)
    leak_err[5:] = 1.0
    leak_delta[5:] = 1.0
    leak_truth, leak_teacher, leak_auto = _window(leak_err, leak_delta)
    raw = _score([leak_truth], [leak_teacher], [leak_auto], variance=(1.0, 1.0))
    assert raw["cos_per_step"][0, 0] == pytest.approx(0.8)
    hidden = _score(
        [leak_truth],
        [leak_teacher],
        [leak_auto],
        variance=(1.0, 1.0),
        observation_process="only_x_indicate",
    )
    assert np.isnan(hidden["cos_per_step"][0, 0])

    cfg = configparser.ConfigParser()
    cfg["Network"] = {
        "x_dim": "2",
        "dim_rnn": "4",
        "type_rnn": "RNN",
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "off",
    }
    cfg["Training"] = {}
    torch.manual_seed(0)
    model = MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=2,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=4,
        cfg=cfg,
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )
    model.eval()
    window = np.random.default_rng(1).normal(size=(12, 2))
    batch = torch.as_tensor(window, dtype=torch.float32).unsqueeze(1)
    _truth, recon_tf, recon_auto = fork_tf_and_auto(
        model, batch, flip_point=4, device="cpu"
    )
    tf = recon_tf[:, 0].detach().cpu().numpy()
    auto = recon_auto[:, 0].detach().cpu().numpy()
    assert np.max(np.abs(tf[:4] - auto[:4])) == 0.0
    rolled = _score([window], [tf], [auto], variance=np.var(window, axis=0), flip=4, horizon=5)
    assert np.isnan(rolled["cos_per_step"][:, :4]).all()
    assert np.isfinite(rolled["cos_per_step"][:, 5:]).any()


def test_cosine_panel_is_linear_on_minus_one_to_one_with_a_finite_mean():
    cos = np.array(
        [
            [np.nan, np.nan, 1.0, -1.0],
            [np.nan, np.nan, np.nan, 1.0],
        ],
        dtype=float,
    )
    terms = {
        "e2": np.ones_like(cos),
        "ed": np.zeros_like(cos),
        "d2": np.ones_like(cos),
        "cos": cos,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        fig = build_drift_timestep_figure(terms, flip=2)
    try:
        assert not any("Mean of empty slice" in str(item.message) for item in caught)
        ax = fig.axes[3]
        assert ax.get_yscale() == "linear"
        assert ax.get_ylim() == (-1.0, 1.0)
        lines = _trajectory_lines(ax)
        faint = [line for line in lines if line.get_alpha() == pytest.approx(0.2)]
        solid = [line for line in lines if line.get_alpha() == pytest.approx(1.0)]
        assert len(faint) == 2
        assert len(solid) == 1
        np.testing.assert_allclose(faint[0].get_ydata(), cos[0], equal_nan=True)
        np.testing.assert_allclose(faint[1].get_ydata(), cos[1], equal_nan=True)
        expected = _finite_mean(cos)
        assert np.isnan(expected[:2]).all()
        assert expected[2] == pytest.approx(1.0)
        assert expected[3] == pytest.approx(0.0)
        np.testing.assert_allclose(solid[0].get_ydata(), expected, equal_nan=True)
    finally:
        plt.close(fig)


def test_varnorm_figure_scales_by_sqrt_variance_on_positive_channels():
    err = np.zeros((10, 2))
    delta = np.zeros((10, 2))
    err[0] = (3.0, 4.0)
    delta[0] = (0.0, 4.0)
    err[5:] = 1.0
    delta[5:] = 1.0
    truth, teacher, autonomous = _window(err, delta)
    # Channel 1 has var 0, so it is dropped. The remaining channel has d = 0.
    dropped = _score([truth], [teacher], [autonomous], variance=(1.0, 0.0))
    assert dropped["cos_per_step"][0, 0] == pytest.approx(0.8)
    assert np.isnan(dropped["cos_per_step_varnorm"][0, 0])
    assert dropped["e2_per_step_varnorm"][0, 0] == pytest.approx(9.0)

    scaled_err = np.zeros((10, 2))
    scaled_delta = np.zeros((10, 2))
    scaled_err[0] = (3.0, 4.0)
    scaled_delta[0] = (3.0, 0.0)
    scaled_err[5:] = 1.0
    scaled_delta[5:] = 1.0
    scaled = _window(scaled_err, scaled_delta)
    metrics = _score([scaled[0]], [scaled[1]], [scaled[2]], variance=(1.0, 4.0))
    # e/sqrt(var) = (3, 2), d/sqrt(var) = (3, 0).
    assert metrics["cos_per_step"][0, 0] == pytest.approx(0.6)
    assert metrics["cos_per_step_varnorm"][0, 0] == pytest.approx(3.0 / np.sqrt(13.0))
    assert metrics["e2_per_step_varnorm"][0, 0] == pytest.approx((9.0 + 4.0) / 2.0)
    assert metrics["ed_per_step_varnorm"][0, 0] == pytest.approx(9.0 / 2.0)
    assert metrics["d2_per_step_varnorm"][0, 0] == pytest.approx(9.0 / 2.0)

    # A negative variance is not a usable channel. Only channel 0 remains,
    # and its drift is 0, so the variance-normalised cosine is NaN.
    # Keeping channel 1 would make the cosine 0.8.
    negative = _score([truth], [teacher], [autonomous], variance=(1.0, -2.0))
    assert negative["cos_per_step"][0, 0] == pytest.approx(0.8)
    assert np.isnan(negative["cos_per_step_varnorm"][0, 0])

    terms = {
        "e2": metrics["e2_per_step_varnorm"],
        "ed": metrics["ed_per_step_varnorm"],
        "d2": metrics["d2_per_step_varnorm"],
        "cos": metrics["cos_per_step_varnorm"],
    }
    fig = build_drift_timestep_figure(terms, flip=5, normalized=True)
    try:
        assert fig.axes[0].get_ylabel() == r"$e_t^{2}$ / var"
        assert fig.axes[3].get_ylabel() == r"$\cos_t$"
        assert fig.axes[3].get_ylim() == (-1.0, 1.0)
        solid = [
            line
            for line in _trajectory_lines(fig.axes[3])
            if line.get_alpha() == pytest.approx(1.0)
        ]
        assert solid[0].get_ydata()[0] == pytest.approx(3.0 / np.sqrt(13.0))
    finally:
        plt.close(fig)


def test_per_step_arrays_are_not_checkpointed_and_paths_are_unchanged(tmp_path):
    for name in _STEP_KEYS:
        assert name not in EPOCH_SERIES_KEYS
        assert name not in DRIFT_SERIES_KEYS
        assert name not in DRIFT_METRIC_NAMES
    series = {key: np.zeros(2, dtype=float) for key in EPOCH_SERIES_KEYS}
    for name in _STEP_KEYS:
        series[name] = np.ones((2, 8), dtype=float)
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
        drift_horizon=60,
        drift_flip=4,
    )
    for name in _STEP_KEYS:
        assert name not in history
    path = tmp_path / "loss_model.pckl"
    save_training_history(path, history)
    loaded = load_training_history(path)
    for name in _STEP_KEYS:
        assert name not in loaded
    checkpoint = tmp_path / "RNN_checkpoint.pt"
    torch.save(
        {
            "epoch": 0,
            "best_val_loss": 1.0,
            "cpt_patience": 0,
            "model_state_dict": {},
            "optim_state_dict": {},
            "loss_log": history,
        },
        checkpoint,
    )
    restored = torch.load(checkpoint, weights_only=False)
    assert "cos_per_step" not in restored["loss_log"]
    assert "cos_per_step_varnorm" not in restored["loss_log"]

    tree = ast.parse(ALGO_PATH.read_text())
    saved_keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "save" and node.args and isinstance(node.args[0], ast.Dict):
                saved_keys.append(
                    [key.value for key in node.args[0].keys if isinstance(key, ast.Constant)]
                )
    assert saved_keys
    for keys in saved_keys:
        assert "cos" not in keys
        assert not any("per_step" in str(key) for key in keys)

    vis = tmp_path / "vis_during_training"
    raw = drift_timestep_figure_path(vis, 7, normalized=False)
    norm = drift_timestep_figure_path(vis, 7, normalized=True)
    assert raw == os.path.join(
        str(vis), "drift_timestep", "vis_drift_timestep_epoch_7.png"
    )
    assert norm == os.path.join(
        str(vis), "drift_timestep", "vis_drift_timestep_varnorm_epoch_7.png"
    )
    length = 10
    blank = np.zeros((1, length))
    terms = {"e2": blank, "ed": blank, "d2": blank, "cos": blank}
    written = save_drift_timestep_figures(
        terms, vis, 7, flip=4, varnorm_terms=terms
    )
    assert written == [raw, norm]
    assert os.path.isfile(raw)
    assert os.path.isfile(norm)
