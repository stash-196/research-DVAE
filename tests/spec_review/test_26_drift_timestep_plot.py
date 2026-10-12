"""Pin the per-timestep drift figure from commit 9a29fad.

Synthetic arrays only. These tests lock the checklist for the
``e_t^2``, ``e_t d_t``, ``d_t^2`` checkpoint plot: full-window traces,
faint windows plus a solid nanmean, the save path, history and
checkpoint exclusion, unchanged Appendix B scalars, gap NaNs, paper
style, and the sign of ``e d`` against ``error_drift_cross``.
"""

from __future__ import annotations

import ast
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import to_rgba

from dvae.eval.aggregate_plot_style import choose_y_axis_scale
from dvae.eval.utils.local_drift_analysis import (
    DRIFT_METRIC_NAMES,
    drift_metrics_from_series,
)
from dvae.eval.utils.spectrum_validation import select_tf_clean_windows
from dvae.run_options import parse_optional_int, parse_val_drift_horizon
from dvae.training_history import (
    DRIFT_SERIES_KEYS,
    EPOCH_SERIES_KEYS,
    build_training_history,
    load_training_history,
    regenerate_training_figures,
    save_training_history,
)
from dvae.visualizers.visualize_training_metrics import (
    build_drift_timestep_figure,
    drift_timestep_figure_path,
    save_drift_timestep_figure,
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
    "val_e2_per_step",
    "val_ed_per_step",
    "val_d2_per_step",
    "val_cos_per_step",
    "val_e2_per_step_varnorm",
    "val_ed_per_step_varnorm",
    "val_d2_per_step_varnorm",
    "val_cos_per_step_varnorm",
)
_SCALAR_KEYS = (
    "base_error",
    "auto_drift",
    "error_drift_cross",
    "error_drift_total",
    "base_error_varnorm",
    "auto_drift_varnorm",
    "error_drift_cross_varnorm",
    "error_drift_total_varnorm",
    "recon_k",
)
_PANEL_COLORS = {
    "e2": "tab:blue",
    "ed": "tab:green",
    "d2": "tab:orange",
}


def _entry_mask(values, observation_process=None):
    """Observed entries: finite ground truth, and signal channels only."""
    arr = np.asarray(values)
    mask = ~np.isnan(arr)
    indicate = observation_process is not None and (
        observation_process == "only_x_indicate"
        or str(observation_process).endswith("_indicate")
    )
    if indicate and arr.shape[-1] > 1:
        keep = np.ones(arr.shape[-1], dtype=bool)
        keep[1::2] = False
        mask = mask & keep.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return mask


def _masked_mean(values, mask) -> float:
    chosen = np.asarray(mask, dtype=bool)
    count = int(chosen.sum())
    assert count > 0
    return float(np.sum(np.where(chosen, values, 0.0)) / count)


def _channel_mean(values, mask) -> np.ndarray:
    chosen = np.asarray(mask, dtype=bool)
    count = chosen.sum(axis=-1)
    summed = np.sum(np.where(chosen, values, 0.0), axis=-1)
    out = np.full(count.shape, np.nan, dtype=np.float64)
    ok = count > 0
    out[ok] = summed[ok] / count[ok]
    return out


def _legacy_scalars(truth, teacher, autonomous, flip, horizon, variance, observation_process=None):
    """Pre-commit scalar reduction: post-flip horizon only, pooled entries."""
    sl = slice(flip, flip + horizon)
    err = teacher[:, sl] - truth[:, sl]
    delta = autonomous[:, sl] - teacher[:, sl]
    auto_err = autonomous[:, sl] - truth[:, sl]
    mask = _entry_mask(truth[:, sl], observation_process)
    e2 = err * err
    d2 = delta * delta
    cross = 2.0 * delta * err
    scale = np.asarray(variance, dtype=np.float64).reshape(-1)
    safe = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    usable = mask & np.isfinite(scale) & (scale > 0.0)
    weights = np.where(usable, 1.0 / safe.reshape(1, 1, -1), 0.0)
    base = _masked_mean(e2, mask)
    drift = _masked_mean(d2, mask)
    cross_term = _masked_mean(cross, mask)
    if int(usable.sum()) == 0:
        base_n = drift_n = cross_n = float("nan")
    else:
        base_n = _masked_mean(e2 * weights, usable)
        drift_n = _masked_mean(d2 * weights, usable)
        cross_n = _masked_mean(cross * weights, usable)
    return {
        "base_error": base,
        "auto_drift": drift,
        "error_drift_cross": cross_term,
        "error_drift_total": base + drift + cross_term,
        "base_error_varnorm": base_n,
        "auto_drift_varnorm": drift_n,
        "error_drift_cross_varnorm": cross_n,
        "error_drift_total_varnorm": base_n + drift_n + cross_n,
        "recon_k": _masked_mean(auto_err * auto_err, mask),
    }


def _as_lists(truth, teacher, autonomous):
    return (
        [truth[i] for i in range(truth.shape[0])],
        [teacher[i] for i in range(truth.shape[0])],
        [autonomous[i] for i in range(truth.shape[0])],
    )


def _score(truth, teacher, autonomous, flip, horizon, variance, observation_process=None):
    truths, teachers, autos = _as_lists(truth, teacher, autonomous)
    return drift_metrics_from_series(
        truths,
        teachers,
        autos,
        flip=flip,
        horizon=horizon,
        variance=variance,
        observation_process=observation_process,
    )


def _production_windows():
    """Two windows at the production length. Horizon stays the default 60."""
    rng = np.random.default_rng(11)
    n_windows, length, channels = 2, 2000, 2
    truth = rng.normal(size=(n_windows, length, channels))
    teacher = truth + rng.normal(scale=0.2, size=truth.shape)
    autonomous = teacher + rng.normal(scale=0.3, size=truth.shape)
    # A one-step gap in window 0, and a channel gap in window 1.
    truth[0, 40, :] = np.nan
    teacher[0, 40, :] = np.nan
    autonomous[0, 40, :] = np.nan
    truth[1, 1500, 1] = np.nan
    teacher[1, 1500, 1] = np.nan
    autonomous[1, 1500, 1] = np.nan
    # Opposite signs on the scored horizon so e*d is negative there.
    flip, horizon = 1000, 60
    teacher[:, flip : flip + horizon, :] = truth[:, flip : flip + horizon, :] + 1.0
    autonomous[:, flip : flip + horizon, :] = teacher[:, flip : flip + horizon, :] - 2.0
    variance = np.array([1.5, 0.5])
    return truth, teacher, autonomous, flip, horizon, variance


def _axes_share_x(left, right):
    group = left.get_shared_x_axes()
    if hasattr(group, "joined"):
        return bool(group.joined(left, right))
    return right in group.get_siblings(left)


def _trajectory_lines(ax):
    return [line for line in ax.lines if line.get_linestyle() not in ("--", "dashed")]


def _algo_tree():
    return ast.parse(ALGO_PATH.read_text())


def _call_name(node):
    func = getattr(node, "func", None)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_ifs(node, parents):
    found = []
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.If):
            found.append(cur)
    return found


def _enclosing_function(node, parents):
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, ast.FunctionDef):
            return cur.name
    return None


def test_traces_cover_the_production_window_not_the_drift_horizon():
    assert parse_optional_int(None, 2000, "val_spectrum_window_len") == 2000
    assert parse_val_drift_horizon(None) == 60
    tree = _algo_tree()
    defaults = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "parse_optional_int"
        and len(node.args) >= 3
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value == "val_spectrum_window_len"
    ]
    assert defaults
    assert defaults[0].args[1].value == 2000

    truth, teacher, autonomous, flip, horizon, variance = _production_windows()
    metrics = _score(truth, teacher, autonomous, flip, horizon, variance)
    for key in ("e2_per_step", "ed_per_step", "d2_per_step"):
        assert metrics[key].shape == (truth.shape[0], truth.shape[1])
        assert metrics[key].shape[1] == 2000
        assert metrics[key].shape[1] != horizon

    terms = {
        "e2": metrics["e2_per_step"],
        "ed": metrics["ed_per_step"],
        "d2": metrics["d2_per_step"],
        "cos": metrics["cos_per_step"],
    }
    fig = build_drift_timestep_figure(terms, flip=flip)
    try:
        assert len(fig.axes) == 4
        positions = [ax.get_position().y0 for ax in fig.axes]
        assert positions[0] > positions[1] > positions[2]
        assert _axes_share_x(fig.axes[0], fig.axes[1])
        assert _axes_share_x(fig.axes[1], fig.axes[2])
        for ax in fig.axes:
            lines = _trajectory_lines(ax)
            assert lines
            xdata = np.asarray(lines[0].get_xdata())
            assert xdata.shape == (2000,)
            assert xdata[0] == 0
            assert xdata[-1] == 1999
            for line in lines[1:]:
                np.testing.assert_array_equal(line.get_xdata(), xdata)
        assert fig.axes[-1].get_xlabel() == "$t$"
        assert fig.axes[0].get_xlabel() == ""
    finally:
        plt.close(fig)


def test_each_window_is_faint_and_the_mean_is_solid():
    truth, teacher, autonomous, flip, horizon, variance = _production_windows()
    metrics = _score(truth, teacher, autonomous, flip, horizon, variance)
    terms = {
        "e2": metrics["e2_per_step"],
        "ed": metrics["ed_per_step"],
        "d2": metrics["d2_per_step"],
        "cos": metrics["cos_per_step"],
    }
    fig = build_drift_timestep_figure(terms, flip=flip)
    try:
        for ax, key in zip(fig.axes, ("e2", "ed", "d2")):
            lines = _trajectory_lines(ax)
            faint = [line for line in lines if line.get_alpha() == pytest.approx(0.2)]
            solid = [line for line in lines if line.get_alpha() == pytest.approx(1.0)]
            assert len(faint) == truth.shape[0]
            assert len(solid) == 1
            assert solid[0].get_linestyle() == "-"
            with np.errstate(all="ignore"):
                mean = np.nanmean(terms[key], axis=0)
            np.testing.assert_allclose(solid[0].get_ydata(), mean, equal_nan=True)
            # The all-channel gap in window 0 must not pull the mean to zero.
            assert np.isnan(terms[key][0, 40])
            assert solid[0].get_ydata()[40] == pytest.approx(terms[key][1, 40])
            assert to_rgba(solid[0].get_color()) == to_rgba(_PANEL_COLORS[key])
            assert to_rgba(faint[0].get_color()) == to_rgba(_PANEL_COLORS[key])
    finally:
        plt.close(fig)


def test_checkpoint_path_contains_the_epoch(tmp_path):
    truth, teacher, autonomous, flip, horizon, variance = _production_windows()
    metrics = _score(truth, teacher, autonomous, flip, horizon, variance)
    terms = {key: metrics[key + "_per_step"] for key in ("e2", "ed", "d2", "cos")}
    varnorm = {
        key: metrics[key + "_per_step_varnorm"] for key in ("e2", "ed", "d2", "cos")
    }
    epoch = 12
    save_dir = tmp_path / "run"
    # The training loop passes ``{save_dir}/vis_during_training``.
    vis_dir = save_dir / "vis_during_training"
    written = save_drift_timestep_figures(
        terms, vis_dir, epoch, flip=flip, varnorm_terms=varnorm
    )
    raw = drift_timestep_figure_path(vis_dir, epoch, normalized=False)
    assert written[0] == raw
    assert os.path.isfile(raw)
    assert raw.endswith(
        os.path.join(
            "vis_during_training",
            "drift_timestep",
            "vis_drift_timestep_epoch_12.png",
        )
    )
    assert "12" in os.path.basename(raw)
    norm = drift_timestep_figure_path(vis_dir, epoch, normalized=True)
    assert norm in written
    assert norm.endswith("vis_drift_timestep_varnorm_epoch_12.png")


def test_draw_is_on_every_save_frequency_checkpoint_for_any_criterion():
    tree = _algo_tree()
    parents = _parents(tree)
    draws = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "save_drift_timestep_figures"
    ]
    assert len(draws) == 1
    tests = [ast.unparse(node.test) for node in _enclosing_ifs(draws[0], parents)]
    assert tests == [
        "self._latest_drift_steps is not None",
        "epoch % save_frequency == 0",
    ]
    assert _enclosing_function(draws[0], parents) == "train"
    for text in tests:
        assert "val_criteria" not in text
        assert "criterion" not in text

    assigns = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_latest_drift_steps"
                and isinstance(node.value, ast.Dict)
            ):
                assigns.append(node)
    assert len(assigns) == 1
    keys = []
    for key in assigns[0].value.keys:
        assert isinstance(key, ast.Constant)
        keys.append(key.value)
    assert keys == ["e2", "ed", "d2", "cos"]
    guard = [ast.unparse(node.test) for node in _enclosing_ifs(assigns[0], parents)]
    assert guard == ["self._spectrum_windows is not None"]
    assert _enclosing_function(assigns[0], parents) == "_epoch_selection_value"

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "_epoch_selection_value"
    ]
    assert len(calls) == 1
    selection_guards = [ast.unparse(node.test) for node in _enclosing_ifs(calls[0], parents)]
    assert selection_guards == []


def test_per_step_arrays_stay_out_of_history_and_the_checkpoint(tmp_path):
    for name in _STEP_KEYS:
        assert name not in EPOCH_SERIES_KEYS
        assert name not in DRIFT_SERIES_KEYS
        assert name not in DRIFT_METRIC_NAMES
    series = {key: np.zeros(2, dtype=float) for key in EPOCH_SERIES_KEYS}
    series["train_loss"] = np.ones(2)
    series["val_loss"] = np.ones(2)
    cube = np.ones((2, 3, 2000), dtype=float)
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
        drift_horizon=60,
        drift_flip=1000,
    )
    for name in _STEP_KEYS:
        assert name not in history
    blob = pickle.dumps(history)
    for name in _STEP_KEYS:
        assert name.encode() not in blob
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
    assert set(restored) == {
        "epoch",
        "best_val_loss",
        "cpt_patience",
        "model_state_dict",
        "optim_state_dict",
        "loss_log",
    }
    for name in _STEP_KEYS:
        assert name not in restored["loss_log"]

    tree = _algo_tree()
    saved_keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "save" and node.args:
            payload = node.args[0]
            if isinstance(payload, ast.Dict):
                saved_keys.append([ast.unparse(key) for key in payload.keys])
    assert saved_keys
    for keys in saved_keys:
        for key in keys:
            assert "per_step" not in key
            assert key.strip("'\"") not in {"e2", "ed", "d2", "cos"}

    fig_dir = tmp_path / "redraw"
    written = regenerate_training_figures(history, fig_dir, tag="synthetic")
    assert written
    assert not any("drift_timestep" in path for path in written)
    assert not (fig_dir / "drift_timestep").exists()


def test_scalar_terms_ignore_steps_outside_the_horizon():
    truth, teacher, autonomous, flip, horizon, variance = _production_windows()
    original = _score(truth, teacher, autonomous, flip, horizon, variance)
    expected = _legacy_scalars(truth, teacher, autonomous, flip, horizon, variance)
    for key in _SCALAR_KEYS:
        assert original[key] == pytest.approx(expected[key])
    assert original["error_drift_total"] == pytest.approx(
        original["base_error"] + original["auto_drift"] + original["error_drift_cross"]
    )
    assert original["recon_k"] == pytest.approx(original["error_drift_total"])

    bumped = teacher.copy()
    bumped_auto = autonomous.copy()
    bumped[:, :flip, :] += 9.0
    bumped_auto[:, :flip, :] -= 9.0
    bumped[:, flip + horizon :, :] += 4.0
    bumped_auto[:, flip + horizon :, :] -= 4.0
    shifted = _score(truth, bumped, bumped_auto, flip, horizon, variance)
    for key in _SCALAR_KEYS:
        assert shifted[key] == pytest.approx(original[key])
    assert not np.allclose(
        shifted["e2_per_step"][:, 0],
        original["e2_per_step"][:, 0],
        equal_nan=True,
    )
    assert not np.allclose(
        shifted["d2_per_step"][:, -1],
        original["d2_per_step"][:, -1],
        equal_nan=True,
    )


def test_gap_checked_windows_leave_nans_and_nanmean_skips_them():
    window_len, tf_steps, horizon = 30, 20, 8
    series = np.ones((90, 1), dtype=np.float64)
    # One missing teacher-forced step still qualifies (limit is 4).
    series[5, 0] = np.nan
    # A long teacher-forced gap does not qualify.
    series[32:42, 0] = np.nan
    # A gap after the flip stays in a qualified window.
    series[80, 0] = np.nan
    chosen, relaxed = select_tf_clean_windows(
        series,
        np.array([0, 30, 60]),
        n_windows=2,
        window_len=window_len,
        tf_steps=tf_steps,
        min_observed=0.95,
        max_gap_frac=0.2,
        seed=0,
    )
    assert relaxed is False
    assert [record["start"] for record in chosen] == [0, 60]
    truth = np.stack(
        [series[start : start + window_len] for start in (0, 60)], axis=0
    )
    teacher = np.where(np.isnan(truth), np.nan, truth + 0.4)
    autonomous = np.where(np.isnan(teacher), np.nan, teacher - 0.5)
    variance = np.array([1.0])
    metrics = _score(truth, teacher, autonomous, tf_steps, horizon, variance)
    assert metrics["e2_per_step"].shape == (2, window_len)
    # The rejected window is not a row. Its 10-step gap would have been t=2.
    assert not np.isnan(metrics["e2_per_step"][1, 2])
    # Qualified gaps stay NaN rather than zero.
    assert np.isnan(metrics["e2_per_step"][0, 5])
    assert np.isnan(metrics["ed_per_step"][0, 5])
    assert np.isnan(metrics["d2_per_step"][1, 20])
    with np.errstate(all="ignore"):
        mean = np.nanmean(metrics["e2_per_step"], axis=0)
    assert mean[5] == pytest.approx(metrics["e2_per_step"][1, 5])
    assert mean[5] != pytest.approx(0.0)
    both = metrics["e2_per_step"].copy()
    both[:, 7] = np.nan
    with np.errstate(all="ignore"):
        empty = np.nanmean(both, axis=0)
    assert np.isnan(empty[7])

    terms = {
        "e2": metrics["e2_per_step"],
        "ed": metrics["ed_per_step"],
        "d2": metrics["d2_per_step"],
        "cos": metrics["cos_per_step"],
    }
    fig = build_drift_timestep_figure(terms, flip=tf_steps)
    try:
        solid = [
            line
            for line in _trajectory_lines(fig.axes[0])
            if line.get_alpha() == pytest.approx(1.0)
        ]
        assert len(solid) == 1
        assert solid[0].get_ydata()[5] == pytest.approx(mean[5])
        # Window 1 is missing at the flip. The mean keeps window 0.
        assert np.isnan(metrics["e2_per_step"][1, 20])
        assert solid[0].get_ydata()[20] == pytest.approx(metrics["e2_per_step"][0, 20])
    finally:
        plt.close(fig)


def test_indicate_channels_and_dead_variance_are_left_out():
    rng = np.random.default_rng(3)
    length = 16
    truth = rng.normal(size=(2, length, 2))
    teacher = truth + 0.2
    autonomous = teacher + 0.5
    # Mask channel would dominate if it were kept.
    teacher[:, :, 1] += 8.0
    autonomous[:, :, 1] += 8.0
    truth[0, 3, 0] = np.nan
    teacher[0, 3, 0] = np.nan
    autonomous[0, 3, 0] = np.nan
    process = "only_x_indicate"
    variance = np.array([2.0, 0.0])
    flip, horizon = 4, 6
    metrics = _score(
        truth, teacher, autonomous, flip, horizon, variance, observation_process=process
    )
    expected = _legacy_scalars(
        truth, teacher, autonomous, flip, horizon, variance, observation_process=process
    )
    for key in _SCALAR_KEYS:
        assert metrics[key] == pytest.approx(expected[key], nan_ok=True)
    mask = _entry_mask(truth, process)
    assert not mask[:, :, 1].any()
    err = teacher - truth
    delta = autonomous - teacher
    np.testing.assert_allclose(
        metrics["e2_per_step"],
        _channel_mean(err * err, mask),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        metrics["ed_per_step"],
        _channel_mean(err * delta, mask),
        equal_nan=True,
    )
    # Only the signal channel remains, so the mask-channel spike is absent.
    signal = (teacher[1, 1, 0] - truth[1, 1, 0]) ** 2
    assert metrics["e2_per_step"][1, 1] == pytest.approx(signal)
    assert metrics["e2_per_step"][1, 1] < 1.0
    usable = mask & np.isfinite(variance) & (variance > 0.0)
    safe = np.where(np.isfinite(variance) & (variance > 0.0), variance, 1.0)
    inv = (1.0 / safe).reshape(1, 1, -1)
    np.testing.assert_allclose(
        metrics["e2_per_step_varnorm"],
        _channel_mean(err * err * inv, usable),
        equal_nan=True,
    )


def test_ed_sign_matches_error_drift_cross():
    truth, teacher, autonomous, flip, horizon, variance = _production_windows()
    metrics = _score(truth, teacher, autonomous, flip, horizon, variance)
    sl = slice(flip, flip + horizon)
    err = teacher[:, sl] - truth[:, sl]
    delta = autonomous[:, sl] - teacher[:, sl]
    product = err * delta
    mask = _entry_mask(truth[:, sl])
    pooled = _masked_mean(product, mask)
    # e = +1, d = -2 on the horizon, so e*d is -2 and the cross term is -4.
    assert pooled == pytest.approx(-2.0)
    assert metrics["error_drift_cross"] == pytest.approx(2.0 * pooled)
    assert metrics["error_drift_cross"] < 0
    horizon_ed = metrics["ed_per_step"][:, sl]
    assert np.all(horizon_ed < 0)
    np.testing.assert_allclose(horizon_ed, product.mean(axis=-1), equal_nan=True)
    assert not np.allclose(horizon_ed, 2.0 * product.mean(axis=-1))

    fig = build_drift_timestep_figure(
        {
            "e2": metrics["e2_per_step"],
            "ed": metrics["ed_per_step"],
            "d2": metrics["d2_per_step"],
            "cos": metrics["cos_per_step"],
        },
        flip=flip,
    )
    try:
        assert fig.axes[0].get_ylabel() == r"$e_t^{2}$"
        assert fig.axes[1].get_ylabel() == r"$e_t d_t$"
        assert fig.axes[2].get_ylabel() == r"$d_t^{2}$"
        solid = [
            line
            for line in _trajectory_lines(fig.axes[1])
            if line.get_alpha() == pytest.approx(1.0)
        ]
        assert solid[0].get_ydata()[flip] < 0
    finally:
        plt.close(fig)


def test_panels_follow_aggregate_plot_style(tmp_path):
    rng = np.random.default_rng(5)
    n_windows, length = 3, 40
    e2 = np.abs(rng.normal(loc=20.0, scale=30.0, size=(n_windows, length))) + 1e-3
    ed = rng.normal(scale=80.0, size=(n_windows, length))
    ed[:, 4] = np.nan
    d2 = np.full((n_windows, length), 0.2)
    terms = {
        "e2": e2,
        "ed": ed,
        "d2": d2,
        "cos": np.clip(ed, -1.0, 1.0),
    }
    fig = build_drift_timestep_figure(terms, flip=10)
    try:
        fig.canvas.draw()
        assert fig._suptitle is None
        for ax, key in zip(fig.axes, ("e2", "ed", "d2")):
            assert ax.get_title() == ""
            assert ax.yaxis.label.get_fontsize() == pytest.approx(24)
            assert "Serif" in ax.yaxis.label.get_fontname()
            assert ax.get_yscale() == choose_y_axis_scale(terms[key])
            legend = ax.get_legend()
            assert [text.get_text() for text in legend.get_texts()] == ["window", "mean"]
            assert legend.get_texts()[0].get_fontsize() == pytest.approx(20)
            assert "Serif" in legend.get_texts()[0].get_fontname()
            solid = [
                line
                for line in _trajectory_lines(ax)
                if line.get_alpha() == pytest.approx(1.0)
            ]
            assert solid[0].get_linewidth() == pytest.approx(2.0)
        yticks = fig.axes[1].get_yticklabels()
        assert yticks
        assert yticks[0].get_fontsize() == pytest.approx(20)
        assert "Serif" in yticks[0].get_fontname()
        xticks = fig.axes[-1].get_xticklabels()
        assert xticks
        assert xticks[0].get_fontsize() == pytest.approx(20)
        # Top panels keep the shared axis and do not repeat tick labels.
        assert fig.axes[0].get_xticklabels() == []
    finally:
        plt.close(fig)

    out = tmp_path / "drift-style.png"
    save_drift_timestep_figure(terms, out)
    assert out.is_file()
    assert out.stat().st_size > 0
