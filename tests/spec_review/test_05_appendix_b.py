"""Spec 5. Appendix B on the fixed spectrum windows.

e = x_tf - x_true, base_error = ||e||^2
d = x_auto - x_tf, auto_drift = ||d||^2
error_drift_cross = 2 d·e
error_drift_total = ||d+e||^2 = the sum of those three
recon_k equals the total
*_varnorm divides by the per-channel ground-truth variance
The horizon is the K steps after the flip. Default K is 60.

Windows need a clean teacher-forced half, one segment, and an observed
start. Their indices are stored on the history.
"""

from __future__ import annotations

import numpy as np
import pytest

from dvae.eval.utils.local_drift_analysis import (
    drift_metrics_from_series,
    x_true_variance,
)
from dvae.eval.utils.spectrum_validation import (
    capture_fixed_spectrum_windows,
    select_tf_clean_windows,
)
from dvae.run_options import parse_val_drift_horizon
from dvae.training_history import build_training_history


def _mask(truth):
    return np.isfinite(truth)


def _sums(truth, teacher, autonomous, flip, horizon):
    sl = slice(flip, flip + horizon)
    err = teacher[sl] - truth[sl]
    delta = autonomous[sl] - teacher[sl]
    mask = _mask(truth[sl])
    base = float(np.sum(err[mask] ** 2))
    drift = float(np.sum(delta[mask] ** 2))
    cross = float(2.0 * np.sum(delta[mask] * err[mask]))
    total = float(np.sum((delta + err)[mask] ** 2))
    return base, drift, cross, total, mask


def _series_windows():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(12, 2))
    truth[8, 0] = np.nan
    teacher = truth + rng.normal(scale=0.1, size=truth.shape)
    teacher[8, 0] = 0.0
    autonomous = teacher.copy()
    autonomous[6:] += rng.normal(scale=0.2, size=autonomous[6:].shape)
    variance = x_true_variance([truth])
    return truth, teacher, autonomous, variance


def test_default_horizon_is_60_and_only_those_steps_are_scored():
    assert parse_val_drift_horizon(None) == 60
    length = 80
    flip = 10
    horizon = 60
    truth = np.zeros((length, 1))
    teacher = np.zeros_like(truth)
    autonomous = np.zeros_like(truth)
    autonomous[flip : flip + horizon, 0] = 0.5
    autonomous[flip + horizon :, 0] = 100.0
    quiet = drift_metrics_from_series(
        [truth],
        [teacher],
        [autonomous],
        flip=flip,
        horizon=horizon,
        variance=np.array([1.0]),
    )
    autonomous_tail = autonomous.copy()
    autonomous_tail[flip + horizon :, 0] = -100.0
    shifted = drift_metrics_from_series(
        [truth],
        [teacher],
        [autonomous_tail],
        flip=flip,
        horizon=horizon,
        variance=np.array([1.0]),
    )
    assert quiet["base_error"] == pytest.approx(shifted["base_error"])
    assert quiet["auto_drift"] == pytest.approx(shifted["auto_drift"])
    assert quiet["recon_k"] == pytest.approx(shifted["recon_k"])
    assert quiet["auto_drift"] > 0.0


def test_appendix_b_is_the_squared_norm_identity():
    truth, teacher, autonomous, variance = _series_windows()
    flip, horizon = 6, 6
    metrics = drift_metrics_from_series(
        [truth],
        [teacher],
        [autonomous],
        flip=flip,
        horizon=horizon,
        variance=variance,
    )
    base, drift, cross, total, mask = _sums(
        truth, teacher, autonomous, flip, horizon
    )
    assert total == pytest.approx(base + drift + cross)
    # Stored terms are means over observed entries, the autonomous MSE
    # identity. They are not the un-normalised squared norms.
    count = int(mask.sum())
    base, drift, cross, total = (
        value / count for value in (base, drift, cross, total)
    )
    problems = []
    for name, expected in (
        ("base_error", base),
        ("auto_drift", drift),
        ("error_drift_cross", cross),
        ("error_drift_total", total),
        ("recon_k", total),
    ):
        got = float(metrics[name])
        if not got == pytest.approx(expected):
            mean = expected / int(mask.sum())
            problems.append(
                f"{name}: spec ||.||^2={expected}, code={got}, mean={mean}"
            )
    scale = variance.astype(np.float64)
    err = (teacher - truth)[flip : flip + horizon]
    delta = (autonomous - teacher)[flip : flip + horizon]
    usable = mask.copy()
    usable[:, scale <= 0] = False
    usable[:, ~np.isfinite(scale)] = False
    channel_index = np.where(usable)[1]
    var_count = int(usable.sum())
    base_n = float(np.sum((err ** 2)[usable] / scale[channel_index])) / var_count
    drift_n = float(np.sum((delta ** 2)[usable] / scale[channel_index])) / var_count
    cross_n = float(np.sum((2.0 * delta * err)[usable] / scale[channel_index])) / var_count
    for name, expected in (
        ("base_error_varnorm", base_n),
        ("auto_drift_varnorm", drift_n),
        ("error_drift_cross_varnorm", cross_n),
        ("error_drift_total_varnorm", base_n + drift_n + cross_n),
    ):
        got = float(metrics[name])
        if not got == pytest.approx(expected):
            problems.append(f"{name}: spec sum/var={expected}, code={got}")
    assert not problems, "\n".join(problems)


def _segmented_series():
    # Two segments of 30. Window 10, teacher-forced prefix 10.
    series = np.ones((60, 1), dtype=np.float64)
    series[30:, 0] = 2.0
    group = np.array([0] * 30 + [1] * 30)

    class _Dataset:
        seq = series
        overlap = False
        shuffle = False
        val_indices = 0.0
        split = "test"
        coarse_mode = False
        stay_id_per_row = None
        subject_index_per_row = group

    return _Dataset()


def test_fixed_windows_are_clean_in_one_segment_and_indices_are_stored():
    dataset = _segmented_series()
    # Punch a long gap into the first candidate so it cannot qualify,
    # and leave later candidates clean. Ask for fewer windows than qualify.
    dataset.seq = dataset.seq.copy()
    dataset.seq[0:3, 0] = np.nan
    windows, info = capture_fixed_spectrum_windows(
        dataset,
        n_windows=2,
        window_len=10,
        tf_steps=10,
        min_observed=0.95,
        max_gap_frac=0.01,
        return_info=True,
    )
    assert len(windows) == 2
    gap_limit = 0.01 * 10
    for start, frac, gap, qualified in zip(
        info["starts"], info["tf_observed"], info["tf_max_gap"], info["qualified"]
    ):
        assert bool(qualified)
        assert frac + 1e-12 >= 0.95
        assert gap <= gap_limit
        assert np.isfinite(dataset.seq[int(start), 0])
        segment = dataset.subject_index_per_row[int(start) : int(start) + 10]
        assert np.unique(segment).size == 1
    history = _history_with_windows(info)
    np.testing.assert_array_equal(history["val_window_starts"], info["starts"])


def test_a_shortfall_keeps_best_ranked_windows_and_marks_the_clean_ones():
    series = np.ones((40, 1))
    series[0, 0] = np.nan
    series[12:14, 0] = np.nan
    starts = np.array([0, 10, 20, 30])
    # Start 0 is missing, so it is not a candidate. Only starts 20 and
    # 30 have a fully observed prefix of 10. Fewer than requested
    # qualify, so the best-ranked set is kept, including the unclean
    # prefix at 10, and each record says whether it qualified.
    chosen, relaxed = select_tf_clean_windows(
        series,
        starts,
        n_windows=4,
        window_len=10,
        tf_steps=10,
        min_observed=0.95,
        max_gap_frac=0.01,
    )
    assert relaxed is True
    assert len(chosen) == 3
    clean = [record for record in chosen if record["qualified"]]
    assert len(clean) == 2
    assert {record["start"] for record in clean} == {20, 30}
    unclean = [record for record in chosen if not record["qualified"]]
    assert [record["start"] for record in unclean] == [10]
    for record in clean:
        assert record["observed_frac"] + 1e-12 >= 0.95
        assert record["max_gap"] <= 0.01 * 10


def _history_with_windows(info):
    n = 2
    series = {
        "train_loss": np.ones(n),
        "val_loss": np.ones(n),
        "train_recon": np.ones(n),
        "train_kl": np.zeros(n),
        "val_recon": np.ones(n),
        "val_kl": np.zeros(n),
        "val_spectrum": np.ones(n),
        "val_criteria_score": np.ones(n),
        "kl_warm_values": np.zeros(n),
        "auto_warm_values": np.zeros(n),
        "noise_warm_values": np.zeros(n),
        "sequence_len_values": np.full(n, 10.0),
        "cpt_patience_epochs": np.zeros(n),
        "patience_threshold_epochs": np.full(n, 40.0),
        "delta_per_epoch": np.zeros(n),
        "best_state_epochs": np.zeros(n),
        "windows_kept": np.ones(n),
        "windows_dropped_nan": np.zeros(n),
        "windows_dropped_shift": np.zeros(n),
        "gap_splits": np.zeros(n),
        "mean_start_shift": np.zeros(n),
        "time_train": np.ones(n),
        "time_val": np.ones(n),
        "time_val_forward": np.ones(n),
        "time_spectrum": np.ones(n),
        "time_drift": np.ones(n),
        "time_rosenstein": np.zeros(n),
        "time_plot": np.zeros(n),
        "time_eval": np.zeros(n),
    }
    return build_training_history(
        end=n,
        model_name="spec",
        tag="synthetic",
        auto_sampling_method="ss",
        noise_sampling_method="none",
        delta_threshold=1e-4,
        val_criteria=("recon", "spectrum"),
        val_reference={"recon": 1.0, "spectrum": 1.0},
        series=series,
        dataset_stats={"dataset_missing_rate": 0.0},
        window_info=info,
        drift_horizon=60,
        drift_flip=1000,
    )
