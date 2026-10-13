"""Spectrum validation with masked (NaN) free-run segments.

Synthetic only. Nothing is filled or resampled: skipped segments leave the
average, and a fully skipped set gives NaN.
"""

import types
import warnings

import numpy as np
import pytest

from dvae.eval.utils.spectrum_validation import (
    free_run_spectrum_error,
    segment_has_enough_finite,
)

TF = 64
LEN = 320


def _windows(n=4, channels=2, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(LEN)
    gts, preds = [], []
    for i in range(n):
        gt = np.stack(
            [np.sin(2 * np.pi * (0.03 + 0.01 * c) * t + i) for c in range(channels)],
            axis=1,
        ) + 0.1 * rng.randn(LEN, channels)
        gts.append(gt)
        preds.append(gt + 0.2 * rng.randn(LEN, channels))
    return gts, preds


def test_no_nan_unchanged_and_zero_skipped():
    gts, preds = _windows()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = free_run_spectrum_error(preds, gts, TF)
        err, skipped = free_run_spectrum_error(preds, gts, TF, return_skipped=True)
    assert isinstance(plain, float) and np.isfinite(plain)
    assert err == plain
    assert skipped == 0


def test_all_nan_window_is_skipped_not_raised():
    gts, preds = _windows()
    expected = free_run_spectrum_error(preds[1:], gts[1:], TF)
    gts[0] = gts[0].copy()
    gts[0][TF:, :] = np.nan
    with pytest.warns(UserWarning, match="skipped 2"):
        err, skipped = free_run_spectrum_error(preds, gts, TF, return_skipped=True)
    assert skipped == 2  # one window x two channels
    assert err == pytest.approx(expected, rel=0, abs=0)
    # Input is untouched: still NaN, nothing filled.
    assert np.isnan(gts[0][TF:]).all()


def test_partial_nan_one_channel():
    gts, preds = _windows()
    gts[2] = gts[2].copy()
    gts[2][TF:, 1] = np.nan  # channel 1 of window 2 fully missing
    gts[3] = gts[3].copy()
    gts[3][TF : TF + 20, 0] = np.nan  # short gap: kept
    with pytest.warns(UserWarning, match="skipped 1"):
        err, skipped = free_run_spectrum_error(preds, gts, TF, return_skipped=True)
    assert skipped == 1
    assert np.isfinite(err)


def test_mostly_missing_segment_is_skipped():
    seg = np.full(100, np.nan)
    seg[:10] = 1.0
    assert not segment_has_enough_finite(seg)
    seg[:60] = 1.0
    assert segment_has_enough_finite(seg)
    assert not segment_has_enough_finite(np.full(10, np.nan))
    assert not segment_has_enough_finite(np.array([1.0]))


def test_nan_prediction_also_skips_pair():
    gts, preds = _windows()
    preds[1] = preds[1].copy()
    preds[1][TF:, 0] = np.nan
    with pytest.warns(UserWarning):
        _err, skipped = free_run_spectrum_error(preds, gts, TF, return_skipped=True)
    assert skipped == 1


def test_all_skipped_gives_nan():
    gts, preds = _windows()
    gts = [g.copy() for g in gts]
    for g in gts:
        g[TF:] = np.nan
    with pytest.warns(UserWarning, match="skipped 8"):
        err, skipped = free_run_spectrum_error(preds, gts, TF, return_skipped=True)
    assert np.isnan(err)
    assert skipped == 8


def _selector(criteria):
    """Bare object with just what _epoch_selection_value needs."""
    from dvae.learning_algo import LearningAlgorithm

    obj = types.SimpleNamespace()
    obj._spectrum_windows = None
    obj._metric_reference = None
    obj._val_reference = None
    obj._selection_skipped = False
    obj._any_selection_eligible = False
    obj._val_spectrum_skipped = None
    obj._val_criteria = tuple(criteria)
    obj._lle_gt = None
    obj.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    obj._raw_criterion = types.MethodType(LearningAlgorithm._raw_criterion, obj)
    obj.select = types.MethodType(LearningAlgorithm._epoch_selection_value, obj)
    return obj


def test_spectrum_nan_epoch_is_skipped_not_scored():
    sel = _selector(("recon", "spectrum"))
    logged = []
    sel.logger.warning = lambda msg, *a, **k: logged.append(msg)
    val_loss = np.array([2.0, 1.0, 0.1, 0.5])
    val_spec = np.array([np.nan, 4.0, np.nan, 2.0])
    score_arr = np.zeros(4)
    # Epoch 0: spectrum NaN before any reference -> skipped, no reference.
    s0, _raw0, norm0 = sel.select(0, val_loss, val_spec, score_arr, {})
    assert np.isnan(s0) and np.isnan(score_arr[0])
    assert all(np.isnan(v) for v in norm0.values())
    assert sel._selection_skipped is True
    assert sel._any_selection_eligible is False
    assert sel._val_reference is None
    assert any("selection skipped: spectrum NaN" in m for m in logged)
    # Epoch 1: first finite spectrum sets the full reference.
    s1, _, _ = sel.select(1, val_loss, val_spec, score_arr, {})
    assert sel._selection_skipped is False
    assert sel._any_selection_eligible is True
    assert sel._val_reference == {"recon": 1.0, "spectrum": 4.0}
    assert s1 == pytest.approx(2.0)
    # Epoch 2: very low recon, but spectrum NaN -> still skipped (no recon fallback).
    s2, _, _ = sel.select(2, val_loss, val_spec, score_arr, {})
    assert np.isnan(s2) and sel._selection_skipped is True
    # Epoch 3: normal sum.
    s3, _, _ = sel.select(3, val_loss, val_spec, score_arr, {})
    assert s3 == pytest.approx(0.5 + 0.5)


def test_single_spectrum_criterion_nan_is_skipped():
    sel = _selector(("spectrum",))
    s0, _, _ = sel.select(0, np.array([1.0]), np.array([np.nan]), np.zeros(1), {})
    assert np.isnan(s0) and sel._selection_skipped is True


def test_recon_only_never_skipped():
    sel = _selector(("recon",))
    s0, _, _ = sel.select(0, np.array([1.0]), np.array([np.nan]), np.zeros(1), {})
    assert s0 == 1.0 and sel._selection_skipped is False


def _train_loop_source():
    import inspect

    from dvae.learning_algo import LearningAlgorithm

    return inspect.getsource(LearningAlgorithm.train)


def test_train_loop_skipped_epoch_cannot_be_best_or_reset_patience():
    src = _train_loop_source()
    assert "selection_skipped = bool(self._selection_skipped)" in src
    assert "if warm_up_happened and not selection_skipped:" in src
    assert "improved = (not selection_skipped) and delta < -delta_threshold" in src


def test_train_loop_warns_when_no_epoch_eligible():
    src = _train_loop_source()
    assert "if ran_epoch and not self._any_selection_eligible:" in src
    assert "NO EPOCH HAD A FINITE SPECTRUM SCORE" in src


def test_recon_spectrum_selection_unchanged_without_nan():
    sel = _selector(("recon", "spectrum"))
    val_loss = np.array([2.0, 1.0])
    val_spec = np.array([4.0, 1.0])
    score_arr = np.zeros(2)
    sel.select(0, val_loss, val_spec, score_arr, {})
    assert sel._val_reference == {"recon": 2.0, "spectrum": 4.0}
    s1, _, norm = sel.select(1, val_loss, val_spec, score_arr, {})
    assert s1 == pytest.approx(0.5 + 0.25)
    assert norm == {"recon": 0.5, "spectrum": 0.25}
