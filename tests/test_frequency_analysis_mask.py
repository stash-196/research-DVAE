"""Shared-mask spectra keep one frequency grid for ground truth and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from dvae.eval.utils.frequency_analysis import (
    _spectrum_pair_errors,
    aligned_power_spectra,
)


def test_gap_does_not_drop_frequency_bins():
    rng = np.random.RandomState(0)
    n = 32
    gt = np.sin(np.linspace(0, 4 * np.pi, n)) + 0.01 * rng.randn(n)
    gt = gt.copy()
    gt[:8] = np.nan
    pred = np.sin(np.linspace(0, 4 * np.pi, n))
    gt_ps, pred_ps = aligned_power_spectra(gt, pred, sampling_rate=1.0)
    full = np.fft.fftfreq(n, d=1.0)
    assert gt_ps.shape == pred_ps.shape
    assert gt_ps.shape[0] == int(np.sum(full > 0))


def test_identical_gapped_series_have_zero_hellinger():
    n = 64
    gt = np.sin(np.linspace(0, 6 * np.pi, n))
    gt = gt.copy()
    gt[8:16] = np.nan
    pred = gt.copy()
    pred[np.isnan(gt)] = 999.0
    err = _spectrum_pair_errors(gt, pred, sampling_rate=1.0)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="share length"):
        aligned_power_spectra(np.ones(16), np.ones(15), sampling_rate=1.0)
