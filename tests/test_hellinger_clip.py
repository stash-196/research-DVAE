"""Hellinger spectrum distance stays finite for near-identical spectra. Synthetic only."""

import warnings

import numpy as np

from dvae.eval.utils.durstewitz_eval_metrics import hellinger_distance, power_spectrum_error


def _spectrum(n=256):
    f = np.arange(1, n + 1, dtype=np.float64)
    return (np.exp(-((f - 40.0) / 6.0) ** 2) + 0.01).reshape(1, -1)


def test_identical_and_tiny_offset_spectra_are_finite_zero():
    ps = _spectrum()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        same = power_spectrum_error(ps, ps)
        near = power_spectrum_error(ps + 1e-12, ps)
    assert np.isfinite(same) and same == 0.0
    assert np.isfinite(near) and 0.0 <= near < 1e-4


def test_inner_value_slightly_negative_clips_to_zero():
    p = np.full((1, 4), 0.25 + 1e-10)
    assert hellinger_distance(p, p).tolist() == [0.0]


def test_distinct_spectra_unchanged():
    p = np.array([[0.7, 0.2, 0.1]])
    q = np.array([[0.1, 0.2, 0.7]])
    expected = np.sqrt(1 - np.sum(np.sqrt(p * q), axis=1))
    np.testing.assert_allclose(hellinger_distance(p, q), expected)
    assert 0.0 < float(expected[0]) < 1.0
