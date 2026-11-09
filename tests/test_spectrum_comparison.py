"""Original vs output spectrum plots from the same waveform arrays.

Synthetic only. No wearable data. Missing stays NaN; excluded channels are
annotated; half_half uses the autonomous second half.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dvae.visualizers.spectrum_comparison import (
    autonomous_time_slice,
    compute_original_vs_output_spectra,
    save_original_vs_output_spectrum,
)


def _sine_pair(n=256, channels=2, fs=32.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    original = np.stack(
        [np.sin(2 * np.pi * (1.5 + 0.7 * c) * t) for c in range(channels)],
        axis=1,
    )
    output = original + 0.05 * rng.randn(n, channels)
    return original, output, fs


def test_autonomous_slice_prefers_second_half():
    n = 100
    sel = np.zeros(n)
    sel[50:] = 1.0
    assert autonomous_time_slice(n, sel) == slice(50, 100)
    assert autonomous_time_slice(n, None) == slice(50, 100)


def test_compute_includes_channels_and_excludes_sparse():
    original, output, fs = _sine_pair()
    # Channel 1 almost all missing on the autonomous half.
    original = original.copy()
    original[128:, 1] = np.nan
    result = compute_original_vs_output_spectra(
        original, output, region="autonomous", sample_rate=fs
    )
    assert result["region"] == "autonomous"
    assert result["time_slice_start"] == 128
    assert result["channel_included"][0]
    assert not result["channel_included"][1]
    assert result["n_excluded_channels"] == 1
    assert result["excluded_channel_fraction"] == pytest.approx(0.5)
    assert np.isfinite(result["original_psd"][0]).all()
    assert np.isnan(result["original_psd"][1]).all()
    # Input untouched.
    assert np.isnan(original[128:, 1]).all()


def test_no_resampling_of_nans():
    original, output, fs = _sine_pair(n=64, channels=1)
    original = original.copy()
    original[10:20, 0] = np.nan
    before = original.copy()
    compute_original_vs_output_spectra(
        original, output, region="full", sample_rate=fs
    )
    assert np.array_equal(np.isnan(original), np.isnan(before))
    assert np.isnan(original[10:20, 0]).all()


def test_save_writes_png_and_npz(tmp_path):
    original, output, fs = _sine_pair()
    mode_selector = np.zeros((original.shape[0], original.shape[1]))
    mode_selector[original.shape[0] // 2 :] = 1.0
    result = save_original_vs_output_spectrum(
        original,
        output,
        str(tmp_path),
        explain="half_half_epoch:0",
        mode_selector=mode_selector,
        region="autonomous",
        sample_rate=fs,
        epoch=0,
        mode_name="half_half",
    )
    assert os.path.isfile(result["png_path"])
    assert os.path.isfile(result["npz_path"])
    loaded = np.load(result["npz_path"], allow_pickle=True)
    assert "frequencies_hz" in loaded
    assert "original_psd" in loaded
    assert "output_psd" in loaded
    assert float(loaded["excluded_channel_fraction"]) == pytest.approx(
        result["excluded_channel_fraction"]
    )
    assert int(loaded["epoch"]) == 0
    assert str(loaded["region"]) == "autonomous"
