"""Teacher-forced and autonomous spectra from the same batch.

Synthetic only. Missing stays NaN. The ground-truth gap mask is applied
to the model output. Channels that fail the finiteness rule are excluded.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from dvae.visualizers.spectrum_comparison import (
    alpha_line_frequencies,
    autonomous_time_slice,
    compute_original_vs_output_spectra,
    dataset_sampling_freq_hz,
    save_original_vs_output_spectrum,
    timescale_alphas_from_model,
)


def _batch(n=128, sequences=3, channels=2, fs=32.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    original = np.stack(
        [
            np.stack(
                [
                    np.sin(2 * np.pi * (1.5 + 0.4 * c + 0.05 * s) * t)
                    for c in range(channels)
                ],
                axis=1,
            )
            for s in range(sequences)
        ],
        axis=0,
    )
    output = original + 0.02 * rng.randn(sequences, n, channels)
    return original, output, fs


def _half_selector(n_seq, n_time, n_ch):
    sel = np.zeros((n_seq, n_time, n_ch), dtype=np.float64)
    sel[:, n_time // 2 :, :] = 1.0
    return sel


def test_autonomous_slice_prefers_second_half():
    n = 100
    sel = np.zeros(n)
    sel[50:] = 1.0
    assert autonomous_time_slice(n, sel) == slice(50, 100)
    assert autonomous_time_slice(n, None) == slice(50, 100)


def test_even_bursts_are_not_a_contiguous_span():
    original, output, fs = _batch(n=100, sequences=1, channels=1)
    sel = np.zeros(100)
    sel[10:20] = 1.0
    sel[40:50] = 1.0
    result = compute_original_vs_output_spectra(
        original[0], output[0], mode_selector=sel, sample_rate=fs
    )
    # 20 autonomous steps, not the 40-step span from the first to the last.
    assert int(result["segment_length_autonomous"][0]) == 20
    assert int(result["segment_length_tf"][0]) == 80
    assert autonomous_time_slice(100, sel) == slice(10, 50)


def test_batch_average_counts_every_sequence():
    original, output, fs = _batch()
    sel = _half_selector(*original.shape)
    result = compute_original_vs_output_spectra(
        original, output, mode_selector=sel, sample_rate=fs
    )
    assert result["n_sequences"] == 3
    assert result["channel_included_tf"].all()
    assert result["channel_included_autonomous"].all()
    assert result["n_sequences_averaged_tf"].tolist() == [3, 3]
    assert result["n_sequences_averaged_autonomous"].tolist() == [3, 3]
    assert result["frequency_unit"] == "Hz"
    # Item 0 alone is a different average than the full batch.
    alone = compute_original_vs_output_spectra(
        original[0], output[0], mode_selector=sel[0], sample_rate=fs
    )
    assert not np.allclose(
        result["original_psd_tf"][0], alone["original_psd_tf"][0]
    )


def test_shared_mask_makes_gap_fill_irrelevant():
    original, output, fs = _batch(n=64, sequences=2, channels=1)
    original = original.copy()
    original[:, 10:18, 0] = np.nan
    filled = original.copy()
    filled[np.isnan(original)] = 999.0
    # Prediction equals the original, including a bogus fill in the gaps.
    pred = filled.copy()
    pred[np.isfinite(original)] = original[np.isfinite(original)]
    result = compute_original_vs_output_spectra(
        original, pred, mode_selector=None, sample_rate=fs
    )
    assert result["channel_included_tf"][0]
    assert np.allclose(
        result["original_psd_tf"][0],
        result["output_psd_tf"][0],
        equal_nan=True,
    )
    assert np.isnan(original[:, 10:18, 0]).all()


def test_sparse_channel_is_excluded_and_inputs_stay_nan():
    original, output, fs = _batch(sequences=1, channels=2)
    original = original.copy()
    original[:, original.shape[1] // 2 :, 1] = np.nan
    before = original.copy()
    result = compute_original_vs_output_spectra(
        original[0],
        output[0],
        mode_selector=_half_selector(1, original.shape[1], 2)[0],
        sample_rate=fs,
    )
    assert result["channel_included_autonomous"][0]
    assert not result["channel_included_autonomous"][1]
    assert result["excluded_channel_fraction_autonomous"] == pytest.approx(0.5)
    assert np.array_equal(np.isnan(original), np.isnan(before))


def test_unknown_rate_is_cycles_per_step():
    original, _output, _fs = _batch(sequences=1, channels=1)
    result = compute_original_vs_output_spectra(
        original[0], original[0], sample_rate=None
    )
    assert result["frequency_unit"] == "cycles/step"
    assert np.isnan(result["sample_rate"])
    n_auto = int(result["segment_length_autonomous"][0])
    expected = np.fft.rfftfreq(n_auto, d=1.0)[1:]
    got = result["frequencies_autonomous"][0]
    got = got[np.isfinite(got)]
    assert got.shape == expected.shape
    assert np.allclose(got, expected)


def test_alpha_lines_use_sampling_rate():
    alphas = np.array([0.25, 0.5])
    assert np.allclose(alpha_line_frequencies(alphas, 32.0), alphas * 32.0)
    assert np.allclose(alpha_line_frequencies(alphas, None), alphas)

    class _MT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmas = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

    got = timescale_alphas_from_model(_MT())
    assert got.shape == (2,)
    assert got[0] == pytest.approx(0.5)
    assert timescale_alphas_from_model(torch.nn.Linear(2, 2)).size == 0
    assert dataset_sampling_freq_hz(object()) is None


def test_save_writes_split_npz(tmp_path):
    original, output, fs = _batch()
    sel = _half_selector(*original.shape)
    alphas = np.array([2.0, 80.0])  # 80 Hz is above the Nyquist of this series
    result = save_original_vs_output_spectrum(
        original,
        output,
        str(tmp_path),
        explain="half_half_epoch:0",
        mode_selector=sel,
        sample_rate=fs,
        epoch=0,
        mode_name="half_half",
        alpha_frequencies=alpha_line_frequencies(alphas, fs),
    )
    assert os.path.isfile(result["png_path"])
    assert os.path.isfile(result["npz_path"])
    loaded = np.load(result["npz_path"], allow_pickle=True)
    for key in (
        "frequencies_tf",
        "frequencies_autonomous",
        "original_psd_tf",
        "output_psd_tf",
        "original_psd_autonomous",
        "output_psd_autonomous",
        "alpha_frequencies",
        "channel_included_tf",
        "channel_included_autonomous",
        "excluded_channel_fraction_tf",
        "excluded_channel_fraction_autonomous",
        "missing_sample_fraction_tf",
        "missing_sample_fraction_autonomous",
        "n_sequences",
        "n_sequences_averaged_tf",
        "n_sequences_averaged_autonomous",
        "frequency_unit",
    ):
        assert key in loaded
    assert int(loaded["n_sequences"]) == 3
    assert int(loaded["epoch"]) == 0
    assert str(loaded["frequency_unit"]) == "Hz"
    assert loaded["n_sequences_averaged_tf"].tolist() == [3, 3]
    # The in-band line is stored; drawing must not require the out-of-band one.
    assert np.allclose(loaded["alpha_frequencies"], [2.0 * fs, 80.0 * fs])


def test_timescale_lines_at_or_above_nyquist_are_noted_not_drawn(tmp_path):
    original, output, fs = _batch()
    # fs/2 is Nyquist. 2 Hz is in band; 80 Hz is above it. The axis stays
    # on the rFFT grid. The NPZ keeps every alpha frequency.
    alpha_hz = np.array([2.0, 80.0])
    result = save_original_vs_output_spectrum(
        original,
        output,
        str(tmp_path),
        explain="half_half_epoch:0",
        mode_selector=_half_selector(*original.shape),
        sample_rate=fs,
        epoch=0,
        mode_name="half_half",
        alpha_frequencies=alpha_hz,
    )
    assert result["n_timescale_lines_above_nyquist"] >= 1
    assert "timescale lines above Nyquist not shown" in result["annotation"]
    assert result["axis_xmax"] is not None
    assert result["axis_xmax"] < 80.0
    loaded = np.load(result["npz_path"], allow_pickle=True)
    assert np.allclose(loaded["alpha_frequencies"], alpha_hz)
