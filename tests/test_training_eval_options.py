"""Default-off training switches, spectrum windows, and eval output suffix."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from dvae.eval.utils.spectrum_validation import (
    capture_fixed_spectrum_windows,
    score_spectrum_windows,
)
from dvae.run_options import (
    eval_output_dir,
    parse_checkpoint_every_n_epochs,
    parse_val_criterion,
    resolve_mask_label_override,
)


class _Series:
    def __init__(self, seq, split="valid", overlap=False, shuffle=False, val_indices=0.0):
        self.seq = np.asarray(seq, dtype=np.float64)
        self.split = split
        self.overlap = overlap
        self.shuffle = shuffle
        self.val_indices = val_indices
        self.coarse_mode = False

    def update_sequence_length(self, *args, **kwargs):
        raise AssertionError("spectrum windows must not call update_sequence_length")


class _Subjects(_Series):
    def __init__(self, blocks, **kwargs):
        parts = [np.asarray(block, dtype=np.float64) for block in blocks]
        seq = np.concatenate(parts, axis=0)
        super().__init__(seq, **kwargs)
        index = []
        for i, block in enumerate(parts):
            index.extend([i] * len(block))
        self.subject_index_per_row = np.asarray(index, dtype=np.int64)


def test_val_criterion_and_checkpoint_defaults_are_off():
    assert parse_val_criterion(None) == "recon"
    assert parse_val_criterion("") == "recon"
    assert parse_val_criterion("None") == "recon"
    assert parse_val_criterion("recon") == "recon"
    assert parse_val_criterion("spectrum") == "spectrum"
    with pytest.raises(ValueError):
        parse_val_criterion("both")
    assert parse_checkpoint_every_n_epochs(None) == 0
    assert parse_checkpoint_every_n_epochs("") == 0
    assert parse_checkpoint_every_n_epochs("off") == 0
    assert parse_checkpoint_every_n_epochs("0") == 0
    assert parse_checkpoint_every_n_epochs("5") == 5


def test_mask_label_suffix_only_when_override_differs():
    label, suffix = resolve_mask_label_override("realtime", None, None)
    assert label == "realtime"
    assert suffix is None
    label, suffix = resolve_mask_label_override("realtime", "realtime", "custom")
    assert suffix is None
    label, suffix = resolve_mask_label_override("realtime", "retrans", None)
    assert label == "retrans"
    assert suffix == "retrans"
    label, suffix = resolve_mask_label_override("realtime", "retrans", "cross")
    assert suffix == "cross"
    assert eval_output_dir("/runs/exp", None) == "/runs/exp"
    assert eval_output_dir("/runs/exp", "retrans") == "/runs/exp/eval_mask_retrans"


def test_spectrum_windows_are_copies_and_do_not_resize():
    series = np.arange(80, dtype=np.float64).reshape(80, 1)
    dataset = _Series(series, split="valid", overlap=False, val_indices=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        windows = capture_fixed_spectrum_windows(dataset, n_windows=10, window_len=20)
    assert len(windows) == 4  # non-overlap, full series is the validation split
    assert any(issubclass(item.category, UserWarning) for item in caught)
    windows[0][0, 0] = -999.0
    assert dataset.seq[0, 0] == 0.0
    assert windows[0].shape == (20, 1)


def test_spectrum_windows_do_not_cross_subjects():
    dataset = _Subjects(
        [np.ones((30, 1)), np.full((30, 1), 2.0)],
        split="valid",
        overlap=True,
        val_indices=1.0,
    )
    windows = capture_fixed_spectrum_windows(dataset, n_windows=4, window_len=20)
    assert windows
    for window in windows:
        assert set(np.unique(window[:, 0])).issubset({1.0, 2.0})
        assert len(set(np.unique(window[:, 0]).tolist())) == 1
    with pytest.raises(ValueError):
        capture_fixed_spectrum_windows(dataset, n_windows=2, window_len=40)


class _Identity:
    device = "cpu"

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        return batch.clone()


class _ReplaceTail:
    def __init__(self, tf_steps):
        self.device = "cpu"
        self.tf_steps = tf_steps

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        out = batch.clone()
        steps = out.shape[0] - self.tf_steps
        t = torch.arange(steps, dtype=out.dtype)
        out[self.tf_steps :, :, 0] = torch.sin(t / 3.0).unsqueeze(-1)
        if out.shape[-1] > 1:
            out[self.tf_steps :, :, 1] = (0.25 * torch.sin(t / 7.0)).unsqueeze(-1)
        return out


class _ZeroHead:
    def __init__(self, tf_steps):
        self.device = "cpu"
        self.tf_steps = tf_steps

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        out = batch.clone()
        out[: self.tf_steps] = 0.0
        return out


def _sine_windows():
    t = np.arange(64, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * t / 16.0)
    window = np.column_stack([wave, 0.5 * wave])
    return [window.copy(), window.copy()]


def test_spectrum_score_ignores_teacher_forced_half():
    windows = _sine_windows()
    tf_steps = 32
    same = score_spectrum_windows(_Identity(), windows, tf_steps, None, "cpu")
    head = score_spectrum_windows(_ZeroHead(tf_steps), windows, tf_steps, None, "cpu")
    tail = score_spectrum_windows(
        _ReplaceTail(tf_steps), windows, tf_steps, None, "cpu"
    )
    # Identical free-run halves score the same. Hellinger on a perfect match
    # is NaN in power_spectrum_error (the +1e-10 sits outside the normalizer).
    assert np.isnan(same) and np.isnan(head)
    assert np.isfinite(tail) and tail > 0.02
