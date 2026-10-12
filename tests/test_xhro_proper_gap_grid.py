"""Waveform gap-grid keys and re-cuts. Synthetic NPZs only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvae.dataset.utils.data_utils import find_indices
from dvae.dataset.xhro_proper_dataset import XhroProper, load_proper_frame


def _save_stage(path: Path, n: int, **extra):
    t = np.arange(n, dtype=np.float64) * 0.004
    t_abs = pd.date_range(
        "2026-06-22 14:00:00", periods=n, freq="4ms", tz="Asia/Tokyo"
    )
    utc_ns = pd.DatetimeIndex(t_abs).tz_convert("UTC").asi8
    payload = {
        "t": t,
        "datetime_ns": np.asarray(utc_ns, dtype=np.int64),
        "tz": np.asarray("Asia/Tokyo"),
        "ch1": np.arange(n, dtype=np.float64) + 1.0,
        "ch2": np.linspace(-1.0, 1.0, n),
        "ecg": np.linspace(0.5, -0.5, n),
    }
    payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _session(root: Path, n: int = 100, **extra) -> Path:
    sess = root / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    _save_stage(sess / "intermediates" / "stage05_ecg_filtered.npz", n, **extra)
    return sess


def _load(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=10,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    params.update(overrides)
    return XhroProper(**params)


def _crosses(starts, seq_len, boundaries) -> bool:
    for start in starts:
        s = int(start)
        for cut in boundaries:
            if s < int(cut) < s + int(seq_len):
                return True
    return False


def _assert_inside(ds):
    bounds = [] if ds.split_boundaries is None else ds.split_boundaries
    assert not _crosses(ds.data_idx, ds.seq_len, bounds)
    group = ds.segment_index_per_row
    if group is None:
        return
    for start in ds.data_idx:
        window = group[int(start) : int(start) + ds.seq_len]
        assert len(np.unique(window)) == 1


def test_find_indices_without_boundaries_is_the_single_span_rule():
    got = find_indices(20, 5, 20 // 5)
    assert list(np.asarray(got).tolist()) == [0, 5, 10, 15]
    assert list(np.asarray(find_indices(20, 5, 4, boundaries=None)).tolist()) == [
        0, 5, 10, 15
    ]
    assert list(np.asarray(find_indices(20, 5, 4, boundaries=[])).tolist()) == [
        0, 5, 10, 15
    ]


def test_find_indices_stays_inside_each_piece():
    # The unsplit picker includes start 5, and [5, 10) crosses cut 7.
    plain = set(int(i) for i in find_indices(20, 5, 4))
    assert 5 in plain
    bounded = find_indices(20, 5, 4, boundaries=[7])
    assert not _crosses(bounded, 5, [7])
    assert 5 not in set(int(i) for i in bounded)
    # Same local seed on a second call.
    again = find_indices(20, 5, 4, boundaries=[7])
    np.testing.assert_array_equal(bounded, again)


def test_old_npz_recut_matches_find_indices(tmp_path, capsys):
    from tests.test_xhro_proper import _fake_session

    _fake_session(tmp_path)
    capsys.readouterr()
    ds = _load(tmp_path, seq_len=16)
    assert ds.split_boundaries is None
    assert ds.segment_index_per_row is None
    n = len(np.asarray(ds.seq))
    assert list(ds.data_idx) == list(np.asarray(find_indices(n, 16, n // 16)).tolist())
    out = capsys.readouterr().out
    assert "present_frac=1.000" in out
    assert "segments=1" in out
    assert "split_points=0" in out
    assert "long_filled_windows=na" in out
    assert "threshold_sec=unset" in out

    capsys.readouterr()
    ds.update_sequence_length(20)
    assert ds.seq_len == 20
    assert list(ds.data_idx) == list(np.asarray(find_indices(n, 20, n // 20)).tolist())
    assert not _crosses(ds.data_idx, ds.seq_len, [])
    recut = capsys.readouterr().out
    assert f"re-cut seq_len=20 windows={len(ds.data_idx)}" in recut
    assert "long_filled_windows=na" in recut


def test_segment_boundary_survives_recut(tmp_path, capsys):
    n = 100
    segment = np.zeros(n, dtype=np.int64)
    segment[40:] = 1
    _session(tmp_path, n, segment_id=segment, present=np.ones(n, dtype=bool))
    ds = _load(tmp_path, seq_len=10)
    assert list(np.asarray(ds.split_boundaries).tolist()) == [40]
    _assert_inside(ds)
    expected = find_indices(len(ds.seq), 10, 1, boundaries=ds.split_boundaries)
    assert list(ds.data_idx) == list(np.asarray(expected).tolist())

    capsys.readouterr()
    for new_len in (15, 8, 15):
        ds.update_sequence_length(new_len)
        assert ds.seq_len == new_len
        _assert_inside(ds)
        expected = find_indices(
            len(ds.seq), new_len, 1, boundaries=ds.split_boundaries
        )
        assert list(ds.data_idx) == list(np.asarray(expected).tolist())
    # Repeating a length is deterministic (local RandomState, not global).
    first = list(ds.data_idx)
    ds.update_sequence_length(8)
    ds.update_sequence_length(15)
    assert list(ds.data_idx) == first
    out = capsys.readouterr().out
    assert "re-cut seq_len=15 windows=" in out
    assert "long_filled_windows=" in out
    assert "threshold_sec=" in out


def test_overlap_recut_does_not_cross_segment(tmp_path):
    n = 100
    segment = np.zeros(n, dtype=np.int64)
    segment[40:] = 1
    _session(tmp_path, n, segment_id=segment, present=np.ones(n, dtype=bool))
    # Two channels stay (time, channel), so overlap applies to seq_len
    # windows rather than the 1-d feature pack.
    ds = _load(
        tmp_path,
        seq_len=10,
        overlap=True,
        observation_process="raw_ecg",
        x_dim=2,
    )
    _assert_inside(ds)
    assert 30 in ds.data_idx
    assert 31 not in ds.data_idx
    assert 40 in ds.data_idx
    ds.update_sequence_length(16)
    _assert_inside(ds)
    assert 24 in ds.data_idx
    assert 25 not in ds.data_idx
    assert 39 not in ds.data_idx
    assert 40 in ds.data_idx


def test_present_false_is_missing_even_if_the_sample_is_finite(tmp_path):
    n = 100
    present = np.ones(n, dtype=bool)
    present[5] = False
    ch1 = np.arange(n, dtype=np.float64) + 1.0
    _session(tmp_path, n, present=present, ch1=ch1)
    ds = _load(tmp_path, seq_len=10)
    seq = np.asarray(ds.seq).reshape(-1)
    mask = np.asarray(ds.missing_mask)
    if mask.ndim == 2:
        mask = mask[:, 0]
    assert np.isnan(seq[5])
    assert bool(mask[5])
    assert np.isfinite(seq[4]) and np.isfinite(seq[6])
    assert not bool(mask[4])


def test_long_filled_gap_is_logged_when_unset_and_split_when_set(tmp_path, capsys):
    n = 100
    present = np.ones(n, dtype=bool)
    present[30:38] = False
    ch1 = np.arange(n, dtype=np.float64) + 1.0
    gap = dict(
        gap_t_start=np.array([30 * 0.004], dtype=np.float64),
        gap_n_missing=np.array([8], dtype=np.int64),
        gap_duration_s=np.array([0.032], dtype=np.float64),
        gap_filled=np.array([True]),
        gap_kind=np.array(["packet_loss"]),
        gap_index=np.array([30], dtype=np.int64),
        segment_id=np.zeros(n, dtype=np.int64),
        present=present,
        ch1=ch1,
    )
    _session(tmp_path, n, **gap)
    capsys.readouterr()
    kept = _load(tmp_path, seq_len=10, filled_nan_log_sec=0.01)
    assert kept.split_boundaries is None or len(kept.split_boundaries) == 0
    seq = np.asarray(kept.seq).reshape(-1)
    assert np.isnan(seq[30:38]).all()
    assert np.isfinite(seq[29]) and np.isfinite(seq[38])
    out = capsys.readouterr().out
    assert "long_filled_windows=0" not in out.split("re-cut")[-1]
    assert "threshold_sec=0.01" in out
    # Some kept window covers the 8-sample hole.
    assert any(
        start < 38 and start + kept.seq_len > 30 for start in kept.data_idx
    )

    capsys.readouterr()
    split = _load(tmp_path, seq_len=4, max_fill_gap_sec=0.01)
    assert list(np.asarray(split.split_boundaries).tolist()) == [30, 38]
    _assert_inside(split)
    assert any(30 <= int(s) < 38 for s in split.data_idx)
    split_out = capsys.readouterr().out
    assert "threshold_sec=0.01" in split_out
    # The in-gap window contains the 0.032 s run, which is over the cutoff.
    assert "long_filled_windows=0" not in split_out.split("re-cut")[-1]

    capsys.readouterr()
    split.update_sequence_length(10)
    _assert_inside(split)
    assert split.seq_len == 10
    assert not any(30 <= int(s) < 38 for s in split.data_idx)
    recut = capsys.readouterr().out
    assert "re-cut seq_len=10 windows=" in recut
    assert "long_filled_windows=0" in recut
    split.update_sequence_length(24)
    _assert_inside(split)


def test_crop_stays_on_the_clock_and_keeps_the_boundary(tmp_path):
    n = 100
    segment = np.zeros(n, dtype=np.int64)
    segment[50:] = 1
    sess = _session(tmp_path, n, segment_id=segment, present=np.ones(n, dtype=bool))
    frame = load_proper_frame(sess, "raw_ch1")
    elapsed = (frame["datetime"] - frame["datetime"].min()).dt.total_seconds()
    start = float(elapsed.iloc[20])
    end = float(elapsed.iloc[70])
    kept_rows = frame.loc[(elapsed.to_numpy() >= start) & (elapsed.to_numpy() <= end)]
    ds = _load(
        tmp_path,
        seq_len=8,
        crop_start_sec=start,
        crop_end_sec=end,
    )
    train_n = len(kept_rows) + ((-len(kept_rows)) // 5)
    assert len(ds.full_sequence) == train_n
    pd.testing.assert_series_equal(
        ds.full_sequence["datetime"].reset_index(drop=True),
        kept_rows["datetime"].iloc[:train_n].reset_index(drop=True),
        check_names=False,
    )
    # Original row 50 is local row 30 inside the cropped frame, and the
    # train head still contains that cut.
    assert 30 in set(int(c) for c in ds.split_boundaries)
    _assert_inside(ds)
    ds.update_sequence_length(12)
    _assert_inside(ds)


def test_negative_max_fill_is_rejected(tmp_path):
    _session(tmp_path, 20, present=np.ones(20, dtype=bool))
    with pytest.raises(ValueError, match="max_fill_gap_sec"):
        _load(tmp_path, max_fill_gap_sec=-0.1)


def test_template_has_gap_keys():
    text = (
        Path(__file__).resolve().parents[1]
        / "config/general_signal/cfg_base_template.ini"
    ).read_text()
    assert "max_fill_gap_sec = {max_fill_gap_sec}" in text
    assert "filled_nan_log_sec = {filled_nan_log_sec}" in text
