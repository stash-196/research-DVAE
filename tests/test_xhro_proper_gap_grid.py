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
    ds = _load(tmp_path, seq_len=16, max_window_nan_ratio="off")
    assert ds.split_boundaries is None
    assert ds.segment_index_per_row is None
    assert ds.strict_non_overlap is True
    assert ds.start_on_observed is True
    assert ds.max_gap_window_frac == pytest.approx(0.25)
    n = len(np.asarray(ds.seq))
    assert list(ds.data_idx) == list(
        np.asarray(find_indices(n, 16, n // 16, strict_non_overlap=True)).tolist()
    )
    # This fixture's first window is mostly NaN. The relative default
    # (missing rate + 0.2) still drops it. The rate itself is 5/200.
    filtered = _load(tmp_path, seq_len=16)
    assert filtered.max_window_nan_ratio is None
    assert filtered.max_window_nan_excess == pytest.approx(0.2)
    assert filtered.dataset_missing_rate == pytest.approx(5 / 200)
    assert 0 not in filtered.data_idx
    legacy = _load(tmp_path, seq_len=16, legacy_windowing=True)
    assert legacy.strict_non_overlap is False
    assert legacy.start_on_observed is False
    assert legacy.max_window_nan_ratio is None
    assert legacy.max_window_nan_excess is None
    assert legacy.max_gap_window_frac is None
    assert list(legacy.data_idx) == list(
        np.asarray(find_indices(n, 16, n // 16)).tolist()
    )
    out = capsys.readouterr().out
    assert "present_frac=1.000" in out
    assert "segments=1" in out
    assert "split_points=0" in out
    assert "long_filled_windows=na" in out
    assert "threshold_sec=unset" in out

    capsys.readouterr()
    ds.update_sequence_length(20)
    assert ds.seq_len == 20
    assert list(ds.data_idx) == list(
        np.asarray(find_indices(n, 20, n // 20, strict_non_overlap=True)).tolist()
    )
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
    expected = find_indices(
        len(ds.seq),
        10,
        1,
        boundaries=ds.split_boundaries,
        strict_non_overlap=True,
    )
    assert list(ds.data_idx) == list(np.asarray(expected).tolist())

    capsys.readouterr()
    for new_len in (15, 8, 15):
        ds.update_sequence_length(new_len)
        assert ds.seq_len == new_len
        _assert_inside(ds)
        expected = find_indices(
            len(ds.seq),
            new_len,
            1,
            boundaries=ds.split_boundaries,
            strict_non_overlap=True,
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
        strict_non_overlap=False,
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
    kept = _load(
        tmp_path, seq_len=10, filled_nan_log_sec=0.01, legacy_windowing=True
    )
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
    split = _load(
        tmp_path, seq_len=4, max_fill_gap_sec=0.01, legacy_windowing=True
    )
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
    assert "max_gap_window_frac = {max_gap_window_frac}" in text
    assert "gap_split_quantile = 0.99" in text
    assert "start_on_observed = {start_on_observed}" in text
    assert "max_window_nan_ratio = {max_window_nan_ratio}" in text
    assert "max_window_nan_excess = {max_window_nan_excess}" in text
    assert "strict_non_overlap = {strict_non_overlap}" in text
    assert "legacy_windowing = {legacy_windowing}" in text


def _overlaps(starts, seq_len) -> bool:
    ordered = sorted(int(s) for s in starts)
    return any(b < a + int(seq_len) for a, b in zip(ordered, ordered[1:]))


def _piece_is_tiled(starts, seq_len, left, right) -> bool:
    """Abutting windows, and only a final remainder shorter than ``seq_len``."""
    length = int(seq_len)
    got = sorted(int(s) for s in starts if left <= int(s) < right)
    expect = list(range(left, right - length + 1, length))
    return got == expect


def test_default_find_indices_is_unchanged_and_can_overlap():
    historical = find_indices(40, 10, 4)
    assert list(np.asarray(historical).tolist()) == [0, 8, 16, 24]
    assert list(np.asarray(find_indices(40, 10, 4, strict_non_overlap=False)).tolist()) == [
        0, 8, 16, 24
    ]
    assert _overlaps(historical, 10)
    bounded = find_indices(20, 5, 4, boundaries=[7])
    assert list(np.asarray(bounded).tolist()) == list(
        np.asarray(find_indices(20, 5, 4, boundaries=[7], strict_non_overlap=False)).tolist()
    )


def test_default_picker_still_builds_the_local_rng(monkeypatch):
    import dvae.dataset.utils.data_utils as data_utils

    calls = []
    real = data_utils.np.random.RandomState

    def wrapped(seed=None):
        calls.append(seed)
        return real(seed)

    monkeypatch.setattr(data_utils.np.random, "RandomState", wrapped)
    find_indices(40, 10, 4)
    assert calls == [1234567890]
    calls.clear()
    find_indices(20, 5, 4, boundaries=[7])
    assert calls == [1234567890, 1234567890]
    calls.clear()
    find_indices(40, 10, 4, strict_non_overlap=True)
    find_indices(20, 5, 4, boundaries=[7], strict_non_overlap=True)
    assert calls == []


def test_strict_non_overlap_tiles_each_piece_without_overlap():
    tiled = find_indices(40, 10, 99, strict_non_overlap=True)
    assert list(tiled.tolist()) == [0, 10, 20, 30]
    assert not _overlaps(tiled, 10)
    assert _piece_is_tiled(tiled, 10, 0, 40)
    # Remainder is the only uncovered tail: 41 % 10 == 1.
    odd = find_indices(41, 10, 4, strict_non_overlap=True)
    assert list(odd.tolist()) == [0, 10, 20, 30]
    assert not _overlaps(odd, 10)
    pieces = find_indices(20, 5, 4, boundaries=[7], strict_non_overlap=True)
    assert not _overlaps(pieces, 5)
    assert not _crosses(pieces, 5, [7])
    assert _piece_is_tiled(pieces, 5, 0, 7)
    assert _piece_is_tiled(pieces, 5, 7, 20)


def test_strict_non_overlap_steps_over_a_missing_start(tmp_path):
    n = 100
    segment = np.zeros(n, dtype=np.int64)
    segment[40:] = 1
    present = np.ones(n, dtype=bool)
    present[20:30] = False
    _session(tmp_path, n, present=present, segment_id=segment)
    old = _load(tmp_path, seq_len=10, legacy_windowing=True)
    length = len(np.asarray(old.seq))
    assert list(old.data_idx) == list(
        np.asarray(find_indices(length, 10, length // 10, boundaries=[40])).tolist()
    )
    ds = _load(
        tmp_path,
        seq_len=10,
        strict_non_overlap=True,
        start_on_observed=True,
    )
    assert not _overlaps(ds.data_idx, ds.seq_len)
    _assert_inside(ds)
    assert 20 not in ds.data_idx
    assert 30 in ds.data_idx
    seq = np.asarray(ds.seq).reshape(-1)
    for start in ds.data_idx:
        assert np.isfinite(seq[int(start)])
    # The gap is stepped over; the windows on either side still abut the gap.
    assert 10 in ds.data_idx
    assert any(int(s) == 30 for s in ds.data_idx)


def test_shift_start_off_a_gap_or_drop_it():
    from dvae.dataset.gap_grid import shift_starts_onto_observed

    observed = np.ones(20, dtype=bool)
    observed[0] = False
    observed[3:6] = False
    got = shift_starts_onto_observed([0, 3, 8], observed, 4, None, 20)
    assert list(got.tolist()) == [1, 6, 8]
    # The gap runs to the segment cut, so the window cannot stay inside it.
    blocked = np.ones(20, dtype=bool)
    blocked[6:10] = False
    dropped = shift_starts_onto_observed([6], blocked, 4, [10], 20)
    assert list(dropped.tolist()) == []
    # An observed start is kept even when a later sample in the window is missing.
    kept = shift_starts_onto_observed([0], blocked, 4, None, 20)
    assert list(kept.tolist()) == [0]


def test_frac_cuts_follow_the_sequence_length():
    from dvae.dataset.gap_grid import filled_gap_frac_cuts, resolve_recut_length

    span = [np.arange(30, 38)]
    assert filled_gap_frac_cuts(span, 20, 0.5, 80).size == 0
    assert list(filled_gap_frac_cuts(span, 10, 0.5, 80).tolist()) == [30, 38]
    assert filled_gap_frac_cuts(span, 10, None, 80).size == 0
    # Capping the window re-evaluates the fraction and can add the same cuts.
    length, bounds = resolve_recut_length(30, 20, None, [np.arange(9, 21)], 0.5)
    assert length == 12
    assert list(np.asarray(bounds).tolist()) == [9, 21]


def test_start_on_observed_moves_off_present_false(tmp_path):
    n = 100
    present = np.ones(n, dtype=bool)
    present[0] = False
    present[2] = False
    _session(tmp_path, n, present=present, segment_id=np.zeros(n, dtype=np.int64))
    plain = _load(tmp_path, seq_len=10, legacy_windowing=True)
    assert 0 in plain.data_idx
    assert any(s <= 2 < s + plain.seq_len for s in plain.data_idx)
    moved = _load(tmp_path, seq_len=10)
    assert 0 not in moved.data_idx
    assert 1 in moved.data_idx
    # The hole at row 2 stays inside the window that now starts at 1.
    assert any(s <= 2 < s + moved.seq_len for s in moved.data_idx)
    seq = np.asarray(moved.seq).reshape(-1)
    for start in moved.data_idx:
        assert np.isfinite(seq[int(start)])


def test_start_on_observed_drops_a_start_that_cannot_fit(tmp_path):
    n = 100
    segment = np.zeros(n, dtype=np.int64)
    segment[40:] = 1
    present = np.ones(n, dtype=bool)
    present[24:40] = False
    _session(tmp_path, n, present=present, segment_id=segment)
    ds = _load(
        tmp_path,
        seq_len=10,
        max_gap_window_frac="off",
        max_window_nan_ratio="off",
    )
    _assert_inside(ds)
    assert 24 not in ds.data_idx
    assert not any(24 <= int(s) < 40 for s in ds.data_idx)
    # The window that starts before the hole still contains it.
    assert any(int(s) < 24 < int(s) + ds.seq_len for s in ds.data_idx)


def test_max_gap_window_frac_is_recomputed_on_recut(tmp_path):
    n = 100
    present = np.ones(n, dtype=bool)
    present[30:38] = False
    _session(
        tmp_path,
        n,
        present=present,
        segment_id=np.zeros(n, dtype=np.int64),
        gap_t_start=np.array([30 * 0.004]),
        gap_n_missing=np.array([8], dtype=np.int64),
        gap_duration_s=np.array([0.032]),
        gap_filled=np.array([True]),
        gap_kind=np.array(["packet_loss"]),
        gap_index=np.array([30], dtype=np.int64),
    )
    ds = _load(
        tmp_path,
        seq_len=20,
        max_gap_window_frac=0.5,
        gap_split_quantile="none",
        max_window_nan_ratio="off",
    )
    # 8 samples is not longer than 0.5 * 20, so the hole stays inside a window.
    assert ds.split_boundaries is None or len(ds.split_boundaries) == 0
    assert any(start < 38 and start + ds.seq_len > 30 for start in ds.data_idx)
    ds.update_sequence_length(10)
    assert ds.seq_len == 10
    _assert_inside(ds)
    assert not any(start < 38 and start + ds.seq_len > 30 for start in ds.data_idx)
    assert not any(30 <= int(s) < 38 for s in ds.data_idx)
    ds.update_sequence_length(20)
    assert any(start < 38 and start + ds.seq_len > 30 for start in ds.data_idx)


def test_max_fill_and_frac_both_split(tmp_path):
    n = 100
    present = np.ones(n, dtype=bool)
    present[30:38] = False
    _session(
        tmp_path,
        n,
        present=present,
        segment_id=np.zeros(n, dtype=np.int64),
        gap_t_start=np.array([30 * 0.004]),
        gap_n_missing=np.array([8], dtype=np.int64),
        gap_duration_s=np.array([0.032]),
        gap_filled=np.array([True]),
        gap_kind=np.array(["packet_loss"]),
        gap_index=np.array([30], dtype=np.int64),
    )
    ds = _load(
        tmp_path,
        seq_len=20,
        max_fill_gap_sec=0.01,
        max_gap_window_frac=0.9,
    )
    assert list(np.asarray(ds.split_boundaries).tolist()) == [30, 38]
    _assert_inside(ds)
    assert not any(start < 38 and start + ds.seq_len > 30 for start in ds.data_idx)


def test_max_window_nan_ratio_filters_recuts_unless_overridden(tmp_path):
    n = 100
    present = np.ones(n, dtype=bool)
    present[5] = False
    _session(tmp_path, n, present=present)
    open_windows = _load(tmp_path, seq_len=10)
    assert any(s <= 5 < s + open_windows.seq_len for s in open_windows.data_idx)
    filtered = _load(tmp_path, seq_len=10, max_window_nan_ratio=0.0)
    assert not any(s <= 5 < s + filtered.seq_len for s in filtered.data_idx)
    filtered.update_sequence_length(10, minimum_nan_ratio=1.0)
    assert any(s <= 5 < s + filtered.seq_len for s in filtered.data_idx)
    text = (
        Path(__file__).resolve().parents[1] / "src/dvae/learning_algo.py"
    ).read_text()
    assert "minimum_nan_ratio=0.3" in text
    assert open_windows.max_window_nan_ratio is None
    assert open_windows.max_window_nan_excess == pytest.approx(0.2)


def test_default_fraction_splits_and_legacy_keeps_an_explicit_key(tmp_path):
    from dvae.dataset.xhro_proper_dataset import resolve_windowing, windowing_from_config
    import configparser

    blank = resolve_windowing()
    assert blank["legacy_windowing"] is False
    assert blank["start_on_observed"] is True
    assert blank["strict_non_overlap"] is True
    assert blank["max_window_nan_ratio"] is None
    assert blank["max_window_nan_ratio_off"] is False
    assert blank["max_window_nan_excess"] == pytest.approx(0.2)
    assert blank["max_gap_window_frac"] == pytest.approx(0.25)
    assert blank["gap_split_quantile"] == pytest.approx(0.99)
    legacy = resolve_windowing(legacy_windowing=True)
    assert legacy["start_on_observed"] is False
    assert legacy["strict_non_overlap"] is False
    assert legacy["max_window_nan_ratio"] is None
    assert legacy["max_window_nan_ratio_off"] is False
    assert legacy["max_window_nan_excess"] is None
    assert legacy["max_gap_window_frac"] is None
    assert legacy["gap_split_quantile"] == pytest.approx(0.99)
    assert resolve_windowing(gap_split_quantile="none")["gap_split_quantile"] is None
    assert resolve_windowing(gap_split_quantile="")["gap_split_quantile"] is None
    assert resolve_windowing(gap_split_quantile="empty")["gap_split_quantile"] is None
    assert resolve_windowing(max_window_nan_ratio="off")["max_window_nan_ratio_off"]
    assert resolve_windowing(max_window_nan_excess="off")["max_window_nan_excess"] is None
    mixed = resolve_windowing(legacy_windowing=True, start_on_observed=True)
    assert mixed["start_on_observed"] is True
    assert mixed["strict_non_overlap"] is False
    assert resolve_windowing(max_gap_window_frac="off")["max_gap_window_frac"] is None

    cfg = configparser.ConfigParser()
    cfg.add_section("DataFrame")
    cfg.set("DataFrame", "legacy_windowing", "true")
    cfg.set("DataFrame", "max_gap_window_frac", "0.5")
    from_cfg = windowing_from_config(cfg)
    assert from_cfg["legacy_windowing"] is True
    assert from_cfg["max_gap_window_frac"] == pytest.approx(0.5)
    assert from_cfg["gap_split_quantile"] == pytest.approx(0.99)
    assert from_cfg["start_on_observed"] is False
    cfg.set("DataFrame", "gap_split_quantile", "none")
    assert windowing_from_config(cfg)["gap_split_quantile"] is None
    cfg.set("DataFrame", "gap_split_quantile", "")
    assert windowing_from_config(cfg)["gap_split_quantile"] is None

    n = 100
    present = np.ones(n, dtype=bool)
    present[32:40] = False
    _session(tmp_path, n, present=present, segment_id=np.zeros(n, dtype=np.int64))
    # The only missing run is 8 samples, so the 0.99 quantile is 8 and
    # 8 > 8 does not split. The tile at 30 still contains the hole.
    # Dropping the quantile restores the fraction split.
    lone = _load(tmp_path, seq_len=10, max_window_nan_ratio="off")
    assert lone.gap_split_quantile == pytest.approx(0.99)
    assert lone.dataset_gap_quantile == pytest.approx(8)
    assert any(start < 40 and start + lone.seq_len > 32 for start in lone.data_idx)
    split = _load(
        tmp_path,
        seq_len=10,
        max_window_nan_ratio="off",
        gap_split_quantile="none",
    )
    assert split.dataset_gap_quantile is None
    assert split.max_gap_window_frac == pytest.approx(0.25)
    assert not any(start < 40 and start + split.seq_len > 32 for start in split.data_idx)


def test_missing_rate_counts_absent_rows_and_ignores_inf():
    from dvae.dataset.gap_grid import dataset_missing_rate

    values = np.array([[1.0, np.nan], [2.0, 3.0], [np.inf, 4.0]])
    assert dataset_missing_rate(values) == pytest.approx(1 / 6)
    present = np.array([True, False, True])
    # The finite row with present=false counts. inf does not.
    assert dataset_missing_rate(values, present) == pytest.approx(0.5)


def test_relative_nan_excess_uses_the_loaded_series(tmp_path, capsys):
    n = 100
    ch1 = np.arange(n, dtype=np.float64) + 1.0
    ch1[0:4] = np.nan
    ch1[10:12] = np.nan
    ch1[90:94] = np.nan
    _session(tmp_path, n, ch1=ch1)
    common = dict(
        seq_len=10,
        start_on_observed=False,
        max_gap_window_frac="off",
        val_indices=0.0,
    )
    capsys.readouterr()
    ds = _load(tmp_path, **common)
    # 10 NaNs on the full 100 rows, including the held-out tail.
    assert ds.dataset_missing_rate == pytest.approx(0.1)
    assert ds.max_window_nan_ratio is None
    assert ds.max_window_nan_excess == pytest.approx(0.2)
    # 4/10 is above 0.3; 2/10 is not.
    assert 0 not in ds.data_idx
    assert 10 in ds.data_idx
    out = capsys.readouterr().out
    assert "dataset_missing_rate=0.1" in out
    assert "threshold=0.3 (relative)" in out
    assert "windows_dropped=1" in out

    held = _load(tmp_path, split="test", **common)
    assert held.dataset_missing_rate == pytest.approx(ds.dataset_missing_rate)

    capsys.readouterr()
    rate_before = ds.dataset_missing_rate
    ds.update_sequence_length(20)
    assert ds.dataset_missing_rate == pytest.approx(rate_before)
    recut = capsys.readouterr().out
    assert "dataset_missing_rate=0.1" in recut
    assert "windows_dropped=" in recut

    capsys.readouterr()
    wide = _load(tmp_path, max_window_nan_excess=0.95, **common)
    assert 0 in wide.data_idx
    wide_out = capsys.readouterr().out
    assert "threshold=1.05 (relative, >=1)" in wide_out
    assert "windows_dropped=0" in wide_out

    absolute = _load(
        tmp_path,
        legacy_windowing=True,
        max_window_nan_ratio=0.0,
        max_window_nan_excess=0.2,
        **common,
    )
    assert absolute.max_window_nan_ratio == pytest.approx(0.0)
    assert 0 not in absolute.data_idx
    assert 10 not in absolute.data_idx

    legacy = _load(
        tmp_path,
        legacy_windowing=True,
        max_window_nan_excess=0.0,
        seq_len=10,
        start_on_observed=False,
        max_gap_window_frac="off",
        val_indices=0.0,
    )
    assert legacy.max_window_nan_ratio is None
    assert legacy.max_window_nan_excess == pytest.approx(0.0)
    assert 0 in legacy.data_idx

    off = _load(tmp_path, max_window_nan_ratio="off", **common)
    assert off.max_window_nan_ratio_off is True
    assert 0 in off.data_idx

    sess = tmp_path / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    frame = load_proper_frame(sess, "raw_ch1")
    elapsed = (frame["datetime"] - frame["datetime"].min()).dt.total_seconds()
    cropped = _load(tmp_path, crop_end_sec=float(elapsed.iloc[79]), **common)
    assert cropped.dataset_missing_rate == pytest.approx(6 / 80)

    indicated = _load(
        tmp_path,
        observation_process="raw_ch1_indicate",
        x_dim=2,
        **common,
    )
    # The 0/1 indicate channel is finite, and imputation clears the NaNs the
    # filter sees. The rate stays the pre-process observation rate.
    assert indicated.dataset_missing_rate == pytest.approx(0.1)
    assert not np.isnan(np.asarray(indicated.seq)).any()
    assert 0 in indicated.data_idx


def _covers(ds, start, end) -> bool:
    return any(int(s) < end and int(s) + ds.seq_len > start for s in ds.data_idx)


def test_gap_split_uses_the_run_length_quantile(tmp_path, capsys):
    from dvae.dataset.gap_grid import (
        PRESENT_COL,
        contiguous_run_lengths,
        gap_length_quantile,
        missing_row_mask,
    )

    # A finite present=false row is a missing run. inf is not.
    values = np.array([1.0, 2.0, np.inf, np.nan])
    row_present = np.array([True, False, True, True])
    mask = missing_row_mask(values, row_present)
    assert list(mask.tolist()) == [False, True, False, True]
    assert list(contiguous_run_lengths(mask).tolist()) == [1, 1]

    n = 200
    present = np.ones(n, dtype=bool)
    present[40:52] = False
    present[list(range(80, 140, 2)) + list(range(170, 200, 2))] = False
    _session(tmp_path, n, present=present, segment_id=np.zeros(n, dtype=np.int64))
    sess = tmp_path / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    frame = load_proper_frame(sess, "raw_ch1")
    full_q = gap_length_quantile(
        contiguous_run_lengths(
            missing_row_mask(
                frame["ch1"].to_numpy(dtype=np.float64),
                frame[PRESENT_COL].to_numpy(dtype=bool),
            )
        ),
        0.99,
    )
    head = 160
    head_q = gap_length_quantile(
        contiguous_run_lengths(
            missing_row_mask(
                frame["ch1"].to_numpy(dtype=np.float64)[:head],
                frame[PRESENT_COL].to_numpy(dtype=bool)[:head],
            )
        ),
        0.99,
    )
    assert full_q != pytest.approx(head_q)

    capsys.readouterr()
    ds = _load(
        tmp_path,
        seq_len=20,
        val_indices=0.0,
        max_window_nan_ratio="off",
    )
    assert ds.gap_split_quantile == pytest.approx(0.99)
    assert ds.dataset_gap_quantile == pytest.approx(full_q)
    assert not _covers(ds, 40, 52)
    out = capsys.readouterr().out
    limit = max(0.25 * ds.seq_len, ds.dataset_gap_quantile)
    assert f"gap_quantile_len={ds.dataset_gap_quantile:.6g}" in out
    assert f"gap_split_threshold={limit:.6g}" in out
    assert "gap_splits=2" in out

    held = _load(tmp_path, split="test", seq_len=20, max_window_nan_ratio="off")
    assert held.dataset_gap_quantile == pytest.approx(ds.dataset_gap_quantile)

    capsys.readouterr()
    ds.update_sequence_length(60)
    assert ds.dataset_gap_quantile == pytest.approx(full_q)
    assert _covers(ds, 40, 52)
    recut = capsys.readouterr().out
    wide_limit = max(0.25 * 60, full_q)
    assert f"gap_quantile_len={full_q:.6g}" in recut
    assert f"gap_split_threshold={wide_limit:.6g}" in recut
    assert "gap_splits=0" in recut

    fraction_only = _load(
        tmp_path,
        seq_len=20,
        val_indices=0.0,
        max_window_nan_ratio="off",
        gap_split_quantile="none",
    )
    assert fraction_only.dataset_gap_quantile is None
    assert not _covers(fraction_only, 40, 52)

    legacy = _load(
        tmp_path,
        seq_len=20,
        val_indices=0.0,
        legacy_windowing=True,
        max_window_nan_ratio="off",
    )
    assert legacy.max_gap_window_frac is None
    assert _covers(legacy, 40, 52)

    elapsed = (frame["datetime"] - frame["datetime"].min()).dt.total_seconds()
    cropped = _load(
        tmp_path,
        seq_len=20,
        val_indices=0.0,
        max_window_nan_ratio="off",
        crop_end_sec=float(elapsed.iloc[head - 1]),
    )
    assert cropped.dataset_gap_quantile == pytest.approx(head_q)
