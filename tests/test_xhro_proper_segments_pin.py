"""Pinning checks for the XhroProper waveform ``segments`` key.

Synthetic NPZ only. Each test names the spec item it locks.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dvae.dataset.dataset_builder import DATASET_REGISTRY, DatasetConfig
from dvae.dataset.xhro_dataset import Xhro
from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    crop_proper_frame,
    crop_waveform_segment,
    optional_config_str,
    parse_waveform_segments,
    training_initial_sequence_len,
    _waveform_head_length,
)
from dvae.eval.utils.spectrum_validation import capture_fixed_spectrum_windows


def _clock(n: int, freq: str = "4ms") -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq=freq, tz="Asia/Tokyo")


def _elapsed(n: int, freq: str = "4ms") -> np.ndarray:
    clock = _clock(n, freq)
    return (clock - clock[0]).total_seconds().to_numpy(dtype=np.float64)


def _write_subject(
    root: Path, name: str, values: np.ndarray, *, freq: str = "4ms"
) -> Path:
    sess = Path(root) / name
    inter = sess / "intermediates"
    inter.mkdir(parents=True)
    n = int(len(values))
    step = pd.Timedelta(freq).total_seconds()
    t = np.arange(n, dtype=np.float64) * float(step)
    clock = _clock(n, freq)
    utc_ns = pd.DatetimeIndex(clock).tz_convert("UTC").as_unit("ns").asi8
    signal = np.asarray(values, dtype=np.float64)
    np.savez_compressed(
        inter / "stage05_ecg_filtered.npz",
        t=t,
        datetime_ns=np.asarray(utc_ns, dtype=np.int64),
        tz=np.asarray("Asia/Tokyo"),
        ch1=signal,
        ch2=np.zeros(n, dtype=np.float64),
        ecg=np.zeros(n, dtype=np.float64),
    )
    return sess


def _load(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label=None,
        mask_label=None,
        split="train",
        seq_len=16,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=False,
        data_root=str(root),
        corpus="multi",
    )
    params.update(overrides)
    return XhroProper(**params)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _same(a, b, *, atol: float = 0.0) -> None:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    assert left.shape == right.shape
    if atol:
        np.testing.assert_allclose(left, right, atol=atol, equal_nan=True)
        return
    assert np.array_equal(left, right, equal_nan=True)


def _same_clock(got, expected) -> None:
    left = pd.DatetimeIndex(pd.Series(got)).tz_convert("UTC").as_unit("ns")
    right = pd.DatetimeIndex(pd.Series(expected)).tz_convert("UTC").as_unit("ns")
    assert left.equals(right)


def _windows_stay_inside_boundaries(dataset) -> None:
    bounds = [int(cut) for cut in np.asarray(dataset.split_boundaries).tolist()]
    length = int(dataset.seq_len)
    assert dataset.data_idx
    n = int(len(dataset.seq))
    for start in dataset.data_idx:
        start = int(start)
        assert start + length <= n
        for cut in bounds:
            assert not (start < cut < start + length), (start, cut, length)


def _windows_one_segment(dataset) -> set[int]:
    group = np.asarray(dataset.segment_index_per_row)
    length = int(dataset.seq_len)
    seen: set[int] = set()
    assert dataset.data_idx
    for start in dataset.data_idx:
        piece = group[int(start) : int(start) + length]
        assert len(piece) == length
        ids = {int(v) for v in piece.tolist()}
        assert len(ids) == 1
        seen.add(next(iter(ids)))
    return seen


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"datetime": _clock(len(values)), "ch1": np.asarray(values, dtype=np.float64)}
    )


# ---------------------------------------------------------------------------
# (1) segments unset: rows and values match dataset_label + crop
# ---------------------------------------------------------------------------


def test_pin_unset_segments_matches_crop_rows_and_values(tmp_path):
    n = 200
    values = np.linspace(-2.0, 3.0, n)
    values[40:46] = np.nan
    _write_subject(tmp_path, "REC1", values)
    elapsed = _elapsed(n)
    start = float(elapsed[20])
    end = float(elapsed[149])
    common = dict(
        dataset_label="REC1",
        crop_start_sec=start,
        crop_end_sec=end,
        shuffle=True,
        val_indices=0.2,
        seq_len=16,
    )
    omitted = _load(tmp_path, **common)
    explicit_none = _load(tmp_path, segments=None, **common)
    blank = _load(tmp_path, segments="None", **common)
    spaces = _load(tmp_path, segments="  ", **common)
    for other in (explicit_none, blank, spaces):
        _same(omitted.seq, other.seq)
        _same(omitted.missing_mask, other.missing_mask)
        assert omitted.data_idx == other.data_idx
        pd.testing.assert_series_equal(
            omitted.full_sequence["datetime"].reset_index(drop=True),
            other.full_sequence["datetime"].reset_index(drop=True),
        )
        assert omitted.split_boundaries is None or np.array_equal(
            np.asarray(omitted.split_boundaries), np.asarray(other.split_boundaries)
        )
    # The crop is inclusive, and NaNs inside it stay NaN on the unset path.
    assert omitted.full_sequence["datetime"].iloc[0] == _clock(n)[20]
    assert omitted.missing_mask[20:26].all()
    assert np.isnan(np.asarray(omitted.seq)[20:26]).all()


def test_pin_unset_path_does_not_emit_per_segment_log(tmp_path, capsys):
    n = 80
    _write_subject(tmp_path, "REC1", np.linspace(-1.0, 1.0, n))
    _load(tmp_path, dataset_label="REC1")
    assert "[XhroProper][segment] subject=" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (2) inline / yaml / csv, inclusive end, same comparison as crop_proper_frame
# ---------------------------------------------------------------------------


def test_pin_inline_yaml_and_csv_parse_the_same_rows(tmp_path):
    inline = "subj_a:0-1.5,3-4; subj_b:10-12"
    rows = parse_waveform_segments(inline)
    assert [(row["subject"], row["start"], row["end"]) for row in rows] == [
        ("subj_a", 0.0, 1.5),
        ("subj_a", 3.0, 4.0),
        ("subj_b", 10.0, 12.0),
    ]
    yaml_path = tmp_path / "segments.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "subj_a": [[0, 1.5], {"start": 3, "end": 4}],
                "subj_b": [{"start_sec": 10, "end_sec": 12}],
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "subject,start,end\n"
        "subj_a,0,1.5\n"
        "subj_a,3,4\n"
        "subj_b,10,12\n",
        encoding="utf-8",
    )
    yml_path = tmp_path / "segments.yml"
    yml_path.write_text(
        yaml.safe_dump(
            [
                {"subject": "subj_a", "start": 0, "end": 1.5},
                {"subject": "subj_a", "start": 3, "end": 4},
                {"dataset_label": "subj_b", "crop_start_sec": 10, "crop_end_sec": 12},
            ]
        ),
        encoding="utf-8",
    )
    for path in (yaml_path, csv_path, yml_path):
        parsed = parse_waveform_segments(str(path))
        assert [(row["subject"], row["start"], row["end"]) for row in parsed] == [
            (row["subject"], row["start"], row["end"]) for row in rows
        ]


def test_pin_end_is_inclusive_and_matches_crop_proper_frame():
    values = np.linspace(-1.0, 1.0, 40)
    values[8] = np.nan
    frame = _frame(values)
    elapsed = (frame["datetime"] - frame["datetime"].iloc[0]).dt.total_seconds()
    start = float(elapsed.iloc[5])
    end = float(elapsed.iloc[12])
    cropped = crop_proper_frame(frame, start, end).reset_index(drop=True)
    segmented = crop_waveform_segment(frame, start, end, "subj_a").reset_index(drop=True)
    pd.testing.assert_series_equal(cropped["datetime"], segmented["datetime"])
    _same(cropped["ch1"], segmented["ch1"])
    # Sample 5 and sample 12 are inside. Sample 13 is past the inclusive end.
    assert cropped["datetime"].iloc[0] == frame["datetime"].iloc[5]
    assert cropped["datetime"].iloc[-1] == frame["datetime"].iloc[12]
    assert len(cropped) == 8
    assert np.isnan(cropped["ch1"]).any()
    # The next timestamp is excluded by both crops.
    past = crop_proper_frame(frame, start, float(elapsed.iloc[11]))
    assert past["datetime"].iloc[-1] == frame["datetime"].iloc[11]


def test_pin_sidecar_relative_to_data_dir_matches_inline(tmp_path, monkeypatch):
    n = 80
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(0.0, 1.0, n))
    a0, a1 = float(elapsed[0]), float(elapsed[39])
    b0, b1 = float(elapsed[50]), float(elapsed[79])
    inline = f"subj_a:{_fmt(a0)}-{_fmt(a1)},{_fmt(b0)}-{_fmt(b1)}"
    sidecar = tmp_path / "ranges.csv"
    sidecar.write_text(
        "subject,start,end\n"
        f"subj_a,{a0},{a1}\n"
        f"subj_a,{b0},{b1}\n",
        encoding="utf-8",
    )
    empty = tmp_path / "cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)
    from_name = _load(tmp_path, segments="ranges.csv")
    from_inline = _load(tmp_path, segments=inline)
    _same(from_name.seq, from_inline.seq)
    assert from_name.data_idx == from_inline.data_idx


# ---------------------------------------------------------------------------
# (3) independent loads; no window crosses a subject or segment join
# ---------------------------------------------------------------------------


def _two_subject_spec(n: int = 80):
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    return f"subj_a:0-{end}; subj_b:0-{end}", elapsed


def test_pin_train_valid_and_test_windows_do_not_cross_joins(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    # Two ranges on one subject, then a second subject. Joins stay in train.
    a1 = _fmt(float(elapsed[39]))
    b0 = _fmt(float(elapsed[40]))
    spec = f"subj_a:0-{a1},{b0}-{end}; subj_b:0-{end}"
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    _write_subject(tmp_path, "subj_b", np.linspace(10.0, 11.0, n))
    for split in ("train", "valid", "test"):
        dataset = _load(
            tmp_path,
            segments=spec,
            split=split,
            val_indices=0.25,
            shuffle=True,
            seq_len=16,
        )
        seen = _windows_one_segment(dataset)
        _windows_stay_inside_boundaries(dataset)
        assert len(seen) >= 1
    # The train head contains both of subj_a's ranges and the join.
    train = _load(tmp_path, segments=spec, split="train", val_indices=0.0)
    assert len(_windows_one_segment(train)) >= 2


def test_pin_overlap_windows_do_not_cross_joins(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    spec = f"subj_a:0-{_fmt(float(elapsed[39]))},{_fmt(float(elapsed[40]))}-{end}"
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    dataset = _load(
        tmp_path,
        segments=spec,
        overlap=True,
        strict_non_overlap=False,
        seq_len=16,
        val_indices=0.0,
    )
    _windows_one_segment(dataset)
    _windows_stay_inside_boundaries(dataset)
    # Overlapping starts still stop at the join: adjacent starts differ by 1,
    # and none of them sit on the boundary itself.
    assert len(dataset.data_idx) > 2


def test_pin_fixed_spectrum_windows_do_not_cross_segment_joins(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    spec = f"subj_a:0-{_fmt(float(elapsed[39]))},{_fmt(float(elapsed[40]))}-{end}"
    _write_subject(tmp_path, "subj_a", np.linspace(-2.0, 2.0, n))
    dataset = _load(
        tmp_path,
        segments=spec,
        split="valid",
        val_indices=0.5,
        shuffle=True,
        seq_len=16,
        max_window_nan_ratio=1.0,
    )
    group = np.asarray(dataset.segment_index_per_row)
    assert group is not None
    window_len = 16
    _windows, info = capture_fixed_spectrum_windows(
        dataset,
        n_windows=2,
        window_len=window_len,
        tf_steps=8,
        return_info=True,
    )
    assert len(info["starts"]) >= 1
    for start in info["starts"]:
        piece = group[int(start) : int(start) + window_len]
        assert len(piece) == window_len
        assert len(set(int(v) for v in piece.tolist())) == 1


# ---------------------------------------------------------------------------
# (4) per-subject non-test norm; each segment holds out its own tail
# ---------------------------------------------------------------------------


def _piece_index(n_piece: int, n_pieces: int, *, head: bool) -> np.ndarray:
    cut = _waveform_head_length(n_piece)
    parts = []
    for i in range(n_pieces):
        lo = i * n_piece
        if head:
            parts.append(np.arange(lo, lo + cut))
        else:
            parts.append(np.arange(lo + cut, lo + n_piece))
    return np.concatenate(parts)


def test_pin_g2_each_piece_contributes_its_own_tail(tmp_path):
    """Each segment holds out its own last 20% of rows.

    At 250 Hz a 2 minute piece is 30_000 samples, so that piece's tail is
    6_000 rows. Five pieces contribute five tails. The scale is fit on
    every non-test head of the subject, not on one piece's own rows.
    Eval with ``seq_len`` unset keeps the same tails.
    """
    fs = 250
    piece_rows = fs * 120
    assert piece_rows == 30000
    piece_head = _waveform_head_length(piece_rows)
    assert piece_head == 24000
    assert piece_rows - piece_head == 6000

    n_piece = 40
    n = 5 * n_piece
    head = _waveform_head_length(n_piece)
    tail = n_piece - head
    assert head == 32 and tail == 8
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "g.2", np.linspace(-5.0, 5.0, n))
    ranges = []
    for i in range(5):
        lo = i * n_piece
        hi = lo + n_piece - 1
        ranges.append(f"{_fmt(float(elapsed[lo]))}-{_fmt(float(elapsed[hi]))}")
    spec = "g.2:" + ",".join(ranges)
    train = _load(tmp_path, segments=spec, split="train", seq_len=16)
    held = _load(tmp_path, segments=spec, split="test", seq_len=16)
    eval_like = _load(
        tmp_path, segments=spec, split="test", seq_len=None, segment_min_len=16
    )
    assert len(train.full_sequence) == 5 * head
    assert len(held.full_sequence) == 5 * tail
    clock = _clock(n)
    _same_clock(train.full_sequence["datetime"], clock[_piece_index(n_piece, 5, head=True)])
    _same_clock(held.full_sequence["datetime"], clock[_piece_index(n_piece, 5, head=False)])
    _same_clock(
        eval_like.full_sequence["datetime"], clock[_piece_index(n_piece, 5, head=False)]
    )
    _same(held.seq, eval_like.seq, atol=1e-5)
    for i, row in enumerate(held.waveform_segments):
        assert row["n_samples"] == n_piece
        assert row["row_lo"] == i * tail
        assert row["row_hi"] == (i + 1) * tail
    raw = np.linspace(-5.0, 5.0, n)
    heads = raw[_piece_index(n_piece, 5, head=True)]
    tails = raw[_piece_index(n_piece, 5, head=False)]
    mean = heads.mean()
    std = heads.std()
    _same(np.asarray(held.seq)[:, 0], (tails - mean) / std, atol=1e-5)
    # One piece's own tail is not the scale.
    own = tails[-tail:]
    own_scaled = (own - own.mean()) / own.std()
    assert not np.allclose(np.asarray(held.seq)[-tail:, 0], own_scaled, atol=1e-3)


def test_pin_valid_windows_come_from_the_non_test_head_only(tmp_path):
    n_piece = 40
    n = 5 * n_piece
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "g.2", np.linspace(-5.0, 5.0, n))
    ranges = []
    for i in range(5):
        lo = i * n_piece
        hi = lo + n_piece - 1
        ranges.append(f"{_fmt(float(elapsed[lo]))}-{_fmt(float(elapsed[hi]))}")
    spec = "g.2:" + ",".join(ranges)
    valid = _load(
        tmp_path,
        segments=spec,
        split="valid",
        val_indices=0.25,
        shuffle=True,
        seq_len=16,
    )
    train = _load(
        tmp_path,
        segments=spec,
        split="train",
        val_indices=0.25,
        shuffle=True,
        seq_len=16,
    )
    heads = _clock(n)[_piece_index(n_piece, 5, head=True)]
    for dataset in (train, valid):
        _same_clock(dataset.full_sequence["datetime"], heads)
        _windows_one_segment(dataset)
        _windows_stay_inside_boundaries(dataset)
    assert set(train.data_idx).isdisjoint(valid.data_idx)
    assert train.data_idx and valid.data_idx


def test_pin_each_segment_keeps_its_own_tail_in_written_order(tmp_path):
    """Each segment contributes its own tail, in the order written.

    Listing the later interval first still holds out that interval's own
    last 20%, then the earlier interval's last 20%. The earlier interval
    is not the whole test set.
    """
    n = 100
    elapsed = _elapsed(n)
    values = np.arange(n, dtype=np.float64)
    _write_subject(tmp_path, "subj_a", values)
    # 80 later samples, then 20 earlier ones.
    early = f"{_fmt(float(elapsed[0]))}-{_fmt(float(elapsed[19]))}"
    late = f"{_fmt(float(elapsed[20]))}-{_fmt(float(elapsed[99]))}"
    spec = f"subj_a:{late},{early}"
    late_head = _waveform_head_length(80)
    early_head = _waveform_head_length(20)
    assert late_head == 64 and early_head == 16
    clock = _clock(n)
    test_idx = np.concatenate(
        [np.arange(20 + late_head, 100), np.arange(early_head, 20)]
    )
    train_idx = np.concatenate(
        [np.arange(20, 20 + late_head), np.arange(0, early_head)]
    )
    held = _load(tmp_path, segments=spec, split="test", seq_len=16)
    _same_clock(held.full_sequence["datetime"], clock[test_idx])
    train = _load(tmp_path, segments=spec, split="train", seq_len=16)
    _same_clock(train.full_sequence["datetime"], clock[train_idx])
    eval_like = _load(
        tmp_path, segments=spec, split="test", seq_len=None, segment_min_len=16
    )
    _same_clock(eval_like.full_sequence["datetime"], clock[test_idx])
    _same(held.seq, eval_like.seq)


def test_pin_head_cut_matches_single_session_iloc(tmp_path):
    """The segment head is ``iloc[: -n // 5]``, unary minus included.

    ``-n // 5`` is ``(-n) // 5``, not ``-(n // 5)``. For n = 101 that is a
    head of 80 and a tail of 21, on both the segment path and the crop path.
    """
    for n in range(1, 40):
        frame = pd.DataFrame({"a": np.arange(n)})
        assert _waveform_head_length(n) == len(frame.iloc[: -n // 5])
        assert n - _waveform_head_length(n) == len(frame.iloc[-n // 5 :])

    n = 101
    elapsed = _elapsed(n)
    values = np.linspace(-1.0, 1.0, n)
    _write_subject(tmp_path, "subj_a", values)
    spec = f"subj_a:0-{_fmt(float(elapsed[-1]))}"
    seg_train = _load(tmp_path, segments=spec, split="train", seq_len=16)
    leg_train = _load(tmp_path, dataset_label="subj_a", split="train", seq_len=16)
    seg_test = _load(tmp_path, segments=spec, split="test", seq_len=16)
    leg_test = _load(tmp_path, dataset_label="subj_a", split="test", seq_len=16)
    assert len(seg_train.full_sequence) == len(leg_train.full_sequence) == 80
    assert len(seg_test.full_sequence) == len(leg_test.full_sequence) == 21
    _same_clock(seg_train.full_sequence["datetime"], leg_train.full_sequence["datetime"])
    _same_clock(seg_test.full_sequence["datetime"], leg_test.full_sequence["datetime"])
    # Train rows are the non-test rows, so the scales match. The test split
    # on the segment path keeps the train scale; the unset path rescales
    # the tail on its own.
    _same(seg_train.seq, leg_train.seq, atol=1e-5)
    assert not np.allclose(np.asarray(seg_test.seq), np.asarray(leg_test.seq), atol=1e-3)


def test_pin_norm_uses_non_test_rows_and_does_not_pool_subjects(tmp_path):
    n = 100
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    left = np.linspace(-3.0, 3.0, n)
    right = np.linspace(50.0, 70.0, n)
    _write_subject(tmp_path, "subj_a", left)
    _write_subject(tmp_path, "subj_b", right)
    spec = f"subj_a:0-{end}; subj_b:0-{end}"
    train = _load(tmp_path, segments=spec, split="train")
    held = _load(tmp_path, segments=spec, split="test")
    head = _waveform_head_length(n)
    seq = np.asarray(train.seq)[:, 0]
    _same(seq[:head], (left[:head] - left[:head].mean()) / left[:head].std(), atol=1e-5)
    _same(seq[head:], (right[:head] - right[:head].mean()) / right[:head].std(), atol=1e-5)
    test_seq = np.asarray(held.seq)[:, 0]
    _same(
        test_seq[: n - head],
        (left[head:] - left[:head].mean()) / left[:head].std(),
        atol=1e-5,
    )
    pooled = np.concatenate([left[:head], right[:head]])
    pooled_left = (left[:head] - pooled.mean()) / pooled.std()
    assert not np.allclose(seq[:head], pooled_left, atol=1e-2)


def test_pin_constant_non_test_std_blanks_the_subject(tmp_path):
    n = 100
    elapsed = _elapsed(n)
    # Head is 80 constant samples. The tail varies, and must still go to NaN.
    values = np.concatenate([np.ones(80), np.linspace(-2.0, 2.0, 20)])
    _write_subject(tmp_path, "subj_a", values)
    spec = f"subj_a:0-{_fmt(float(elapsed[-1]))}"
    with pytest.warns(UserWarning, match="non-test std"):
        held = _load(tmp_path, segments=spec, split="test", seq_len=16)
    assert np.isnan(np.asarray(held.seq)).all()


# ---------------------------------------------------------------------------
# (5) NaNs stay; nothing is resampled or inserted
# ---------------------------------------------------------------------------


def test_pin_nans_stay_and_no_samples_are_inserted(tmp_path):
    n = 120
    elapsed = _elapsed(n)
    values = np.linspace(-1.0, 1.0, n)
    values[4:7] = np.nan
    values[70:74] = np.nan
    _write_subject(tmp_path, "subj_a", values)
    a1 = _fmt(float(elapsed[49]))
    b0 = _fmt(float(elapsed[80]))
    b1 = _fmt(float(elapsed[119]))
    spec = f"subj_a:0-{a1},{b0}-{b1}"
    with pytest.warns(UserWarning, match="resample_hz is ignored"):
        dataset = _load(tmp_path, segments=spec, resample_hz=50.0, split="train")
    # Heads are 40 and 32, so train keeps 72.
    assert len(dataset.full_sequence) == 72
    clock = _clock(n)
    # The join is the original gap, not a filled sample.
    join = int(dataset.waveform_segments[1]["row_lo"])
    delta = (
        dataset.full_sequence["datetime"].iloc[join]
        - dataset.full_sequence["datetime"].iloc[join - 1]
    )
    assert delta > pd.Timedelta("4ms") * 2
    assert dataset.full_sequence["datetime"].iloc[0] == clock[0]
    # Leading NaNs are kept on the segment path (rows 4:7 are inside the head).
    assert dataset.missing_mask[4:7].all()
    assert np.isnan(np.asarray(dataset.seq)[4:7]).all()


def test_pin_interpolate_does_not_cross_the_join(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    values = np.linspace(0.0, 1.0, n)
    values[30:40] = np.nan
    _write_subject(tmp_path, "subj_a", values)
    spec = (
        f"subj_a:0-{_fmt(float(elapsed[39]))},"
        f"{_fmt(float(elapsed[40]))}-{_fmt(float(elapsed[79]))}"
    )
    dataset = _load(
        tmp_path,
        segments=spec,
        observation_process="raw_ch1_interpolate",
        seq_len=16,
        split="train",
        max_window_nan_ratio=1.0,
        gap_split_quantile="none",
        max_gap_window_frac="off",
    )
    # Z-score runs on both segments' heads. Interpolation then fills
    # inside the already sliced piece, so the first head cannot borrow
    # a value from the second.
    raw = np.linspace(0.0, 1.0, n)
    raw[30:40] = np.nan
    head = _waveform_head_length(40)
    assert head == 32
    non_test = np.concatenate([raw[:head], raw[40 : 40 + head]])
    mean = np.nanmean(non_test)
    std = np.nanstd(non_test)
    scaled = (raw[:head] - mean) / std
    idx = np.arange(head)
    missing = np.isnan(scaled)
    valid = ~missing
    scaled[missing] = np.interp(idx[missing], idx[valid], scaled[valid])
    got = np.asarray(dataset.seq)[:head, 0]
    np.testing.assert_allclose(got, scaled, atol=1e-5)
    assert not np.isnan(got).any()
    # NaNs at the end of the head take the last observed sample.
    assert abs(float(got[-1]) - float((raw[29] - mean) / std)) < 1e-6
    second = float(np.asarray(dataset.seq)[head, 0])
    assert abs(second - float((raw[40] - mean) / std)) < 1e-6


# ---------------------------------------------------------------------------
# (6) validation errors, and the sidecar must resolve inside data_dir
# ---------------------------------------------------------------------------


def test_pin_validation_errors(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    end = _fmt(float(elapsed[-1]))
    with pytest.raises(ValueError, match="unknown subject"):
        _load(tmp_path, segments=f"no_such:0-{end}")
    with pytest.raises(ValueError, match="out of range"):
        _load(tmp_path, segments="subj_a:0-50")
    with pytest.raises(ValueError, match="start >= end"):
        parse_waveform_segments("subj_a:5-5")
    with pytest.raises(ValueError, match="start >= end"):
        parse_waveform_segments("subj_a:4-1")
    with pytest.raises(ValueError, match="overlapping segments"):
        parse_waveform_segments("subj_a:0-1,1-2")
    with pytest.raises(ValueError, match="crop_start_sec"):
        _load(
            tmp_path,
            segments=f"subj_a:0-{end}",
            crop_start_sec=0.0,
            crop_end_sec=0.1,
        )
    with pytest.raises(ValueError, match="waveform"):
        _load(tmp_path, observation_process="coarse_eeg", segments=f"subj_a:0-{end}")
    with pytest.raises(ValueError, match="dataset_label"):
        _load(tmp_path, dataset_label="other", segments=f"subj_a:0-{end}")


def test_pin_short_segment_dropped_and_all_short_raises(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    short_end = float(elapsed[7])
    spec = (
        f"subj_a:0-{_fmt(short_end)},"
        f"{_fmt(float(elapsed[20]))}-{_fmt(float(elapsed[79]))}"
    )
    with pytest.warns(UserWarning, match="shorter than seq_len"):
        dataset = _load(tmp_path, segments=spec, seq_len=16)
    assert dataset.waveform_segments[0]["dropped"] == "shorter_than_seq_len"
    assert dataset.waveform_segments[0]["windows"] == 0
    assert dataset.waveform_segments[1]["dropped"] is None
    assert dataset.full_sequence["datetime"].iloc[0] == _clock(n)[20]
    tiny = (
        f"subj_a:0-{_fmt(short_end)},"
        f"{_fmt(float(elapsed[10]))}-{_fmt(float(elapsed[15]))}"
    )
    with pytest.warns(UserWarning, match="shorter than seq_len"):
        with pytest.raises(ValueError, match="shorter than seq_len"):
            _load(tmp_path, segments=tiny, seq_len=16)


def test_pin_sidecar_must_resolve_inside_data_dir(tmp_path, monkeypatch):
    """A sidecar that resolves outside data_dir is rejected.

    Absolute paths and ``..`` relatives that leave data_dir raise. An
    absolute path that stays inside data_dir is still read.
    """
    data = tmp_path / "data"
    data.mkdir()
    n = 40
    elapsed = _elapsed(n)
    _write_subject(data, "subj_a", np.linspace(-1.0, 1.0, n))
    end = float(elapsed[-1])
    outside = tmp_path / "outside"
    outside.mkdir()
    sidecar = outside / "segments.yaml"
    sidecar.write_text(
        yaml.safe_dump({"subj_a": [[0.0, end]]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside data_dir"):
        _load(data, segments=str(sidecar), seq_len=16)
    relative = os.path.relpath(sidecar, data)
    assert relative.startswith("..")
    with pytest.raises(ValueError, match="outside data_dir"):
        _load(data, segments=relative, seq_len=16)
    inside = data / "segments.yaml"
    inside.write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")
    kept = _load(data, segments=str(inside), seq_len=16)
    assert kept.dataset_labels == ["subj_a"]


def test_pin_symlink_sidecar_and_nested_dotdot_stay_inside_data_dir(tmp_path):
    """Resolution, not the written string, decides confinement.

    A symlink inside data_dir that points at a file outside it raises.
    ``foo/../../outside.yaml`` raises the same way. A relative name that
    exists only in the process cwd is not opened.
    """
    data = tmp_path / "data"
    data.mkdir()
    n = 40
    elapsed = _elapsed(n)
    _write_subject(data, "subj_a", np.linspace(-1.0, 1.0, n))
    end = float(elapsed[-1])
    outside = tmp_path / "outside"
    outside.mkdir()
    sidecar = outside / "segments.yaml"
    body = yaml.safe_dump({"subj_a": [[0.0, end]]})
    sidecar.write_text(body, encoding="utf-8")
    link = data / "link.yaml"
    link.symlink_to(sidecar)
    with pytest.raises(ValueError, match="outside data_dir"):
        _load(data, segments="link.yaml", seq_len=16)
    (data / "foo").mkdir()
    nested = "foo/../../outside/segments.yaml"
    with pytest.raises(ValueError, match="outside data_dir"):
        _load(data, segments=nested, seq_len=16)
    cwd_only = tmp_path / "cwd_only.yaml"
    cwd_only.write_text(body, encoding="utf-8")
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError, match="not found"):
            _load(data, segments="cwd_only.yaml", seq_len=16)
        # When both exist, the file under data_dir wins. The cwd copy names
        # a subject that is not on disk, so reading it would raise.
        (data / "both.yaml").write_text(body, encoding="utf-8")
        (tmp_path / "both.yaml").write_text(
            yaml.safe_dump({"no_such": [[0.0, end]]}),
            encoding="utf-8",
        )
        kept = _load(data, segments="both.yaml", seq_len=16)
    finally:
        os.chdir(old)
    assert kept.dataset_labels == ["subj_a"]


def test_pin_parser_without_data_dir_is_not_confined(tmp_path):
    """``data_dir=None`` is the parser-only case and still opens any file.

    The loader always passes ``path_to_data``, so this does not open a
    hole in ``XhroProper``. It stays unconfined on purpose.
    """
    outside = tmp_path / "outside.yaml"
    outside.write_text(
        yaml.safe_dump({"subj_a": [[0.0, 1.0]]}),
        encoding="utf-8",
    )
    rows = parse_waveform_segments(str(outside), data_dir=None)
    assert rows == [{"subject": "subj_a", "start": 0.0, "end": 1.0}]


def test_pin_subject_token_must_stay_inside_data_root(tmp_path):
    """A subject token cannot be a path that walks out of data_root.

    A session id that is a symlink to a directory outside the session
    parent is rejected after resolution.
    """
    n = 40
    elapsed = _elapsed(n)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_subject(tmp_path / "escaped", "evil", np.linspace(-1.0, 1.0, n))
    end = _fmt(float(elapsed[-1]))
    spec = f"../escaped/evil:0-{end}"
    for token in (".", "..", ".hidden", "-bad", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="session id"):
            parse_waveform_segments(f"{token}:0-1")
    # Session ids used by the corpus stay valid tokens.
    assert parse_waveform_segments("g.2:0-1")[0]["subject"] == "g.2"
    assert parse_waveform_segments("a+b:0-1")[0]["subject"] == "a+b"
    with pytest.raises(ValueError, match="session id"):
        parse_waveform_segments(spec)
    with pytest.raises(ValueError, match="session id"):
        _load(
            tmp_path,
            data_root=str(sessions),
            segments=spec,
            seq_len=16,
        )
    link = sessions / "legit"
    link.symlink_to(tmp_path / "escaped" / "evil")
    with pytest.raises(ValueError, match="outside data_root"):
        _load(
            tmp_path,
            data_root=str(sessions),
            segments=f"legit:0-{end}",
            seq_len=16,
        )


# ---------------------------------------------------------------------------
# (7) gap splits still apply inside a segment
# ---------------------------------------------------------------------------


def test_pin_gap_split_inside_a_segment_and_not_across_the_join(tmp_path):
    n = 160
    elapsed = _elapsed(n)
    values = np.linspace(-1.0, 1.0, n)
    # 8-sample hole inside the first segment. Floor at seq_len 16 is 4.
    values[30:38] = np.nan
    _write_subject(tmp_path, "subj_a", values)
    spec = (
        f"subj_a:0-{_fmt(float(elapsed[79]))},"
        f"{_fmt(float(elapsed[80]))}-{_fmt(float(elapsed[159]))}"
    )
    dataset = _load(
        tmp_path,
        segments=spec,
        seq_len=16,
        split="train",
        val_indices=0.0,
        max_window_nan_ratio=1.0,
        gap_split_quantile="none",
        max_gap_window_frac=0.25,
    )
    # Each segment of 80 holds out 16 rows, so train keeps 64 + 64.
    # The join of those heads is the split. The hole is applied at re-cut
    # time: no window covers rows 30:38, and none crosses the join.
    join = _waveform_head_length(80)
    assert join == 64
    assert len(dataset.full_sequence) == 128
    bounds = {int(cut) for cut in np.asarray(dataset.split_boundaries).tolist()}
    assert bounds == {join}
    covered = np.zeros(len(dataset.seq), dtype=bool)
    length = int(dataset.seq_len)
    for start in dataset.data_idx:
        covered[int(start) : int(start) + length] = True
    assert not covered[30:38].any()
    assert covered[:16].all()
    assert covered[38:54].all()
    _windows_stay_inside_boundaries(dataset)
    _windows_one_segment(dataset)
    assert dataset.missing_mask[30:38].all()
    assert np.isnan(np.asarray(dataset.seq)[30:38]).all()


# ---------------------------------------------------------------------------
# (8) per-segment log
# ---------------------------------------------------------------------------


def test_pin_per_segment_log_fields(tmp_path, capsys):
    n = 80
    elapsed = _elapsed(n)
    values = np.linspace(-1.0, 1.0, n)
    values[10:12] = np.nan
    _write_subject(tmp_path, "subj_a", values)
    _write_subject(tmp_path, "subj_b", np.linspace(2.0, 3.0, n))
    end = _fmt(float(elapsed[-1]))
    short = _fmt(float(elapsed[5]))
    spec = f"subj_a:0-{short},0.1-{end}; subj_b:0-{end}"
    # 0 to elapsed[5] is 6 samples, shorter than seq_len 16.
    with pytest.warns(UserWarning, match="shorter than seq_len"):
        dataset = _load(tmp_path, segments=spec, seq_len=16)
    logged = capsys.readouterr().out
    lines = [
        line
        for line in logged.splitlines()
        if line.startswith("[XhroProper][segment] subject=")
    ]
    assert len(lines) == 3
    for line in lines:
        assert "start=" in line
        assert "end=" in line
        assert "n_samples=" in line
        assert "missing_rate=" in line
        assert "windows=" in line
    assert "dropped=shorter_than_seq_len" in lines[0]
    assert "windows=0" in lines[0]
    assert "subject=subj_a" in lines[0]
    assert "subject=subj_a" in lines[1]
    assert "subject=subj_b" in lines[2]
    assert dataset.waveform_segments[1]["windows"] > 0
    assert dataset.waveform_segments[2]["windows"] > 0


# ---------------------------------------------------------------------------
# (9) eval drops the same short ranges as training, so the test rows match
# ---------------------------------------------------------------------------


def _config_segments(text: str) -> str | None:
    import configparser

    parser = configparser.ConfigParser()
    parser.read_string(f"[DataFrame]\nsegments = {text}\n")
    return optional_config_str(parser, "DataFrame", "segments")


def test_pin_eval_and_training_read_the_same_segments_string():
    spec = "subj_a:0-1.5,3-4; subj_b:10-12"
    assert _config_segments(spec) == spec
    # Both call sites pass optional_config_str into DatasetConfig.segments.
    train_src = Path("src/dvae/learning_algo.py").read_text()
    eval_src = Path("src/dvae/eval/eval_signal.py").read_text()
    assert 'segments=optional_config_str(self.cfg, "DataFrame", "segments")' in train_src
    assert 'segments=optional_config_str(cfg, "DataFrame", "segments")' in eval_src
    # Eval keeps seq_len unset and drops short ranges with the training cutoff.
    assert "seq_len=None" in eval_src
    assert "segment_min_len=training_initial_sequence_len(cfg)" in eval_src
    assert "training_initial_sequence_len(self.cfg)" in train_src
    config = DatasetConfig(
        data_dir=".",
        x_dim=1,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.2,
        observation_process="raw_ch1",
        overlap=False,
        with_nan=False,
        seq_len=None,
        device="cpu",
        segments=spec,
    )
    assert config.segments == spec


def test_pin_long_segments_match_between_training_seq_len_and_eval_none(tmp_path):
    n = 100
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    spec = f"subj_a:0-{_fmt(float(elapsed[39]))},{_fmt(float(elapsed[40]))}-{_fmt(float(elapsed[-1]))}"
    training = _load(tmp_path, segments=spec, split="test", seq_len=16)
    eval_like = _load(tmp_path, segments=spec, split="test", seq_len=None)
    pd.testing.assert_series_equal(
        training.full_sequence["datetime"].reset_index(drop=True),
        eval_like.full_sequence["datetime"].reset_index(drop=True),
    )
    _same(training.seq, eval_like.seq, atol=1e-5)


def test_pin_eval_segment_min_len_matches_training_test_rows(tmp_path):
    """Eval keeps ``seq_len`` unset and passes the training cutoff.

    ``segment_min_len`` is the constructor ``seq_len`` training used, so
    the short range is dropped before the 20% cut and the test clocks match.
    """
    n = 100
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.arange(n, dtype=np.float64))
    short = f"0-{_fmt(float(elapsed[7]))}"
    long = f"{_fmt(float(elapsed[20]))}-{_fmt(float(elapsed[99]))}"
    spec = f"subj_a:{short},{long}"
    training = _load(tmp_path, segments=spec, split="test", seq_len=16)
    eval_like = _load(
        tmp_path, segments=spec, split="test", seq_len=None, segment_min_len=16
    )
    train_clock = training.full_sequence["datetime"].reset_index(drop=True)
    eval_clock = eval_like.full_sequence["datetime"].reset_index(drop=True)
    assert train_clock.equals(eval_clock)
    _same(training.seq, eval_like.seq)
    assert training.waveform_segments[0]["dropped"] == "shorter_than_seq_len"
    assert eval_like.waveform_segments[0]["dropped"] == "shorter_than_seq_len"


def test_pin_omitting_segment_min_len_keeps_the_short_range(tmp_path):
    """The match depends on ``segment_min_len``.

    ``seq_len=None`` with the cutoff omitted keeps the short range and
    shifts the 20% tail. Eval does not take this path: it passes
    ``training_initial_sequence_len``.
    """
    n = 100
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.arange(n, dtype=np.float64))
    short = f"0-{_fmt(float(elapsed[7]))}"
    long = f"{_fmt(float(elapsed[20]))}-{_fmt(float(elapsed[99]))}"
    spec = f"subj_a:{short},{long}"
    training = _load(tmp_path, segments=spec, split="test", seq_len=16)
    omitted = _load(tmp_path, segments=spec, split="test", seq_len=None)
    assert training.waveform_segments[0]["dropped"] == "shorter_than_seq_len"
    assert omitted.waveform_segments[0]["dropped"] is None
    assert len(omitted.full_sequence) != len(training.full_sequence)
    train_clock = training.full_sequence["datetime"].reset_index(drop=True)
    omit_clock = omitted.full_sequence["datetime"].reset_index(drop=True)
    assert not train_clock.equals(omit_clock)


def test_pin_training_initial_sequence_len_is_the_shared_cutoff():
    """Training and eval share one cutoff.

    Unset ``initial_sequence_len`` falls back to ``sequence_len // 10``,
    the same fallback the training loader used before the helper. Training
    passes that value as ``seq_len`` and does not set ``segment_min_len``.
    Eval passes it as ``segment_min_len`` and leaves ``seq_len`` unset.
    """
    import configparser

    fallback = configparser.ConfigParser()
    fallback.read_string("[DataFrame]\nsequence_len = 160\n")
    assert training_initial_sequence_len(fallback) == 160 // 10
    explicit = configparser.ConfigParser()
    explicit.read_string(
        "[DataFrame]\nsequence_len = 160\ninitial_sequence_len = 40\n"
    )
    assert training_initial_sequence_len(explicit) == 40
    train_src = Path("src/dvae/learning_algo.py").read_text()
    start = train_src.index(
        "initial_sequence_len = training_initial_sequence_len(self.cfg)"
    )
    block = train_src[start : start + 1800]
    assert "seq_len=initial_sequence_len" in block
    assert "segment_min_len" not in block


def test_pin_strict_test_windows_use_the_val_slice(tmp_path):
    """Default strict windowing sends split='test' through Xhro.split_dataset.

    With ``val_indices=0`` that slice is empty, so a strict test loader has
    no windows. The test rows are still both subjects' tails. The overlap
    segment path keeps every test window instead. This window slice predates
    the per-segment tail and is not the row-match rule.
    """
    n = 100
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    _write_subject(tmp_path, "subj_b", np.linspace(2.0, 3.0, n))
    spec = f"subj_a:0-{end}; subj_b:0-{end}"
    common = dict(
        segments=spec,
        split="test",
        seq_len=16,
        val_indices=0.0,
        shuffle=False,
    )
    strict = _load(tmp_path, overlap=False, strict_non_overlap=True, **common)
    overlap = _load(tmp_path, overlap=True, strict_non_overlap=False, **common)
    # Both subjects contribute a 20-row tail, so the test series has a join.
    assert len(strict.full_sequence) == len(overlap.full_sequence) == 40
    assert strict.data_idx == []
    assert overlap.data_idx
    _windows_one_segment(overlap)
    _windows_stay_inside_boundaries(overlap)


# ---------------------------------------------------------------------------
# (10) the old Xhro loader is untouched
# ---------------------------------------------------------------------------


def test_pin_old_xhro_loader_has_no_segments_key():
    assert DATASET_REGISTRY["Xhro"] is Xhro
    assert DATASET_REGISTRY["XhroProper"] is XhroProper
    assert issubclass(XhroProper, Xhro)
    params = inspect.signature(Xhro.__init__).parameters
    assert "segments" not in params
    source = Path("src/dvae/dataset/xhro_dataset.py").read_text()
    assert "parse_waveform_segments" not in source
    assert "_init_waveform_segments" not in source
    assert "waveform_segments" not in source
