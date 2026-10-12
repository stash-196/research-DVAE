"""XhroProper waveform segments. Synthetic NPZ only, no wearable files."""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    _waveform_head_length,
    optional_config_str,
    parse_waveform_segments,
)


def _elapsed(n: int) -> np.ndarray:
    clock = pd.date_range("2026-01-01", periods=n, freq="4ms", tz="Asia/Tokyo")
    return (clock - clock[0]).total_seconds().to_numpy(dtype=np.float64)


def _write_subject(root: Path, name: str, values: np.ndarray) -> Path:
    sess = Path(root) / name
    inter = sess / "intermediates"
    inter.mkdir(parents=True)
    n = int(len(values))
    t = np.arange(n, dtype=np.float64) * 0.004
    clock = pd.date_range("2026-01-01", periods=n, freq="4ms", tz="Asia/Tokyo")
    # Loader reads ``datetime_ns`` as UTC nanoseconds. Pandas 3 indexes are
    # microseconds, so ``asi8`` is not already in ns.
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


def _assert_windows_inside_one_segment(dataset) -> None:
    group = np.asarray(dataset.segment_index_per_row)
    length = int(dataset.seq_len)
    assert dataset.data_idx
    seen = set()
    for start in dataset.data_idx:
        piece = group[int(start) : int(start) + length]
        assert len(piece) == length
        ids = set(int(v) for v in piece.tolist())
        assert len(ids) == 1
        seen.add(next(iter(ids)))
    assert len(seen) >= 2


def test_inline_spec_and_config_roundtrip():
    rows = parse_waveform_segments("subj_a:0-1.5,3-4; subj_b:10-12")
    assert [(row["subject"], row["start"], row["end"]) for row in rows] == [
        ("subj_a", 0.0, 1.5),
        ("subj_a", 3.0, 4.0),
        ("subj_b", 10.0, 12.0),
    ]
    listed = parse_waveform_segments(
        [("subj_a", 0, 1.5), {"subject": "subj_b", "start": 2, "end": 3}]
    )
    assert [row["subject"] for row in listed] == ["subj_a", "subj_b"]

    parser = configparser.ConfigParser()
    parser.read_string(
        "[DataFrame]\nsegments = subj_a:0-1.5,3-4; subj_b:10-12\n"
    )
    text = optional_config_str(parser, "DataFrame", "segments")
    assert text == "subj_a:0-1.5,3-4; subj_b:10-12"
    assert parse_waveform_segments(text) == rows


def test_parser_rejects_bad_ranges():
    with pytest.raises(ValueError, match="start >= end"):
        parse_waveform_segments("subj_a:5-5")
    with pytest.raises(ValueError, match="start >= end"):
        parse_waveform_segments("subj_a:4-1")
    with pytest.raises(ValueError, match="overlapping segments"):
        parse_waveform_segments("subj_a:0-2,1-3")
    with pytest.raises(ValueError, match="overlapping segments"):
        parse_waveform_segments("subj_a:0-1,1-2")
    # Different subjects may cover the same clock time.
    both = parse_waveform_segments("subj_a:0-1; subj_b:0-1")
    assert len(both) == 2


def test_crop_path_matches_itself_when_segments_unset(tmp_path):
    n = 200
    values = np.random.default_rng(0).normal(size=n)
    values[10:15] = np.nan
    _write_subject(tmp_path, "REC1", values)
    elapsed = _elapsed(n)
    start = float(elapsed[25])
    end = float(elapsed[100])
    plain = _load(tmp_path, dataset_label="REC1")
    again = _load(tmp_path, dataset_label="REC1", segments=None)
    blank = _load(tmp_path, dataset_label="REC1", segments="None")
    empty = _load(tmp_path, dataset_label="REC1", segments="  ")
    for other in (again, blank, empty):
        np.testing.assert_array_equal(np.asarray(plain.seq), np.asarray(other.seq))
        assert plain.data_idx == other.data_idx
        np.testing.assert_array_equal(plain.missing_mask, other.missing_mask)
        pd.testing.assert_series_equal(
            plain.full_sequence["datetime"].reset_index(drop=True),
            other.full_sequence["datetime"].reset_index(drop=True),
        )
    cropped = _load(
        tmp_path, dataset_label="REC1", crop_start_sec=start, crop_end_sec=end
    )
    cropped_again = _load(
        tmp_path,
        dataset_label="REC1",
        crop_start_sec=start,
        crop_end_sec=end,
        segments=None,
    )
    np.testing.assert_array_equal(np.asarray(cropped.seq), np.asarray(cropped_again.seq))
    assert cropped.data_idx == cropped_again.data_idx
    np.testing.assert_array_equal(cropped.missing_mask, cropped_again.missing_mask)
    assert len(np.asarray(plain.seq)) == 160
    assert len(np.asarray(cropped.seq)) == len(np.asarray(cropped_again.seq))


def test_single_segment_keeps_the_crop_rows_and_nans(tmp_path):
    n = 200
    values = np.linspace(-2.0, 2.0, n)
    values[30:35] = np.nan
    _write_subject(tmp_path, "subj_a", values)
    elapsed = _elapsed(n)
    start = float(elapsed[25])
    end = float(elapsed[100])
    cropped = _load(
        tmp_path,
        dataset_label="subj_a",
        crop_start_sec=start,
        crop_end_sec=end,
    )
    spec = f"subj_a:{_fmt(start)}-{_fmt(end)}"
    segmented = _load(tmp_path, dataset_label="subj_a", segments=spec)
    pd.testing.assert_series_equal(
        segmented.full_sequence["datetime"].reset_index(drop=True),
        cropped.full_sequence["datetime"].reset_index(drop=True),
    )
    np.testing.assert_array_equal(segmented.missing_mask, cropped.missing_mask)
    # The crop does not invent or delete timestamps.
    assert len(segmented.full_sequence) == len(cropped.full_sequence)
    assert np.isnan(np.asarray(segmented.seq)[segmented.missing_mask]).all()

    whole = f"subj_a:0-{_fmt(float(elapsed[-1]))}"
    full = _load(tmp_path, segments=whole)
    legacy = _load(tmp_path, dataset_label="subj_a")
    assert len(full.full_sequence) == len(legacy.full_sequence) == 160
    pd.testing.assert_series_equal(
        full.full_sequence["datetime"].reset_index(drop=True),
        legacy.full_sequence["datetime"].reset_index(drop=True),
    )
    np.testing.assert_array_equal(full.missing_mask, legacy.missing_mask)
    # Rows 30:35 are inside the non-test head and stay NaN on the native grid.
    assert np.isnan(np.asarray(full.seq)[30:35]).all()
    assert full.missing_mask[30:35].all()


def test_multi_subject_normalises_separately_and_does_not_cross(tmp_path, capsys):
    n = 200
    elapsed = _elapsed(n)
    end = _fmt(float(elapsed[-1]))
    left = np.linspace(-3.0, 3.0, n)
    right = 80.0 + np.linspace(-1.0, 1.0, n)
    _write_subject(tmp_path, "subj_a", left)
    _write_subject(tmp_path, "subj_b", right)
    spec = f"subj_a:0-{end}; subj_b:0-{end}"
    dataset = _load(tmp_path, dataset_label="subj_a,subj_b", segments=spec)
    seq = np.asarray(dataset.seq)[:, 0]
    assert seq.shape == (320,)
    head = 160
    np.testing.assert_allclose(seq[:head].mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(seq[head:].mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(seq[:head].std(), 1.0, atol=1e-5)
    np.testing.assert_allclose(seq[head:].std(), 1.0, atol=1e-5)
    pooled = np.concatenate([left[:head], right[:head]])
    pooled_left = (left[:head] - pooled.mean()) / pooled.std()
    assert not np.allclose(seq[:head], pooled_left, atol=1e-2)
    assert dataset.dataset_labels == ["subj_a", "subj_b"]
    _assert_windows_inside_one_segment(dataset)
    assert head in set(int(cut) for cut in dataset.split_boundaries)
    logged = capsys.readouterr().out
    assert "subject=subj_a" in logged
    assert "subject=subj_b" in logged
    assert "n_samples=200" in logged
    assert "windows=" in logged
    assert dataset.waveform_segments[0]["windows"] > 0
    assert dataset.waveform_segments[1]["windows"] > 0
    # The held-out tail uses the same per-subject non-test statistics.
    held = _load(
        tmp_path,
        dataset_label="subj_a,subj_b",
        segments=spec,
        split="test",
        val_indices=0.0,
    )
    test_seq = np.asarray(held.seq)[:, 0]
    assert test_seq.shape == (80,)
    left_mean, left_std = left[:head].mean(), left[:head].std()
    right_mean, right_std = right[:head].mean(), right[:head].std()
    np.testing.assert_allclose(
        test_seq[:40], (left[head:] - left_mean) / left_std, atol=1e-5
    )
    np.testing.assert_allclose(
        test_seq[40:], (right[head:] - right_mean) / right_std, atol=1e-5
    )
    assert 40 in set(int(cut) for cut in held.split_boundaries)


def test_multi_segment_one_subject_windows_do_not_span(tmp_path):
    n = 400
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    # 120 samples, then a gap, then 120 samples. Closed ranges do not meet.
    a0, a1 = float(elapsed[0]), float(elapsed[119])
    b0, b1 = float(elapsed[200]), float(elapsed[319])
    spec = f"subj_a:{_fmt(a0)}-{_fmt(a1)},{_fmt(b0)}-{_fmt(b1)}"
    dataset = _load(tmp_path, segments=spec)
    assert len(dataset.waveform_segments) == 2
    assert dataset.waveform_segments[0]["n_samples"] == 120
    assert dataset.waveform_segments[1]["n_samples"] == 120
    # Each segment holds out its own last 20%, so the join is that head.
    head = _waveform_head_length(120)
    assert head == 96
    join = int(dataset.waveform_segments[1]["row_lo"])
    assert join == head
    assert join in set(int(cut) for cut in dataset.split_boundaries)
    _assert_windows_inside_one_segment(dataset)
    clock = dataset.full_sequence["datetime"]
    gap = clock.diff().dt.total_seconds().iloc[join]
    assert gap > 0.004 * 2
    # No sample was inserted to close that gap.
    assert len(dataset.full_sequence) == head + head


def test_yaml_and_csv_sidecars_match_inline(tmp_path):
    n = 200
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(0.0, 1.0, n))
    a0, a1 = float(elapsed[0]), float(elapsed[79])
    b0, b1 = float(elapsed[100]), float(elapsed[199])
    inline = f"subj_a:{_fmt(a0)}-{_fmt(a1)},{_fmt(b0)}-{_fmt(b1)}"
    yaml_path = tmp_path / "segments.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "subj_a": [[a0, a1], {"start": b0, "end": b1}],
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "segments.csv"
    csv_path.write_text(
        "subject,start,end\n"
        f"subj_a,{a0},{a1}\n"
        f"subj_a,{b0},{b1}\n",
        encoding="utf-8",
    )
    from_inline = _load(tmp_path, segments=inline)
    from_yaml = _load(tmp_path, segments=str(yaml_path))
    from_csv = _load(tmp_path, segments=str(csv_path))
    np.testing.assert_allclose(np.asarray(from_yaml.seq), np.asarray(from_inline.seq))
    np.testing.assert_allclose(np.asarray(from_csv.seq), np.asarray(from_inline.seq))
    assert from_yaml.data_idx == from_inline.data_idx == from_csv.data_idx
    _assert_windows_inside_one_segment(from_yaml)


def test_validation_errors(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    end = _fmt(float(elapsed[-1]))
    with pytest.raises(ValueError, match="unknown subject"):
        _load(tmp_path, segments=f"no_such:0-{end}")
    with pytest.raises(ValueError, match="out of range"):
        _load(tmp_path, segments="subj_a:0-50")
    with pytest.raises(ValueError, match="start >= end"):
        _load(tmp_path, segments="subj_a:1-0.2")
    with pytest.raises(ValueError, match="overlapping segments"):
        _load(tmp_path, segments=f"subj_a:0-{end},0-{end}")
    with pytest.raises(ValueError, match="crop_start_sec"):
        _load(
            tmp_path,
            segments=f"subj_a:0-{end}",
            crop_start_sec=0.0,
            crop_end_sec=0.1,
        )
    with pytest.raises(ValueError, match="dataset_label"):
        _load(tmp_path, dataset_label="other", segments=f"subj_a:0-{end}")
    with pytest.raises(ValueError, match="waveform"):
        _load(tmp_path, observation_process="coarse_eeg", segments=f"subj_a:0-{end}")
    with pytest.raises(ValueError, match="segments file not found"):
        _load(tmp_path, segments=str(tmp_path / "missing.yaml"))


def test_short_segment_is_warned_and_dropped(tmp_path):
    n = 80
    elapsed = _elapsed(n)
    _write_subject(tmp_path, "subj_a", np.linspace(-1.0, 1.0, n))
    short_end = float(elapsed[7])
    long_start = float(elapsed[20])
    long_end = float(elapsed[79])
    spec = (
        f"subj_a:0-{_fmt(short_end)},{_fmt(long_start)}-{_fmt(long_end)}"
    )
    with pytest.warns(UserWarning, match="shorter than seq_len"):
        dataset = _load(tmp_path, segments=spec, seq_len=16)
    assert dataset.waveform_segments[0]["dropped"] == "shorter_than_seq_len"
    assert dataset.waveform_segments[0]["windows"] == 0
    assert dataset.waveform_segments[0]["n_samples"] == 8
    assert dataset.waveform_segments[1]["dropped"] is None
    assert dataset.waveform_segments[1]["windows"] > 0
    assert dataset.waveform_segments[1]["n_samples"] == 60
    # The dropped samples are not on the stored time axis.
    first = dataset.full_sequence["datetime"].iloc[0]
    clock = pd.date_range("2026-01-01", periods=n, freq="4ms", tz="Asia/Tokyo")
    assert first == clock[20]

    tiny = f"subj_a:0-{_fmt(short_end)},{_fmt(float(elapsed[10]))}-{_fmt(float(elapsed[15]))}"
    with pytest.warns(UserWarning, match="shorter than seq_len"):
        with pytest.raises(ValueError, match="shorter than seq_len"):
            _load(tmp_path, segments=tiny, seq_len=16)


def test_template_and_generator_example():
    template = (
        Path(__file__).resolve().parents[1] / "config/general_signal/cfg_base_template.ini"
    ).read_text()
    assert "segments = {segments}" in template
    from bin.generate_config_file_for_multiple import xhro_proper_waveform_segments_example

    example = xhro_proper_waveform_segments_example()
    spec = example["segments"][0]
    rows = parse_waveform_segments(spec)
    assert [row["subject"] for row in rows] == [
        "xhro_04_XH058",
        "xhro_04_XH058",
        "xhro_01_XH032",
    ]
    assert example["crop_start_sec"] == ["None"]
    assert example["dataset_label"] == ["xhro_04_XH058,xhro_01_XH032"]
