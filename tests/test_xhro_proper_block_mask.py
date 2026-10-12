"""Block masks on synthetic waveform NPZs. No wearable files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvae.dataset.gap_grid import PRESENT_COL
from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    block_flags,
    block_mask_rows,
    parse_block_mask_mode,
)


def test_block_rows_follow_inferred_rate_and_spacing():
    # 200 Hz and 0.5 s blocks: 100 rows, not a hardcoded 250 Hz / 10 s grid.
    masked = block_mask_rows(
        500,
        200.0,
        np.array([0.0, 0.5, 1.0]),
        np.array([False, True, False]),
    )
    assert masked[100:200].all()
    assert not masked[:100].any()
    assert not masked[200:].any()


def test_block_modes_are_independent():
    assert parse_block_mask_mode(None) == "not_good"
    assert parse_block_mask_mode("") == "not_good"
    assert parse_block_mask_mode("Dirty") == "dirty"
    with pytest.raises(ValueError, match="block_mask_mode"):
        parse_block_mask_mode("both")
    dirty = np.array([True, False, False])
    lowamp = np.array([False, True, False])
    good = np.array([True, True, False])
    assert list(block_flags(dirty, lowamp, good, "not_good")) == [False, False, True]
    assert list(block_flags(dirty, lowamp, good, "dirty")) == [True, False, False]
    assert list(block_flags(dirty, lowamp, good, "lowamp")) == [False, True, False]
    assert list(block_flags(dirty, lowamp, good, "dirty_or_lowamp")) == [
        True,
        True,
        False,
    ]


def _write_stage(path: Path, n: int, fs: float, **extra):
    t = np.arange(n, dtype=np.float64) / fs
    base = pd.Timestamp("2026-01-01", tz="UTC").value
    payload = {
        "t": t,
        "datetime_ns": base + np.rint(t * 1e9).astype(np.int64),
        "tz": np.asarray("UTC"),
        "ch1": np.arange(n, dtype=np.float64) + 1.0,
        "ch2": np.linspace(-1.0, 1.0, n),
        "ecg": np.linspace(0.5, -0.5, n),
    }
    payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _session(root: Path, variant: str, n: int, fs: float, **extra) -> Path:
    sess = root / "xhro_packet_loss" / "grok_output" / f"REC1_{variant}"
    _write_stage(sess / "intermediates" / "stage05_ecg_filtered.npz", n, fs, **extra)
    return sess


def _mask_file(path: Path, t0, dirty, lowamp, good):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t0=np.asarray(t0, dtype=np.float64),
        dirty=np.asarray(dirty, dtype=bool),
        lowamp=np.asarray(lowamp, dtype=bool),
        good=np.asarray(good, dtype=bool),
    )


def _load(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=20,
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


def test_unset_block_mask_leaves_the_series(tmp_path, capsys):
    fs = 200.0
    _session(tmp_path, "realtime", 400, fs)
    capsys.readouterr()
    ds = _load(tmp_path)
    seq = np.asarray(ds.seq).reshape(-1)
    assert np.isfinite(seq).all()
    assert ds.observed_per_row is None
    assert "block mask" not in capsys.readouterr().out


def test_mask_nans_present_and_skips_uncovered_rows(tmp_path, capsys):
    fs = 200.0
    n = 500
    present = np.ones(n, dtype=bool)
    present[10] = False
    _session(tmp_path, "realtime", n, fs, present=present)
    # 0.5 s blocks at 200 Hz are 100 rows. The first block is not flagged,
    # so rows [0, 100) stay unflagged except the pre-existing hole at 10.
    mask = tmp_path / "blocks.npz"
    _mask_file(
        mask,
        t0=[0.0, 0.5, 1.0],
        dirty=[False, True, False],
        lowamp=[False, False, False],
        good=[True, False, True],
    )
    capsys.readouterr()
    ds = _load(tmp_path, block_mask_file=str(mask), block_mask_mode="not_good")
    seq = np.asarray(ds.seq).reshape(-1)
    # Train head keeps 400 rows, so the masked block [100, 200) is intact.
    assert np.isnan(seq[100:200]).all()
    assert np.isfinite(seq[11:100]).all()
    assert np.isfinite(seq[200:400]).all()
    assert np.isnan(seq[10])
    # One present=false row plus the 100-row masked block, on all 500 rows.
    assert ds.dataset_missing_rate == pytest.approx(101 / 500)
    assert not ds.observed_per_row[100:200].any()
    assert ds.observed_per_row[11]
    assert not ds.observed_per_row[10]
    logged = capsys.readouterr().out
    assert "block mask mode=not_good" in logged
    assert "masked=100/500 (0.200)" in logged
    assert "fs≈200" in logged


def test_dirty_mode_does_not_use_the_good_flag(tmp_path):
    fs = 200.0
    _session(tmp_path, "realtime", 400, fs)
    mask = tmp_path / "blocks.npz"
    _mask_file(
        mask,
        t0=[0.0, 0.5],
        dirty=[False, False],
        lowamp=[True, False],
        good=[False, True],
    )
    dirty = _load(tmp_path, block_mask_file=str(mask), block_mask_mode="dirty")
    low = _load(tmp_path, block_mask_file=str(mask), block_mask_mode="lowamp")
    either = _load(
        tmp_path, block_mask_file=str(mask), block_mask_mode="dirty_or_lowamp"
    )
    not_good = _load(tmp_path, block_mask_file=str(mask), block_mask_mode="not_good")
    assert np.isfinite(np.asarray(dirty.seq).reshape(-1)).all()
    assert np.isnan(np.asarray(low.seq).reshape(-1)[:100]).all()
    assert np.isnan(np.asarray(either.seq).reshape(-1)[:100]).all()
    assert np.isnan(np.asarray(not_good.seq).reshape(-1)[:100]).all()
    assert np.isfinite(np.asarray(not_good.seq).reshape(-1)[100:200]).all()


def test_mask_is_applied_before_crop_and_shared_by_variants(tmp_path, capsys):
    fs = 200.0
    n = 500
    _session(tmp_path, "realtime", n, fs)
    _session(tmp_path, "retrans", n, fs)
    mask = tmp_path / "blocks.npz"
    _mask_file(
        mask,
        t0=[0.0, 0.5],
        dirty=[True, False],
        lowamp=[False, False],
        good=[False, True],
    )
    capsys.readouterr()
    # Crop starts at 0.5 s, after the masked block, measured on this clock.
    cropped = _load(
        tmp_path,
        block_mask_file=str(mask),
        block_mask_mode="dirty",
        crop_start_sec=0.5,
    )
    seq = np.asarray(cropped.seq).reshape(-1)
    assert np.isfinite(seq).all()
    # The masked block sits before the crop, so it is not in the rate.
    assert cropped.dataset_missing_rate == pytest.approx(0.0)
    assert "masked=100/500 (0.200)" in capsys.readouterr().out
    other = _load(
        tmp_path,
        mask_label="retrans",
        block_mask_file=str(mask),
        block_mask_mode="dirty",
    )
    other_seq = np.asarray(other.seq).reshape(-1)
    assert np.isnan(other_seq[:100]).all()
    assert np.isfinite(other_seq[100:]).all()
    assert other.dataset_missing_rate == pytest.approx(100 / 500)


def test_masked_rows_are_gaps_for_starts_fraction_and_nan_ratio(tmp_path):
    fs = 200.0
    n = 400
    _session(tmp_path, "realtime", n, fs)
    mask = tmp_path / "blocks.npz"
    # Rows [40, 140).
    _mask_file(
        mask,
        t0=[0.2, 0.7],
        dirty=[True, False],
        lowamp=[False, False],
        good=[False, True],
    )
    common = dict(block_mask_file=str(mask), block_mask_mode="dirty")
    moved = _load(
        tmp_path,
        seq_len=20,
        strict_non_overlap=True,
        start_on_observed=True,
        **common,
    )
    assert 40 not in moved.data_idx
    assert 140 in moved.data_idx
    seq = np.asarray(moved.seq).reshape(-1)
    for start in moved.data_idx:
        assert np.isfinite(seq[int(start)])
    # Row 20 is observed. Its window ends where the hole begins.
    assert 20 in moved.data_idx

    dropped = _load(tmp_path, seq_len=20, max_window_nan_ratio=0.0, **common)
    assert not any(s < 140 and s + dropped.seq_len > 40 for s in dropped.data_idx)

    # 100 masked rows exceed 0.5 * 120, so the run is a split. The interior
    # piece is shorter than the window, so nothing starts inside it.
    split = _load(
        tmp_path,
        seq_len=120,
        max_gap_window_frac=0.5,
        gap_split_quantile="none",
        **common,
    )
    assert split.seq_len == 120
    assert split.split_boundaries is None
    assert not any(s < 140 and s + split.seq_len > 40 for s in split.data_idx)
    assert not any(40 <= int(s) < 140 for s in split.data_idx)
    assert any(int(s) >= 140 for s in split.data_idx)


def test_missing_file_and_bad_columns_raise(tmp_path):
    _session(tmp_path, "realtime", 50, 200.0)
    with pytest.raises(FileNotFoundError, match="block_mask_file"):
        _load(tmp_path, block_mask_file=str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError, match="block_mask_mode"):
        _load(tmp_path, block_mask_mode="nope")
    bad = tmp_path / "bad.npz"
    np.savez(bad, t0=np.array([0.0, 1.0]), dirty=np.array([True, False]))
    with pytest.raises(ValueError, match="missing keys"):
        _load(tmp_path, block_mask_file=str(bad))


def test_template_has_block_mask_keys():
    text = (
        Path(__file__).resolve().parents[1]
        / "config/general_signal/cfg_base_template.ini"
    ).read_text()
    assert "block_mask_file = {block_mask_file}" in text
    assert "block_mask_mode = {block_mask_mode}" in text
    assert PRESENT_COL == "__present"
