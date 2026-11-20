"""Band-matched re-filter of stored XhroProper *_raw series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    crop_proper_frame,
    filter_bank_sos,
    filter_column_runs,
    load_proper_frame,
    sosfiltfilt_padlen,
)


def _tone(freq: float, n: int, fs: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return np.sin(2.0 * np.pi * freq * t)


def _mid_rms(values: np.ndarray) -> float:
    n = len(values)
    seg = np.asarray(values[n // 4 : n - n // 4], dtype=np.float64)
    return float(np.sqrt(np.mean(seg**2)))


def _write_stage(path: Path, n: int, fs: float, **extra) -> None:
    t = np.arange(n, dtype=np.float64) / fs
    base = pd.Timestamp("2026-01-01", tz="UTC").value
    payload = {
        "t": t,
        "fs": np.asarray(fs),
        "datetime_ns": base + np.rint(t * 1e9).astype(np.int64),
        "tz": np.asarray("UTC"),
        "ch1": np.zeros(n, dtype=np.float64),
        "ch2": np.zeros(n, dtype=np.float64),
        "ecg": np.zeros(n, dtype=np.float64),
    }
    payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _session(root: Path, variant: str = "realtime", n: int = 200, fs: float = 250.0, **extra) -> Path:
    sess = root / "xhro_packet_loss" / "grok_output" / f"REC1_{variant}"
    _write_stage(sess / "intermediates" / "stage05_ecg_filtered.npz", n, fs, **extra)
    return sess


def _eeg_session(root: Path, raw: np.ndarray, fs: float = 250.0) -> Path:
    n = len(raw)
    sess = root / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    t = np.arange(n, dtype=np.float64) / fs
    base = pd.Timestamp("2026-01-01", tz="UTC").value
    path = sess / "intermediates" / "stage06_eeg_filtered.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=t,
        fs=np.asarray(fs),
        datetime_ns=base + np.rint(t * 1e9).astype(np.int64),
        tz=np.asarray("UTC"),
        ch3=np.zeros(n),
        ch4=np.zeros(n),
        eeg=np.zeros(n),
        ch3_raw=raw,
        ch4_raw=raw,
        eeg_raw=raw,
        present=np.ones(n, dtype=bool),
        segment_id=np.zeros(n, dtype=np.int64),
    )
    return sess


def _load(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label="REC1",
        mask_label="realtime",
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
    )
    params.update(overrides)
    return XhroProper(**params)


def test_stored_bank_is_bit_identical_without_raw_keys(tmp_path, capsys):
    sess = _session(tmp_path, n=200, fs=250.0, ch1=np.linspace(-1.0, 1.0, 200))
    capsys.readouterr()
    plain = load_proper_frame(sess, "raw_ch1")
    for bank in (None, "", "stored", "None"):
        other = load_proper_frame(
            sess, "raw_ch1", filter_bank=bank, filter_min_run_sec=None
        )
        np.testing.assert_array_equal(
            plain["ch1"].to_numpy(), other["ch1"].to_numpy()
        )
    capsys.readouterr()
    left = _load(tmp_path)
    right = _load(tmp_path, filter_bank="stored")
    blank = _load(tmp_path, filter_bank="")
    np.testing.assert_array_equal(np.asarray(left.seq), np.asarray(right.seq))
    np.testing.assert_array_equal(np.asarray(left.seq), np.asarray(blank.seq))
    assert "filter_bank=" not in capsys.readouterr().out


def test_band_matched_passes_10hz_and_rejects_drift_and_line(tmp_path, capsys):
    fs = 250.0
    n = 2000
    tone10 = _tone(10.0, n, fs)
    sess = _eeg_session(tmp_path, tone10, fs)
    capsys.readouterr()
    frame = load_proper_frame(sess, "raw_ch3", filter_bank="band_matched_v1")
    logged = capsys.readouterr().out
    assert "filter_bank=band_matched_v1" in logged
    assert "stage=stage06_eeg_filtered.npz" in logged
    passed = frame["ch3"].to_numpy()
    assert _mid_rms(passed) == pytest.approx(1.0 / np.sqrt(2.0), abs=0.15)
    drift = filter_column_runs(
        _tone(0.2, n, fs),
        fs=fs,
        kind="eeg",
        bank="band_matched_v1",
        min_run_sec=2.0,
    )
    line = filter_column_runs(
        _tone(50.0, n, fs),
        fs=fs,
        kind="eeg",
        bank="band_matched_v1",
        min_run_sec=2.0,
    )
    assert _mid_rms(drift) < 0.05
    assert _mid_rms(line) < 0.05
    drift_hp05 = filter_column_runs(
        _tone(0.2, n, fs),
        fs=fs,
        kind="eeg",
        bank="band_matched_hp05_v1",
        min_run_sec=2.0,
    )
    assert _mid_rms(drift) < _mid_rms(drift_hp05) < 1.0 / np.sqrt(2.0)


def test_nan_gap_and_segment_and_present_do_not_leak(tmp_path):
    fs = 250.0
    n = 1600
    left = _tone(10.0, n, fs)
    right = _tone(7.0, n, fs)
    full = left.copy()
    full[900:] = right[900:]
    gapped = full.copy()
    gapped[800:900] = np.nan
    both = filter_column_runs(
        gapped, fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
    )
    alone_left = filter_column_runs(
        gapped[:800], fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
    )
    alone_right = filter_column_runs(
        gapped[900:], fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
    )
    np.testing.assert_allclose(both[:800], alone_left, atol=1e-10, rtol=0)
    np.testing.assert_allclose(both[900:], alone_right, atol=1e-10, rtol=0)
    assert np.isnan(both[800:900]).all()

    split_ids = np.array([0] * 800 + [1] * 800)
    split = filter_column_runs(
        full,
        fs=fs,
        kind="ecg",
        bank="band_matched_v1",
        min_run_sec=2.0,
        segment_id=split_ids,
    )
    np.testing.assert_allclose(
        split[:800],
        filter_column_runs(
            full[:800], fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
        ),
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        split[800:],
        filter_column_runs(
            full[800:], fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
        ),
        atol=1e-10,
        rtol=0,
    )

    present = np.ones(n, dtype=bool)
    present[800:900] = False
    held = filter_column_runs(
        full,
        fs=fs,
        kind="eeg",
        bank="band_matched_v1",
        min_run_sec=2.0,
        present=present,
    )
    assert np.isnan(held[800:900]).all()
    np.testing.assert_allclose(
        held[:800],
        filter_column_runs(
            full[:800], fs=fs, kind="eeg", bank="band_matched_v1", min_run_sec=2.0
        ),
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        held[900:],
        filter_column_runs(
            full[900:], fs=fs, kind="eeg", bank="band_matched_v1", min_run_sec=2.0
        ),
        atol=1e-10,
        rtol=0,
    )


def test_short_runs_and_padlen_become_nan():
    fs = 250.0
    short = _tone(10.0, 250, fs)
    out = filter_column_runs(
        short, fs=fs, kind="eeg", bank="band_matched_v1", min_run_sec=2.0
    )
    assert np.isnan(out).all()
    sos = filter_bank_sos("band_matched_v1", "eeg", fs)
    padlen = max(sosfiltfilt_padlen(section) for section in sos)
    series = _tone(10.0, padlen + 1, fs)
    blocked = filter_column_runs(
        series[:padlen],
        fs=fs,
        kind="eeg",
        bank="band_matched_v1",
        min_run_sec=0.0,
    )
    kept = filter_column_runs(
        series,
        fs=fs,
        kind="eeg",
        bank="band_matched_v1",
        min_run_sec=0.0,
    )
    assert np.isnan(blocked).all()
    assert np.isfinite(kept).all()
    assert np.nanmax(np.abs(kept)) > 0.0


def test_filter_is_applied_before_the_crop(tmp_path):
    fs = 250.0
    n = 2000
    raw = _tone(10.0, n, fs)
    sess = _session(
        tmp_path,
        n=n,
        fs=fs,
        ch1_raw=raw,
        ch2_raw=raw,
        ecg_raw=raw,
        present=np.ones(n, dtype=bool),
    )
    frame = load_proper_frame(sess, "raw_ch1", filter_bank="band_matched_v1")
    filtered = filter_column_runs(
        raw, fs=fs, kind="ecg", bank="band_matched_v1", min_run_sec=2.0
    )
    np.testing.assert_allclose(frame["ch1"].to_numpy(), filtered, atol=1e-10, rtol=0)
    cropped = crop_proper_frame(frame, 1.0, 3.0)
    elapsed = (frame["datetime"] - frame["datetime"].min()).dt.total_seconds().to_numpy()
    keep = (elapsed >= 1.0) & (elapsed <= 3.0)
    np.testing.assert_allclose(
        cropped["ch1"].to_numpy(), filtered[keep], atol=1e-10, rtol=0
    )


def test_missing_raw_and_unknown_bank_and_non_waveform(tmp_path):
    sess = _session(tmp_path, n=400, fs=250.0)
    with pytest.raises(ValueError, match="ch1_raw"):
        load_proper_frame(sess, "raw_ch1", filter_bank="band_matched_v1")
    with pytest.raises(ValueError, match="Unknown filter_bank"):
        load_proper_frame(sess, "raw_ch1", filter_bank="not_a_bank")
    with pytest.raises(ValueError, match="filter_min_run_sec"):
        load_proper_frame(sess, "raw_ch1", filter_min_run_sec=-1)
    empty = tmp_path / "empty"
    (empty / "xhro_packet_loss" / "grok_output" / "REC1_realtime").mkdir(parents=True)
    for process in ("ppg_ch1", "acc_xyz", "ecg_removed_eeg"):
        with pytest.raises(ValueError, match="filter_bank"):
            _load(empty, observation_process=process, filter_bank="band_matched_v1")
    with pytest.raises(ValueError, match="filter_bank"):
        _load(
            tmp_path,
            observation_process="coarse_eeg",
            filter_bank="band_matched_v1",
            x_dim=4,
        )
    with pytest.raises(ValueError, match="Unknown filter_bank"):
        _load(tmp_path, observation_process="coarse_eeg", filter_bank="nope", x_dim=4)


def _mask_file(path: Path) -> None:
    t0 = np.arange(8, dtype=np.float64)
    good = np.ones(8, dtype=bool)
    good[1] = False
    np.savez(
        path,
        t0=t0,
        dirty=np.zeros(8, dtype=bool),
        lowamp=np.zeros(8, dtype=bool),
        good=good,
    )


def test_packet_loss_retrans_segments_mask_and_filter(tmp_path):
    fs = 250.0
    n = 2000
    t = np.arange(n, dtype=np.float64) / fs
    raw = _tone(10.0, n, fs)
    raw[1000:1100] = np.nan
    sess = _session(
        tmp_path,
        variant="retrans",
        n=n,
        fs=fs,
        ch1=np.zeros(n),
        ch1_raw=raw,
        ch2_raw=raw,
        ecg_raw=np.zeros(n),
        present=np.ones(n, dtype=bool),
        segment_id=np.zeros(n, dtype=np.int64),
    )
    assert sess.name.endswith("_retrans")
    mask = tmp_path / "blocks.npz"
    _mask_file(mask)
    common = dict(
        data_dir=str(tmp_path),
        dataset_label="REC1",
        mask_label="retrans",
        seq_len=40,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.5,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=False,
        segments="REC1:0-7.996",
        block_mask_file=str(mask),
        block_mask_mode="not_good",
        contiguous_val_split=True,
    )
    stored = XhroProper(split="train", filter_bank=None, **common)
    assert np.isnan(np.asarray(stored.seq)).all()
    train = XhroProper(split="train", filter_bank="band_matched_v1", **common)
    valid = XhroProper(split="valid", filter_bank="band_matched_v1", **common)
    test = XhroProper(split="test", filter_bank="band_matched_v1", **common)
    assert train.variant == "retrans"
    assert str(train.session_dir).endswith("REC1_retrans")
    series = np.asarray(train.seq)
    head = series.reshape(-1)
    assert series.shape[0] == 1600
    assert np.isnan(head[250:500]).all()
    assert np.isnan(head[1000:1100]).all()
    assert np.isfinite(head[:250]).all()
    assert np.isfinite(head[1100:1280]).all()
    assert np.isfinite(np.asarray(test.seq)).all()
    assert len(test.seq) == 400
    train_starts = np.asarray(train.data_idx)
    val_starts = np.asarray(valid.data_idx)
    assert len(val_starts) == 8
    assert np.all(val_starts >= 1280)
    assert np.all(val_starts + 40 <= 1600)
    assert np.all(train_starts < 1280)
    assert np.all(train_starts + 40 <= 1280)
    # The shared head is the same series; val_indices 0.5 did not reshuffle it.
    np.testing.assert_array_equal(np.asarray(valid.seq), series)


def test_filter_min_run_sec_null_is_default():
    import configparser
    from dvae.dataset.xhro_proper_dataset import (
        optional_config_float,
        resolve_filter_min_run_sec,
    )

    cfg = configparser.ConfigParser()
    cfg.read_string("[DataFrame]\nfilter_min_run_sec = null\nfilter_bank = null\n")
    assert optional_config_float(cfg, "DataFrame", "filter_min_run_sec") is None
    assert resolve_filter_min_run_sec("null") == 2.0
