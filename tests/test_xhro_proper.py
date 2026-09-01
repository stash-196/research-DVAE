"""XhroProper loads grok intermediates (not old filtered_data.parquet)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.utils"] = MagicMock()
    sys.modules["torch.utils.data"] = MagicMock()
    sys.modules["torch.utils.data"].Dataset = object

from dvae.dataset.xhro_proper_dataset import (  # noqa: E402
    XhroProper,
    _resolve_variant,
    datetime_from_npz,
    load_proper_frame,
)


def _write_stage(path: Path, t, t_abs, **cols):
    utc_ns = pd.DatetimeIndex(t_abs).tz_convert("UTC").asi8
    extra = {}
    for k, v in cols.items():
        if isinstance(v, str):
            extra[k] = np.array(v)
        else:
            extra[k] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(
        path,
        t=np.asarray(t, dtype=np.float64),
        datetime_ns=np.asarray(utc_ns, dtype=np.int64),
        tz=np.asarray("Asia/Tokyo"),
        **extra,
    )


def _fake_session(root: Path, recording: str = "REC1", variant: str = "realtime") -> Path:
    sid = f"{recording}_{variant}"
    sess = root / "xhro_packet_loss" / "grok_output" / sid
    inter = sess / "intermediates"
    inter.mkdir(parents=True)
    n = 200
    t = np.arange(n, dtype=np.float64) * 0.004
    t_abs = pd.date_range("2026-06-22 14:00:00", periods=n, freq="4ms", tz="Asia/Tokyo")
    rng = np.random.default_rng(0)
    ch1 = rng.normal(size=n)
    ch1[10:15] = np.nan
    _write_stage(
        inter / "stage05_ecg_filtered.npz",
        t,
        t_abs,
        ch1=ch1,
        ch2=rng.normal(size=n),
        ecg=rng.normal(size=n),
    )
    _write_stage(
        inter / "stage06_eeg_filtered.npz",
        t,
        t_abs,
        ch3=rng.normal(size=n),
        ch4=rng.normal(size=n),
        eeg=rng.normal(size=n),
    )
    t_ppg = np.arange(40, dtype=np.float64) * 0.02
    t_abs_p = pd.date_range("2026-06-22 14:00:00", periods=40, freq="20ms", tz="Asia/Tokyo")
    _write_stage(
        inter / "stage07_ppg_filtered.npz",
        t_ppg,
        t_abs_p,
        filt_ch1=rng.normal(size=40),
        filt_ch2=rng.normal(size=40),
        filt_ch3=rng.normal(size=40),
        filt_ch4=rng.normal(size=40),
        primary="ch1",
    )
    t_acc = np.arange(20, dtype=np.float64) * 0.04
    t_abs_a = pd.date_range("2026-06-22 14:00:00", periods=20, freq="40ms", tz="Asia/Tokyo")
    _write_stage(
        inter / "stage08_acc.npz",
        t_acc,
        t_abs_a,
        x=rng.normal(size=20),
        y=rng.normal(size=20),
        z=rng.normal(size=20),
        motion_energy=np.abs(rng.normal(size=20)),
    )
    return sess


def test_registry():
    if not HAS_TORCH:
        pytest.skip("torch required to import dataset_builder")
    from dvae.dataset.dataset_builder import DATASET_REGISTRY

    assert DATASET_REGISTRY["XhroProper"] is XhroProper


def test_recovered_alias():
    assert _resolve_variant("recovered") == "retrans"
    assert _resolve_variant("retrans") == "retrans"


def test_load_frame_has_datetime(tmp_path):
    sess = _fake_session(tmp_path)
    df = load_proper_frame(sess, "raw_ch1")
    assert "datetime" in df.columns
    assert "ch1" in df.columns
    assert df["ch1"].isna().sum() == 5
    assert df["datetime"].dt.tz is not None


def test_bipolar_and_ppg_acc_frames(tmp_path):
    sess = _fake_session(tmp_path)
    ecg = load_proper_frame(sess, "bipolar_ecg")
    assert "ecg" in ecg.columns
    both = load_proper_frame(sess, "raw_all")
    assert list(both.columns)[:1] == ["datetime"]
    assert {"ch1", "ch2", "ch3", "ch4"}.issubset(both.columns)
    ppg = load_proper_frame(sess, "ppg_primary")
    assert "ppg_primary" in ppg.columns
    acc = load_proper_frame(sess, "acc_xyz")
    assert {"acc_x", "acc_y", "acc_z"}.issubset(acc.columns)


def test_datetime_from_npz_fallback(tmp_path):
    t = np.array([0.0, 1.0])
    p = tmp_path / "old.npz"
    np.savez_compressed(p, t=t, ch1=np.ones(2))
    z = np.load(p)
    dt = datetime_from_npz(z, t0=pd.Timestamp("2026-01-01 00:00:00+09:00"))
    assert (dt[1] - dt[0]).total_seconds() == 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_raw_ch1_dataset_shape(tmp_path):
    _fake_session(tmp_path)
    ds = XhroProper(
        data_dir=str(tmp_path),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=16,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    item = np.asarray(ds[0])
    assert item.shape[-1] == 1 or item.ndim == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_indicate_adds_mask_channel(tmp_path):
    _fake_session(tmp_path)
    ds = XhroProper(
        data_dir=str(tmp_path),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=16,
        x_dim=2,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_ch1_indicate",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    assert np.asarray(ds[0]).shape[-1] == 2
