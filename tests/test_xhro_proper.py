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
    _require_session_dir,
    _resolve_variant,
    _session_dir,
    datetime_from_npz,
    load_proper_frame,
    optional_config_str,
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


def _write_intermediates(sess: Path) -> Path:
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


def _fake_session(root: Path, recording: str = "REC1", variant: str = "realtime") -> Path:
    sid = f"{recording}_{variant}"
    sess = root / "xhro_packet_loss" / "grok_output" / sid
    return _write_intermediates(sess)


def _ctor_kwargs(**overrides):
    params = dict(
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
    params.update(overrides)
    return params


def test_registry():
    if not HAS_TORCH:
        pytest.skip("torch required to import dataset_builder")
    from dvae.dataset.dataset_builder import DATASET_REGISTRY

    assert DATASET_REGISTRY["XhroProper"] is XhroProper


def test_recovered_alias():
    assert _resolve_variant("recovered") == "retrans"
    assert _resolve_variant("retrans") == "retrans"


def test_default_packet_loss_session_dir(tmp_path):
    recording = "XHRO3506_20260622T142410000+0900"
    expected = (
        tmp_path / "xhro_packet_loss" / "grok_output" / f"{recording}_realtime"
    )
    assert _session_dir(str(tmp_path), recording, "realtime") == expected
    assert (
        _session_dir(str(tmp_path), recording, "retrans", corpus="packet_loss")
        == tmp_path / "xhro_packet_loss" / "grok_output" / f"{recording}_retrans"
    )
    # Config sentinels must not change the historical path.
    assert (
        _session_dir(
            str(tmp_path), recording, "realtime", data_root="None", corpus="None"
        )
        == expected
    )


def test_data_root_replaces_session_parent(tmp_path):
    root = tmp_path / "custom" / "grok_output"
    got = _session_dir(
        str(tmp_path / "unused"),
        "REC1",
        "retrans",
        data_root=str(root),
    )
    assert got == root / "REC1_retrans"
    # Relative data_root is joined onto path_to_data. corpus still picks the name.
    multi = _session_dir(
        str(tmp_path),
        "xhro_01_XH015",
        "realtime",
        data_root="custom/grok_output",
        corpus="multi",
    )
    assert multi == tmp_path / "custom" / "grok_output" / "xhro_01_XH015"


def test_corpus_multi_session_dir_has_no_variant_suffix(tmp_path):
    label = "xhro_01_XH015"
    expected = tmp_path / "suntory" / "xhro_dataset_v2" / "grok_output" / label
    assert (
        _session_dir(str(tmp_path), label, "realtime", corpus="multi") == expected
    )
    assert _session_dir(str(tmp_path), label, None, corpus="suntory") == expected
    assert _session_dir(str(tmp_path), label, corpus="Multi") == expected


def test_missing_session_dir_names_the_path_tried(tmp_path):
    recording = "REC1"
    expected = tmp_path / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    with pytest.raises(FileNotFoundError) as exc:
        _require_session_dir(str(tmp_path), recording, "realtime")
    assert str(expected) in str(exc.value)

    label = "xhro_01_XH015"
    multi = tmp_path / "suntory" / "xhro_dataset_v2" / "grok_output" / label
    with pytest.raises(FileNotFoundError) as exc:
        _require_session_dir(str(tmp_path), label, "realtime", corpus="multi")
    assert str(multi) in str(exc.value)


def test_unknown_corpus_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown XhroProper corpus"):
        _session_dir(str(tmp_path), "REC1", "realtime", corpus="nope")


def test_optional_config_str():
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read_string(
        """
        [DataFrame]
        corpus = multi
        data_root = None
        blank =
        """
    )
    assert optional_config_str(cfg, "DataFrame", "corpus") == "multi"
    assert optional_config_str(cfg, "DataFrame", "data_root") is None
    assert optional_config_str(cfg, "DataFrame", "blank") is None
    assert optional_config_str(cfg, "DataFrame", "missing") is None


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
    assert ds.session_dir == (
        tmp_path / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    )


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


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_data_root_init_uses_explicit_parent(tmp_path):
    root = tmp_path / "sessions"
    sess = _write_intermediates(root / "REC1_retrans")
    ds = XhroProper(
        data_dir=str(tmp_path / "unused"),
        dataset_label="REC1",
        mask_label="recovered",
        data_root=str(root),
        **_ctor_kwargs(),
    )
    assert ds.session_dir == sess
    assert ds.variant == "retrans"
    assert ds.corpus == "packet_loss"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_corpus_multi_init_ignores_mask_label(tmp_path):
    label = "xhro_01_XH015"
    sess = _write_intermediates(
        tmp_path / "suntory" / "xhro_dataset_v2" / "grok_output" / label
    )
    ds = XhroProper(
        data_dir=str(tmp_path),
        dataset_label=label,
        mask_label="not-a-variant",
        corpus="multi",
        **_ctor_kwargs(),
    )
    assert ds.session_dir == sess
    assert ds.variant is None
    assert ds.corpus == "multi"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_build_dataloader_passes_corpus_and_data_root(tmp_path):
    from dvae.dataset.dataset_builder import DatasetConfig, build_dataloader

    label = "xhro_01_XH015"
    root = tmp_path / "multi_root"
    sess = _write_intermediates(root / label)
    cfg = DatasetConfig(
        data_dir=str(tmp_path / "unused"),
        x_dim=1,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_ch1",
        overlap=False,
        with_nan=True,
        seq_len=16,
        device="cpu",
        dataset_label=label,
        mask_label="None",
        data_root=str(root),
        corpus="multi",
    )
    train_dl, _val_dl, n_train, _n_val = build_dataloader("XhroProper", cfg, "train")
    assert n_train > 0
    assert train_dl.dataset.session_dir == sess

    # Default DatasetConfig fields keep the packet-loss path.
    _fake_session(tmp_path)
    default_cfg = DatasetConfig(
        data_dir=str(tmp_path),
        x_dim=1,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_ch1",
        overlap=False,
        with_nan=True,
        seq_len=16,
        device="cpu",
        dataset_label="REC1",
        mask_label="realtime",
    )
    train_dl, _, _, _ = build_dataloader("XhroProper", default_cfg, "train")
    assert train_dl.dataset.session_dir == (
        tmp_path / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    )
