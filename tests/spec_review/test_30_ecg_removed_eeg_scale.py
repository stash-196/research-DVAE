"""Pin 5bce303: ecg_removed_eeg_scale reads stage09b eeg_clean_scale.

Synthetic NPZ only. The template process still reads eeg_clean_template.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvae.dataset.xhro_proper_dataset import (
    GROK_EXTRA_COLUMNS,
    GROK_EXTRA_PROCESSES,
    XhroProper,
    load_proper_frame,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE = "stage09b_ecg_remove.npz"
N = 20
NAN_AT = (2, 7)


def _irregular_clock():
    """Twenty samples on an irregular clock so a 250 Hz regrid would differ."""
    deltas = np.resize(np.array([0.003, 0.009], dtype=np.float64), N - 1)
    t = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(deltas)])
    start = pd.Timestamp("2026-06-22 14:00:00", tz="Asia/Tokyo")
    t_abs = start + pd.to_timedelta(t, unit="s")
    return t, pd.DatetimeIndex(t_abs)


def _series():
    scale = np.linspace(2.0, 4.0, N)
    template = np.linspace(-1.0, 1.0, N)
    for i in NAN_AT:
        scale[i] = np.nan
    template[4] = np.nan
    return scale, template


def _write_stage(path: Path, t, t_abs, **cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    utc_ns = pd.DatetimeIndex(t_abs).tz_convert("UTC").asi8
    payload = {
        "t": np.asarray(t, dtype=np.float64),
        "datetime_ns": np.asarray(utc_ns, dtype=np.int64),
        "tz": np.asarray("Asia/Tokyo"),
    }
    for name, values in cols.items():
        payload[name] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(path, **payload)


def _session(root: Path) -> Path:
    t, t_abs = _irregular_clock()
    scale, template = _series()
    sess = root / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    _write_stage(
        sess / "intermediates" / STAGE,
        t,
        t_abs,
        eeg_clean_template=template,
        eeg_clean_scale=scale,
    )
    return sess


def test_scale_and_template_read_their_own_stage09b_columns(tmp_path):
    sess = _session(tmp_path)
    scale, template = _series()

    assert GROK_EXTRA_COLUMNS["ecg_removed_eeg_scale"] == ["eeg_clean_scale"]
    assert GROK_EXTRA_COLUMNS["ecg_removed_eeg"] == ["eeg_clean_template"]
    assert "ecg_removed_eeg_scale" in GROK_EXTRA_PROCESSES

    scale_frame = load_proper_frame(sess, "ecg_removed_eeg_scale")
    template_frame = load_proper_frame(sess, "ecg_removed_eeg")

    assert list(scale_frame.columns) == ["datetime", "eeg_clean_scale"]
    assert list(template_frame.columns) == ["datetime", "eeg_clean_template"]
    assert np.array_equal(
        scale_frame["eeg_clean_scale"].to_numpy(), scale, equal_nan=True
    )
    assert np.array_equal(
        template_frame["eeg_clean_template"].to_numpy(), template, equal_nan=True
    )


def test_nans_stay_and_the_default_path_does_not_resample(tmp_path):
    sess = _session(tmp_path)
    scale, _template = _series()
    _t, t_abs = _irregular_clock()

    frame = load_proper_frame(sess, "ecg_removed_eeg_scale")
    got = frame["eeg_clean_scale"].to_numpy()
    assert len(frame) == N
    assert pd.DatetimeIndex(frame["datetime"]).equals(t_abs)
    assert np.array_equal(got, scale, equal_nan=True)
    for i in NAN_AT:
        assert np.isnan(got[i])

    regridded = load_proper_frame(sess, "ecg_removed_eeg_scale", resample_hz=250.0)
    assert len(regridded) != N

    assert inspect.signature(XhroProper.__init__).parameters["resample_hz"].default is None
    ds = XhroProper(
        data_dir=str(tmp_path),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=8,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="ecg_removed_eeg_scale",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    kept = N - N // 5
    assert ds.resample_hz is None
    assert len(ds.full_sequence) == kept
    assert pd.DatetimeIndex(ds.full_sequence["datetime"]).equals(t_abs[:kept])
    assert ds.sampling_freq == pytest.approx(1.0 / 0.003)
    seq = np.asarray(ds.seq)
    assert seq.shape == (kept, 1)
    nan_rows = np.flatnonzero(np.isnan(seq[:, 0]))
    assert list(nan_rows) == list(NAN_AT)


def test_scale_shares_the_template_time_axis(tmp_path):
    sess = _session(tmp_path)
    scale, template = _series()
    _t, t_abs = _irregular_clock()

    scale_frame = load_proper_frame(sess, "ecg_removed_eeg_scale")
    template_frame = load_proper_frame(sess, "ecg_removed_eeg")

    assert pd.DatetimeIndex(scale_frame["datetime"]).equals(
        pd.DatetimeIndex(template_frame["datetime"])
    )
    assert pd.DatetimeIndex(scale_frame["datetime"]).equals(t_abs)
    assert len(scale_frame) == len(t_abs) == len(template_frame)
    assert np.array_equal(
        scale_frame["eeg_clean_scale"].to_numpy(), scale, equal_nan=True
    )
    assert np.array_equal(
        template_frame["eeg_clean_template"].to_numpy(), template, equal_nan=True
    )

    short = scale[:-1].copy()
    bad = tmp_path / "short"
    _write_stage(
        bad / "intermediates" / STAGE,
        _t,
        t_abs,
        eeg_clean_scale=short,
    )
    with pytest.raises(ValueError, match="eeg_clean_scale"):
        load_proper_frame(bad, "ecg_removed_eeg_scale")


def test_missing_file_or_key_raises(tmp_path):
    t, t_abs = _irregular_clock()
    scale, template = _series()

    bare = tmp_path / "bare"
    bare.mkdir()
    missing = bare / "intermediates" / STAGE
    with pytest.raises(FileNotFoundError, match="eeg_clean_scale") as exc:
        load_proper_frame(bare, "ecg_removed_eeg_scale")
    assert str(missing) in str(exc.value)

    template_only = tmp_path / "template_only"
    _write_stage(
        template_only / "intermediates" / STAGE,
        t,
        t_abs,
        eeg_clean_template=template,
    )
    with pytest.raises(KeyError, match="eeg_clean_scale"):
        load_proper_frame(template_only, "ecg_removed_eeg_scale")
    template_frame = load_proper_frame(template_only, "ecg_removed_eeg")
    assert np.array_equal(
        template_frame["eeg_clean_template"].to_numpy(), template, equal_nan=True
    )

    both = tmp_path / "both"
    _write_stage(
        both / "intermediates" / STAGE,
        t,
        t_abs,
        eeg_clean_template=template,
        eeg_clean_scale=scale,
    )
    assert "eeg_clean_scale" in load_proper_frame(both, "ecg_removed_eeg_scale").columns


def test_generator_lists_the_scale_process():
    text = (ROOT / "bin" / "generate_config_file_for_multiple.py").read_text()
    scale_at = text.index('# "ecg_removed_eeg_scale"')
    start = text.rfind('elif dataset_name[0] == "XhroProper"', 0, scale_at)
    end = text.index("elif dataset_name", scale_at)
    block = text[start:end]
    assert "observation_process" in block
    entries = re.findall(r'#?\s*"([^"]+)"', block)
    assert "ecg_removed_eeg" in entries
    assert "ecg_removed_eeg_scale" in entries
    assert entries.index("ecg_removed_eeg_scale") == entries.index("ecg_removed_eeg") + 1
