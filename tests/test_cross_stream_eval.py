"""Cross-stream XhroProper eval when realtime and retrans ids differ.

Synthetic session folders only. Crop seconds and segment start/end stay
on the config; the recording id in the folder name is what changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    parse_stream_dataset_labels,
    recording_id_for_stream,
)
from dvae.eval.eval_signal import Options
from dvae.run_options import resolve_cross_stream_eval


REALTIME_ID = "XHRO3506_20260622T142410000+0900"
RETRANS_ID = "XHRO3506_20260622T142411000+0900"
STREAM_MAP = f"realtime:{REALTIME_ID},retrans:{RETRANS_ID}"


def _write_stage(path: Path, t, t_abs, **cols):
    # Loader reads datetime_ns as UTC nanoseconds. Pandas 3 stores this
    # clock in microseconds, so .asi8 would shrink a 0.1 s crop to nothing.
    utc_ns = pd.DatetimeIndex(t_abs).tz_convert("UTC").as_unit("ns").asi8
    extra = {
        key: np.asarray(value, dtype=np.float64) for key, value in cols.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.asarray(t, dtype=np.float64),
        datetime_ns=np.asarray(utc_ns, dtype=np.int64),
        tz=np.asarray("Asia/Tokyo"),
        **extra,
    )


def _session(root: Path, recording_id: str, variant: str, *, value: float, t0: str) -> Path:
    sess = root / "xhro_packet_loss" / "grok_output" / f"{recording_id}_{variant}"
    n = 200
    t = np.arange(n, dtype=np.float64) * 0.004
    t_abs = pd.date_range(t0, periods=n, freq="4ms", tz="Asia/Tokyo")
    channel = np.full(n, value, dtype=np.float64)
    channel[40] = value + 5.0
    _write_stage(
        sess / "intermediates" / "stage05_ecg_filtered.npz",
        t,
        t_abs,
        ch1=channel,
        ch2=channel,
        ecg=channel,
    )
    return sess


def _pair(root: Path) -> None:
    _session(
        root,
        REALTIME_ID,
        "realtime",
        value=1.0,
        t0="2026-06-22 14:24:10",
    )
    _session(
        root,
        RETRANS_ID,
        "retrans",
        value=2.0,
        t0="2026-06-22 14:24:11",
    )


def _dataset(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label=REALTIME_ID,
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
        crop_start_sec=0.1,
        crop_end_sec=0.3,
    )
    params.update(overrides)
    return XhroProper(**params)


def test_parse_stream_map_and_leave_other_subjects():
    parsed = parse_stream_dataset_labels(
        f" realtime:{REALTIME_ID} , recovered:{RETRANS_ID} "
    )
    assert parsed == {"realtime": REALTIME_ID, "retrans": RETRANS_ID}
    assert (
        recording_id_for_stream(
            REALTIME_ID,
            "retrans",
            parsed,
            corpus="packet_loss",
            anchor_ids=[REALTIME_ID],
        )
        == RETRANS_ID
    )
    assert (
        recording_id_for_stream(
            "OTHER",
            "retrans",
            parsed,
            corpus="packet_loss",
            anchor_ids=[REALTIME_ID],
        )
        == "OTHER"
    )
    assert (
        recording_id_for_stream(
            REALTIME_ID,
            None,
            parsed,
            corpus="multi",
            anchor_ids=[REALTIME_ID],
        )
        == REALTIME_ID
    )
    with pytest.raises(ValueError, match="variant:recording_id"):
        parse_stream_dataset_labels("retrans")


def test_mask_label_alone_looks_for_the_training_id(tmp_path):
    _pair(tmp_path)
    missing = (
        tmp_path
        / "xhro_packet_loss"
        / "grok_output"
        / f"{REALTIME_ID}_retrans"
    )
    with pytest.raises(FileNotFoundError) as exc:
        _dataset(tmp_path, mask_label="retrans")
    assert str(missing) in str(exc.value)
    assert not missing.exists()


def test_stream_map_loads_the_other_id_and_keeps_crop_times(tmp_path):
    _pair(tmp_path)
    home = _dataset(tmp_path, mask_label="realtime", stream_dataset_labels=STREAM_MAP)
    other = _dataset(tmp_path, mask_label="retrans", stream_dataset_labels=STREAM_MAP)
    assert home.session_dir.name == f"{REALTIME_ID}_realtime"
    assert other.session_dir.name == f"{RETRANS_ID}_retrans"
    assert other.dataset_label == REALTIME_ID
    assert other.crop_start_sec == 0.1
    assert other.crop_end_sec == 0.3
    assert home.crop_start_sec == other.crop_start_sec
    assert home.crop_end_sec == other.crop_end_sec
    assert len(home.full_sequence) == len(other.full_sequence)
    assert len(home.full_sequence) > 0
    assert other.resolved_recording_ids[REALTIME_ID] == RETRANS_ID


def test_segments_keep_start_and_end_on_the_other_stream(tmp_path):
    _pair(tmp_path)
    spec = f"{REALTIME_ID}:0.1-0.3"
    other = _dataset(
        tmp_path,
        mask_label="retrans",
        stream_dataset_labels=STREAM_MAP,
        crop_start_sec=None,
        crop_end_sec=None,
        segments=spec,
    )
    assert other.segments == spec
    assert other.session_dir.name == f"{RETRANS_ID}_retrans"
    assert other.waveform_segments[0]["subject"] == REALTIME_ID
    assert other.waveform_segments[0]["start"] == pytest.approx(0.1)
    assert other.waveform_segments[0]["end"] == pytest.approx(0.3)
    home = _dataset(
        tmp_path,
        mask_label="realtime",
        stream_dataset_labels=STREAM_MAP,
        crop_start_sec=None,
        crop_end_sec=None,
        segments=spec,
    )
    assert len(home.full_sequence) == len(other.full_sequence)


def test_eval_option_sets_the_target_recording_id_and_keeps_times():
    resolved = resolve_cross_stream_eval(
        REALTIME_ID,
        "realtime",
        "retrans",
        RETRANS_ID,
        None,
        None,
    )
    assert resolved["dataset_label"] == REALTIME_ID
    assert resolved["mask_label"] == "retrans"
    assert resolved["resolved_recording_id"] == RETRANS_ID
    assert resolved["stream_dataset_labels"]["retrans"] == RETRANS_ID
    assert resolved["eval_output_suffix"] == "retrans"

    from_map = resolve_cross_stream_eval(
        REALTIME_ID,
        "realtime",
        "retrans",
        None,
        None,
        STREAM_MAP,
    )
    assert from_map["resolved_recording_id"] == RETRANS_ID
    assert from_map["dataset_label"] == REALTIME_ID
    assert from_map["eval_output_suffix"] == "retrans"

    same = resolve_cross_stream_eval(
        REALTIME_ID,
        "realtime",
        None,
        None,
        None,
        STREAM_MAP,
    )
    assert same["resolved_recording_id"] == REALTIME_ID
    assert same["eval_output_suffix"] is None

    multi = resolve_cross_stream_eval(
        REALTIME_ID,
        "realtime",
        "retrans",
        None,
        None,
        STREAM_MAP,
        corpus="suntory",
    )
    assert multi["resolved_recording_id"] == REALTIME_ID
    assert multi["mask_label"] == "retrans"
    assert multi["eval_output_suffix"] == "retrans"

    options = Options()
    options._initial()
    args = options.parser.parse_args(
        [
            "--saved_dict",
            "model.pt",
            "--mask-label",
            "retrans",
            "--dataset-label",
            RETRANS_ID,
        ]
    )
    assert args.mask_label == "retrans"
    assert args.dataset_label == RETRANS_ID
