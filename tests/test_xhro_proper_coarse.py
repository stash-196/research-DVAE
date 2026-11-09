"""Stage-15 coarse features on XhroProper, with synthetic parquet only."""

from __future__ import annotations

import configparser
import importlib.util
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
    COARSE_MAD_TO_SIGMA,
    COARSE_OBS_COLUMNS,
    COARSE_SAMPLING_FREQ,
    COARSE_WINDOW_RNG_SEED,
    XhroProper,
    assign_shuffled_coarse_split,
    _apply_coarse_suffix,
    _provisional_nontest_mask,
    apply_rel_alpha_eeg_mask,
    build_coarse_subject,
    coarse_dataframe_kwargs,
    coarse_test_cut,
    join_hr_mean_onto_epochs,
    rebuild_tod_from_clock,
    _parse_coarse_labels,
)
from dvae.eval.window_length import apply_eval_seq_len, cap_eval_seq_len  # noqa: E402


def _wave(n, loc, scale=0.02, period=10.0):
    phase = np.arange(n, dtype=np.float64)
    return loc + scale * np.sin(2.0 * np.pi * phase / period)


def make_frames(n, label, seed=0):
    """Finite, varying coarse tables. QC flags are deliberately unusable."""
    rng = np.random.default_rng(seed)
    t_start = np.arange(n, dtype=np.float64) * 30.0
    rel_delta = np.clip(_wave(n, 0.25, 0.03), 0.05, 0.8)
    rel_theta = np.clip(_wave(n, 0.20, 0.02), 0.05, 0.8)
    rel_alpha = np.clip(_wave(n, 0.30, 0.02), 0.05, 0.8)
    rel_beta = np.clip(_wave(n, 0.25, 0.02), 0.05, 0.8)
    abs_delta = 10 ** _wave(n, 1.3, 0.05)
    abs_theta = 10 ** _wave(n, 1.1, 0.04)
    abs_alpha = 10 ** _wave(n, 1.4, 0.04)
    abs_beta = 10 ** _wave(n, 1.0, 0.04)
    abs_total = 10 ** _wave(n, 3.0, 0.03)
    epochs = pd.DataFrame(
        {
            "session_id": label,
            "session_kind": "multi",
            "t_start": t_start,
            "t_end": t_start + 30.0,
            "abs_delta": abs_delta,
            "abs_theta": abs_theta,
            "abs_alpha": abs_alpha,
            "abs_beta": abs_beta,
            "rel_delta": rel_delta,
            "rel_theta": rel_theta,
            "rel_alpha": rel_alpha,
            "rel_beta": rel_beta,
            "abs_total_0.5-30": abs_total,
            "iaf_hz": _wave(n, 10.0, 0.2),
            "eeg_rms_mad": np.abs(_wave(n, 5.0, 0.3)) + 0.2,
            "eeg_median": np.full(n, 12345.0),
            "eeg_iqr": np.full(n, 99.0),
            "gap_frac": np.ones(n),
            "loff_frac": np.ones(n),
            "valid_frac": np.zeros(n),
            "epoch_eeg_good": np.zeros(n, dtype=bool),
            "acc_motion_rms": np.abs(_wave(n, 0.4, 0.05)) + 0.05,
            "acc_motion_p95": np.full(n, -1.0),
            "tod_sin": np.sin(2.0 * np.pi * np.arange(n) / max(n, 1)),
            "tod_cos": np.cos(2.0 * np.pi * np.arange(n) / max(n, 1)),
        }
    )
    # A little extra jitter so a flat sine over a short head cannot be constant.
    epochs["rel_delta"] = epochs["rel_delta"] + 0.01 * rng.standard_normal(n)
    n_win = int(np.ceil(n / 10)) or 1
    hr_t = np.arange(n_win, dtype=np.float64) * 300.0
    hrv = pd.DataFrame(
        {
            "session_id": label,
            "session_kind": "multi",
            "t_start": hr_t,
            "t_end": hr_t + 300.0,
            "hr_mean_bpm": 50.0 + 3.0 * np.arange(n_win),
            "sdnn_ms": np.full(n_win, -999.0),
            "rmssd_ms": np.full(n_win, -999.0),
            "pnn50": np.full(n_win, -999.0),
            "n_beats_kept": np.full(n_win, 100),
            "coverage_frac": np.ones(n_win),
            "window_ok": np.ones(n_win, dtype=bool),
            "gap_frac": np.ones(n_win),
        }
    )
    return epochs, hrv


def write_session(tmp_path: Path, label: str, epochs: pd.DataFrame, hrv: pd.DataFrame) -> Path:
    folder = (
        tmp_path
        / "suntory"
        / "xhro_dataset_v2"
        / "grok_output"
        / label
        / "15_coarse_features"
    )
    folder.mkdir(parents=True)
    epochs.to_parquet(folder / "coarse_epochs_30s.parquet")
    hrv.to_parquet(folder / "coarse_hrv_300s.parquet")
    return folder.parent


def _base_name(process: str) -> str:
    for suffix in ("_interpolate", "_indicate"):
        if process.endswith(suffix):
            return process[: -len(suffix)]
    return process


def _x_dim(process: str) -> int:
    width = len(COARSE_OBS_COLUMNS[_base_name(process)])
    if process.endswith("_indicate"):
        width *= 2
    return width


def _load(tmp_path, label, process, **overrides):
    params = dict(
        data_dir=str(tmp_path),
        dataset_label=label,
        mask_label=None,
        corpus="multi",
        split="train",
        seq_len=8,
        x_dim=_x_dim(process),
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process=process,
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    params.update(overrides)
    base = _base_name(process)
    if (
        "coarse_include_hr" not in overrides
        and "hr_mean_bpm" in COARSE_OBS_COLUMNS.get(base, [])
    ):
        params["coarse_include_hr"] = True
    return XhroProper(**params)


def _col(process, name):
    return COARSE_OBS_COLUMNS[_base_name(process)].index(name)


def _seq(ds):
    arr = ds.seq.detach().cpu().numpy() if hasattr(ds.seq, "detach") else np.asarray(ds.seq)
    return np.asarray(arr)


def _assert_windows_stay_inside(ds):
    assert len(ds.window_subject) == len(ds.data_idx)
    for index, start in enumerate(ds.data_idx):
        assert ds.window_subject_id(index) == ds.window_subject[index]
        block = np.asarray(ds.subject_per_row[start : start + ds.seq_len])
        assert len(block) == ds.seq_len
        assert set(block.tolist()) == {ds.window_subject[index]}


@pytest.fixture(scope="module")
def _torch_required():
    if not HAS_TORCH:
        pytest.skip("torch not installed")


def test_parse_coarse_labels_one_equals_list():
    assert _parse_coarse_labels("xhro_01_XH032") == ["xhro_01_XH032"]
    assert _parse_coarse_labels("xhro_04_XH058,xhro_01_XH032") == [
        "xhro_04_XH058",
        "xhro_01_XH032",
    ]
    assert _parse_coarse_labels("xhro_04_XH058, xhro_01_XH032") == [
        "xhro_04_XH058",
        "xhro_01_XH032",
    ]
    assert _parse_coarse_labels("only,") == ["only"]


def test_hr_join_uses_containing_window_and_window_ok():
    epochs = pd.DataFrame(
        {
            "t_start": [0.0, 270.0, 300.0, 570.0, 600.0, 630.0],
            "t_end": [30.0, 300.0, 330.0, 600.0, 630.0, 660.0],
        }
    )
    hrv = pd.DataFrame(
        {
            "t_start": [0.0, 300.0, 600.0],
            "t_end": [300.0, 600.0, 900.0],
            "hr_mean_bpm": [55.0, np.nan, 80.0],
            "window_ok": [True, True, False],
            "sdnn_ms": [1.0, 2.0, 3.0],
            "gap_frac": [1.0, 1.0, 1.0],
        }
    )
    out = join_hr_mean_onto_epochs(epochs, hrv)
    got = out["hr_mean_bpm"].to_numpy()
    assert got[0] == pytest.approx(55.0)
    assert got[1] == pytest.approx(55.0)
    # t_start == previous t_end belongs to the next window. NaN HR stays NaN.
    assert np.isnan(got[2]) and np.isnan(got[3])
    # window_ok False, even with a finite rate.
    assert np.isnan(got[4]) and np.isnan(got[5])


def test_rel_alpha_mask_clears_eeg_and_ignores_epoch_eeg_good():
    frame = pd.DataFrame(
        {
            "rel_alpha": [0.2, np.nan],
            "rel_delta": [0.1, 5.0],
            "abs_delta": [3.0, 4.0],
            "abs_total_0.5-30": [10.0, 12.0],
            "iaf_hz": [9.0, 11.0],
            "eeg_rms_mad": [1.0, 2.0],
            "eeg_median": [0.0, 7.0],
            "hr_mean_bpm": [60.0, 61.0],
            "acc_motion_rms": [0.2, 0.3],
            "tod_sin": [0.1, 0.2],
            "epoch_eeg_good": [False, True],
        }
    )
    out = apply_rel_alpha_eeg_mask(frame)
    assert out.loc[0, "rel_delta"] == pytest.approx(0.1)
    assert np.isnan(out.loc[1, "rel_delta"])
    assert np.isnan(out.loc[1, "abs_delta"])
    assert np.isnan(out.loc[1, "abs_total_0.5-30"])
    assert np.isnan(out.loc[1, "iaf_hz"])
    assert np.isnan(out.loc[1, "eeg_rms_mad"])
    assert np.isnan(out.loc[1, "eeg_median"])
    assert out.loc[1, "hr_mean_bpm"] == pytest.approx(61.0)
    assert out.loc[1, "acc_motion_rms"] == pytest.approx(0.3)
    assert out.loc[1, "tod_sin"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "process,width",
    [
        ("coarse_hr", 1),
        ("coarse_hr_acc", 2),
        ("coarse_hr_acc_tod", 4),
        ("coarse_eeg", 4),
        ("coarse_eeg_acc", 5),
        ("coarse_eeg_acc_tod", 7),
        ("coarse_eeg_hr", 5),
        ("coarse_eeg_hr_acc", 6),
        ("coarse_eeg_hr_acc_tod", 8),
        ("coarse_full", 15),
    ],
)
def test_column_mapping_per_mode(tmp_path, _torch_required, process, width):
    epochs, hrv = make_frames(40, "subj_a", seed=1)
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(tmp_path, "subj_a", process, seq_len=8)
    seq = _seq(ds)
    assert seq.shape[1] == width
    assert ds.feature_columns == COARSE_OBS_COLUMNS[process]
    assert ds.sampling_freq == pytest.approx(COARSE_SAMPLING_FREQ)
    # epoch_eeg_good is false and valid_frac is 0; rows still load.
    assert np.isfinite(seq[0]).all()


def test_hr_join_and_eeg_mask_survive_normalization(tmp_path, _torch_required):
    epochs, hrv = make_frames(40, "subj_a", seed=2)
    epochs = epochs.iloc[::-1].reset_index(drop=True)
    hrv.loc[1, "window_ok"] = False
    epochs.loc[epochs["t_start"] == 150.0, "rel_alpha"] = np.nan
    epochs.loc[epochs["t_start"] == 150.0, "rel_delta"] = 50.0
    epochs.loc[epochs["t_start"] == 150.0, "epoch_eeg_good"] = True
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg_hr",
        seq_len=8,
        coarse_test_split="last20_rows",
    )
    seq = _seq(ds)
    hr = seq[:, _col("coarse_eeg_hr", "hr_mean_bpm")]
    # Time order is restored. 30 s epochs 0-9 share a window; 10-19 are not ok.
    assert np.allclose(hr[0:10], hr[0])
    assert np.isnan(hr[10:20]).all()
    assert np.allclose(hr[20:30], hr[20])
    assert hr[0] != pytest.approx(hr[20])
    # t_start 150 s is epoch 5. EEG cleared, HR kept. epoch_eeg_good ignored.
    assert np.isnan(seq[5, :4]).all()
    assert np.isfinite(hr[5])
    assert np.isfinite(seq[0, :4]).all()


def test_transforms_clip_and_zscore_use_nontest_only(tmp_path, _torch_required):
    n = 50
    epochs, hrv = make_frames(n, "subj_a", seed=3)
    epochs["abs_delta"] = 10 ** np.linspace(1.0, 2.0, n)
    epochs.loc[40:, "abs_delta"] = 1e9
    epochs["acc_motion_rms"] = 0.2 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 8.0)
    epochs.loc[40, "acc_motion_rms"] = 1e6
    epochs["tod_sin"] = np.arange(n, dtype=np.float64)
    epochs = epochs.iloc[::-1].reset_index(drop=True)
    write_session(tmp_path, "subj_a", epochs, hrv)
    train = _load(
        tmp_path,
        "subj_a",
        "coarse_full",
        seq_len=8,
        coarse_test_split="last20_rows",
        val_indices=0.0,
    )
    test = _load(
        tmp_path,
        "subj_a",
        "coarse_full",
        split="test",
        seq_len=4,
        coarse_test_split="last20_rows",
    )
    assert train.coarse_reports[0]["test_cut"] == 40
    report = train.coarse_reports[0]
    acc_i = _col("coarse_full", "acc_motion_rms")
    abs_i = _col("coarse_full", "abs_delta")
    tod_i = _col("coarse_full", "tod_sin")
    raw_acc = np.array(0.2 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 8.0))
    raw_acc[40] = 1e6
    log_acc = np.log1p(raw_acc[:40])
    expected = (np.log1p(raw_acc[0]) - np.nanmean(log_acc)) / np.nanstd(log_acc)
    assert _seq(train)[0, acc_i] == pytest.approx(expected, rel=1e-5)
    assert report["norm_mean"]["acc_motion_rms"] == pytest.approx(np.nanmean(log_acc))

    # Tail spike is not clipped (log1p only). Inverse z-score recovers log1p(1e6).
    mean = report["norm_mean"]["acc_motion_rms"]
    std = report["norm_std"]["acc_motion_rms"]
    recon_acc = float(_seq(test)[0, acc_i]) * std + mean
    assert recon_acc == pytest.approx(np.log1p(1e6), rel=1e-4)

    logs = np.linspace(1.0, 2.0, n)[:40]
    hi = float(np.percentile(logs, 99.5))
    assert report["clip_hi"]["abs_delta"] == pytest.approx(hi)
    recon_abs = float(_seq(test)[0, abs_i]) * report["norm_std"]["abs_delta"] + report[
        "norm_mean"
    ]["abs_delta"]
    assert recon_abs == pytest.approx(hi, rel=1e-4)
    assert recon_abs < 4.0
    # tod is sorted back to 0,1,2,... and not z-scored.
    assert _seq(train)[:5, tod_i].tolist() == pytest.approx([0, 1, 2, 3, 4])


def test_hr_modes_require_opt_in_and_warn_when_enabled(tmp_path, _torch_required):
    epochs, hrv = make_frames(40, "subj_a", seed=7)
    write_session(tmp_path, "subj_a", epochs, hrv)
    with pytest.raises(ValueError, match="hr_mean_bpm"):
        _load(tmp_path, "subj_a", "coarse_eeg_hr", coarse_include_hr=False)
    with pytest.raises(ValueError, match="hr_mean_bpm"):
        _load(tmp_path, "subj_a", "coarse_full", coarse_include_hr=False)
    with pytest.warns(UserWarning, match="hr_mean_bpm"):
        opted = _load(tmp_path, "subj_a", "coarse_full", coarse_include_hr=True)
    assert "hr_mean_bpm" in opted.feature_columns
    ecg_free = _load(tmp_path, "subj_a", "coarse_eeg_acc_tod")
    assert ecg_free.feature_columns == COARSE_OBS_COLUMNS["coarse_eeg_acc_tod"]
    assert "hr_mean_bpm" not in ecg_free.feature_columns


def test_std_guard_sets_constant_column_to_nan(tmp_path, _torch_required):
    epochs, hrv = make_frames(40, "subj_b", seed=4)
    hrv["hr_mean_bpm"] = 60.0
    write_session(tmp_path, "subj_b", epochs, hrv)
    with pytest.warns(UserWarning, match="hr_mean_bpm"):
        ds = _load(tmp_path, "subj_b", "coarse_eeg_hr", seq_len=8, val_indices=0.0)
    hr = _seq(ds)[:, _col("coarse_eeg_hr", "hr_mean_bpm")]
    assert np.isnan(hr).all()
    assert np.isfinite(_seq(ds)[:, _col("coarse_eeg_hr", "rel_alpha")]).any()
    assert "hr_mean_bpm" in ds.coarse_reports[0]["std_guarded"]


def test_no_eeg_modes_ignore_eeg_and_keep_usable_tail(tmp_path, _torch_required, capsys):
    epochs, hrv = make_frames(100, "subj_a", seed=11)
    epochs.loc[70:, "rel_alpha"] = np.nan
    epochs.loc[10:12, "abs_total_0.5-30"] = 1e12
    epochs.loc[5, "rel_delta"] = 0.99
    write_session(tmp_path, "subj_a", epochs, hrv)
    dropped = epochs.drop(
        columns=[c for c in epochs.columns if c in {
            "rel_delta", "rel_theta", "rel_alpha", "rel_beta",
            "abs_delta", "abs_theta", "abs_alpha", "abs_beta",
            "abs_total_0.5-30", "iaf_hz", "eeg_rms_mad", "eeg_median", "eeg_iqr",
        }]
    )
    write_session(tmp_path, "subj_b", dropped, hrv)

    plain = _load(tmp_path, "subj_a", "coarse_hr_acc_tod", seq_len=8)
    ruled = _load(
        tmp_path,
        "subj_a",
        "coarse_hr_acc_tod",
        seq_len=8,
        coarse_eeg_artifact_mask="log_total_power_robust",
        coarse_eeg_artifact_k=3.0,
        coarse_eeg_max_rel_delta=0.4,
    )
    assert plain.coarse_reports[0]["test_cut"] == 80
    assert plain.coarse_reports[0]["n_test_usable"] == 20
    assert ruled.coarse_reports[0]["test_cut"] == 80
    assert ruled.coarse_reports[0]["artifact"] == {}
    assert ruled.coarse_reports[0]["eeg_columns"] is False
    np.testing.assert_allclose(_seq(plain), _seq(ruled))
    logged = capsys.readouterr().out
    assert "artifact rules not applied (no EEG columns)" in logged

    eeg = _load(tmp_path, "subj_a", "coarse_eeg", seq_len=8)
    assert eeg.coarse_reports[0]["test_cut"] == 56

    hr = _load(tmp_path, "subj_a", "coarse_hr", seq_len=8)
    hr_no_cols = _load(tmp_path, "subj_b", "coarse_hr", seq_len=8)
    assert hr.coarse_reports[0]["test_cut"] == 80
    assert hr_no_cols.coarse_reports[0]["test_cut"] == 80
    np.testing.assert_allclose(_seq(hr), _seq(hr_no_cols))

    indicated = _load(tmp_path, "subj_a", "coarse_hr_indicate", seq_len=8)
    assert _seq(indicated).shape[1] == 2
    assert indicated.feature_columns == ["hr_mean_bpm"]
    interpolated = _load(tmp_path, "subj_a", "coarse_hr_acc_interpolate", seq_len=8)
    assert _seq(interpolated).shape[1] == 2
    tod = _load(tmp_path, "subj_a", "coarse_hr_acc_tod_indicate", seq_len=8)
    assert _seq(tod).shape[1] == 8


def test_usable_tail_versus_last20_rows(tmp_path, _torch_required):
    epochs, hrv = make_frames(100, "subj_a", seed=5)
    epochs.loc[70:, "rel_alpha"] = np.nan
    write_session(tmp_path, "subj_a", epochs, hrv)
    usable = _load(tmp_path, "subj_a", "coarse_eeg", seq_len=8, coarse_test_split="usable_tail")
    last = _load(tmp_path, "subj_a", "coarse_eeg", seq_len=8, coarse_test_split="last20_rows")
    assert usable.coarse_reports[0]["test_cut"] == 56
    assert usable.coarse_reports[0]["n_test_usable"] == 14
    assert last.coarse_reports[0]["test_cut"] == 80
    assert last.coarse_reports[0]["n_test_usable"] == 0
    usable_test = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        split="test",
        seq_len=8,
        coarse_test_split="usable_tail",
    )
    last_test = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        split="test",
        seq_len=8,
        coarse_test_split="last20_rows",
    )
    assert _seq(usable_test).shape[0] == 44
    assert _seq(last_test).shape[0] == 20


def test_per_subject_norm_and_windows_do_not_cross(tmp_path, _torch_required):
    for label, loc in (("subj_a", 0.2), ("subj_b", 0.8)):
        epochs, hrv = make_frames(100, label, seed=6)
        epochs["rel_alpha"] = loc + 0.05 * np.sin(2 * np.pi * np.arange(100) / 10.0)
        write_session(tmp_path, label, epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a, subj_b",
        "coarse_eeg",
        seq_len=10,
        overlap=True,
        shuffle=False,
        val_indices=0.0,
    )
    assert ds.dataset_labels == ["subj_a", "subj_b"]
    seq = _seq(ds)
    alpha = _col("coarse_eeg", "rel_alpha")
    for label in ("subj_a", "subj_b"):
        block = seq[ds.subject_per_row == label, alpha]
        assert np.nanmean(block) == pytest.approx(0.0, abs=1e-4)
    _assert_windows_stay_inside(ds)
    # overlap=True would emit starts in (boundary - seq_len, boundary) if the
    # two heads were one series.
    boundary = int(np.sum(ds.subject_per_row == "subj_a"))
    assert all(
        start <= boundary - ds.seq_len or start >= boundary for start in ds.data_idx
    )


def test_train_valid_split_is_deterministic_and_spares_global_rng(tmp_path, _torch_required):
    for label in ("subj_a", "subj_b"):
        epochs, hrv = make_frames(40, label, seed=7)
        write_session(tmp_path, label, epochs, hrv)
    np.random.seed(12345)
    expected = np.random.rand(4)
    np.random.seed(12345)
    train = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        seq_len=8,
        overlap=True,
        shuffle=True,
        val_indices=0.25,
    )
    got = np.random.rand(4)
    assert np.allclose(expected, got)
    train2 = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        seq_len=8,
        overlap=True,
        shuffle=True,
        val_indices=0.25,
    )
    valid = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        split="valid",
        seq_len=8,
        overlap=True,
        shuffle=True,
        val_indices=0.25,
    )
    pooled = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        seq_len=8,
        overlap=True,
        shuffle=False,
        val_indices=0.0,
    )
    assert train.data_idx == train2.data_idx
    assert train.window_subject == train2.window_subject
    assert set(train.data_idx).isdisjoint(valid.data_idx)
    assert set(train.data_idx) | set(valid.data_idx) == set(pooled.data_idx)
    _assert_windows_stay_inside(train)
    _assert_windows_stay_inside(valid)


def test_test_split_keeps_every_intra_subject_window(tmp_path, _torch_required):
    for label in ("subj_a", "subj_b"):
        epochs, hrv = make_frames(40, label, seed=8)
        write_session(tmp_path, label, epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        split="test",
        seq_len=4,
        overlap=False,
        shuffle=True,
        val_indices=0.25,
    )
    # 8-row tails, non-overlapping length 4 → 2 windows each. Not val_indices of that.
    assert len(ds.data_idx) == 4
    _assert_windows_stay_inside(ds)


def test_indicate_doubles_and_matches_plain_on_observed(tmp_path, _torch_required):
    epochs, hrv = make_frames(40, "subj_a", seed=9)
    epochs.loc[4, "rel_alpha"] = np.nan
    write_session(tmp_path, "subj_a", epochs, hrv)
    plain = _load(
        tmp_path, "subj_a", "coarse_eeg", seq_len=8, coarse_test_split="last20_rows"
    )
    ind = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg_indicate",
        seq_len=8,
        coarse_test_split="last20_rows",
    )
    assert _seq(ind).shape[1] == 8
    assert ind.missing_mask.shape[1] == 4
    assert np.allclose(_seq(ind)[0, 0::2], _seq(plain)[0])
    assert np.allclose(_seq(ind)[0, 1::2], 1.0)
    assert np.isnan(_seq(plain)[4]).all()
    assert np.allclose(_seq(ind)[4, 0::2], 0.0)
    assert np.allclose(_seq(ind)[4, 1::2], 0.0)
    assert ind.missing_mask[4].all()
    assert not ind.missing_mask[0].any()


def test_interpolate_does_not_cross_subjects(tmp_path, _torch_required):
    epochs_a, hrv_a = make_frames(40, "subj_a", seed=10)
    epochs_b, hrv_b = make_frames(40, "subj_b", seed=11)
    epochs_a["tod_sin"] = 0.1
    epochs_a.loc[30, "tod_sin"] = np.nan
    epochs_a.loc[31, "tod_sin"] = np.nan
    epochs_b["tod_sin"] = 0.9
    write_session(tmp_path, "subj_a", epochs_a, hrv_a)
    write_session(tmp_path, "subj_b", epochs_b, hrv_b)
    ds = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg_hr_acc_tod_interpolate",
        seq_len=8,
        coarse_test_split="last20_rows",
        val_indices=0.0,
    )
    tod = _seq(ds)[:, _col("coarse_eeg_hr_acc_tod", "tod_sin")]
    # Edge hold inside A, not a blend toward B's 0.9.
    assert tod[30] == pytest.approx(0.1)
    assert tod[31] == pytest.approx(0.1)
    assert tod[32] == pytest.approx(0.9)
    assert ds.missing_mask[30, _col("coarse_eeg_hr_acc_tod", "tod_sin")]
    assert ds.missing_mask[31, _col("coarse_eeg_hr_acc_tod", "tod_sin")]
    assert not ds.missing_mask[29, _col("coarse_eeg_hr_acc_tod", "tod_sin")]


def test_missing_mask_indexes_flat_multi_subject_series(tmp_path, _torch_required):
    epochs_a, hrv_a = make_frames(40, "subj_a", seed=12)
    epochs_b, hrv_b = make_frames(40, "subj_b", seed=13)
    epochs_b.loc[0, "rel_alpha"] = np.nan
    write_session(tmp_path, "subj_a", epochs_a, hrv_a)
    write_session(tmp_path, "subj_b", epochs_b, hrv_b)
    ds = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        seq_len=8,
        coarse_test_split="last20_rows",
        shuffle=False,
        val_indices=0.0,
        start_on_observed=False,
    )
    boundary = int(np.sum(ds.subject_per_row == "subj_a"))
    assert not ds.missing_mask[0].any()
    assert ds.missing_mask[boundary].all()
    b_windows = [i for i, start in enumerate(ds.data_idx) if start == boundary]
    assert b_windows
    mask = ds.get_missing_mask(b_windows[0])
    start = ds.data_idx[b_windows[0]]
    assert np.array_equal(mask, ds.missing_mask[start : start + ds.seq_len])
    window = _seq(ds)[start : start + ds.seq_len]
    assert np.array_equal(mask, ~np.isfinite(window))
    assert mask[0].all()
    a_mask = ds.get_missing_mask(0)
    assert not a_mask[0].any()


def test_single_label_matches_one_element_list(tmp_path, _torch_required):
    epochs, hrv = make_frames(40, "subj_a", seed=14)
    write_session(tmp_path, "subj_a", epochs, hrv)
    one = _load(tmp_path, "subj_a", "coarse_eeg", seq_len=8, shuffle=True, val_indices=0.25)
    listed = _load(tmp_path, "subj_a,", "coarse_eeg", seq_len=8, shuffle=True, val_indices=0.25)
    spaced = _load(
        tmp_path, "subj_a, ", "coarse_eeg", seq_len=8, shuffle=True, val_indices=0.25
    )
    assert one.dataset_labels == ["subj_a"]
    np.testing.assert_array_equal(_seq(one), _seq(listed))
    np.testing.assert_array_equal(_seq(one), _seq(spaced))
    assert one.data_idx == listed.data_idx == spaced.data_idx


def test_update_sequence_length_round_trip_and_nan_filter(tmp_path, _torch_required, capsys):
    epochs, hrv = make_frames(40, "subj_a", seed=15)
    epochs.loc[0:9, "rel_alpha"] = np.nan
    write_session(tmp_path, "subj_a", epochs, hrv)
    for label in ("subj_b",):
        eb, hb = make_frames(40, label, seed=16)
        write_session(tmp_path, label, eb, hb)
    ds = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        seq_len=10,
        overlap=True,
        shuffle=True,
        val_indices=0.2,
        coarse_test_split="last20_rows",
    )
    ds.update_sequence_length(8)
    first_idx = list(ds.data_idx)
    first_subj = list(ds.window_subject)
    _assert_windows_stay_inside(ds)
    ds.update_sequence_length(11)
    assert ds.seq_len == 11
    _assert_windows_stay_inside(ds)
    ds.update_sequence_length(8)
    assert ds.seq_len == 8
    assert list(ds.data_idx) == first_idx
    assert list(ds.window_subject) == first_subj

    filtered = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        seq_len=10,
        overlap=False,
        shuffle=False,
        val_indices=0.0,
        coarse_test_split="last20_rows",
        start_on_observed=False,
    )
    capsys.readouterr()
    filtered.update_sequence_length(10)
    # 10 all-NaN epochs out of 40, four EEG columns: rate 0.25, threshold 0.45.
    # The all-NaN prefix is above that, so it is dropped. The rate is the
    # full subject, not the post-cut train rows (those would be 10/32).
    assert filtered.max_window_nan_ratio is None
    assert filtered.max_window_nan_excess == pytest.approx(0.2)
    assert filtered.dataset_missing_rate == pytest.approx(0.25)
    assert 0 not in filtered.data_idx
    logged = capsys.readouterr().out
    assert "dataset_missing_rate=0.25" in logged
    assert "threshold=0.45 (relative)" in logged
    assert "windows_dropped=" in logged
    filtered.update_sequence_length(10, minimum_nan_ratio=1.0)
    assert 0 in filtered.data_idx
    filtered.update_sequence_length(10)
    assert 0 not in filtered.data_idx
    indicated = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg_indicate",
        seq_len=10,
        overlap=False,
        shuffle=False,
        val_indices=0.0,
        coarse_test_split="last20_rows",
        start_on_observed=False,
    )
    # Indicate doubles the width with a finite 0/1 channel. The rate stays
    # on the pre-indicate feature matrix.
    assert indicated.dataset_missing_rate == pytest.approx(filtered.dataset_missing_rate)


def test_artifact_mask_off_by_default_keeps_high_power(tmp_path, _torch_required):
    epochs, hrv = make_frames(80, "subj_a", seed=17)
    epochs.loc[2:4, "abs_total_0.5-30"] = 1e8
    epochs.loc[2:4, "rel_alpha"] = 0.95
    write_session(tmp_path, "subj_a", epochs, hrv)
    default = _load(tmp_path, "subj_a", "coarse_eeg", seq_len=8, val_indices=0.0)
    explicit = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        seq_len=8,
        val_indices=0.0,
        coarse_eeg_artifact_mask="none",
        coarse_eeg_artifact_k=None,
        coarse_eeg_max_rel_delta=None,
    )
    np.testing.assert_array_equal(_seq(default), _seq(explicit))
    assert default.coarse_reports[0]["artifact"] == {}
    assert np.isfinite(_seq(default)[2:5, _col("coarse_eeg", "rel_alpha")]).all()


def test_artifact_rule_removes_high_power_and_logs(tmp_path, _torch_required, capsys):
    epochs, hrv = make_frames(80, "subj_a", seed=18)
    epochs["abs_total_0.5-30"] = 1000.0 * (1.0 + 0.01 * np.sin(np.arange(80)))
    epochs.loc[2:4, "abs_total_0.5-30"] = 1e8
    epochs.loc[2:4, "rel_alpha"] = 0.95
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        seq_len=8,
        val_indices=0.0,
        coarse_eeg_artifact_mask="log_total_power_robust",
        coarse_eeg_artifact_k=8,
    )
    alpha = _seq(ds)[:, _col("coarse_eeg", "rel_alpha")]
    assert np.isnan(alpha[2:5]).all()
    assert np.isfinite(alpha[0]) and np.isfinite(alpha[6])
    stats = ds.coarse_reports[0]["artifact"]["log_total_power_robust"]
    assert stats["removed"] == 3
    assert stats["k"] == pytest.approx(8)
    assert stats["threshold"] < 4.0
    out = capsys.readouterr().out
    assert "artifact log_total_power_robust removed 3/" in out
    assert "usable EEG epochs" in out


def test_artifact_threshold_ignores_provisional_tail(tmp_path, _torch_required):
    """Tail power must not set the threshold, or log10==6 would not be flagged."""
    n = 50
    epochs, hrv = make_frames(n, "subj_a", seed=19)
    epochs["rel_alpha"] = np.nan
    epochs.loc[0, "rel_alpha"] = 0.2
    epochs.loc[1, "rel_alpha"] = 0.4
    epochs.loc[0, "rel_delta"] = 0.2
    epochs.loc[1, "rel_delta"] = 0.5
    epochs.loc[0, "rel_theta"] = 0.2
    epochs.loc[1, "rel_theta"] = 0.5
    epochs.loc[0, "rel_beta"] = 0.2
    epochs.loc[1, "rel_beta"] = 0.5
    epochs.loc[40:, "rel_alpha"] = 0.9
    epochs.loc[40:, "rel_delta"] = 0.3
    epochs.loc[40:, "rel_theta"] = 0.2
    epochs.loc[40:, "rel_beta"] = 0.2
    epochs["abs_total_0.5-30"] = 1e6
    epochs.loc[0:1, "abs_total_0.5-30"] = 1000.0
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        seq_len=4,
        val_indices=0.0,
        coarse_eeg_artifact_mask="log_total_power_robust",
        coarse_eeg_artifact_k=1,
    )
    stats = ds.coarse_reports[0]["artifact"]["log_total_power_robust"]
    assert stats["n_fit"] == 2
    assert stats["median"] == pytest.approx(3.0)
    assert stats["threshold"] == pytest.approx(3.0)
    assert stats["removed"] == 10
    alpha = _seq(ds)[:, _col("coarse_eeg", "rel_alpha")]
    assert np.isfinite(alpha[0]) and np.isfinite(alpha[1])
    assert np.isnan(alpha[40:]).all()
    # Masked 0.9s are not in the z-score. Mean of the two clean values is 0.3.
    assert ds.coarse_reports[0]["norm_mean"]["rel_alpha"] == pytest.approx(0.3)


def test_artifact_stats_exclude_masked_epochs(tmp_path, _torch_required):
    n = 80
    epochs, hrv = make_frames(n, "subj_a", seed=20)
    phase = np.arange(n)
    epochs["rel_alpha"] = 0.30 + 0.02 * np.sin(2 * np.pi * phase / 10.0)
    epochs["abs_delta"] = 10 ** (1.2 + 0.05 * np.sin(2 * np.pi * phase / 7.0))
    epochs["abs_total_0.5-30"] = 10 ** (3.0 + 0.01 * np.sin(2 * np.pi * phase / 9.0))
    artifact_at = np.array([5, 6, 7, 8])
    epochs.loc[artifact_at, "abs_total_0.5-30"] = 1e8
    epochs.loc[artifact_at, "abs_delta"] = 1e12
    epochs.loc[artifact_at, "rel_alpha"] = 0.95
    write_session(tmp_path, "subj_a", epochs, hrv)
    off = _load(
        tmp_path,
        "subj_a",
        "coarse_full",
        seq_len=8,
        val_indices=0.0,
        coarse_eeg_artifact_mask="none",
    )
    on = _load(
        tmp_path,
        "subj_a",
        "coarse_full",
        seq_len=8,
        val_indices=0.0,
        coarse_eeg_artifact_mask="log_total_power_robust",
        coarse_eeg_artifact_k=6,
    )
    assert np.isnan(_seq(on)[artifact_at, _col("coarse_full", "rel_alpha")]).all()
    cut = on.coarse_reports[0]["test_cut"]
    off_cut = off.coarse_reports[0]["test_cut"]
    assert artifact_at.max() < cut
    raw = epochs.sort_values("t_start")["rel_alpha"].to_numpy()
    raw_masked = raw.copy()
    raw_masked[artifact_at] = np.nan
    expected = np.nanmean(raw_masked[:cut])
    contaminated = np.nanmean(raw[:off_cut])
    assert on.coarse_reports[0]["norm_mean"]["rel_alpha"] == pytest.approx(expected)
    assert contaminated != pytest.approx(expected)
    assert off.coarse_reports[0]["norm_mean"]["rel_alpha"] == pytest.approx(contaminated)
    assert on.coarse_reports[0]["clip_hi"]["abs_delta"] < 4.0
    assert off.coarse_reports[0]["clip_hi"]["abs_delta"] > 6.0
    # k * scaled MAD is the threshold definition (not a hard-coded cutoff).
    fit_log = np.log10(epochs.sort_values("t_start")["abs_total_0.5-30"].to_numpy())
    # Provisional non-test is the leading n - n//5 rows, including artifacts.
    provisional = fit_log[: n - n // 5]
    med = np.median(provisional)
    scale = COARSE_MAD_TO_SIGMA * np.median(np.abs(provisional - med))
    assert on.coarse_reports[0]["artifact"]["log_total_power_robust"][
        "threshold"
    ] == pytest.approx(med + 6 * scale)


def test_artifact_missing_k_and_unknown_rule_raise(tmp_path, _torch_required):
    epochs, hrv = make_frames(20, "subj_a", seed=21)
    write_session(tmp_path, "subj_a", epochs, hrv)
    with pytest.raises(ValueError, match="coarse_eeg_artifact_k"):
        _load(
            tmp_path,
            "subj_a",
            "coarse_eeg",
            coarse_eeg_artifact_mask="log_total_power_robust",
        )
    with pytest.raises(ValueError, match="coarse_eeg_artifact_mask"):
        _load(
            tmp_path,
            "subj_a",
            "coarse_eeg",
            coarse_eeg_artifact_mask="not_a_rule",
            coarse_eeg_artifact_k=3,
        )


def test_max_rel_delta_caps_without_requiring_k(tmp_path, _torch_required, capsys):
    epochs, hrv = make_frames(40, "subj_a", seed=22)
    epochs.loc[8:11, "rel_delta"] = 0.95
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        seq_len=8,
        val_indices=0.0,
        coarse_test_split="last20_rows",
        coarse_eeg_max_rel_delta=0.5,
    )
    delta = _seq(ds)[:, _col("coarse_eeg", "rel_delta")]
    assert np.isnan(delta[8:12]).all()
    assert np.isfinite(delta[0])
    stats = ds.coarse_reports[0]["artifact"]["max_rel_delta"]
    assert stats["removed"] == 4
    assert stats["cap"] == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "artifact max_rel_delta removed 4/" in out


def test_coarse_config_keys_reach_dataloader(tmp_path, _torch_required):
    from dvae.dataset.dataset_builder import DatasetConfig, build_dataloader

    for label in ("subj_a", "subj_b"):
        epochs, hrv = make_frames(40, label, seed=23)
        write_session(tmp_path, label, epochs, hrv)
    cfg = DatasetConfig(
        data_dir=str(tmp_path),
        x_dim=4,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="coarse_eeg",
        overlap=False,
        with_nan=True,
        seq_len=8,
        device="cpu",
        dataset_label="subj_a,subj_b",
        mask_label=None,
        corpus="multi",
        coarse_test_split="usable_tail",
        coarse_eeg_artifact_mask="none",
    )
    train_dl, _val_dl, n_train, _n_val = build_dataloader("XhroProper", cfg, "train")
    assert n_train > 0
    assert train_dl.dataset.dataset_labels == ["subj_a", "subj_b"]
    assert train_dl.dataset.coarse_test_split == "usable_tail"
    assert train_dl.dataset.sampling_freq == pytest.approx(1.0 / 30.0)
    _assert_windows_stay_inside(train_dl.dataset)

    parser = configparser.ConfigParser()
    parser.read_string(
        """
        [DataFrame]
        coarse_test_split = last20_rows
        coarse_eeg_artifact_mask = log_total_power_robust
        coarse_eeg_artifact_k = 6.5
        coarse_eeg_max_rel_delta = 0.9
        blank_k =
        """
    )
    got = coarse_dataframe_kwargs(parser)
    assert got["coarse_test_split"] == "last20_rows"
    assert got["coarse_eeg_artifact_mask"] == "log_total_power_robust"
    assert got["coarse_eeg_artifact_k"] == pytest.approx(6.5)
    assert got["coarse_eeg_max_rel_delta"] == pytest.approx(0.9)
    assert coarse_dataframe_kwargs(configparser.ConfigParser())["coarse_eeg_artifact_k"] is None

    template = Path(__file__).resolve().parents[1] / "config/general_signal/cfg_base_template.ini"
    text = template.read_text()
    for key in (
        "coarse_test_split",
        "coarse_eeg_artifact_mask",
        "coarse_eeg_artifact_k",
        "coarse_eeg_max_rel_delta",
        "coarse_include_hr",
        "coarse_tod_from_clock",
    ):
        assert key in text


def _legacy_global_split(starts, val_indices, shuffle, split):
    chosen = np.array(list(starts), copy=True)
    if shuffle and chosen.size:
        rng = np.random.RandomState(COARSE_WINDOW_RNG_SEED)
        rng.shuffle(chosen)
    split_point = int(len(chosen) * (1.0 - float(val_indices)))
    if split == "train":
        return [int(i) for i in chosen[:split_point]]
    return [int(i) for i in chosen[split_point:]]


def _window_starts(tmp_path, label, seq_len=8, val_indices=0.2, shuffle=True, **overrides):
    epochs, hrv = None, None  # callers write sessions first
    del epochs, hrv
    train = _load(
        tmp_path,
        label,
        "coarse_eeg",
        seq_len=seq_len,
        overlap=False,
        shuffle=False,
        val_indices=0.0,
        **overrides,
    )
    return list(train.data_idx), train


def test_assign_shuffled_coarse_split_default_matches_global_slice():
    starts = np.arange(20)
    subjects = np.array([f"s{i}" for i in starts])
    for split in ("train", "valid"):
        got = assign_shuffled_coarse_split(
            starts, subjects, 0.2, split, val_split_per_subject=False
        )
        split_point = int(20 * 0.8)
        expected = starts[:split_point] if split == "train" else starts[split_point:]
        assert np.array_equal(got, expected)


def test_assign_shuffled_coarse_split_gives_each_subject_a_val_window():
    # Ten already-shuffled starts. Subject A has two windows at the front,
    # which a global 80/20 cut (split_point=8) would keep entirely in train.
    starts = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17])
    subject_per_row = np.empty(18, dtype=object)
    subject_per_row[0:2] = "A"
    subject_per_row[10:18] = "B"
    train = assign_shuffled_coarse_split(
        starts, subject_per_row, 0.2, "train", val_split_per_subject=True
    )
    valid = assign_shuffled_coarse_split(
        starts, subject_per_row, 0.2, "valid", val_split_per_subject=True
    )
    label = {int(i): str(subject_per_row[int(i)]) for i in starts}
    assert sorted(int(i) for i in train) + sorted(int(i) for i in valid)
    assert set(int(i) for i in train).isdisjoint(set(int(i) for i in valid))
    assert set(int(i) for i in train) | set(int(i) for i in valid) == set(int(i) for i in starts)
    for subj in ("A", "B"):
        n = sum(1 for i in starts if label[int(i)] == subj)
        n_val = sum(1 for i in valid if label[int(i)] == subj)
        n_train = sum(1 for i in train if label[int(i)] == subj)
        assert n_train + n_val == n
        if n >= 2:
            assert n_val >= 1


def test_assign_shuffled_coarse_split_forces_one_val_when_ratio_would_take_none():
    starts = np.array([0, 1])
    subject_per_row = np.array(["A", "A"])
    valid = assign_shuffled_coarse_split(
        starts, subject_per_row, 0.0, "valid", val_split_per_subject=True
    )
    train = assign_shuffled_coarse_split(
        starts, subject_per_row, 0.0, "train", val_split_per_subject=True
    )
    assert list(valid) == [1]
    assert list(train) == [0]


def test_existing_configs_keep_assignment_when_flag_unset(tmp_path, _torch_required):
    """A bcd-style waveform INI and a single-subject coarse config.

    With val_split_per_subject unset or false, train/val window assignment
    matches the historical global shuffle-then-slice (coarse) or the
    unchanged waveform path (bcd).
    """
    from dvae.dataset.xhro_proper_dataset import optional_config_bool
    import subprocess
    import sys
    import configparser

    root = Path(__file__).resolve().parents[1]
    gen = root / "bin" / "generate_config_file_for_multiple.py"
    out = tmp_path / "bcd_inis"
    out.mkdir()
    subprocess.run(
        [sys.executable, str(gen), "--rows", "bcd", "--output-dir", str(out)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    bcd_files = sorted(out.glob("cfg_*.ini"))
    assert bcd_files
    bcd = configparser.ConfigParser()
    bcd.read(bcd_files[0])
    assert optional_config_bool(bcd, "DataFrame", "val_split_per_subject", False) is False
    assert bcd["DataFrame"].get("val_split_per_subject", "").strip() == ""

    # Packet-loss waveform session, same split knobs a bcd INI uses except
    # a short synthetic seq_len the fixture can support.
    from tests.test_xhro_proper import _fake_session

    wave_root = tmp_path / "bcd_wave"
    wave_root.mkdir()
    _fake_session(wave_root)
    wave_kwargs = dict(
        data_dir=str(wave_root),
        dataset_label="REC1",
        mask_label="realtime",
        seq_len=8,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=float(bcd["DataFrame"]["val_indices"]),
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=True,
        contiguous_val_split=optional_config_bool(
            bcd, "DataFrame", "contiguous_val_split", False
        ),
    )
    for split in ("train", "valid"):
        omitted = XhroProper(split=split, **wave_kwargs)
        explicit = XhroProper(split=split, val_split_per_subject=False, **wave_kwargs)
        assert omitted.data_idx == explicit.data_idx

    # Single-subject coarse config: omitted == False == legacy formula.
    epochs, hrv = make_frames(80, "subj_a", seed=11)
    write_session(tmp_path, "subj_a", epochs, hrv)
    starts, _base = _window_starts(tmp_path, "subj_a", seq_len=8, val_indices=0.2, shuffle=True)
    assert starts
    for split in ("train", "valid"):
        omitted = _load(
            tmp_path,
            "subj_a",
            "coarse_eeg",
            split=split,
            seq_len=8,
            overlap=False,
            shuffle=True,
            val_indices=0.2,
        )
        explicit = _load(
            tmp_path,
            "subj_a",
            "coarse_eeg",
            split=split,
            seq_len=8,
            overlap=False,
            shuffle=True,
            val_indices=0.2,
            val_split_per_subject=False,
        )
        legacy = _legacy_global_split(starts, 0.2, True, split)
        assert omitted.data_idx == explicit.data_idx == legacy


def test_val_split_per_subject_covers_every_subject(tmp_path, _torch_required):
    labels = ["subj_a", "subj_b", "subj_c", "subj_d", "subj_e"]
    for i, label in enumerate(labels):
        epochs, hrv = make_frames(50, label, seed=20 + i)
        write_session(tmp_path, label, epochs, hrv)
    joined = ",".join(labels)
    train = _load(
        tmp_path,
        joined,
        "coarse_eeg",
        split="train",
        seq_len=8,
        overlap=False,
        shuffle=True,
        val_indices=0.2,
        val_split_per_subject=True,
    )
    valid = _load(
        tmp_path,
        joined,
        "coarse_eeg",
        split="valid",
        seq_len=8,
        overlap=False,
        shuffle=True,
        val_indices=0.2,
        val_split_per_subject=True,
    )
    off_train = _load(
        tmp_path,
        joined,
        "coarse_eeg",
        split="train",
        seq_len=8,
        overlap=False,
        shuffle=True,
        val_indices=0.2,
        val_split_per_subject=False,
    )
    assert set(train.data_idx).isdisjoint(valid.data_idx)
    pooled, _ = _window_starts(tmp_path, joined, seq_len=8)
    assert set(train.data_idx) | set(valid.data_idx) == set(pooled)
    for label in labels:
        n = sum(1 for s in train.window_subject if s == label) + sum(
            1 for s in valid.window_subject if s == label
        )
        n_val = sum(1 for s in valid.window_subject if s == label)
        assert n >= 2
        assert n_val >= 1
    # Enabling the flag changes the pooled assignment (otherwise the
    # i.2 gap would not need a new option).
    assert train.data_idx != off_train.data_idx


def test_config_generator_coarse_example_is_separate():
    path = Path(__file__).resolve().parents[1] / "bin/generate_config_file_for_multiple.py"
    spec = importlib.util.spec_from_file_location("gen_multi_coarse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    example = module.xhro_proper_coarse_example()
    assert example["dataset_label"] == ["xhro_04_XH058,xhro_01_XH032"]
    assert example["observation_process"] == ["coarse_eeg_acc_tod"]
    assert example["x_dim"] == [7]
    assert "hr_mean_bpm" not in COARSE_OBS_COLUMNS["coarse_eeg_acc_tod"]
    assert example["coarse_eeg_artifact_mask"] == ["none"]
    assert example["coarse_test_split"] == ["usable_tail"]


def test_last20_rows_uses_ceil_tail_when_n_not_divisible_by_five():
    usable = np.ones(41, dtype=bool)
    # Xhro's ``-n // 5`` tail: 41 -> 9 rows, start 32. Floor ``n - n // 5`` is 33.
    assert coarse_test_cut(usable, "last20_rows") == 32
    assert 41 - coarse_test_cut(usable, "last20_rows") == 9
    provisional = _provisional_nontest_mask(41)
    assert int((~provisional).sum()) == 41 // 5
    assert int(provisional.sum()) == 41 - 41 // 5


def test_last20_rows_dataset_tail_when_n_not_divisible_by_five(tmp_path, _torch_required):
    epochs, hrv = make_frames(41, "subj_a", seed=41)
    write_session(tmp_path, "subj_a", epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg",
        split="test",
        seq_len=4,
        coarse_test_split="last20_rows",
    )
    assert ds.coarse_reports[0]["test_cut"] == 32
    assert _seq(ds).shape[0] == 9


def test_indicate_zeros_nonfinite_values():
    features = np.array([[1.0, 2.0], [np.inf, 3.0], [np.nan, -np.inf]], dtype=np.float64)
    out = _apply_coarse_suffix(features, "indicate")
    assert out.shape == (3, 4)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(3.0)
    assert out[2, 0] == pytest.approx(0.0)
    assert out[2, 2] == pytest.approx(0.0)
    assert np.isfinite(out).all()


def test_multi_subject_request_longer_than_tail_still_windows(tmp_path, _torch_required):
    """Two 40-epoch subjects have 8-row test tails. A 16-row request must cap."""
    for label in ("subj_a", "subj_b"):
        epochs, hrv = make_frames(40, label, seed=16)
        write_session(tmp_path, label, epochs, hrv)
    ds = _load(
        tmp_path,
        "subj_a,subj_b",
        "coarse_eeg",
        split="test",
        seq_len=16,
        overlap=False,
        shuffle=False,
        coarse_test_split="last20_rows",
    )
    assert _seq(ds).shape[0] == 16  # pooled tails
    assert ds.seq_len == 8
    assert cap_eval_seq_len(ds, 16) == 8
    assert cap_eval_seq_len(ds, 1000) == 8
    assert apply_eval_seq_len(ds, 16) == 8
    assert len(ds.data_idx) > 0
    _assert_windows_stay_inside(ds)
    if HAS_TORCH:
        from torch.utils.data import DataLoader

        batch = next(iter(DataLoader(ds, batch_size=1)))
        assert tuple(batch.shape)[1] == 8


_TOD_OMEGA = 2.0 * np.pi / 86400.0


def _phase_steps(sin, cos):
    phase = np.arctan2(np.asarray(sin, dtype=np.float64), np.asarray(cos, dtype=np.float64))
    return np.angle(np.exp(1j * np.diff(phase)))


def _clock_export_with_gap(n=48, gap=(12, 28), offset=np.pi - 1e-3):
    """True 30 s clock, with a freeze-then-jump gap in the export.

    The first half of the gap holds the pre-gap phase. The second half
    holds the post-gap phase. ``offset`` sits just below pi so the true
    phase crosses the branch cut inside the series.
    """
    t = np.arange(n, dtype=np.float64) * 30.0
    phase = _TOD_OMEGA * t + float(offset)
    gap_lo, gap_hi = gap
    mid = (gap_lo + gap_hi) // 2
    exported = phase.copy()
    exported[gap_lo:mid] = phase[gap_lo - 1]
    exported[mid:gap_hi] = phase[gap_hi]
    return {
        "t": t,
        "phase": phase,
        "sin": np.sin(exported),
        "cos": np.cos(exported),
        "gap": np.arange(gap_lo, gap_hi),
        "outside": np.r_[np.arange(0, gap_lo), np.arange(gap_hi, n)],
    }


def test_rebuild_tod_smooths_freeze_jump_gap():
    clock = _clock_export_with_gap()
    t0 = pd.Timestamp("2026-06-22 18:00:00")
    sin_n, cos_n, info = rebuild_tod_from_clock(
        clock["t"], clock["sin"], clock["cos"], t0=t0
    )
    steps = _phase_steps(sin_n, cos_n)
    assert np.allclose(steps, _TOD_OMEGA * 30.0, atol=1e-9)
    outside = clock["outside"]
    assert np.max(np.abs(sin_n[outside] - clock["sin"][outside])) < 1e-9
    assert np.max(np.abs(cos_n[outside] - clock["cos"][outside])) < 1e-9
    gap = clock["gap"]
    assert np.max(np.abs(sin_n[gap] - clock["sin"][gap])) > 1e-3
    assert np.allclose(sin_n**2 + cos_n**2, 1.0, atol=1e-12)
    assert info["source"] == "fit"
    assert info["n_rows"] == len(clock["t"])
    assert info["n_fit_rows"] == len(clock["outside"])
    assert info["n_rows_changed"] == len(gap)
    assert info["max_dev_outside_gaps_rad"] < 1e-9
    assert "t0_offset_rad" in info
    assert info["fit_minus_t0_rad"] == pytest.approx(
        float(np.angle(np.exp(1j * (info["offset_rad"] - info["t0_offset_rad"]))))
    )


def test_rebuild_tod_t0_fallback_when_fewer_than_two_normal_rows():
    t = np.arange(6, dtype=np.float64) * 30.0
    # Constant phase: no pair steps by omega*30.
    sin = np.zeros(len(t))
    cos = np.ones(len(t))
    sin[-1] = np.nan
    t0 = pd.Timestamp("2026-06-22 14:24:10.500000")
    hours = 14.0 + 24.0 / 60.0 + 10.0 / 3600.0 + 500000.0 / 3.6e9
    offset = 2.0 * np.pi * (hours / 24.0)
    sin_n, cos_n, info = rebuild_tod_from_clock(t, sin, cos, t0=t0)
    assert info["source"] == "t0"
    assert info["n_fit_rows"] < 2
    assert info["offset_rad"] == pytest.approx(offset)
    assert info["t0_offset_rad"] == pytest.approx(offset)
    assert np.isnan(info["fit_minus_t0_rad"])
    assert np.isnan(info["max_dev_outside_gaps_rad"])
    phase = np.arctan2(sin_n, cos_n)
    expected = np.angle(np.exp(1j * (_TOD_OMEGA * t + offset)))
    assert np.allclose(phase, expected, atol=1e-9)
    assert np.allclose(sin_n**2 + cos_n**2, 1.0, atol=1e-12)
    # 06:00 is a quarter turn, including through the previously missing row.
    noon = rebuild_tod_from_clock(t[:1], [0.0], [1.0], t0=pd.Timestamp("2026-01-01 06:00:00"))
    assert noon[2]["source"] == "t0"
    assert noon[2]["offset_rad"] == pytest.approx(np.pi / 2.0)
    assert noon[0][0] == pytest.approx(1.0)
    assert noon[1][0] == pytest.approx(0.0, abs=1e-12)


def test_rebuild_tod_export_fallback_without_t0():
    t = np.arange(4, dtype=np.float64) * 30.0
    sin = np.array([0.1, 0.2, np.nan, -0.3])
    cos = np.array([0.9, 0.8, 0.1, 0.4])
    sin_n, cos_n, info = rebuild_tod_from_clock(t, sin, cos, t0=None)
    assert info["source"] == "export"
    assert np.isnan(info["offset_rad"])
    assert info["n_fit_rows"] < 2
    assert np.allclose(sin_n, sin, equal_nan=True)
    assert np.allclose(cos_n, cos, equal_nan=True)
    assert info["n_rows_changed"] == 0
    assert "t0_offset_rad" not in info
    assert "fit_minus_t0_rad" not in info


def test_coarse_tod_from_clock_config_blank_is_true():
    assert coarse_dataframe_kwargs(configparser.ConfigParser())["coarse_tod_from_clock"] is True
    blank = configparser.ConfigParser()
    blank.read_string("[DataFrame]\ncoarse_tod_from_clock =\n")
    assert coarse_dataframe_kwargs(blank)["coarse_tod_from_clock"] is True
    explicit = configparser.ConfigParser()
    explicit.read_string("[DataFrame]\ncoarse_tod_from_clock = true\n")
    assert coarse_dataframe_kwargs(explicit)["coarse_tod_from_clock"] is True
    off = configparser.ConfigParser()
    off.read_string("[DataFrame]\ncoarse_tod_from_clock = false\n")
    assert coarse_dataframe_kwargs(off)["coarse_tod_from_clock"] is False
    template = (
        Path(__file__).resolve().parents[1] / "config/general_signal/cfg_base_template.ini"
    ).read_text()
    assert "coarse_tod_from_clock = {coarse_tod_from_clock}" in template


def _subject_with_tod_gap(tmp_path, label="subj_a"):
    clock = _clock_export_with_gap()
    epochs, hrv = make_frames(len(clock["t"]), label, seed=3)
    epochs["t_start"] = clock["t"]
    epochs["t_end"] = clock["t"] + 30.0
    epochs["tod_sin"] = clock["sin"]
    epochs["tod_cos"] = clock["cos"]
    session = write_session(tmp_path, label, epochs, hrv)
    return session, clock


def test_build_coarse_subject_tod_flag_off_keeps_export(tmp_path):
    session, clock = _subject_with_tod_gap(tmp_path)
    columns = list(COARSE_OBS_COLUMNS["coarse_eeg_acc_tod"])
    common = dict(
        test_split_mode="last20_rows",
        artifact_mask="none",
        artifact_k=None,
        max_rel_delta=None,
        subject_label="subj_a",
    )
    on = build_coarse_subject(session, columns, tod_from_clock=True, **common)
    off = build_coarse_subject(session, columns, tod_from_clock=False, **common)
    sin_i = columns.index("tod_sin")
    cos_i = columns.index("tod_cos")
    on_sin = on["features"][:, sin_i]
    on_cos = on["features"][:, cos_i]
    off_sin = off["features"][:, sin_i]
    off_cos = off["features"][:, cos_i]
    assert on["report"]["tod_clock"]["source"] == "fit"
    assert "tod_clock" not in off["report"]
    steps = _phase_steps(on_sin, on_cos)
    assert np.allclose(steps, _TOD_OMEGA * 30.0, atol=1e-9)
    outside = clock["outside"]
    assert np.max(np.abs(on_sin[outside] - clock["sin"][outside])) < 1e-9
    assert np.max(np.abs(on_cos[outside] - clock["cos"][outside])) < 1e-9
    assert np.allclose(on_sin**2 + on_cos**2, 1.0, atol=1e-12)
    assert np.allclose(off_sin, clock["sin"])
    assert np.allclose(off_cos, clock["cos"])
    assert np.max(np.abs(on_sin[clock["gap"]] - off_sin[clock["gap"]])) > 1e-3


def test_coarse_tod_flag_reaches_loader_and_logs(tmp_path, _torch_required, capsys):
    session, clock = _subject_with_tod_gap(tmp_path)
    del session
    import warnings

    capsys.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        on = _load(
            tmp_path,
            "subj_a",
            "coarse_eeg_acc_tod",
            seq_len=8,
            val_indices=0.0,
            coarse_test_split="last20_rows",
        )
    assert on.coarse_tod_from_clock is True
    assert on.coarse_reports[0]["tod_clock"]["source"] == "fit"
    logged = capsys.readouterr().out
    report = on.coarse_reports[0]["tod_clock"]
    assert "tod_clock source=fit" in logged
    assert f"offset_rad={report['offset_rad']:.3g}" in logged
    assert f"max_dev_outside_gaps_rad={report['max_dev_outside_gaps_rad']:.3g}" in logged
    assert f"n_rows_changed={report['n_rows_changed']}/{report['n_rows']}" in logged
    assert not any("tod_clock" in str(item.message) for item in caught)

    capsys.readouterr()
    off = _load(
        tmp_path,
        "subj_a",
        "coarse_eeg_acc_tod",
        seq_len=8,
        val_indices=0.0,
        coarse_test_split="last20_rows",
        coarse_tod_from_clock=False,
    )
    assert off.coarse_tod_from_clock is False
    assert "tod_clock" not in off.coarse_reports[0]
    assert "tod_clock" not in capsys.readouterr().out
    sin_i = _col("coarse_eeg_acc_tod", "tod_sin")
    # Train head still contains the gap. Off keeps the frozen export.
    gap_in_head = clock["gap"][clock["gap"] < on.coarse_reports[0]["test_cut"]]
    assert np.max(np.abs(_seq(on)[gap_in_head, sin_i] - _seq(off)[gap_in_head, sin_i])) > 1e-3


def test_nonclock_tod_export_warns(tmp_path, _torch_required, capsys):
    epochs, hrv = make_frames(40, "subj_a", seed=1)
    write_session(tmp_path, "subj_a", epochs, hrv)
    capsys.readouterr()
    with pytest.warns(UserWarning, match="tod_clock source=export"):
        ds = _load(
            tmp_path,
            "subj_a",
            "coarse_eeg_acc_tod",
            seq_len=8,
            val_indices=0.0,
            coarse_test_split="last20_rows",
        )
    info = ds.coarse_reports[0]["tod_clock"]
    assert info["source"] == "export"
    assert info["n_rows_changed"] == 0
    logged = capsys.readouterr().out
    assert "WARNING" in logged
    assert "tod_clock source=export" in logged
    # Raw export is not a clock, and the fallback leaves it unscaled.
    sin_i = _col("coarse_eeg_acc_tod", "tod_sin")
    raw = np.sin(2.0 * np.pi * np.arange(40) / 40.0)
    cut = ds.coarse_reports[0]["test_cut"]
    assert np.allclose(_seq(ds)[:5, sin_i], raw[:5])
    assert cut > 5


def test_t0_snapshot_fallback_fills_a_frozen_clock(tmp_path):
    epochs, hrv = make_frames(12, "subj_a", seed=4)
    epochs["tod_sin"] = 0.0
    epochs["tod_cos"] = 1.0
    epochs.loc[3, "tod_sin"] = np.nan
    epochs.loc[3, "tod_cos"] = np.nan
    session = write_session(tmp_path, "subj_a", epochs, hrv)
    (session / "config_snapshot.json").write_text('{"t0": "2026-06-22T06:00:00"}')
    columns = list(COARSE_OBS_COLUMNS["coarse_eeg_acc_tod"])
    built = build_coarse_subject(
        session,
        columns,
        test_split_mode="last20_rows",
        artifact_mask="none",
        artifact_k=None,
        max_rel_delta=None,
        subject_label="subj_a",
        tod_from_clock=True,
    )
    info = built["report"]["tod_clock"]
    assert info["source"] == "t0"
    assert info["n_fit_rows"] < 2
    assert info["offset_rad"] == pytest.approx(np.pi / 2.0)
    sin_i = columns.index("tod_sin")
    cos_i = columns.index("tod_cos")
    t = built["t_start"]
    phase = np.arctan2(built["features"][:, sin_i], built["features"][:, cos_i])
    expected = np.angle(np.exp(1j * (_TOD_OMEGA * t + np.pi / 2.0)))
    assert np.allclose(phase, expected, atol=1e-9)
    assert np.isfinite(built["features"][:, sin_i]).all()
    assert np.allclose(
        built["features"][:, sin_i] ** 2 + built["features"][:, cos_i] ** 2,
        1.0,
        atol=1e-12,
    )
