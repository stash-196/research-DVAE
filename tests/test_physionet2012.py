"""PhysioNet2012 hourly parquet: native NaNs, indicate, registry, dataloaders."""

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

from dvae.dataset.physionet2012_dataset import (  # noqa: E402
    HOURS_PER_STAY,
    PHYSINET_OBS_COLUMNS,
    VITAL_COLUMNS,
    PhysioNet2012,
    intra_stay_window_starts,
    parse_challenge_record,
    parse_hhmm_to_hour,
    partition_stay_ids,
    processed_parquet_path,
    records_to_hourly_frame,
)


def _write_record(path: Path, stay_id: str, samples: list[tuple[str, str, float]]) -> None:
    path.write_text(
        "Time,Parameter,Value\n"
        f"00:00,RecordID,{stay_id}\n"
        "00:00,Age,60\n"
        + "".join(f"{t},{p},{v}\n" for t, p, v in samples)
    )


def _write_processed(root: Path, n_stays: int = 10, hours: int = HOURS_PER_STAY) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_stays):
        for h in range(hours):
            row = {"stay_id": f"{132539 + s}", "hour": h}
            for col in VITAL_COLUMNS:
                val = rng.normal()
                if rng.random() < 0.25:
                    val = np.nan
                row[col] = val
            rows.append(row)
    frame = pd.DataFrame(rows)
    out = processed_parquet_path(root, "hourly_v1")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    return out


def _dataset_kwargs(root: Path, **overrides):
    params = dict(
        data_dir=str(root),
        dataset_label="hourly_v1",
        mask_label="None",
        split="train",
        seq_len=16,
        x_dim=5,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_vitals",
        device="cpu",
        overlap=False,
        shuffle=False,
    )
    params.update(overrides)
    return params


def test_parse_hhmm_to_hour():
    assert parse_hhmm_to_hour("00:07") == 0
    assert parse_hhmm_to_hour("01:00") == 1
    assert parse_hhmm_to_hour("47:59") == 47


def test_parse_record_keeps_empty_hours(tmp_path):
    rec = tmp_path / "132539.txt"
    _write_record(
        rec,
        "132539",
        [("00:07", "HR", 80.0), ("00:40", "HR", 82.0), ("02:00", "Temp", 36.5)],
    )
    grid = parse_challenge_record(rec)
    assert len(grid) == HOURS_PER_STAY
    assert grid.loc[grid["hour"] == 0, "HR"].iloc[0] == 82.0  # last in hour
    assert np.isnan(grid.loc[grid["hour"] == 1, "HR"].iloc[0])
    assert grid.loc[grid["hour"] == 2, "Temp"].iloc[0] == 36.5
    assert grid["HR"].isna().sum() == HOURS_PER_STAY - 1


def test_records_to_hourly_missing_raw(tmp_path):
    empty = tmp_path / "raw"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No Challenge 2012"):
        records_to_hourly_frame(empty)


def test_prepare_from_raw_txt(tmp_path):
    raw = tmp_path / "raw" / "set-a"
    raw.mkdir(parents=True)
    _write_record(raw / "1.txt", "1", [("00:00", "HR", 70.0)])
    _write_record(raw / "2.txt", "2", [("01:00", "SysABP", 110.0)])
    frame = records_to_hourly_frame(tmp_path / "raw")
    assert set(frame["stay_id"].astype(str)) == {"1", "2"}
    assert frame["HR"].isna().any()


def test_missing_processed_dir(tmp_path):
    if not HAS_TORCH:
        pytest.skip("torch required")
    with pytest.raises(FileNotFoundError, match="Processed PhysioNet 2012"):
        PhysioNet2012(**_dataset_kwargs(tmp_path))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_registry():
    from dvae.dataset.dataset_builder import DATASET_REGISTRY

    assert DATASET_REGISTRY["PhysioNet2012"] is PhysioNet2012


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_nans_preserved(tmp_path):
    _write_processed(tmp_path)
    ds = PhysioNet2012(**_dataset_kwargs(tmp_path))
    assert np.isnan(ds.seq).any()
    mask = ds.get_missing_mask(0)
    assert mask.dtype == bool
    assert mask.any()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_interpolate_fills_nans(tmp_path):
    _write_processed(tmp_path)
    ds = PhysioNet2012(
        **_dataset_kwargs(tmp_path, observation_process="raw_vitals_interpolate")
    )
    assert not np.isnan(np.asarray(ds.seq)).any()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_indicate_adds_mask_channel(tmp_path):
    _write_processed(tmp_path)
    ds = PhysioNet2012(
        **_dataset_kwargs(
            tmp_path, observation_process="raw_hr_indicate", x_dim=2, seq_len=16
        )
    )
    item = np.asarray(ds[0])
    assert item.shape[-1] == 2
    # observed flag is 0/1
    assert set(np.unique(item[:, 1])).issubset({0.0, 1.0})


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_build_dataloader_splits(tmp_path):
    _write_processed(tmp_path, n_stays=20)
    from dvae.dataset.dataset_builder import DatasetConfig, build_dataloader

    cfg = DatasetConfig(
        data_dir=str(tmp_path),
        x_dim=5,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.25,
        observation_process="raw_vitals",
        overlap=False,
        with_nan=True,
        seq_len=16,
        device="cpu",
        dataset_label="hourly_v1",
        mask_label="None",
    )
    train_dl, val_dl, n_train, n_val = build_dataloader("PhysioNet2012", cfg, "train")
    test_dl = build_dataloader("PhysioNet2012", cfg, "test")
    assert n_train > 0 and n_val > 0
    batch = next(iter(train_dl))
    assert batch.shape[-1] == 5
    test_batch = next(iter(test_dl))
    assert test_batch.shape[-1] == 5


def test_obs_column_map():
    assert PHYSINET_OBS_COLUMNS["raw_hr"] == ["HR"]
    assert PHYSINET_OBS_COLUMNS["raw_vitals"] == VITAL_COLUMNS


def test_row_concat_would_cut_mid_stay_and_span_patients():
    """The old concat+iloc+find_indices protocol mixed stays (11×48 hours)."""
    from dvae.dataset.utils.data_utils import find_indices

    n_stays, hours = 11, HOURS_PER_STAY
    n_rows = n_stays * hours
    cut = n_rows - n_rows // 5
    assert cut % hours != 0
    stay_ids = np.repeat([f"s{i}" for i in range(n_stays)], hours)
    starts = find_indices(n_rows, hours, max(1, n_rows // hours))
    mixed = sum(len(set(stay_ids[s : s + hours])) > 1 for s in starts)
    assert mixed > 0


def test_partition_stay_ids_are_disjoint():
    ids = [f"s{i}" for i in range(11)]
    train, val, test = partition_stay_ids(ids, val_frac=0.2)
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    assert set(train) | set(val) | set(test) == set(ids)
    assert test == ids[-2:]  # 11 // 5 == 2


def test_intra_stay_windows_never_mix_stays():
    stay_ids = np.repeat(["a", "b", "c"], HOURS_PER_STAY)
    starts = intra_stay_window_starts(stay_ids, seq_len=16, overlap=False)
    assert starts.size > 0
    for start in starts:
        window = stay_ids[start : start + 16]
        assert len(set(window)) == 1
    # Start 40 is hours 40–55: last 8 of stay a plus first 8 of stay b.
    assert 40 not in set(starts.tolist())
    full_stay = intra_stay_window_starts(stay_ids, seq_len=HOURS_PER_STAY, overlap=False)
    assert set(full_stay.tolist()) == {0, HOURS_PER_STAY, 2 * HOURS_PER_STAY}


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_splits_are_stay_disjoint_and_windows_are_intra_stay(tmp_path):
    _write_processed(tmp_path, n_stays=11)
    kwargs = _dataset_kwargs(tmp_path, val_indices=0.2, shuffle=False)
    train = PhysioNet2012(**kwargs)
    valid = PhysioNet2012(**{**kwargs, "split": "valid"})
    test = PhysioNet2012(**{**kwargs, "split": "test"})
    train_stays = set(train.stay_id_per_row)
    valid_stays = set(valid.stay_id_per_row)
    test_stays = set(test.stay_id_per_row)
    assert train_stays and valid_stays and test_stays
    assert train_stays.isdisjoint(valid_stays)
    assert train_stays.isdisjoint(test_stays)
    assert valid_stays.isdisjoint(test_stays)
    for ds in (train, valid, test):
        assert len(ds) > 0
        for i in range(len(ds)):
            start = ds.data_idx[i]
            window_ids = ds.stay_id_per_row[start : start + ds.seq_len]
            assert len(np.unique(window_ids)) == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_eval_seq_len_is_capped_to_stay_length(tmp_path):
    _write_processed(tmp_path, n_stays=10)
    ds = PhysioNet2012(**_dataset_kwargs(tmp_path, seq_len=16, shuffle=False))
    ds.update_sequence_length(1000)
    assert ds.seq_len == HOURS_PER_STAY
    for i in range(len(ds)):
        start = ds.data_idx[i]
        window_ids = ds.stay_id_per_row[start : start + ds.seq_len]
        assert len(np.unique(window_ids)) == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed in this env")
def test_interpolate_does_not_cross_stays(tmp_path):
    hours = HOURS_PER_STAY
    rows = []
    for s, fill in enumerate((1.0, 100.0)):
        for h in range(hours):
            row = {"stay_id": f"{1000 + s}", "hour": h}
            for col in VITAL_COLUMNS:
                row[col] = fill
            if s == 0 and h == hours - 1:
                row["HR"] = np.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    out = processed_parquet_path(tmp_path, "hourly_v1")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    ds = PhysioNet2012(
        **_dataset_kwargs(
            tmp_path,
            observation_process="raw_vitals_interpolate",
            seq_len=hours,
            val_indices=0.0,
            shuffle=False,
        )
    )
    # With 2 stays, last 20% is the second stay; train is the first only.
    assert {str(s) for s in ds.stay_id_per_row} == {"1000"}
    item = np.asarray(ds[0])
    assert item.shape[0] == hours
    # Constant 1.0 stay (NaN at last hour filled from the same stay) stays
    # internally consistent; cross-stay interp toward 100 would shift the tail.
    assert abs(item[-1, 0] - item[0, 0]) < 1e-5


def test_eval_channel_keys_and_dt():
    from dvae.eval.utils.benchmark_signals import _get_dt, resolve_channel_keys

    keys = resolve_channel_keys("raw_vitals", 5, "PhysioNet2012")
    assert [k for k, _ in keys] == VITAL_COLUMNS
    class _Dummy:
        sampling_freq = 1.0 / 3600.0
    assert abs(_get_dt(_Dummy(), "PhysioNet2012") - 3600.0) < 1e-6

