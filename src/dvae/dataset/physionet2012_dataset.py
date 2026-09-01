#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PhysioNet Challenge 2012 (ICU vitals) as an Xhro-shaped Dataset.

Raw Challenge records are irregular CSVs. ``bin/prepare_physionet2012.py``
bins them onto an hourly grid **without imputing** (empty hours stay NaN).
This class reads the processed parquet.

Download (open, no login):
  https://physionet.org/content/challenge-2012/1.0.0/

On-disk layout::

    {data_dir}/physionet2012/processed/{dataset_label}/stays.parquet

``dataset_label`` defaults to ``hourly_v1``. Cite Silva et al., CinC 2012.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HOURS_PER_STAY = 48
DEFAULT_LABEL = "hourly_v1"
# 1 sample / hour. eval dt = 1 / sampling_freq = 3600 s.
SAMPLING_FREQ_HZ = 1.0 / 3600.0

VITAL_COLUMNS = ["HR", "SysABP", "DiasABP", "RespRate", "Temp"]
STATIC_PARAMETERS = frozenset(
    {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}
)

PHYSINET_OBS_COLUMNS = {
    "raw_hr": ["HR"],
    "raw_vitals": list(VITAL_COLUMNS),
}


def unique_stay_ids_in_order(stay_ids) -> list:
    """Contiguous run labels, preserving first-seen order."""
    ordered = []
    for sid in stay_ids:
        if not ordered or ordered[-1] != sid:
            ordered.append(sid)
    return ordered


def partition_stay_ids(unique_ids: list, *, val_frac: float = 0.2):
    """Last 20% of stays → test; remaining stays split train/valid.

    Splits are by stay, never by concatenated hour, so a record cannot land
    in two splits and a cut cannot fall mid-stay.
    """
    ids = list(unique_ids)
    n = len(ids)
    n_test = n // 5
    if n >= 2 and n_test == 0:
        n_test = 1
    test_ids = ids[n - n_test :] if n_test else []
    pool = ids[: n - n_test] if n_test else ids
    n_val = int(len(pool) * val_frac) if pool else 0
    if val_frac > 0 and n_val == 0 and len(pool) > 1:
        n_val = 1
    val_ids = pool[len(pool) - n_val :] if n_val else []
    train_ids = pool[: len(pool) - n_val] if n_val else list(pool)
    return train_ids, val_ids, test_ids


def intra_stay_window_starts(
    stay_id_per_row: np.ndarray, seq_len: int, overlap: bool
) -> np.ndarray:
    """Window starts that lie entirely inside one stay."""
    stay_id_per_row = np.asarray(stay_id_per_row)
    starts = []
    n = len(stay_id_per_row)
    i = 0
    while i < n:
        sid = stay_id_per_row[i]
        j = i + 1
        while j < n and stay_id_per_row[j] == sid:
            j += 1
        if j - i >= seq_len:
            step = 1 if overlap else seq_len
            starts.extend(range(i, j - seq_len + 1, step))
        i = j
    return np.asarray(starts, dtype=np.int64)


def _resolve_obs_base(observation_process: str) -> tuple[str, str | None]:
    suffix = None
    base = observation_process
    if observation_process.endswith("_interpolate"):
        base = observation_process[: -len("_interpolate")]
        suffix = "interpolate"
    elif observation_process.endswith("_indicate"):
        base = observation_process[: -len("_indicate")]
        suffix = "indicate"
    if base not in PHYSINET_OBS_COLUMNS:
        raise ValueError(
            f"Invalid observation process: {observation_process!r}. "
            f"Must be one of {sorted(PHYSINET_OBS_COLUMNS)} "
            "or those names plus _interpolate / _indicate."
        )
    return base, suffix


def parse_hhmm_to_hour(time_str: str) -> int | None:
    """Challenge time is HH:MM from ICU admission. Return hour bin 0..47."""
    time_str = time_str.strip()
    if ":" not in time_str:
        return None
    hours_s, mins_s = time_str.split(":", 1)
    try:
        hours = int(hours_s)
        int(mins_s)
    except ValueError:
        return None
    if hours < 0:
        return None
    return min(hours, HOURS_PER_STAY - 1)


def parse_challenge_record(path: Path) -> pd.DataFrame:
    """Parse one Challenge 2012 CSV into a 48-row hourly frame (NaNs kept)."""
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty record: {path}")
        header_l = [h.strip() for h in header]
        # v1.0.0: Time,Parameter,Value  — older dump: RecordID,Parameter,Time,Value
        if header_l[:3] == ["Time", "Parameter", "Value"]:
            time_i, param_i, value_i = 0, 1, 2
        elif "Time" in header_l and "Parameter" in header_l and "Value" in header_l:
            time_i = header_l.index("Time")
            param_i = header_l.index("Parameter")
            value_i = header_l.index("Value")
        else:
            raise ValueError(f"Unrecognized header {header_l} in {path}")

        stay_id = path.stem
        for parts in reader:
            if len(parts) <= max(time_i, param_i, value_i):
                continue
            param = parts[param_i].strip()
            if param in STATIC_PARAMETERS:
                if param == "RecordID":
                    stay_id = parts[value_i].strip()
                continue
            if param not in VITAL_COLUMNS:
                continue
            hour = parse_hhmm_to_hour(parts[time_i])
            if hour is None:
                continue
            try:
                value = float(parts[value_i])
            except ValueError:
                continue
            rows.append((hour, param, value))

    grid = pd.DataFrame(
        {
            "stay_id": stay_id,
            "hour": np.arange(HOURS_PER_STAY, dtype=np.int32),
        }
    )
    for col in VITAL_COLUMNS:
        grid[col] = np.nan
    if rows:
        # Last observation in each hour bin wins; empty bins stay NaN.
        tmp = pd.DataFrame(rows, columns=["hour", "param", "value"])
        last = tmp.groupby(["hour", "param"], sort=False).last().reset_index()
        pivoted = last.pivot(index="hour", columns="param", values="value")
        for col in VITAL_COLUMNS:
            if col in pivoted.columns:
                grid[col] = pivoted[col].reindex(grid["hour"]).to_numpy(dtype=np.float64)
    return grid


def collect_raw_txt_files(raw_dir: Path) -> list[Path]:
    files = []
    for sub in ("set-a", "set-b", "set-A", "set-B"):
        d = raw_dir / sub
        if d.is_dir():
            files.extend(sorted(d.glob("*.txt")))
    files.extend(sorted(p for p in raw_dir.glob("*.txt") if p.name != "Outcomes-a.txt"))
    # de-dupe while preserving order
    seen = set()
    unique = []
    for p in files:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def records_to_hourly_frame(raw_dir: Path) -> pd.DataFrame:
    files = collect_raw_txt_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No Challenge 2012 .txt records under {raw_dir}. "
            "Download set-a/set-b tarballs from "
            "https://physionet.org/content/challenge-2012/1.0.0/ "
            "and extract them into this directory."
        )
    frames = [parse_challenge_record(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def processed_parquet_path(data_dir: str | Path, dataset_label: str | None) -> Path:
    label = (
        DEFAULT_LABEL
        if dataset_label in (None, "None", "")
        else str(dataset_label)
    )
    return (
        Path(data_dir) / "physionet2012" / "processed" / label / "stays.parquet"
    )


def _interp_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    nan_mask = np.isnan(x)
    valid = ~nan_mask
    if nan_mask.any() and valid.any():
        idx = np.arange(x.shape[0])
        x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
    elif nan_mask.all():
        x[:] = 0.0
    return x


class PhysioNet2012(Dataset):
    """Hourly ICU vitals with native missingness. Same constructor as Xhro.

    Stays are split by ``stay_id`` (last 20% test; remaining train/valid).
    Sliding windows never cross a stay boundary, and interpolation is
    applied per stay so NaNs are not filled from another patient.
    """

    def __init__(
        self,
        data_dir,
        dataset_label,
        mask_label,
        split,
        seq_len,
        x_dim,
        sample_rate,
        skip_rate,
        val_indices,
        observation_process,
        device,
        overlap,
        shuffle=True,
        **kwargs,
    ):
        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
        self.x_dim = x_dim
        self.seq_len = seq_len
        self.split = split
        self.sample_rate = sample_rate
        self.skip_rate = skip_rate
        self.val_indices = val_indices
        self.observation_process = observation_process
        self.overlap = overlap
        self.shuffle = shuffle
        self.device = device
        self.sampling_freq = SAMPLING_FREQ_HZ

        parquet = processed_parquet_path(data_dir, dataset_label)
        if not parquet.is_file():
            raise FileNotFoundError(
                f"Processed PhysioNet 2012 parquet not found: {parquet}. "
                "Run bin/prepare_physionet2012.py after downloading the raw "
                "Challenge records."
            )
        the_sequence = pd.read_parquet(parquet)
        required = {"stay_id", "hour", *VITAL_COLUMNS}
        missing = required.difference(the_sequence.columns)
        if missing:
            raise ValueError(f"{parquet} missing columns {sorted(missing)}")
        the_sequence = the_sequence.sort_values(["stay_id", "hour"]).reset_index(
            drop=True
        )

        unique_ids = unique_stay_ids_in_order(the_sequence["stay_id"].tolist())
        train_ids, val_ids, test_ids = partition_stay_ids(
            unique_ids, val_frac=float(self.val_indices)
        )
        if self.split == "test":
            keep = test_ids
        elif self.split == "valid":
            keep = val_ids
        else:
            keep = train_ids
        if not keep:
            raise ValueError(
                f"No stays left for split={self.split!r} "
                f"(n_stays={len(unique_ids)}, val_indices={self.val_indices})."
            )
        keep_set = set(keep)
        the_sequence = the_sequence[
            the_sequence["stay_id"].isin(keep_set)
        ].reset_index(drop=True)

        self.full_sequence = the_sequence
        self.stay_id_per_row = the_sequence["stay_id"].to_numpy()
        self.missing_mask = self._extract_missing_mask(the_sequence)
        processed = self.apply_observation_process(the_sequence)
        if isinstance(processed, torch.Tensor):
            processed = processed.numpy()
        processed = np.asarray(processed)
        if processed.ndim == 1:
            processed = processed.reshape(-1, 1)

        if self.x_dim is None:
            self.x_dim = int(processed.shape[1])
        elif processed.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x_dim={self.x_dim}, got {processed.shape[1]} "
                f"for observation_process={self.observation_process!r}"
            )

        self.seq = processed
        self.update_sequence_length(self.seq_len)

    def _selected_frame(self, sequence: pd.DataFrame) -> pd.DataFrame:
        base, _ = _resolve_obs_base(self.observation_process)
        cols = PHYSINET_OBS_COLUMNS[base]
        return sequence[cols]

    def _extract_missing_mask(self, sequence: pd.DataFrame) -> np.ndarray:
        values = self._selected_frame(sequence).to_numpy(dtype=np.float64)
        return np.isnan(values)

    def apply_observation_process(self, sequence: pd.DataFrame) -> torch.Tensor:
        base, suffix = _resolve_obs_base(self.observation_process)
        cols = PHYSINET_OBS_COLUMNS[base]
        if suffix == "interpolate":
            parts = []
            n_nan_before = 0
            for _, grp in sequence.groupby("stay_id", sort=False):
                block = grp[cols].to_numpy(dtype=np.float64)
                n_nan_before += int(np.isnan(block).sum())
                parts.append(
                    np.stack(
                        [_interp_1d(block[:, i]) for i in range(block.shape[1])],
                        axis=1,
                    )
                )
            data = np.concatenate(parts, axis=0)
            n_nan_after = int(np.isnan(data).sum())
            print(
                f"[PhysioNet2012][{self.observation_process}] "
                f"NaNs before: {n_nan_before}, after: {n_nan_after}"
            )
            return torch.tensor(self.normalize(data), dtype=torch.float32)

        data = sequence[cols].to_numpy(dtype=np.float64)
        n_nan_before = int(np.isnan(data).sum())

        if suffix == "indicate":
            # Match Xhro: indicate flag on the first selected column only.
            x = data[:, 0].astype(np.float32)
            missing_mask = np.isnan(x)
            x_normalized = self.normalize(x.reshape(-1, 1))[:, 0]
            x_imputed = np.nan_to_num(x_normalized, nan=0.0).astype(np.float32)
            is_observed = (~missing_mask).astype(np.float32)
            result = np.stack([x_imputed, is_observed], axis=1)
            print(
                f"[PhysioNet2012][{self.observation_process}] "
                f"missing_count: {int(missing_mask.sum())}, result_shape: {result.shape}"
            )
            return torch.tensor(result, dtype=torch.float32)

        print(
            f"[PhysioNet2012][{self.observation_process}] "
            f"NaNs: {n_nan_before}, shape: {data.shape}"
        )
        return torch.tensor(self.normalize(data), dtype=torch.float32)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std

    def get_missing_mask(self, index):
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.missing_mask))
        return self.missing_mask[start_frame:end_frame]

    def split_dataset(self, indices, val_ratio):
        if self.shuffle:
            np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_ratio))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.seq))
        item = self.seq[start_frame:end_frame]
        return torch.as_tensor(item, dtype=torch.float32)

    def update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
        if new_seq_len is None:
            self.data_idx = [0]
            return
        stay_lens = np.diff(
            np.r_[
                0,
                1 + np.flatnonzero(self.stay_id_per_row[1:] != self.stay_id_per_row[:-1]),
                len(self.stay_id_per_row),
            ]
        )
        max_stay = int(stay_lens.max()) if len(stay_lens) else 0
        self.seq_len = min(int(new_seq_len), max_stay)
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1 after stay-length cap")
        all_indices = intra_stay_window_starts(
            self.stay_id_per_row, self.seq_len, overlap=bool(self.overlap)
        )
        if all_indices.size == 0:
            raise ValueError(
                f"No intra-stay windows of length {self.seq_len} "
                f"(max stay length {max_stay})."
            )
        if minimum_nan_ratio is not None:
            valid_frames = []
            for idx in all_indices:
                seq_slice = self.seq[idx : idx + self.seq_len]
                nan_ratio = np.float16(np.isnan(seq_slice)).mean()
                if nan_ratio <= minimum_nan_ratio:
                    valid_frames.append(idx)
            valid_frames = np.array(valid_frames, dtype=np.int64)
            if valid_frames.size == 0:
                raise ValueError(
                    f"No intra-stay windows of length {self.seq_len} "
                    f"with nan_ratio <= {minimum_nan_ratio}."
                )
        else:
            valid_frames = all_indices
        # Stay membership already selected this split; keep every intra-stay
        # window (do not re-split by val_indices).
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(valid_frames)
        self.data_idx = list(valid_frames)
        return
