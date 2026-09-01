#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load grok-pipeline (xhro-analysis-grok) intermediates for DVAE / OTF.

This is **not** a drop-in for ``Xhro`` / ``XhroPacketLoss`` parquet files.
Those read ``filtered_data.parquet`` (publication or packet-loss port).
``XhroProper`` reads ``grok_output/<session>/intermediates/*.npz`` written by
named-bank, no-fill stages.

Comparison protocol (same recording):
  * Use **parity** observation_process names (``raw_ch1``, ``raw_all``, …)
    when comparing against ``Xhro`` / ``XhroPacketLoss``.
  * Keep seq_len, val_indices, and the last-20% test split the same.
  * Treat filter recipe as a factor (0.05 Hz / possible gap-zero leak vs
    ``ecg_v1`` / ``eeg_rest_v1``, never filter across holes).
  * Do not concatenate the two corpora into one tensor.
  * ``bipolar_*`` is a montage ablation, not a substitute for ``raw_ecg``.

Time axis: prefers ``datetime_ns`` on the NPZ (UTC ns + ``tz``). If missing
(old runs), reconstructs from ``config_snapshot.json`` / inventory ``t_start``
plus relative ``t``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .xhro_dataset import (
    Xhro,
    _resolve_original_observation_process,
    select_columns_for_obs_conditions,
)

VALID_VARIANTS = ("realtime", "retrans")
VARIANT_ALIASES = {"recovered": "retrans"}

# Parity names: same strings as Xhro, so configs can swap dataset_name only.
PARITY_PROCESSES = set(select_columns_for_obs_conditions["original"].keys())

# Grok-only bases (plus _interpolate / _indicate).
GROK_EXTRA_COLUMNS = {
    "bipolar_ecg": ["ecg"],
    "bipolar_eeg": ["eeg"],
    "bipolar_both": ["ecg", "eeg"],
    "ppg_ch1": ["ppg_ch1"],
    "ppg_ch2": ["ppg_ch2"],
    "ppg_ch3": ["ppg_ch3"],
    "ppg_ch4": ["ppg_ch4"],
    "ppg_primary": ["ppg_primary"],
    "ppg_all": ["ppg_ch1", "ppg_ch2", "ppg_ch3", "ppg_ch4"],
    "acc_xyz": ["acc_x", "acc_y", "acc_z"],
    "acc_motion": ["acc_motion"],
}

GROK_EXTRA_PROCESSES = set(GROK_EXTRA_COLUMNS.keys())


def _resolve_variant(mask_label: str | None) -> str:
    if mask_label in (None, "None", ""):
        return "realtime"
    key = str(mask_label)
    key = VARIANT_ALIASES.get(key, key)
    if key not in VALID_VARIANTS:
        raise ValueError(
            f"mask_label must be one of {VALID_VARIANTS} "
            f"(alias recovered→retrans), got: {mask_label!r}"
        )
    return key


def _split_process(name: str) -> tuple[str, str | None]:
    if name.endswith("_interpolate"):
        return name[: -len("_interpolate")], "interpolate"
    if name.endswith("_indicate"):
        return name[: -len("_indicate")], "indicate"
    return name, None


def _session_dir(path_to_data: str, recording_id: str, variant: str) -> Path:
    return (
        Path(path_to_data)
        / "xhro_packet_loss"
        / "grok_output"
        / f"{recording_id}_{variant}"
    )


def _tz_name(raw) -> str:
    if raw is None:
        return ""
    if hasattr(raw, "item"):
        try:
            return str(raw.item())
        except Exception:
            pass
    return str(raw)


def datetime_from_npz(
    npz: np.lib.npyio.NpzFile,
    *,
    t0: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    t = np.asarray(npz["t"], dtype=np.float64)
    if "datetime_ns" in npz.files:
        dt = pd.to_datetime(np.asarray(npz["datetime_ns"]), utc=True, unit="ns")
        tz = _tz_name(npz["tz"]) if "tz" in npz.files else ""
        if tz:
            dt = dt.tz_convert(tz)
        return pd.DatetimeIndex(dt)
    if "datetime" in npz.files:
        return pd.DatetimeIndex(pd.to_datetime(npz["datetime"]))
    if t0 is None:
        raise ValueError(
            "NPZ has no datetime/datetime_ns and no t0 to reconstruct from."
        )
    t0 = pd.Timestamp(t0)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("Asia/Tokyo")
    return pd.DatetimeIndex(t0 + pd.to_timedelta(t, unit="s"))


def _infer_t0(session_dir: Path) -> pd.Timestamp | None:
    snap = session_dir / "config_snapshot.json"
    if snap.exists():
        try:
            cfg = json.loads(snap.read_text())
            for key in ("t0", "session_t0", "datetime_start"):
                if key in cfg and cfg[key]:
                    return pd.Timestamp(cfg[key])
        except Exception:
            pass
    inv = session_dir / "00_inventory" / "metrics.json"
    if inv.exists():
        try:
            met = json.loads(inv.read_text())
            for key in ("t_start", "t0", "biop_t_start"):
                if key in met and met[key]:
                    return pd.Timestamp(met[key])
            streams = met.get("streams") or {}
            biop = streams.get("biop") or {}
            if biop.get("t_start"):
                return pd.Timestamp(biop["t_start"])
        except Exception:
            pass
    return None


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Missing grok intermediate: {path}")
    return np.load(path, allow_pickle=False)

def _series_from_stage05(npz) -> dict[str, np.ndarray]:
    return {
        "ch1": np.asarray(npz["ch1"], dtype=np.float64),
        "ch2": np.asarray(npz["ch2"], dtype=np.float64),
        "ecg": np.asarray(npz["ecg"], dtype=np.float64),
    }


def _series_from_stage06(npz) -> dict[str, np.ndarray]:
    return {
        "ch3": np.asarray(npz["ch3"], dtype=np.float64),
        "ch4": np.asarray(npz["ch4"], dtype=np.float64),
        "eeg": np.asarray(npz["eeg"], dtype=np.float64),
    }


def _series_from_stage07(npz) -> dict[str, np.ndarray]:
    out = {}
    for i in range(1, 5):
        key = f"filt_ch{i}"
        if key in npz.files:
            out[f"ppg_ch{i}"] = np.asarray(npz[key], dtype=np.float64)
    primary = "ch1"
    if "primary" in npz.files:
        raw_p = npz["primary"]
        primary = str(raw_p.item() if hasattr(raw_p, "item") else raw_p)
    src = f"filt_{primary}" if f"filt_{primary}" in npz.files else "filt_ch1"
    if src in npz.files:
        out["ppg_primary"] = np.asarray(npz[src], dtype=np.float64)
    return out


def _series_from_stage08(npz) -> dict[str, np.ndarray]:
    out = {}
    for src, dst in (("x", "acc_x"), ("y", "acc_y"), ("z", "acc_z")):
        if src in npz.files:
            out[dst] = np.asarray(npz[src], dtype=np.float64)
        elif f"{src}_hp" in npz.files:
            out[dst] = np.asarray(npz[f"{src}_hp"], dtype=np.float64)
    if "motion_energy" in npz.files:
        out["acc_motion"] = np.asarray(npz["motion_energy"], dtype=np.float64)
    elif all(k in out for k in ("acc_x", "acc_y", "acc_z")):
        out["acc_motion"] = np.sqrt(
            out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2
        )
    return out


def _df_from_npz(
    npz,
    columns: dict[str, np.ndarray],
    *,
    t0: pd.Timestamp | None,
) -> pd.DataFrame:
    dt = datetime_from_npz(npz, t0=t0)
    data = {"datetime": dt}
    n = len(dt)
    for name, arr in columns.items():
        a = np.asarray(arr, dtype=np.float64)
        if len(a) != n:
            raise ValueError(
                f"Column {name!r} length {len(a)} != datetime length {n}"
            )
        data[name] = a
    return pd.DataFrame(data)


def _align_on_datetime(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(left, right, on="datetime", how="outer", sort=True)


def _maybe_resample(df: pd.DataFrame, resample_hz: float | None) -> pd.DataFrame:
    if resample_hz is None:
        return df
    if float(resample_hz) <= 0:
        raise ValueError(f"resample_hz must be > 0, got {resample_hz!r}")
    if "datetime" not in df.columns or df.empty:
        return df
    period_ns = int(round(1e9 / float(resample_hz)))
    out = df.set_index("datetime").sort_index()
    # asof onto a regular grid; holes stay NaN (no interpolation here)
    t0, t1 = out.index[0], out.index[-1]
    grid = pd.date_range(t0, t1, freq=pd.Timedelta(nanoseconds=period_ns), tz=out.index.tz)
    aligned = out.reindex(grid, method=None)
    aligned.index.name = "datetime"
    return aligned.reset_index()


def load_proper_frame(
    session_dir: Path,
    base_process: str,
    *,
    resample_hz: float | None = None,
) -> pd.DataFrame:
    """Build a datetime-indexed table for one observation-process base name."""
    inter = session_dir / "intermediates"
    t0 = _infer_t0(session_dir)

    def load_stage(name: str):
        return _load_npz(inter / name)

    need05 = base_process in {
        "raw_ch1",
        "raw_ch2",
        "raw_ecg",
        "bipolar_ecg",
        "raw_all",
        "bipolar_both",
    }
    need06 = base_process in {
        "raw_ch3",
        "raw_ch4",
        "raw_eeg",
        "bipolar_eeg",
        "raw_all",
        "bipolar_both",
    }
    if need05 or need06:
        frames = []
        if need05:
            z = load_stage("stage05_ecg_filtered.npz")
            frames.append(_df_from_npz(z, _series_from_stage05(z), t0=t0))
        if need06:
            z = load_stage("stage06_eeg_filtered.npz")
            frames.append(_df_from_npz(z, _series_from_stage06(z), t0=t0))
        df = frames[0] if len(frames) == 1 else _align_on_datetime(frames[0], frames[1])
        return _maybe_resample(df, resample_hz)

    if base_process.startswith("ppg_"):
        z = load_stage("stage07_ppg_filtered.npz")
        df = _df_from_npz(z, _series_from_stage07(z), t0=t0)
        return _maybe_resample(df, resample_hz)

    if base_process.startswith("acc_"):
        z = load_stage("stage08_acc.npz")
        df = _df_from_npz(z, _series_from_stage08(z), t0=t0)
        return _maybe_resample(df, resample_hz)

    raise ValueError(f"Unknown XhroProper observation base: {base_process!r}")


def _columns_for_base(base: str) -> list[str]:
    if base in PARITY_PROCESSES:
        return list(select_columns_for_obs_conditions["original"][base])
    if base in GROK_EXTRA_COLUMNS:
        return list(GROK_EXTRA_COLUMNS[base])
    raise ValueError(f"Unknown observation process base: {base!r}")


class XhroProper(Xhro):
    """Grok-pipeline XHRO sequences with named banks and honest gaps.

    Config mapping (same shape as XhroPacketLoss):
      - dataset_label: recording id (e.g. XHRO3506_20260622T142410000+0900)
      - mask_label: ``realtime`` or ``retrans`` (``recovered`` aliases retrans)
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
        resample_hz=None,
        **kwargs,
    ):
        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
        self.variant = _resolve_variant(mask_label)
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
        self.sampling_freq = None
        self.resample_hz = resample_hz
        self.session_dir = _session_dir(
            self.path_to_data, self.dataset_label, self.variant
        )

        base, _suffix = _split_process(self.observation_process)
        if base not in PARITY_PROCESSES and base not in GROK_EXTRA_PROCESSES:
            raise ValueError(
                f"Invalid observation process: {self.observation_process!r}. "
                f"Parity: {sorted(PARITY_PROCESSES)}; "
                f"grok extras: {sorted(GROK_EXTRA_PROCESSES)}; "
                "each may take _interpolate or _indicate."
            )

        the_sequence = load_proper_frame(
            self.session_dir, base, resample_hz=self.resample_hz
        )
        if "datetime" in the_sequence.columns:
            # native rate from median Δt
            deltas = the_sequence["datetime"].diff().dt.total_seconds().dropna()
            if len(deltas):
                med = float(deltas.median())
                self.sampling_freq = (1.0 / med) if med > 0 else None
        if self.sampling_freq is None:
            self.sampling_freq = 250.0
        print(
            f"[XhroProper][{self.variant}] {self.observation_process} "
            f"from {self.session_dir}  rows={len(the_sequence)}  "
            f"fs≈{self.sampling_freq:.4g} Hz"
        )

        if self.split == "test":
            the_sequence = the_sequence.iloc[-len(the_sequence) // 5 :].reset_index(
                drop=True
            )
        else:
            the_sequence = the_sequence.iloc[: -len(the_sequence) // 5].reset_index(
                drop=True
            )

        self.full_sequence = the_sequence
        self.missing_mask = self._extract_missing_mask(the_sequence)
        the_sequence = self.apply_observation_process(the_sequence)
        the_sequence = the_sequence.squeeze()

        if self.x_dim is None:
            if the_sequence.ndim == 1:
                self.x_dim = 1
            elif the_sequence.ndim == 2:
                self.x_dim = the_sequence.shape[1]
            else:
                raise ValueError(
                    f"Expected x is {the_sequence.ndim} dimensions, got x_dim {self.x_dim}."
                )

        self.is_segmented_1d = False
        if the_sequence.ndim == 1:
            if self.x_dim > 1:
                self.is_segmented_1d = True
            if self.overlap:
                the_sequence = self.create_moving_window_sequences(
                    the_sequence, self.x_dim
                )
            else:
                the_sequence = np.array(
                    [
                        the_sequence[i : i + x_dim]
                        for i in range(0, len(the_sequence), x_dim)
                        if i + x_dim <= len(the_sequence)
                    ]
                )
        elif the_sequence.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x_dim={self.x_dim}, got {the_sequence.shape[1]} "
                f"for process {self.observation_process!r}."
            )

        self.seq = the_sequence
        self.update_sequence_length(self.seq_len)

    def _value_frame(self, sequence: pd.DataFrame) -> pd.DataFrame:
        base, _ = _split_process(self.observation_process)
        cols = _columns_for_base(base)
        missing = [c for c in cols if c not in sequence.columns]
        if missing:
            raise KeyError(
                f"{self.observation_process}: columns {missing} not in loaded frame "
                f"{list(sequence.columns)}"
            )
        return sequence[cols]

    def _extract_missing_mask(self, sequence):
        base, _ = _split_process(self.observation_process)
        if base in PARITY_PROCESSES:
            # Reuse Xhro column names on the constructed frame.
            return super()._extract_missing_mask(sequence)
        values = self._value_frame(sequence).to_numpy(dtype=np.float64)
        return np.isnan(values)

    def apply_observation_process(self, sequence) -> torch.Tensor:
        base, suffix = _split_process(self.observation_process)
        if base in PARITY_PROCESSES:
            return super().apply_observation_process(sequence)

        data = self._value_frame(sequence)
        first_valid = data.first_valid_index()
        if first_valid is not None:
            data = data.loc[first_valid:]
        arr = data.to_numpy(dtype=np.float64)

        if suffix is None:
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            std = np.where((std == 0) | np.isnan(std), 1.0, std)
            normed = (arr - mean) / std
            return torch.tensor(normed.astype(np.float32), dtype=torch.float32)
        if suffix == "interpolate":
            out = arr.copy()
            for j in range(out.shape[1]):
                x = out[:, j]
                nan_mask = np.isnan(x)
                if nan_mask.any():
                    idx = np.arange(len(x))
                    valid = ~nan_mask
                    if valid.any():
                        x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
                    else:
                        x[:] = 0.0
                out[:, j] = x
            return torch.tensor(
                self.normalize(out.astype(np.float32)), dtype=torch.float32
            )

        # indicate: all channels + one mask (row observed iff all finite)
        missing_row = np.isnan(arr).any(axis=1)
        normed = self.normalize(arr.astype(np.float32))
        imputed = np.nan_to_num(normed, nan=0.0).astype(np.float32)
        is_obs = (~missing_row).astype(np.float32)[:, None]
        result = np.concatenate([imputed, is_obs], axis=1)
        print(
            f"[XhroProper][{self.observation_process}] missing_rows={int(missing_row.sum())} "
            f"shape={result.shape}"
        )
        return torch.tensor(result, dtype=torch.float32)
