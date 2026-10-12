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

Lost packets on the native grid (stages 05–09b, also emitted on stage 04):
optional keys ``present``, ``segment_id``, ``segment_bounds``, ``gap_indices``,
and the per-gap table ``gap_t_start`` / ``gap_n_missing`` / ``gap_duration_s`` /
``gap_filled`` / ``gap_kind`` / ``gap_index``. When they are absent the
waveform path is unchanged. ``present=False`` is an inserted NaN row and is
missing in the observation and the missing-data mask. Window starts stay
inside one ``segment_id`` run, the same way coarse windows stay inside one
subject. ``[DataFrame] max_fill_gap_sec`` (unset by default) additionally
splits filled gaps longer than that many seconds. Those split sample
indices stay on the dataset and are applied again whenever
``update_sequence_length`` rebuilds windows (curriculum resizes included).
``[DataFrame] max_gap_window_frac`` (default 0.25) splits a filled gap
whose sample count is longer than that fraction of the current sequence
length, and is recomputed on every re-cut. ``[DataFrame] start_on_observed``
(default true) moves a start that lands on ``present=False`` forward to
the next observed sample in the same segment. ``[DataFrame]
strict_non_overlap`` (default true) tiles each segment with abutting
windows. False keeps the historical ``find_indices`` placement, including
its local ``RandomState(1234567890)``. ``[DataFrame] max_window_nan_ratio``
(default 0.3) is the NaN-fraction filter on every re-cut.
``[DataFrame] legacy_windowing`` (default false) restores the previous
bundle when those four keys are blank: starts may land on a missing
sample, the historical picker is used, and neither the fraction split nor
the NaN filter is applied. An explicit value for one key still wins.
``[DataFrame] block_mask_file`` (unset) is an npz of block flags applied
before the time crop. ``block_mask_mode`` defaults to ``not_good``.
``filled_nan_log_sec`` is the re-cut log cutoff when ``max_fill_gap_sec``
is unset. Crop stays time-based.

Session directory
-----------------
``data_dir`` (``path_to_data``) is the dataset root. The folder that
*contains* session folders is chosen as follows. If neither ``data_root``
nor ``corpus`` is set, the path is unchanged::

    {data_dir}/xhro_packet_loss/grok_output/{dataset_label}_{variant}

``variant`` comes from ``mask_label``: ``realtime`` or ``retrans``
(``recovered`` aliases ``retrans``). That is ``corpus="packet_loss"``.

``corpus="multi"`` is the suntory Multi tree. ``dataset_label`` **is** the
session folder name (``xhro_01_XH015``). ``mask_label`` is ignored — do not
append a ``_realtime`` / ``_retrans`` suffix::

    {data_dir}/suntory/xhro_dataset_v2/grok_output/{dataset_label}

``data_root`` replaces that parent. An absolute path is used as given; a
relative path is joined onto ``data_dir``. Naming still follows ``corpus``
(variant suffix for packet-loss, none for multi). Point a non-default Multi
tree at its ``grok_output`` folder with ``data_root`` and ``corpus="multi"``.

On the cluster, ``corpus="multi"`` still resolves under the repo's ``data/``
unless ``data_root`` points at the ``grok_output`` folder that contains the
session directories. A missing session directory reports ``data_root``.

Optional ``[DataFrame]`` keys ``corpus`` and ``data_root`` (missing, blank,
or ``None`` = unset). Other datasets ignore them.

Coarse features (stage 15)
--------------------------
``observation_process`` selects one entry of ``COARSE_OBS_COLUMNS`` inside
this class (not a second dataset). Each name also accepts the
``_interpolate`` and ``_indicate`` suffixes. Waveform names are unchanged.

Per session, under the same directory ``corpus="multi"`` uses today
(``data_root`` still overrides that parent)::

    {session_dir}/15_coarse_features/coarse_epochs_30s.parquet
    {session_dir}/15_coarse_features/coarse_hrv_300s.parquet

``dataset_label`` may be a comma-separated list of those session folder
names. One name is the same as a one-element list.

Join. Rows stay on the 30 s grid (``sampling_freq = 1/30``). ``hr_mean_bpm``
comes from the 300 s window with ``t_start <= epoch.t_start < t_end``. HR
is NaN when ``window_ok`` is false or the rate is non-finite. SDNN, RMSSD,
pNN50, ``valid_frac``, and the 300 s ``gap_frac`` are not read.

EEG mask. A non-finite ``rel_alpha`` sets every EEG-derived column on that
epoch to NaN. ``epoch_eeg_good`` is ignored. Plain modes leave NaNs in
place.

Artifact mask (optional, default off). ``coarse_eeg_artifact_mask=none``
does nothing. ``log_total_power_robust`` NaNs EEG-derived columns where
``log10(abs_total_0.5-30)`` is above ``median + coarse_eeg_artifact_k *
(1.4826 * MAD)``. Median and MAD are fit on that subject's provisional
non-test EEG-usable epochs — the leading rows of the last-20%-of-rows cut —
because the final ``usable_tail`` cut is chosen only after this mask.
``coarse_eeg_max_rel_delta``, when set, also NaNs EEG columns where
``rel_delta`` exceeds it. ``k`` and the cap are config, never hard-coded;
turning the power rule on without ``k`` raises. Extra named rules register
in ``_ARTIFACT_RULE_FUNCS`` (return a boolean row mask and a details dict;
set ``requires_k = True`` if the rule needs ``coarse_eeg_artifact_k``).

Transforms and normalization are per subject, then applied to every split.
``acc_motion_rms`` is ``log1p``. ``abs_*`` and ``eeg_rms_mad`` are ``log10``
(non-positive becomes NaN) and upper-clipped at the non-test 99.5th
percentile. ``tod_sin`` / ``tod_cos`` are not normalized. Other selected
columns are z-scored with ``nanmean`` / ``nanstd`` (ddof 0) from that
subject's final non-test rows. Those statistics ignore NaNs, so plain,
interpolate, and indicate share one scale. If ``std < 1e-8``, the column
is set to NaN for that subject and a warning is logged.

Test split. Waveform modes keep the last 20% of rows. Coarse modes default
to ``coarse_test_split=usable_tail``: a contiguous tail holding 20%
(``n_usable // 5``) of epochs whose selected columns are all finite after
the masks and transforms. ``last20_rows`` is the waveform rule. Clip and
z-score statistics use that final non-test part. The power-rule fit is the
exception above (provisional last-20% rows).

No-EEG modes ``coarse_hr``, ``coarse_hr_acc``, and ``coarse_hr_acc_tod``
(each also ``_interpolate`` / ``_indicate``) select only HR, motion, and
time-of-day columns. The rel_alpha mask and the artifact rules are not
applied. Jointly usable epochs are those whose selected columns are all
finite, so missing EEG does not move the test cut.

``_interpolate`` fills NaNs inside the subject slice only, after z-scoring
(linear fill commutes with that affine map) and does not cross subjects or
the test cut. ``_indicate`` interleaves ``[x0, m0, x1, m1, ...]`` so
``x_dim`` doubles, matching ``Xhro`` and the config generator.

Windows. Starts are computed inside each subject, like
``intra_stay_window_starts``. Train/valid shuffles the pooled windows with
``np.random.RandomState(1234567890)`` and does not touch the global NumPy
RNG. The test split keeps every intra-subject window in the tail.
``window_subject`` / ``window_subject_id`` record the subject of each
window. ``missing_mask`` is the pre-suffix NaN mask of the selected
columns, indexed by the same flat ``data_idx`` as ``seq``.

ECG-removed EEG (stage 09b)
---------------------------
``ecg_removed_eeg`` (plus ``_interpolate`` / ``_indicate``) reads
``{session_dir}/intermediates/stage09b_ecg_remove.npz`` key
``eeg_clean_template``, on the same 250 Hz time axis as
``stage06_eeg_filtered.npz`` key ``eeg``. ``load_proper_frame`` builds that
frame the same way ``bipolar_eeg`` builds stage 06, and the suffixes follow
that grok-extra path (indicate appends one shared is_observed column). The
file exists for only some sessions; a missing file raises
``FileNotFoundError`` naming the path.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .gap_grid import (
    FILLED_RUN_COL,
    GROUP_COL,
    PRESENT_COL,
    _as_bool,
    _blank_absent_rows,
    attach_gap_grid,
    combine_gap_grid,
    gap_load_stats,
    groups_from_boundaries,
    log_window_recut,
    resolve_recut_length,
    split_boundaries_from_ids,
)
from .utils.data_utils import _boundary_cuts
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
    "ecg_removed_eeg": ["eeg_clean_template"],
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


# Parent of session folders, relative to path_to_data (data_dir).
# packet_loss is the historical default when neither data_root nor corpus is set.
CORPUS_GROK_RELATIVE = {
    "packet_loss": Path("xhro_packet_loss") / "grok_output",
    "multi": Path("suntory") / "xhro_dataset_v2" / "grok_output",
}
CORPUS_ALIASES = {
    "packet_loss": "packet_loss",
    "packet-loss": "packet_loss",
    "xhro_packet_loss": "packet_loss",
    "multi": "multi",
    "suntory": "multi",
}


def _is_unset(value) -> bool:
    """Missing, blank, and the config sentinel ``None`` mean unset."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", "None"):
        return True
    return False


def optional_config_str(cfg, section: str, option: str) -> str | None:
    """Read an optional ini string. Missing, blank, and ``None`` mean unset."""
    if not cfg.has_section(section):
        return None
    raw = cfg.get(section, option, fallback=None)
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    return text


def _normalize_corpus(corpus) -> str | None:
    if _is_unset(corpus):
        return None
    key = str(corpus).strip().lower()
    canonical = CORPUS_ALIASES.get(key)
    if canonical is None:
        known = ", ".join(sorted(CORPUS_GROK_RELATIVE))
        raise ValueError(
            f"Unknown XhroProper corpus {corpus!r}. "
            f"Expected one of: {known} "
            "(aliases: packet-loss→packet_loss, suntory→multi). "
            "Or set data_root to the directory that contains session folders."
        )
    return canonical


def _session_parent(path_to_data: str, data_root, corpus: str | None) -> Path:
    """Directory that contains session folders.

    An absolute ``data_root`` is used as-is. A relative ``data_root`` is
    joined onto ``path_to_data`` (pathlib drops the left side when the right
    side is absolute).
    """
    if not _is_unset(data_root):
        return Path(path_to_data) / str(data_root).strip()
    key = corpus or "packet_loss"
    return Path(path_to_data) / CORPUS_GROK_RELATIVE[key]


def _session_folder_name(
    recording_id: str, variant: str | None, corpus: str | None
) -> str:
    if _is_unset(recording_id):
        raise ValueError(
            "XhroProper dataset_label is required: recording id for "
            "packet_loss, or the session folder name for multi "
            "(e.g. xhro_01_XH015)."
        )
    label = str(recording_id).strip()
    # multi: dataset_label is the folder name. mask_label / variant is ignored.
    if corpus == "multi":
        return label
    if _is_unset(variant):
        raise ValueError(
            "XhroProper packet_loss sessions need mask_label "
            f"(realtime or retrans) to form '{label}_<variant>'."
        )
    return f"{label}_{variant}"


def _session_dir(
    path_to_data: str,
    recording_id: str,
    variant: str | None = None,
    *,
    data_root: str | None = None,
    corpus: str | None = None,
) -> Path:
    """Resolve the grok session directory. Does not check that it exists.

    Default (no ``data_root``, no ``corpus``)::

        {path_to_data}/xhro_packet_loss/grok_output/{recording_id}_{variant}

    ``corpus="multi"``::

        {path_to_data}/suntory/xhro_dataset_v2/grok_output/{recording_id}
    """
    normalized = _normalize_corpus(corpus)
    parent = _session_parent(path_to_data, data_root, normalized)
    name = _session_folder_name(recording_id, variant, normalized)
    return parent / name


def _require_session_dir(
    path_to_data: str,
    recording_id: str,
    variant: str | None = None,
    *,
    data_root: str | None = None,
    corpus: str | None = None,
) -> Path:
    """Resolve the session directory and raise if that path is not a directory."""
    path = _session_dir(
        path_to_data,
        recording_id,
        variant,
        data_root=data_root,
        corpus=corpus,
    )
    if not path.is_dir():
        effective = _normalize_corpus(corpus) or "packet_loss"
        shown_root = None if _is_unset(data_root) else str(data_root).strip()
        hint = ""
        if effective == "multi" and shown_root is None:
            hint = (
                " For corpus='multi' on the cluster, set data_root to the "
                "grok_output folder; otherwise this path is under the repo's data/."
            )
        raise FileNotFoundError(
            "XhroProper session directory not found: "
            f"{path} (corpus={effective!r}, data_root={shown_root!r}, "
            f"dataset_label={recording_id!r}).{hint}"
        )
    return path


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
    max_fill_gap_sec=None,
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
    return attach_gap_grid(pd.DataFrame(data), npz, max_fill_gap_sec)


def _align_on_datetime(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(left, right, on="datetime", how="outer", sort=True)
    return combine_gap_grid(merged)


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
    max_fill_gap_sec=None,
) -> pd.DataFrame:
    """Build a datetime-indexed table for one observation-process base name.

    ``max_fill_gap_sec`` unset leaves filled gaps inside windows. Gap-grid
    keys are read from the stage NPZ when present. ``resample_hz`` is still
    the unchanged exact reindex; segment indices are on the native grid.
    """
    inter = session_dir / "intermediates"
    t0 = _infer_t0(session_dir)

    def load_stage(name: str):
        return _load_npz(inter / name)

    def frame_from(npz, columns: dict[str, np.ndarray]) -> pd.DataFrame:
        return _df_from_npz(
            npz, columns, t0=t0, max_fill_gap_sec=max_fill_gap_sec
        )

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
            frames.append(frame_from(z, _series_from_stage05(z)))
        if need06:
            z = load_stage("stage06_eeg_filtered.npz")
            frames.append(frame_from(z, _series_from_stage06(z)))
        df = frames[0] if len(frames) == 1 else _align_on_datetime(frames[0], frames[1])
        return _maybe_resample(df, resample_hz)

    if base_process == "ecg_removed_eeg":
        path = inter / "stage09b_ecg_remove.npz"
        if not path.exists():
            raise FileNotFoundError(
                "Missing ECG-removed EEG intermediate: "
                f"{path}. ecg_removed_eeg reads key 'eeg_clean_template' "
                "(present for only some sessions)."
            )
        z = np.load(path, allow_pickle=False)
        if "eeg_clean_template" not in z.files:
            raise KeyError(
                f"{path} has no 'eeg_clean_template' array "
                f"(found {sorted(z.files)})."
            )
        df = frame_from(
            z,
            {
                "eeg_clean_template": np.asarray(
                    z["eeg_clean_template"], dtype=np.float64
                )
            },
        )
        return _maybe_resample(df, resample_hz)

    if base_process.startswith("ppg_"):
        z = load_stage("stage07_ppg_filtered.npz")
        df = frame_from(z, _series_from_stage07(z))
        return _maybe_resample(df, resample_hz)

    if base_process.startswith("acc_"):
        z = load_stage("stage08_acc.npz")
        df = frame_from(z, _series_from_stage08(z))
        return _maybe_resample(df, resample_hz)

    raise ValueError(f"Unknown XhroProper observation base: {base_process!r}")


def _columns_for_base(base: str) -> list[str]:
    if base in PARITY_PROCESSES:
        return list(select_columns_for_obs_conditions["original"][base])
    if base in GROK_EXTRA_COLUMNS:
        return list(GROK_EXTRA_COLUMNS[base])
    raise ValueError(f"Unknown observation process base: {base!r}")


# One map from observation_process (without suffix) to model columns.
# ``coarse_full`` adds log10 absolute band powers, IAF, and log10 EEG RMS.
# ``coarse_hr*`` modes select no EEG columns.
COARSE_OBS_COLUMNS = {
    "coarse_hr": ["hr_mean_bpm"],
    "coarse_hr_acc": ["hr_mean_bpm", "acc_motion_rms"],
    "coarse_hr_acc_tod": [
        "hr_mean_bpm",
        "acc_motion_rms",
        "tod_sin",
        "tod_cos",
    ],
    "coarse_eeg": ["rel_delta", "rel_theta", "rel_alpha", "rel_beta"],
    "coarse_eeg_hr": [
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "hr_mean_bpm",
    ],
    "coarse_eeg_hr_acc": [
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "hr_mean_bpm",
        "acc_motion_rms",
    ],
    "coarse_eeg_hr_acc_tod": [
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "hr_mean_bpm",
        "acc_motion_rms",
        "tod_sin",
        "tod_cos",
    ],
    "coarse_full": [
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "hr_mean_bpm",
        "acc_motion_rms",
        "tod_sin",
        "tod_cos",
        "abs_delta",
        "abs_theta",
        "abs_alpha",
        "abs_beta",
        "abs_total_0.5-30",
        "iaf_hz",
        "eeg_rms_mad",
    ],
}

# Blanked wherever rel_alpha is non-finite, and by the optional artifact rules.
# Modes whose selected columns miss this set skip both rules.
EEG_DERIVED_COLUMNS = (
    "rel_delta",
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "abs_delta",
    "abs_theta",
    "abs_alpha",
    "abs_beta",
    "abs_total_0.5-30",
    "iaf_hz",
    "eeg_rms_mad",
    "eeg_median",
    "eeg_iqr",
)
ABS_TOTAL_COLUMN = "abs_total_0.5-30"


def _columns_use_eeg(columns: list[str]) -> bool:
    eeg = set(EEG_DERIVED_COLUMNS)
    return any(col in eeg for col in columns)
COARSE_LOG10_COLUMNS = (
    "abs_delta",
    "abs_theta",
    "abs_alpha",
    "abs_beta",
    "abs_total_0.5-30",
    "eeg_rms_mad",
)
COARSE_LOG1P_COLUMNS = ("acc_motion_rms",)
COARSE_NO_NORM_COLUMNS = ("tod_sin", "tod_cos")
# 1/Φ^{-1}(3/4): MAD * this estimates a Gaussian sigma.
COARSE_MAD_TO_SIGMA = 1.4826
COARSE_LOG_CLIP_PERCENTILE = 99.5
COARSE_STD_EPS = 1e-8
COARSE_SAMPLING_FREQ = 1.0 / 30.0
# Fixed local RNG for the train/valid window shuffle. Never np.random.seed.
COARSE_WINDOW_RNG_SEED = 1234567890
_COARSE_TEST_SPLITS = ("usable_tail", "last20_rows")


def _parse_coarse_labels(dataset_label) -> list[str]:
    """Split a coarse ``dataset_label``. One name equals a one-element list."""
    if _is_unset(dataset_label):
        raise ValueError(
            "XhroProper coarse modes need dataset_label: one session folder "
            "name, or a comma-separated list (e.g. xhro_04_XH058,xhro_01_XH032)."
        )
    labels = []
    for part in str(dataset_label).split(","):
        text = part.strip()
        if text and text.lower() != "none":
            labels.append(text)
    if not labels:
        raise ValueError(
            f"XhroProper coarse dataset_label {dataset_label!r} has no session names."
        )
    return labels


def _normalize_coarse_test_split(value) -> str:
    if _is_unset(value):
        return "usable_tail"
    key = str(value).strip().lower()
    if key not in _COARSE_TEST_SPLITS:
        raise ValueError(
            "coarse_test_split must be one of "
            f"{_COARSE_TEST_SPLITS}, got {value!r}."
        )
    return key


def _optional_bool(value, name: str, default: bool = False) -> bool:
    """Blank and ``None`` take ``default``. ``true`` / ``false`` / ``off`` parse."""
    if _is_unset(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}.")


def optional_config_bool(cfg, section: str, option: str, default: bool = False) -> bool:
    """Optional ini bool. Missing, blank, and ``None`` take ``default``."""
    return _optional_bool(
        optional_config_str(cfg, section, option),
        f"[{section}] {option}",
        default,
    )


def _optional_nonnegative_float(value, name: str) -> float | None:
    out = _optional_user_float(value, name)
    if out is not None and out < 0:
        raise ValueError(f"{name} must be >= 0, got {out!r}.")
    return out


# Blank keys take these. ``legacy_windowing`` true makes a blank key mean off.
DEFAULT_MAX_WINDOW_NAN_RATIO = 0.3
DEFAULT_MAX_GAP_WINDOW_FRAC = 0.25
_WINDOW_DEFAULT = object()
_FLOAT_OFF = {"off", "unset", "false", "no"}


def _window_bool(value, name: str, default: bool) -> bool:
    if value is _WINDOW_DEFAULT or _is_unset(value):
        return default
    return _optional_bool(value, name, default=default)


def _window_float(value, name: str, default: float | None) -> float | None:
    """Blank uses ``default``. ``off`` is an explicit disable."""
    if value is _WINDOW_DEFAULT or _is_unset(value):
        return default
    if isinstance(value, str) and value.strip().lower() in _FLOAT_OFF:
        return None
    return _optional_nonnegative_float(value, name)


def resolve_windowing(
    legacy_windowing=_WINDOW_DEFAULT,
    start_on_observed=_WINDOW_DEFAULT,
    strict_non_overlap=_WINDOW_DEFAULT,
    max_window_nan_ratio=_WINDOW_DEFAULT,
    max_gap_window_frac=_WINDOW_DEFAULT,
) -> dict:
    """Defaults for the four window rules, and the legacy override.

    A blank key takes the new default. ``legacy_windowing`` true changes
    that blank-key default back to the historical one (starts may be
    missing, historical ``find_indices``, no fraction split, no NaN
    filter). A value that is actually set still wins, including ``off``
    on the two float keys.
    """
    legacy = _window_bool(legacy_windowing, "legacy_windowing", False)
    if legacy:
        start_default = False
        strict_default = False
        nan_default = None
        frac_default = None
    else:
        start_default = True
        strict_default = True
        nan_default = DEFAULT_MAX_WINDOW_NAN_RATIO
        frac_default = DEFAULT_MAX_GAP_WINDOW_FRAC
    return {
        "legacy_windowing": legacy,
        "start_on_observed": _window_bool(
            start_on_observed, "start_on_observed", start_default
        ),
        "strict_non_overlap": _window_bool(
            strict_non_overlap, "strict_non_overlap", strict_default
        ),
        "max_window_nan_ratio": _window_float(
            max_window_nan_ratio, "max_window_nan_ratio", nan_default
        ),
        "max_gap_window_frac": _window_float(
            max_gap_window_frac, "max_gap_window_frac", frac_default
        ),
    }


def windowing_from_config(cfg) -> dict:
    """Resolve the ``[DataFrame]`` window rules, including a blank legacy key."""
    return resolve_windowing(
        legacy_windowing=optional_config_str(cfg, "DataFrame", "legacy_windowing"),
        start_on_observed=optional_config_str(cfg, "DataFrame", "start_on_observed"),
        strict_non_overlap=optional_config_str(cfg, "DataFrame", "strict_non_overlap"),
        max_window_nan_ratio=optional_config_str(
            cfg, "DataFrame", "max_window_nan_ratio"
        ),
        max_gap_window_frac=optional_config_str(
            cfg, "DataFrame", "max_gap_window_frac"
        ),
    )


def _optional_user_float(value, name: str) -> float | None:
    if _is_unset(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return out


def optional_config_float(cfg, section: str, option: str) -> float | None:
    """Like ``optional_config_str``, then ``float``. Unset stays None."""
    text = optional_config_str(cfg, section, option)
    if text is None:
        return None
    return _optional_user_float(text, f"[{section}] {option}")


def coarse_dataframe_kwargs(cfg) -> dict:
    """Optional ``[DataFrame]`` keys for coarse XhroProper. Missing = defaults."""
    return {
        "coarse_test_split": optional_config_str(cfg, "DataFrame", "coarse_test_split"),
        "coarse_eeg_artifact_mask": optional_config_str(
            cfg, "DataFrame", "coarse_eeg_artifact_mask"
        ),
        "coarse_eeg_artifact_k": optional_config_float(
            cfg, "DataFrame", "coarse_eeg_artifact_k"
        ),
        "coarse_eeg_max_rel_delta": optional_config_float(
            cfg, "DataFrame", "coarse_eeg_max_rel_delta"
        ),
    }


def coarse_feature_paths(session_dir: Path) -> tuple[Path, Path]:
    root = Path(session_dir) / "15_coarse_features"
    return (
        root / "coarse_epochs_30s.parquet",
        root / "coarse_hrv_300s.parquet",
    )


def load_coarse_parquets(session_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epochs_path, hrv_path = coarse_feature_paths(session_dir)
    if not epochs_path.is_file():
        raise FileNotFoundError(f"Missing coarse epochs parquet: {epochs_path}")
    if not hrv_path.is_file():
        raise FileNotFoundError(f"Missing coarse HRV parquet: {hrv_path}")
    return pd.read_parquet(epochs_path), pd.read_parquet(hrv_path)


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing columns {missing}")


def _as_window_ok(values) -> np.ndarray:
    """True only for an explicit ok flag. NaN, 0, and False are not ok."""
    series = pd.Series(values).reset_index(drop=True)
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return np.isfinite(numeric) & (numeric != 0.0)


def _hr_window_ok_fraction(hrv: pd.DataFrame) -> tuple[int, int, float]:
    if len(hrv) == 0 or "window_ok" not in hrv.columns:
        return 0, 0, float("nan")
    ok = _as_window_ok(hrv["window_ok"])
    n = int(ok.size)
    n_ok = int(ok.sum())
    return n_ok, n, (n_ok / n if n else float("nan"))


def join_hr_mean_onto_epochs(epochs: pd.DataFrame, hrv: pd.DataFrame) -> pd.DataFrame:
    """Attach ``hr_mean_bpm`` from the 300 s window that contains each epoch.

    The match is ``window.t_start <= epoch.t_start < window.t_end`` using the
    latest window start at or before the epoch. HR is NaN when that window is
    missing, ``window_ok`` is false, or the rate is non-finite. Other HRV
    columns are ignored.
    """
    out = epochs.copy()
    t = out["t_start"].to_numpy(dtype=np.float64) if len(out) else np.array([])
    hr_out = np.full(len(out), np.nan, dtype=np.float64)
    if len(hrv) and len(out):
        _require_columns(
            hrv,
            ["t_start", "t_end", "hr_mean_bpm", "window_ok"],
            "coarse HRV parquet",
        )
        ordered = hrv.sort_values(["t_start", "t_end"], kind="mergesort")
        starts = ordered["t_start"].to_numpy(dtype=np.float64)
        ends = ordered["t_end"].to_numpy(dtype=np.float64)
        hr_vals = ordered["hr_mean_bpm"].to_numpy(dtype=np.float64)
        ok = _as_window_ok(ordered["window_ok"])
        pos = np.searchsorted(starts, t, side="right") - 1
        clipped = np.clip(pos, 0, len(starts) - 1)
        contains = (pos >= 0) & (t >= starts[clipped]) & (t < ends[clipped])
        take = contains & ok[clipped] & np.isfinite(hr_vals[clipped])
        hr_out[take] = hr_vals[clipped[take]]
    out["hr_mean_bpm"] = hr_out
    return out


def _blank_eeg_rows(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    rows = np.asarray(rows, dtype=bool)
    if rows.size == 0 or not bool(rows.any()):
        return out
    for col in EEG_DERIVED_COLUMNS:
        if col not in out.columns:
            continue
        values = out[col].to_numpy(dtype=np.float64).copy()
        values[rows] = np.nan
        out[col] = values
    return out


def apply_rel_alpha_eeg_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """NaN every EEG-derived column wherever ``rel_alpha`` is non-finite.

    ``epoch_eeg_good`` is not consulted.
    """
    if "rel_alpha" not in frame.columns:
        raise KeyError("coarse epochs require column 'rel_alpha'")
    bad = ~np.isfinite(frame["rel_alpha"].to_numpy(dtype=np.float64))
    return _blank_eeg_rows(frame, bad)


def _log10_positive(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(x) & (x > 0.0)
    out[ok] = np.log10(x[ok])
    return out


def _provisional_nontest_mask(n_rows: int) -> np.ndarray:
    """Leading rows of the last-20%-of-rows cut (Xhro's ``n - n // 5``)."""
    mask = np.ones(n_rows, dtype=bool)
    n_tail = int(n_rows) // 5
    if n_tail:
        mask[-n_tail:] = False
    return mask


def _removal_record(removed: np.ndarray, usable: np.ndarray) -> dict:
    usable = np.asarray(usable, dtype=bool)
    removed = np.asarray(removed, dtype=bool) & usable
    n_usable = int(usable.sum())
    n_removed = int(removed.sum())
    fraction = float(n_removed / n_usable) if n_usable else float("nan")
    return {"removed": n_removed, "usable": n_usable, "fraction": fraction}


def _rule_log_total_power_robust(ctx: dict) -> tuple[np.ndarray, dict]:
    """Flag epochs with log10 total power above median + k * scaled MAD.

    The fit uses provisional non-test EEG-usable epochs only (finite
    ``rel_alpha`` and finite log10 total power).
    """
    log_total = np.asarray(ctx["log_total"], dtype=np.float64)
    fit = (
        np.asarray(ctx["provisional_non_test"], dtype=bool)
        & np.asarray(ctx["eeg_usable"], dtype=bool)
        & np.isfinite(log_total)
    )
    n_fit = int(fit.sum())
    if n_fit == 0:
        raise ValueError(
            "coarse_eeg_artifact_mask='log_total_power_robust' found no "
            "provisional non-test EEG-usable epochs with finite "
            "log10(abs_total_0.5-30) to estimate median/MAD."
        )
    sample = log_total[fit]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = float(COARSE_MAD_TO_SIGMA * mad)
    k = float(ctx["k"])
    threshold = float(median + k * scale)
    flagged = np.isfinite(log_total) & (log_total > threshold)
    return flagged, {
        "median": median,
        "mad": mad,
        "scale": scale,
        "k": k,
        "threshold": threshold,
        "n_fit": n_fit,
    }


_rule_log_total_power_robust.requires_k = True


def _rule_max_rel_delta(ctx: dict) -> tuple[np.ndarray, dict]:
    cap = float(ctx["max_rel_delta"])
    rel = np.asarray(ctx["rel_delta"], dtype=np.float64)
    return np.isfinite(rel) & (rel > cap), {"cap": cap}


# Named ``coarse_eeg_artifact_mask`` values. Add a function here to grow the set.
# Signature: (ctx) -> (row_flag, details_dict). Set ``requires_k = True`` on the
# function if it needs ``coarse_eeg_artifact_k``.
_ARTIFACT_RULE_FUNCS = {
    "log_total_power_robust": _rule_log_total_power_robust,
}


def _normalize_artifact_mask_name(value) -> str:
    if _is_unset(value):
        return "none"
    name = str(value).strip().lower()
    if name == "none":
        return "none"
    if name not in _ARTIFACT_RULE_FUNCS:
        known = ", ".join(["none", *sorted(_ARTIFACT_RULE_FUNCS)])
        raise ValueError(
            f"Unknown coarse_eeg_artifact_mask {value!r}. Expected one of: {known}."
        )
    return name


def _validate_artifact_config(mask_name: str, k: float | None) -> None:
    if mask_name == "none":
        return
    fn = _ARTIFACT_RULE_FUNCS[mask_name]
    if getattr(fn, "requires_k", False) and k is None:
        raise ValueError(
            f"coarse_eeg_artifact_mask={mask_name!r} requires coarse_eeg_artifact_k "
            "(threshold = median + k * 1.4826 * MAD of log10 total power on the "
            "subject's provisional non-test). Omit the rule, or set "
            "coarse_eeg_artifact_mask to 'none', to leave the mask off."
        )


def eeg_artifact_row_mask(
    frame: pd.DataFrame,
    *,
    mask_name: str,
    k: float | None,
    max_rel_delta: float | None,
) -> tuple[np.ndarray, dict]:
    """Rows to NaN, plus per-rule removal counts on pre-rule EEG-usable epochs.

    Named rules and the optional rel_delta cap are scored independently against
    epochs that still have finite ``rel_alpha``, then combined.
    """
    n = len(frame)
    eeg_usable = np.isfinite(frame["rel_alpha"].to_numpy(dtype=np.float64))
    rel_delta = (
        frame["rel_delta"].to_numpy(dtype=np.float64)
        if "rel_delta" in frame.columns
        else np.full(n, np.nan, dtype=np.float64)
    )
    reports: dict = {}
    flagged = np.zeros(n, dtype=bool)
    if mask_name != "none":
        if ABS_TOTAL_COLUMN not in frame.columns:
            raise KeyError(
                f"coarse_eeg_artifact_mask={mask_name!r} needs column "
                f"{ABS_TOTAL_COLUMN!r}."
            )
        ctx = {
            "log_total": _log10_positive(
                frame[ABS_TOTAL_COLUMN].to_numpy(dtype=np.float64)
            ),
            "provisional_non_test": _provisional_nontest_mask(n),
            "eeg_usable": eeg_usable,
            "k": k,
            "rel_delta": rel_delta,
            "max_rel_delta": max_rel_delta,
        }
        rule_flag, details = _ARTIFACT_RULE_FUNCS[mask_name](ctx)
        record = _removal_record(rule_flag, eeg_usable)
        record.update(details)
        reports[mask_name] = record
        flagged |= np.asarray(rule_flag, dtype=bool)
    if max_rel_delta is not None:
        if "rel_delta" not in frame.columns:
            raise KeyError("coarse_eeg_max_rel_delta needs column 'rel_delta'.")
        rel_flag, details = _rule_max_rel_delta(
            {"rel_delta": rel_delta, "max_rel_delta": max_rel_delta}
        )
        record = _removal_record(rel_flag, eeg_usable)
        record.update(details)
        reports["max_rel_delta"] = record
        flagged |= np.asarray(rel_flag, dtype=bool)
    return flagged & eeg_usable, reports


def _transform_feature(name: str, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if name in COARSE_LOG1P_COLUMNS:
        out = np.full(x.shape, np.nan, dtype=np.float64)
        ok = np.isfinite(x) & (x >= 0.0)
        out[ok] = np.log1p(x[ok])
        return out
    if name in COARSE_LOG10_COLUMNS:
        return _log10_positive(x)
    return x.copy()


def coarse_test_cut(usable: np.ndarray, mode: str) -> int:
    """Row index where the contiguous test tail starts.

    ``usable_tail`` places ``n_usable // 5`` jointly usable epochs in the
    tail (same floor as Xhro's ``n // 5``). ``last20_rows`` uses
    ``n + ((-n) // 5)``, the same ceil tail as Xhro's ``-n // 5`` slice.
    ``_provisional_nontest_mask`` keeps the floor cut ``n - n // 5``.
    """
    usable = np.asarray(usable, dtype=bool)
    n = int(usable.shape[0])
    if mode == "last20_rows":
        return n + ((-n) // 5)
    if mode == "usable_tail":
        idx = np.flatnonzero(usable)
        n_test = int(idx.size // 5)
        if n_test == 0:
            return n
        return int(idx[-n_test])
    raise ValueError(
        f"coarse_test_split must be one of {_COARSE_TEST_SPLITS}, got {mode!r}."
    )


def _clip_log_features(
    features: np.ndarray, columns: list[str], non_test: np.ndarray
) -> tuple[np.ndarray, dict]:
    out = np.array(features, dtype=np.float64, copy=True)
    clip_hi: dict = {}
    non_test = np.asarray(non_test, dtype=bool)
    for j, name in enumerate(columns):
        if name not in COARSE_LOG10_COLUMNS:
            continue
        sample = out[non_test, j]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            clip_hi[name] = None
            continue
        hi = float(np.percentile(finite, COARSE_LOG_CLIP_PERCENTILE))
        clip_hi[name] = hi
        col = out[:, j]
        finite_col = np.isfinite(col)
        col[finite_col] = np.minimum(col[finite_col], hi)
        out[:, j] = col
    return out, clip_hi


def _zscore_features(
    features: np.ndarray,
    columns: list[str],
    non_test: np.ndarray,
    *,
    subject_label: str,
) -> tuple[np.ndarray, dict, dict, list]:
    """Z-score using non-test nanmean/nanstd. Constant columns become NaN."""
    out = np.array(features, dtype=np.float64, copy=True)
    non_test = np.asarray(non_test, dtype=bool)
    norm_mean: dict = {}
    norm_std: dict = {}
    guarded: list[str] = []
    if out.shape[0] == 0 or not bool(non_test.any()):
        for name in columns:
            norm_mean[name] = None
            norm_std[name] = None
        return out, norm_mean, norm_std, guarded
    for j, name in enumerate(columns):
        sample = out[non_test, j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = float(np.nanmean(sample))
            std = float(np.nanstd(sample))
        if name in COARSE_NO_NORM_COLUMNS:
            norm_mean[name] = None
            norm_std[name] = None
            continue
        norm_mean[name] = mean
        norm_std[name] = std
        if (not np.isfinite(std)) or std < COARSE_STD_EPS:
            out[:, j] = np.nan
            guarded.append(name)
            msg = (
                f"[XhroProper][coarse][{subject_label}] column {name!r} "
                f"non-test std {std:.3g} < {COARSE_STD_EPS}; "
                "setting that column to NaN for this subject."
            )
            print("WARNING " + msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        out[:, j] = (out[:, j] - mean) / std
    return out, norm_mean, norm_std, guarded


def _apply_coarse_suffix(features: np.ndarray, suffix: str | None) -> np.ndarray:
    """Plain keeps NaNs. Interpolate fills inside this array. Indicate doubles."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if suffix is None:
        return arr.astype(np.float32)
    if suffix == "interpolate":
        out = arr.copy()
        idx = np.arange(out.shape[0])
        for j in range(out.shape[1]):
            x = out[:, j]
            missing = ~np.isfinite(x)
            if not missing.any():
                continue
            valid = ~missing
            if valid.any():
                x[missing] = np.interp(idx[missing], idx[valid], x[valid])
            else:
                x[:] = 0.0
            out[:, j] = x
        return out.astype(np.float32)
    if suffix == "indicate":
        missing = ~np.isfinite(arr)
        # Zero every non-finite, including +/-inf. nan_to_num leaves inf as
        # the float extreme, and the float32 cast turns that back into inf.
        imputed = np.where(missing, 0.0, arr).astype(np.float32)
        is_obs = (~missing).astype(np.float32)
        result = np.empty((arr.shape[0], arr.shape[1] * 2), dtype=np.float32)
        result[:, 0::2] = imputed
        result[:, 1::2] = is_obs
        return result
    raise ValueError(f"Unknown coarse suffix: {suffix!r}")


def intra_subject_window_starts(
    subject_index: np.ndarray, seq_len: int, overlap: bool
) -> np.ndarray:
    """Window starts that lie entirely inside one subject.

    Same rule as ``intra_stay_window_starts`` in ``physionet2012_dataset.py``.
    """
    subject_index = np.asarray(subject_index)
    starts: list[int] = []
    n = len(subject_index)
    i = 0
    while i < n:
        sid = subject_index[i]
        j = i + 1
        while j < n and subject_index[j] == sid:
            j += 1
        if j - i >= seq_len:
            step = 1 if overlap else seq_len
            starts.extend(range(i, j - seq_len + 1, step))
        i = j
    return np.asarray(starts, dtype=np.int64)


def max_group_span(group_index, n_rows: int) -> int:
    """Longest contiguous run. ``group_index is None`` means one series."""
    n_rows = int(n_rows)
    if n_rows <= 0:
        return 0
    if group_index is None:
        return n_rows
    index = np.asarray(group_index)
    if index.size == 0:
        return 0
    if index.size == 1:
        return 1
    changes = np.flatnonzero(index[1:] != index[:-1]) + 1
    bounds = np.r_[0, changes, index.size]
    return int(np.diff(bounds).max())


def dataset_row_group(dataset):
    """Per-row subject, stay, or waveform segment. None if the series is one piece."""
    for name in (
        "subject_index_per_row",
        "stay_id_per_row",
        "segment_index_per_row",
    ):
        group = getattr(dataset, name, None)
        if group is not None:
            return np.asarray(group)
    return None


def dataset_max_window_length(dataset) -> int:
    """Longest window that stays inside one subject, stay, or waveform segment."""
    seq = getattr(dataset, "seq", None)
    n_rows = 0 if seq is None else int(len(seq))
    return max_group_span(dataset_row_group(dataset), n_rows)


_BLOCK_MASK_MODES = ("not_good", "dirty", "lowamp", "dirty_or_lowamp")


def parse_block_mask_mode(value) -> str:
    """``not_good`` when unset. Other accepted names are the flag columns."""
    if _is_unset(value):
        return "not_good"
    key = str(value).strip().lower()
    if key not in _BLOCK_MASK_MODES:
        raise ValueError(
            "block_mask_mode must be one of "
            f"{_BLOCK_MASK_MODES}, got {value!r}."
        )
    return key


def _frame_sampling_hz(frame: pd.DataFrame) -> float:
    """Median positive sample rate of the frame's datetime column."""
    if frame is None or len(frame) < 2 or "datetime" not in frame.columns:
        raise ValueError(
            "block_mask_file needs a datetime column with at least two rows."
        )
    deltas = frame["datetime"].diff().dt.total_seconds().to_numpy(dtype=np.float64)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        raise ValueError("block_mask_file needs a positive sampling interval.")
    return 1.0 / float(np.median(deltas))


def _block_length_sec(t0: np.ndarray) -> float:
    """Median positive spacing of block start times. Not a fixed 10 s."""
    times = np.asarray(t0, dtype=np.float64).reshape(-1)
    times = times[np.isfinite(times)]
    if times.size < 2:
        raise ValueError(
            "block_mask t0 needs at least two finite times so the block "
            "length can be inferred from their spacing."
        )
    step = np.diff(np.sort(times))
    step = step[np.isfinite(step) & (step > 0)]
    if step.size == 0:
        raise ValueError("block_mask t0 spacing must be positive.")
    return float(np.median(step))


def block_mask_rows(n_rows: int, fs_hz: float, t0, flagged) -> np.ndarray:
    """Rows covered by flagged blocks. Row 0 is session time 0.

    Block ``i`` covers ``[t0, t0 + block_len)`` seconds, which is rows
    ``[rint(fs * t0), rint(fs * (t0 + block_len)))``. ``fs_hz`` and the
    block length are the caller's inferred values. Rows outside every
    block stay False.
    """
    n = int(n_rows)
    masked = np.zeros(n, dtype=bool)
    if n == 0:
        return masked
    fs = float(fs_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {fs_hz!r}.")
    times = np.asarray(t0, dtype=np.float64).reshape(-1)
    flags = np.asarray(flagged, dtype=bool).reshape(-1)
    if times.shape != flags.shape:
        raise ValueError(
            f"block mask t0 length {times.shape[0]} != flag length {flags.shape[0]}"
        )
    block_len = _block_length_sec(times)
    use = flags & np.isfinite(times)
    starts = np.rint(times[use] * fs).astype(np.int64)
    ends = np.rint((times[use] + block_len) * fs).astype(np.int64)
    for start, end in zip(starts, ends):
        lo = max(int(start), 0)
        hi = min(int(end), n)
        if hi > lo:
            masked[lo:hi] = True
    return masked


def block_flags(dirty, lowamp, good, mode: str) -> np.ndarray:
    """Which blocks the mode masks. ``not_good`` is the complement of ``good``."""
    mode = parse_block_mask_mode(mode)
    dirty = np.asarray(dirty, dtype=bool).reshape(-1)
    lowamp = np.asarray(lowamp, dtype=bool).reshape(-1)
    good = np.asarray(good, dtype=bool).reshape(-1)
    if mode == "dirty":
        return dirty
    if mode == "lowamp":
        return lowamp
    if mode == "dirty_or_lowamp":
        return dirty | lowamp
    return ~good


def apply_block_mask(frame: pd.DataFrame, path, mode) -> pd.DataFrame:
    """NaN masked rows and mark them absent. Unset path leaves the frame.

    Applied on the waveform frame before the time crop. The same file is
    used for every variant. Rows the file does not cover stay as they are.
    """
    if _is_unset(path):
        return frame
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"block_mask_file not found: {mask_path}")
    chosen = parse_block_mask_mode(mode)
    with np.load(mask_path, allow_pickle=False) as npz:
        missing = [key for key in ("t0", "dirty", "lowamp", "good") if key not in npz.files]
        if missing:
            raise ValueError(
                f"block_mask_file {mask_path} is missing keys {missing}."
            )
        t0 = np.asarray(npz["t0"], dtype=np.float64).reshape(-1)
        dirty = _as_bool(npz["dirty"]).reshape(-1)
        lowamp = _as_bool(npz["lowamp"]).reshape(-1)
        good = _as_bool(npz["good"]).reshape(-1)
    lengths = {
        "t0": int(t0.size),
        "dirty": int(dirty.size),
        "lowamp": int(lowamp.size),
        "good": int(good.size),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "block mask columns must share one length, got "
            + ", ".join(f"{key}={length}" for key, length in sorted(lengths.items()))
        )
    n = 0 if frame is None else int(len(frame))
    fs = _frame_sampling_hz(frame)
    flagged = block_flags(dirty, lowamp, good, chosen)
    masked = block_mask_rows(n, fs, t0, flagged)
    if PRESENT_COL in frame.columns:
        present = frame[PRESENT_COL].to_numpy(dtype=bool).copy()
    else:
        present = np.ones(n, dtype=bool)
    present[masked] = False
    frame = frame.copy()
    frame[PRESENT_COL] = present
    _blank_absent_rows(frame, present)
    n_masked = int(masked.sum())
    frac = (n_masked / n) if n else 0.0
    print(
        f"[XhroProper] block mask mode={chosen} file={mask_path} "
        f"masked={n_masked}/{n} ({frac:.3f}) "
        f"block_sec={_block_length_sec(t0):.6g} fs≈{fs:.6g} Hz"
    )
    return frame


def crop_proper_frame(
    frame: pd.DataFrame,
    crop_start_sec,
    crop_end_sec,
) -> pd.DataFrame:
    """Keep rows with ``crop_start_sec <= t <= crop_end_sec`` from the first timestamp.

    Both bounds unset leaves the frame unchanged. Seconds are measured from
    ``min(datetime)`` so a later resample does not move the crop. Applied on
    the waveform path only, after ``load_proper_frame`` and before the test split.
    """
    if _is_unset(crop_start_sec) and _is_unset(crop_end_sec):
        return frame
    if frame is None or len(frame) == 0:
        return frame
    if "datetime" not in frame.columns:
        warnings.warn(
            "crop_start_sec/crop_end_sec ignored: frame has no datetime column.",
            UserWarning,
            stacklevel=2,
        )
        return frame
    t0 = frame["datetime"].min()
    elapsed = (frame["datetime"] - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    keep = np.ones(len(frame), dtype=bool)
    if not _is_unset(crop_start_sec):
        keep &= elapsed >= float(crop_start_sec)
    if not _is_unset(crop_end_sec):
        keep &= elapsed <= float(crop_end_sec)
    cropped = frame.loc[keep].reset_index(drop=True)
    if len(cropped) == 0:
        raise ValueError(
            "crop_start_sec/crop_end_sec removed every row "
            f"(start={crop_start_sec!r}, end={crop_end_sec!r}, "
            f"span_sec={float(np.nanmax(elapsed) - np.nanmin(elapsed)):.6g})."
        )
    print(
        f"[XhroProper] crop seconds [{crop_start_sec}, {crop_end_sec}] "
        f"kept {len(cropped)}/{len(frame)} rows"
    )
    return cropped


def build_coarse_subject(
    session_dir: Path,
    columns: list[str],
    *,
    test_split_mode: str,
    artifact_mask: str,
    artifact_k: float | None,
    max_rel_delta: float | None,
    subject_label: str,
) -> dict:
    """Load, mask, transform, and z-score one subject. Rows are not sliced."""
    epochs, hrv = load_coarse_parquets(session_dir)
    uses_eeg = _columns_use_eeg(columns)
    needed = ["t_start", "t_end"]
    if uses_eeg:
        needed.append("rel_alpha")
    for col in columns:
        if col != "hr_mean_bpm" and col not in needed:
            needed.append(col)
    epochs_desc = f"{session_dir}/15_coarse_features/coarse_epochs_30s.parquet"
    _require_columns(epochs, needed, epochs_desc)
    _require_columns(
        hrv,
        ["t_start", "t_end", "hr_mean_bpm", "window_ok"],
        f"{session_dir}/15_coarse_features/coarse_hrv_300s.parquet",
    )
    t_raw = epochs["t_start"].to_numpy(dtype=np.float64)
    if len(t_raw) and not np.isfinite(t_raw).all():
        raise ValueError(f"{subject_label}: coarse t_start has non-finite values")
    joined = join_hr_mean_onto_epochs(epochs, hrv)
    joined = joined.sort_values("t_start", kind="mergesort").reset_index(drop=True)
    if uses_eeg:
        joined = apply_rel_alpha_eeg_mask(joined)
        art_rows, art_report = eeg_artifact_row_mask(
            joined,
            mask_name=artifact_mask,
            k=artifact_k,
            max_rel_delta=max_rel_delta,
        )
        joined = _blank_eeg_rows(joined, art_rows)
    else:
        # No selected EEG column: rel_alpha / artifact rules must not run,
        # and jointly-usable below is only the selected columns.
        art_report = {}
    stacked = [
        _transform_feature(col, joined[col].to_numpy(dtype=np.float64))
        for col in columns
    ]
    features = np.column_stack(stacked) if stacked else np.zeros((len(joined), 0))
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    usable = np.isfinite(features).all(axis=1)
    cut = coarse_test_cut(usable, test_split_mode)
    non_test = np.zeros(features.shape[0], dtype=bool)
    non_test[:cut] = True
    features, clip_hi = _clip_log_features(features, columns, non_test)
    features, norm_mean, norm_std, guarded = _zscore_features(
        features, columns, non_test, subject_label=subject_label
    )
    n = int(features.shape[0])
    n_ok, n_win, hr_frac = _hr_window_ok_fraction(hrv)
    head_n = int(cut)
    tail_n = n - head_n
    n_head_usable = int(usable[:cut].sum()) if n else 0
    n_tail_usable = int(usable[cut:].sum()) if n else 0
    report = {
        "subject": subject_label,
        "session_dir": str(session_dir),
        "n_epochs": n,
        "test_cut": int(cut),
        "test_split_mode": test_split_mode,
        "n_usable": int(usable.sum()) if n else 0,
        "n_non_test_usable": n_head_usable,
        "n_test_usable": n_tail_usable,
        "usable_frac_non_test": (
            float(n_head_usable) / head_n if head_n else float("nan")
        ),
        "usable_frac_test": (float(n_tail_usable) / tail_n if tail_n else float("nan")),
        "hr_n_window_ok": n_ok,
        "hr_n_windows": n_win,
        "hr_window_ok_frac": hr_frac,
        "artifact": art_report,
        "eeg_columns": uses_eeg,
        "columns": list(columns),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "clip_hi": clip_hi,
        "std_guarded": list(guarded),
    }
    return {
        "features": features,
        "t_start": joined["t_start"].to_numpy(dtype=np.float64),
        "usable": usable,
        "report": report,
    }


def _fmt_frac(numer: int, denom: int) -> str:
    if denom == 0:
        return "nan"
    return f"{numer / denom:.3f}"


def _log_coarse_subject(report: dict) -> None:
    label = report["subject"]
    hr_frac = report["hr_window_ok_frac"]
    hr_frac_s = "nan" if not np.isfinite(hr_frac) else f"{hr_frac:.3f}"
    print(
        f"[XhroProper][coarse][{label}] epochs={report['n_epochs']} "
        "usable_frac non_test(train/valid)="
        f"{_fmt_frac(report['n_non_test_usable'], report['test_cut'])} "
        "test="
        f"{_fmt_frac(report['n_test_usable'], report['n_epochs'] - report['test_cut'])} "
        f"hr_window_ok={report['hr_n_window_ok']}/{report['hr_n_windows']} "
        f"({hr_frac_s}) test_cut={report['test_cut']} "
        f"mode={report['test_split_mode']} n_test_usable={report['n_test_usable']}"
    )
    if not report.get("eeg_columns", True):
        print(
            f"[XhroProper][coarse][{label}] artifact rules not applied "
            "(no EEG columns)"
        )
        return
    artifact = report["artifact"]
    if not artifact:
        print(f"[XhroProper][coarse][{label}] artifact rules off")
        return
    for name, stats in artifact.items():
        extra = ""
        if "threshold" in stats:
            extra = (
                f" threshold={stats['threshold']:.4g} "
                f"(median={stats['median']:.4g} + k={stats['k']:g} "
                f"* scale={stats['scale']:.4g}, n_fit={stats['n_fit']})"
            )
        elif "cap" in stats:
            extra = f" cap={stats['cap']:g}"
        frac = stats["fraction"]
        frac_s = "nan" if not np.isfinite(frac) else f"{frac:.3f}"
        print(
            f"[XhroProper][coarse][{label}] artifact {name} removed "
            f"{stats['removed']}/{stats['usable']} usable EEG epochs "
            f"({frac_s}){extra}"
        )


def _align_missing_mask(mask, processed) -> np.ndarray:
    """Force the mask onto the rows ``apply_observation_process`` returned.

    A longer mask means a leading all-NaN prefix was dropped from the signal.
    Keep that suffix. A shorter mask is a bug: the signal rows are left as
    apply produced them.
    """
    aligned = np.asarray(mask)
    if aligned.ndim == 1:
        aligned = aligned.reshape(-1, 1)
    n_proc = int(processed.shape[0])
    if len(aligned) == n_proc:
        return aligned
    if len(aligned) > n_proc:
        return aligned[-n_proc:]
    raise RuntimeError(
        f"missing_mask length {len(aligned)} is shorter than the processed "
        f"signal ({n_proc}). The mask must follow apply_observation_process."
    )


class XhroProper(Xhro):
    """Grok-pipeline XHRO sequences with named banks and honest gaps.

    Config mapping:
      - dataset_label: packet-loss recording id
        (e.g. ``XHRO3506_20260622T142410000+0900``), or the Multi session
        folder name (e.g. ``xhro_01_XH015``).
      - mask_label: ``realtime`` or ``retrans`` (``recovered`` aliases
        retrans) for packet-loss. Ignored when ``corpus`` is ``multi``.
      - corpus: ``packet_loss`` (default) or ``multi``.
      - data_root: folder that contains session directories. Overrides the
        corpus default under ``data_dir``. For ``corpus="multi"`` on the
        cluster this must be the grok_output folder; otherwise paths resolve
        under the repo's ``data/``.
      - coarse_test_split: ``usable_tail`` (default) or ``last20_rows``.
        Coarse modes only.
      - coarse_eeg_artifact_mask: ``none`` (default) or
        ``log_total_power_robust``.
      - coarse_eeg_artifact_k: required when the power rule is on.
      - coarse_eeg_max_rel_delta: optional rel_delta cap; unset means off.
      - crop_start_sec / crop_end_sec: inclusive seconds from the first
        timestamp. Unset means no crop. Waveform path only, after load and
        before the 20% test split.
      - max_fill_gap_sec: waveform only. Unset keeps filled gaps as NaN
        inside windows. When set, a filled gap with ``gap_duration_s``
        greater than this becomes a window split, like a segment boundary.
        The split indices are stored and reused on every re-cut.
      - legacy_windowing: false by default. True restores the previous
        window bundle when the four keys below are blank.
      - max_gap_window_frac: waveform only. Blank is 0.25 (off when
        ``legacy_windowing`` is true). A filled gap with more samples
        than this fraction of the current sequence length is a split
        for that re-cut only. ``off`` disables it.
      - start_on_observed: waveform and coarse. Blank is true (false when
        ``legacy_windowing`` is true). True moves a start on a missing
        sample forward to the next observed sample that still fits in
        the segment.
      - strict_non_overlap: waveform only. Blank is true (false when
        ``legacy_windowing`` is true). False keeps the historical
        ``find_indices`` starts. True tiles each segment without overlap
        or interior holes; the only unused samples are the final remainder
        of each piece, plus any gap a start had to step over.
      - block_mask_file / block_mask_mode: waveform only. Unset file means
        no block mask. The file's flagged blocks are set to NaN and
        ``present=False`` before the time crop, so they act as gaps.
      - max_window_nan_ratio: waveform and coarse re-cuts. Blank is 0.3
        (off when ``legacy_windowing`` is true). ``off`` disables it.
        An explicit ``minimum_nan_ratio`` argument overrides it.
      - filled_nan_log_sec: waveform log only. When ``max_fill_gap_sec``
        is unset, a re-cut reports how many windows contain a filled NaN
        run longer than this. Unset means that count is ``na``.

    Omit both ``data_root`` and ``corpus`` to keep today's packet-loss path:
    ``{data_dir}/xhro_packet_loss/grok_output/{dataset_label}_{variant}``.
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
        data_root=None,
        corpus=None,
        coarse_test_split=None,
        coarse_eeg_artifact_mask=None,
        coarse_eeg_artifact_k=None,
        coarse_eeg_max_rel_delta=None,
        crop_start_sec=None,
        crop_end_sec=None,
        max_fill_gap_sec=None,
        filled_nan_log_sec=None,
        legacy_windowing=_WINDOW_DEFAULT,
        max_gap_window_frac=_WINDOW_DEFAULT,
        start_on_observed=_WINDOW_DEFAULT,
        strict_non_overlap=_WINDOW_DEFAULT,
        max_window_nan_ratio=_WINDOW_DEFAULT,
        block_mask_file=None,
        block_mask_mode=None,
        **kwargs,
    ):
        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
        # None from config means the historical packet-loss tree.
        self.corpus = _normalize_corpus(corpus) or "packet_loss"
        self.data_root = None if _is_unset(data_root) else str(data_root).strip()
        if self.corpus == "multi":
            # Session folder is dataset_label. Do not invent a variant suffix.
            self.variant = None
        else:
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
        self.crop_start_sec = crop_start_sec
        self.crop_end_sec = crop_end_sec
        self.max_fill_gap_sec = _optional_nonnegative_float(
            max_fill_gap_sec, "max_fill_gap_sec"
        )
        self.filled_nan_log_sec = _optional_nonnegative_float(
            filled_nan_log_sec, "filled_nan_log_sec"
        )
        windowing = resolve_windowing(
            legacy_windowing=legacy_windowing,
            start_on_observed=start_on_observed,
            strict_non_overlap=strict_non_overlap,
            max_window_nan_ratio=max_window_nan_ratio,
            max_gap_window_frac=max_gap_window_frac,
        )
        self.legacy_windowing = windowing["legacy_windowing"]
        self.max_gap_window_frac = windowing["max_gap_window_frac"]
        self.start_on_observed = windowing["start_on_observed"]
        self.strict_non_overlap = windowing["strict_non_overlap"]
        self.max_window_nan_ratio = windowing["max_window_nan_ratio"]
        self.block_mask_file = (
            None if _is_unset(block_mask_file) else str(block_mask_file).strip()
        )
        self.block_mask_mode = parse_block_mask_mode(block_mask_mode)
        self.segment_index_per_row = None
        self.split_boundaries = None
        self.filled_run_sec = None
        self.filled_gap_spans = []
        self.observed_per_row = None
        self.coarse_mode = False
        base, _suffix = _split_process(self.observation_process)
        if base in COARSE_OBS_COLUMNS:
            self.coarse_mode = True
            if self.block_mask_file is not None:
                warnings.warn(
                    "block_mask_file is ignored for coarse observations.",
                    UserWarning,
                    stacklevel=2,
                )
            self._init_coarse(
                base,
                _suffix,
                coarse_test_split=coarse_test_split,
                coarse_eeg_artifact_mask=coarse_eeg_artifact_mask,
                coarse_eeg_artifact_k=coarse_eeg_artifact_k,
                coarse_eeg_max_rel_delta=coarse_eeg_max_rel_delta,
            )
            return

        self.session_dir = _require_session_dir(
            self.path_to_data,
            self.dataset_label,
            self.variant,
            data_root=self.data_root,
            corpus=self.corpus,
        )

        if base not in PARITY_PROCESSES and base not in GROK_EXTRA_PROCESSES:
            raise ValueError(
                f"Invalid observation process: {self.observation_process!r}. "
                f"Parity: {sorted(PARITY_PROCESSES)}; "
                f"grok extras: {sorted(GROK_EXTRA_PROCESSES)}; "
                f"coarse: {sorted(COARSE_OBS_COLUMNS)}; "
                "each may take _interpolate or _indicate."
            )

        the_sequence = load_proper_frame(
            self.session_dir,
            base,
            resample_hz=self.resample_hz,
            max_fill_gap_sec=self.max_fill_gap_sec,
        )
        the_sequence = apply_block_mask(
            the_sequence, self.block_mask_file, self.block_mask_mode
        )
        the_sequence = crop_proper_frame(
            the_sequence, self.crop_start_sec, self.crop_end_sec
        )
        if "datetime" in the_sequence.columns:
            # native rate from median Δt
            deltas = the_sequence["datetime"].diff().dt.total_seconds().dropna()
            if len(deltas):
                med = float(deltas.median())
                self.sampling_freq = (1.0 / med) if med > 0 else None
        if self.sampling_freq is None:
            self.sampling_freq = 250.0
        log_tag = self.variant if self.variant is not None else self.corpus
        gap_stats = gap_load_stats(the_sequence)
        print(
            f"[XhroProper][{log_tag}] {self.observation_process} "
            f"from {self.session_dir}  rows={gap_stats['rows']}  "
            f"fs≈{self.sampling_freq:.4g} Hz  "
            f"present_frac={gap_stats['present_frac']:.3f}  "
            f"segments={gap_stats['segments']}  "
            f"split_points={gap_stats['split_points']}"
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
        self.missing_mask = _align_missing_mask(self.missing_mask, the_sequence)

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
        self.segment_index_per_row = self._aligned_segment_index(
            self.full_sequence, len(self.seq)
        )
        self.filled_run_sec = self._aligned_float_column(
            self.full_sequence, FILLED_RUN_COL, len(self.seq)
        )
        self.observed_per_row = self._aligned_bool_column(
            self.full_sequence, PRESENT_COL, len(self.seq)
        )
        self.filled_gap_spans = self._filled_gap_spans(
            self.full_sequence, len(self.seq)
        )
        if self.segment_index_per_row is not None:
            self.split_boundaries = split_boundaries_from_ids(
                self.segment_index_per_row
            )
        self.update_sequence_length(self.seq_len)

    def _apply_keep_positions(self, sequence: pd.DataFrame) -> np.ndarray:
        """Positions ``apply_observation_process`` keeps, in order."""
        n = len(sequence)
        base, _suffix = _split_process(self.observation_process)
        if base in PARITY_PROCESSES:
            first_valid = sequence.first_valid_index()
        else:
            first_valid = self._value_frame(sequence).first_valid_index()
        if first_valid is None:
            return np.arange(n, dtype=np.int64)
        loc = sequence.index.get_loc(first_valid)
        if isinstance(loc, slice):
            start = 0 if loc.start is None else int(loc.start)
        elif isinstance(loc, np.ndarray):
            start = int(np.flatnonzero(loc)[0])
        else:
            start = int(loc)
        return np.arange(start, n, dtype=np.int64)

    def _aligned_segment_index(self, frame: pd.DataFrame, n_seq: int):
        """Window-group ids on the same rows as ``seq``. None on old files."""
        if GROUP_COL not in frame.columns:
            return None
        group = frame[GROUP_COL].to_numpy(dtype=np.int64)
        group = group[self._apply_keep_positions(frame)]
        if len(group) != int(n_seq):
            warnings.warn(
                f"[XhroProper] window groups ({len(group)}) do not match "
                f"seq ({n_seq}); segment windowing disabled for this series.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return group

    def _aligned_float_column(self, frame: pd.DataFrame, column: str, n_seq: int):
        """One float per ``seq`` row, or None when the frame has no such column."""
        if column not in frame.columns:
            return None
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[self._apply_keep_positions(frame)]
        if len(values) != int(n_seq):
            warnings.warn(
                f"[XhroProper] {column} length {len(values)} does not match "
                f"seq ({n_seq}); filled-gap durations are not logged.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return values

    def _aligned_bool_column(self, frame: pd.DataFrame, column: str, n_seq: int):
        """One bool per ``seq`` row, or None when the frame has no such column."""
        if column not in frame.columns:
            return None
        values = frame[column].to_numpy(dtype=bool)
        values = values[self._apply_keep_positions(frame)]
        if len(values) != int(n_seq):
            warnings.warn(
                f"[XhroProper] {column} length {len(values)} does not match "
                f"seq ({n_seq}); start_on_observed cannot use it.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return values

    def _filled_gap_spans(self, frame: pd.DataFrame, n_seq: int):
        """Contiguous ``present=False`` runs, in ``seq`` row indices.

        These are the filled NaN gaps. ``max_gap_window_frac`` decides
        which of them split a given re-cut. A length mismatch yields no
        spans, so the re-cut keeps the baked boundaries only.
        """
        if PRESENT_COL not in frame.columns:
            return []
        present = frame[PRESENT_COL].to_numpy(dtype=bool)
        keep = self._apply_keep_positions(frame)
        loc = np.full(len(frame), -1, dtype=np.int64)
        loc[keep] = np.arange(len(keep), dtype=np.int64)
        if int((loc >= 0).sum()) != int(n_seq):
            warnings.warn(
                f"[XhroProper] filled-gap rows do not match seq ({n_seq}); "
                "max_gap_window_frac will not add splits.",
                UserWarning,
                stacklevel=2,
            )
            return []
        spans = []
        i = 0
        n = len(present)
        while i < n:
            if present[i]:
                i += 1
                continue
            j = i + 1
            while j < n and not present[j]:
                j += 1
            rows = loc[i:j]
            rows = rows[rows >= 0]
            if rows.size:
                spans.append(rows.astype(np.int64, copy=False))
            i = j
        return spans

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

    def _values_apply_keeps(self, sequence: pd.DataFrame) -> np.ndarray:
        """Value rows ``apply_observation_process`` keeps, before normalize.

        Grok extras trim on the value frame's ``first_valid_index``. Parity
        trims on the full frame, matching ``Xhro``. The missing mask uses
        this same slice so a leading all-NaN value row cannot disagree with
        the processed length. Which signal rows are kept does not change.
        """
        base, _suffix = _split_process(self.observation_process)
        if base in PARITY_PROCESSES:
            cols = select_columns_for_obs_conditions["original"][base]
            data = sequence[cols]
            first_valid = sequence.first_valid_index()
        else:
            data = self._value_frame(sequence)
            first_valid = data.first_valid_index()
        if first_valid is not None:
            data = data.loc[first_valid:]
        arr = data.to_numpy(dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _extract_missing_mask(self, sequence):
        # Same rows apply keeps, including a datetime-valid leading row.
        mask = np.isnan(self._values_apply_keeps(sequence))
        if mask.ndim == 1:
            mask = mask.reshape(-1, 1)
        return mask

    def apply_observation_process(self, sequence) -> torch.Tensor:
        base, suffix = _split_process(self.observation_process)
        if base in PARITY_PROCESSES:
            return super().apply_observation_process(sequence)

        arr = self._values_apply_keeps(sequence)

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

    def _init_coarse(
        self,
        base: str,
        suffix: str | None,
        *,
        coarse_test_split,
        coarse_eeg_artifact_mask,
        coarse_eeg_artifact_k,
        coarse_eeg_max_rel_delta,
    ) -> None:
        self.coarse_test_split = _normalize_coarse_test_split(coarse_test_split)
        self.coarse_eeg_artifact_mask = _normalize_artifact_mask_name(
            coarse_eeg_artifact_mask
        )
        self.coarse_eeg_artifact_k = _optional_user_float(
            coarse_eeg_artifact_k, "coarse_eeg_artifact_k"
        )
        self.coarse_eeg_max_rel_delta = _optional_user_float(
            coarse_eeg_max_rel_delta, "coarse_eeg_max_rel_delta"
        )
        _validate_artifact_config(
            self.coarse_eeg_artifact_mask, self.coarse_eeg_artifact_k
        )
        self.sampling_freq = COARSE_SAMPLING_FREQ
        self.feature_columns = list(COARSE_OBS_COLUMNS[base])
        self.dataset_labels = _parse_coarse_labels(self.dataset_label)
        self.is_segmented_1d = False
        print(
            f"[XhroProper][coarse] {self.observation_process} "
            f"subjects={self.dataset_labels} fs={self.sampling_freq:.6g} Hz "
            f"columns={self.feature_columns}"
        )
        prepared = []
        self.session_dirs = []
        for label in self.dataset_labels:
            session_dir = _require_session_dir(
                self.path_to_data,
                label,
                self.variant,
                data_root=self.data_root,
                corpus=self.corpus,
            )
            self.session_dirs.append(session_dir)
            item = build_coarse_subject(
                session_dir,
                self.feature_columns,
                test_split_mode=self.coarse_test_split,
                artifact_mask=self.coarse_eeg_artifact_mask,
                artifact_k=self.coarse_eeg_artifact_k,
                max_rel_delta=self.coarse_eeg_max_rel_delta,
                subject_label=label,
            )
            _log_coarse_subject(item["report"])
            prepared.append(item)
        self.session_dir = self.session_dirs[0]
        self.coarse_reports = [item["report"] for item in prepared]

        seq_blocks = []
        mask_blocks = []
        frame_blocks = []
        subject_labels = []
        subject_index = []
        for i, (label, item) in enumerate(zip(self.dataset_labels, prepared)):
            features = item["features"]
            cut = int(item["report"]["test_cut"])
            row_slice = slice(cut, None) if self.split == "test" else slice(0, cut)
            block = features[row_slice]
            seq_blocks.append(_apply_coarse_suffix(block, suffix))
            mask_blocks.append(~np.isfinite(block))
            part = pd.DataFrame(block, columns=self.feature_columns)
            part.insert(0, "t_start", item["t_start"][row_slice])
            part.insert(0, "subject", label)
            frame_blocks.append(part)
            subject_labels.append(np.full(len(block), label, dtype=object))
            subject_index.append(np.full(len(block), i, dtype=np.int64))

        self.seq = torch.tensor(
            np.concatenate(seq_blocks, axis=0), dtype=torch.float32
        )
        self.missing_mask = np.concatenate(mask_blocks, axis=0)
        self.full_sequence = pd.concat(frame_blocks, ignore_index=True)
        self.subject_per_row = np.concatenate(subject_labels)
        self.subject_index_per_row = np.concatenate(subject_index)

        got = int(self.seq.shape[1]) if self.seq.ndim == 2 else 1
        if self.x_dim is None:
            self.x_dim = got
        elif got != self.x_dim:
            widened = " indicate doubles that width" if suffix == "indicate" else ""
            raise ValueError(
                f"Expected x_dim={self.x_dim}, got {got} for "
                f"{self.observation_process!r} "
                f"({len(self.feature_columns)} columns{widened})."
            )
        if suffix == "indicate":
            print(
                f"[XhroProper][{self.observation_process}] "
                f"missing_count={int(self.missing_mask.sum())} "
                f"result_shape={tuple(self.seq.shape)}"
            )
        self.window_subject = []
        self.split_boundaries = split_boundaries_from_ids(self.subject_index_per_row)
        self.update_sequence_length(self.seq_len)

    def window_subject_id(self, index: int) -> str:
        """Session folder name for the window at ``data_idx[index]``."""
        return self.window_subject[index]

    def update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
        if not getattr(self, "coarse_mode", False):
            # Overlapping waveform windows use the per-segment stride when
            # this re-cut has any split (segment, max_fill, or frac).
            # Non-overlapping windows, including curriculum resizes, go
            # through ``Xhro.update_sequence_length`` → ``find_indices``.
            if new_seq_len is None:
                return super().update_sequence_length(None, minimum_nan_ratio)
            num_frames = int(self.seq.shape[0])
            requested = int(new_seq_len)
            seq_len, boundaries = resolve_recut_length(
                num_frames,
                requested,
                getattr(self, "split_boundaries", None),
                getattr(self, "filled_gap_spans", None),
                getattr(self, "max_gap_window_frac", None),
            )
            if (
                self.overlap
                and not getattr(self, "strict_non_overlap", False)
                and _boundary_cuts(boundaries, num_frames) is not None
            ):
                if seq_len < requested:
                    print(
                        f"[XhroProper] seq_len {requested} exceeds the longest "
                        f"unbroken span {seq_len}; using {seq_len}."
                    )
                return self._update_waveform_segment_windows(
                    seq_len, minimum_nan_ratio, boundaries
                )
            return super().update_sequence_length(new_seq_len, minimum_nan_ratio)
        if new_seq_len is None:
            self.data_idx = [0]
            if len(self.subject_per_row):
                self.window_subject = [str(self.subject_per_row[0])]
            else:
                self.window_subject = []
            print(
                "[XhroProper][coarse] seq_len is None; one placeholder window at 0."
            )
            log_window_recut(self)
            return
        new_seq_len = int(new_seq_len)
        if new_seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {new_seq_len}.")
        span = max_group_span(
            self.subject_index_per_row, len(self.subject_index_per_row)
        )
        if span < 1:
            self.seq_len = new_seq_len
            self.data_idx = []
            self.window_subject = []
            print("[XhroProper][coarse] no subject rows; window set is empty.")
            log_window_recut(self)
            return
        if new_seq_len > span:
            print(
                f"[XhroProper][coarse] seq_len {new_seq_len} exceeds the longest "
                f"subject span {span}; using {span}."
            )
            new_seq_len = span
        self.seq_len = new_seq_len
        starts = intra_subject_window_starts(
            self.subject_index_per_row,
            self.seq_len,
            overlap=bool(self.overlap),
        )
        starts = self._window_starts(starts, self.split_boundaries)
        minimum_nan_ratio = self._minimum_nan_ratio(minimum_nan_ratio)
        if minimum_nan_ratio is not None:
            seq_np = (
                self.seq.detach().cpu().numpy()
                if torch.is_tensor(self.seq)
                else np.asarray(self.seq)
            )
            kept = []
            for idx in starts:
                window = seq_np[int(idx) : int(idx) + self.seq_len]
                if window.size == 0:
                    continue
                nan_ratio = float(np.float16(np.isnan(window)).mean())
                if nan_ratio <= float(minimum_nan_ratio):
                    kept.append(int(idx))
            starts = np.asarray(kept, dtype=np.int64)
            print(
                f"[XhroProper][coarse] nan_ratio<={minimum_nan_ratio}: "
                f"windows={len(starts)}"
            )
        if self.split == "test":
            chosen = starts
        else:
            chosen = np.array(starts, copy=True)
            if self.shuffle and chosen.size:
                rng = np.random.RandomState(COARSE_WINDOW_RNG_SEED)
                rng.shuffle(chosen)
            split_point = int(len(chosen) * (1.0 - float(self.val_indices)))
            if self.split == "train":
                chosen = chosen[:split_point]
            else:
                chosen = chosen[split_point:]
        self.data_idx = [int(i) for i in np.asarray(chosen).tolist()]
        self.window_subject = [str(self.subject_per_row[i]) for i in self.data_idx]
        log_window_recut(self)
        return

    def _update_waveform_segment_windows(
        self, new_seq_len, minimum_nan_ratio, boundaries=None
    ):
        """Starts that lie inside one segment. Same split rule as coarse windows."""
        if new_seq_len is None:
            self.data_idx = [0]
            print("[XhroProper] seq_len is None; one placeholder window at 0.")
            log_window_recut(self)
            return
        new_seq_len = int(new_seq_len)
        if new_seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {new_seq_len}.")
        if boundaries is not None:
            n_rows = int(self.seq.shape[0])
            group = groups_from_boundaries(n_rows, boundaries)
        else:
            group = np.asarray(self.segment_index_per_row)
            boundaries = self.split_boundaries
        span = max_group_span(group, len(group))
        if span < 1:
            self.seq_len = new_seq_len
            self.data_idx = []
            print("[XhroProper] no rows; window set is empty.")
            log_window_recut(self)
            return
        if new_seq_len > span:
            print(
                f"[XhroProper] seq_len {new_seq_len} exceeds the longest "
                f"segment span {span}; using {span}."
            )
            new_seq_len = span
        self.seq_len = new_seq_len
        starts = intra_subject_window_starts(
            group, self.seq_len, overlap=bool(self.overlap)
        )
        starts = self._window_starts(starts, boundaries)
        minimum_nan_ratio = self._minimum_nan_ratio(minimum_nan_ratio)
        if minimum_nan_ratio is not None:
            kept = []
            for idx in starts:
                window = self.seq[int(idx) : int(idx) + self.seq_len]
                nan_ratio = float(np.float16(np.isnan(window)).mean())
                if nan_ratio <= float(minimum_nan_ratio):
                    kept.append(int(idx))
            starts = np.asarray(kept, dtype=np.int64)
            print(
                f"[XhroProper] nan_ratio<={minimum_nan_ratio}: "
                f"windows={len(starts)}"
            )
        if self.split == "test":
            chosen = starts
        else:
            chosen = np.array(starts, copy=True)
            train_idx, val_idx = self.split_dataset(chosen, self.val_indices)
            chosen = train_idx if self.split == "train" else val_idx
        self.data_idx = [int(i) for i in np.asarray(chosen).tolist()]
        log_window_recut(self)
        return
