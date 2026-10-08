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
        df = _df_from_npz(
            z,
            {
                "eeg_clean_template": np.asarray(
                    z["eeg_clean_template"], dtype=np.float64
                )
            },
            t0=t0,
        )
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
    tail (same floor as Xhro's ``n // 5``). ``last20_rows`` is ``n - n // 5``.
    """
    usable = np.asarray(usable, dtype=bool)
    n = int(usable.shape[0])
    if mode == "last20_rows":
        return n - (n // 5)
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
        imputed = np.nan_to_num(arr, nan=0.0).astype(np.float32)
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
        self.coarse_mode = False
        base, _suffix = _split_process(self.observation_process)
        if base in COARSE_OBS_COLUMNS:
            self.coarse_mode = True
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
        log_tag = self.variant if self.variant is not None else self.corpus
        print(
            f"[XhroProper][{log_tag}] {self.observation_process} "
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
        self.update_sequence_length(self.seq_len)

    def window_subject_id(self, index: int) -> str:
        """Session folder name for the window at ``data_idx[index]``."""
        return self.window_subject[index]

    def update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
        if not getattr(self, "coarse_mode", False):
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
            return
        new_seq_len = int(new_seq_len)
        if new_seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {new_seq_len}.")
        self.seq_len = new_seq_len
        starts = intra_subject_window_starts(
            self.subject_index_per_row,
            self.seq_len,
            overlap=bool(self.overlap),
        )
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
        return
