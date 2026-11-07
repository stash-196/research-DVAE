"""Load masked series for the standalone baseline scripts.

Lorenz paths and observation slices match ``Lorenz63`` without going through
the training dataloader, so windowing (``x_dim > 1`` on ``only_x``) is not
applied. The baseline models the sampled series itself. Channel metrics then
use the same names as ``resolve_channel_keys`` for that observation process
(``x``, or ``x,y``, ...).

PhysioNet 2012 is optional at runtime. The loader imports ``PhysioNet2012``
only when asked, keeps each stay separate so a transition never crosses
patients, and does not modify that dataset module.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dvae.baselines.filtering.lorenz_oracle import parse_lorenz_label

Array = np.ndarray

# Slices the oracle and the Kalman model can both see. Indicate / interpolate
# suffixes are stripped back to the raw masked slice: the baseline does the
# filling, so it must not be handed a pre-filled series.
_SLICE_ALIASES = {
    "only_x_interpolate": "only_x",
    "only_x_indicate": "only_x",
    "only_xy_interpolate": "only_xy",
    "only_xy_indicate": "only_xy",
    "only_xz_interpolate": "only_xz",
    "only_xz_indicate": "only_xz",
    "all_xyz_interpolate": "all_xyz",
    "all_xyz_indicate": "all_xyz",
    "xyz_to_xyz_w_noise": None,
    "xyz_to_x": None,
    "xyz_to_x_w_noise": None,
    "only_x_w_noise": None,
}


def resolve_observation_process(observation_process: str) -> str:
    if observation_process in _SLICE_ALIASES:
        mapped = _SLICE_ALIASES[observation_process]
        if mapped is None:
            raise ValueError(
                f"{observation_process} adds noise or mixes coordinates. "
                "Filtering baselines take a coordinate slice with NaNs left in "
                "place (only_x, only_xy, only_xz, all_xyz, xyz_to_xyz)."
            )
        return mapped
    return observation_process


def _mask_filename(data_dir: Path, mask_label: str, split: str) -> Path:
    # Same construction as Lorenz63.__init__, including the hard-coded folder
    # the dataset class uses for mask files.
    parts = mask_label.split("_")
    distribution = parts[0]
    rate = parts[-1]
    if distribution == "Markov":
        distribution_label = f"{distribution}_{parts[1]}"
    else:
        distribution_label = distribution
    folder = data_dir / "lorenz63" / "data" / "sigma10_rho28_beta8d3_N108k_dt0.01"
    return folder / f"mask_{distribution_label}_pnan{rate}_{split}.pkl"


def load_lorenz63_series(
    data_dir: str,
    *,
    dataset_label: Optional[str],
    mask_label: Optional[str],
    split: str,
    observation_process: str,
) -> dict:
    """Return one observed series, its missing mask, and the full xyz path."""
    from dvae.dataset.lorenz63_dataset import Lorenz63

    process = resolve_observation_process(observation_process)
    label = dataset_label or "sigma10_rho28_beta8d3_N108k_dt0.01"
    root = Path(data_dir)
    split_name = "test" if split == "test" else "train"
    data_path = root / "lorenz63" / "data" / label / f"complete_dataset_{split_name}.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Lorenz series not found: {data_path}. "
            "On Deigo the container bind is /data (host "
            "/bucket/DoyaU/stash/research-DVAE/data)."
        )
    with data_path.open("rb") as handle:
        xyz = np.asarray(pickle.load(handle), dtype=np.float64)
    if (
        mask_label
        and mask_label not in ("None", "")
        and split != "eval_mode_unused"
    ):
        mask_path = _mask_filename(root, mask_label, split_name)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Lorenz mask not found: {mask_path}")
        with mask_path.open("rb") as handle:
            mask = np.asarray(pickle.load(handle), dtype=bool)
        xyz_masked = np.where(mask, np.nan, xyz)
    else:
        xyz_masked = xyz.copy()

    holder = Lorenz63.__new__(Lorenz63)
    holder.observation_process = process
    observed = holder.apply_observation_process(xyz_masked)
    observed = np.asarray(observed, dtype=np.float64)
    if observed.ndim == 1:
        observed = observed.reshape(-1, 1)
    missing = ~np.isfinite(observed)
    params = parse_lorenz_label(label)
    return {
        "series": [observed],
        "missing": [missing],
        "xyz": xyz,
        "dt": 0.01 if params is None else params[3],
        "lorenz_params": params,
        "observation_process": process,
        "dataset_name": "Lorenz63",
        "requested_observation_process": observation_process,
    }


def load_physionet2012_stays(
    data_dir: str,
    *,
    dataset_label: Optional[str],
    split: str,
    observation_process: str,
    seq_len: int = 48,
    val_indices: float = 0.2,
) -> dict:
    """Per-stay arrays from ``PhysioNet2012``.

    Stays are not concatenated. Pass the list to EM / Tresp so a transition
    does not cross patients. ``observation_process`` should be a raw vital
    name (``raw_hr``, ``raw_vitals``), not an interpolate/indicate variant:
    those variants fill NaNs before the baseline sees them.
    """
    import inspect

    from dvae.dataset.physionet2012_dataset import (
        PHYSINET_OBS_COLUMNS,
        PhysioNet2012,
    )

    if observation_process.endswith("_interpolate") or observation_process.endswith(
        "_indicate"
    ):
        raise ValueError(
            "Load a raw PhysioNet observation (raw_hr or raw_vitals). "
            "Interpolate/indicate variants fill or expand the series inside "
            "the dataset, which hides the missingness these baselines model."
        )
    if observation_process not in PHYSINET_OBS_COLUMNS:
        raise ValueError(
            f"Unknown PhysioNet observation {observation_process!r}. "
            f"Expected one of {sorted(PHYSINET_OBS_COLUMNS)}."
        )
    n_cols = len(PHYSINET_OBS_COLUMNS[observation_process])
    candidate = {
        "data_dir": data_dir,
        "dataset_label": dataset_label,
        "mask_label": None,
        "split": split,
        "seq_len": seq_len,
        "x_dim": n_cols,
        "sample_rate": 1,
        "skip_rate": 1,
        "val_indices": val_indices,
        "observation_process": observation_process,
        "device": "cpu",
        "overlap": False,
        "shuffle": False,
    }
    params = inspect.signature(PhysioNet2012.__init__).parameters
    kwargs = {key: value for key, value in candidate.items() if key in params}
    dataset = PhysioNet2012(**kwargs)
    values = np.asarray(dataset.seq, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    stay_ids = np.asarray(dataset.stay_id_per_row)
    series: List[Array] = []
    missing: List[Array] = []
    start = 0
    while start < len(stay_ids):
        stop = start + 1
        while stop < len(stay_ids) and stay_ids[stop] == stay_ids[start]:
            stop += 1
        block = values[start:stop]
        series.append(block)
        missing.append(~np.isfinite(block))
        start = stop
    return {
        "series": series,
        "missing": missing,
        "dt": 3600.0,
        "lorenz_params": None,
        "observation_process": observation_process,
        "dataset_name": "PhysioNet2012",
        "requested_observation_process": observation_process,
    }


def load_npz_series(path: str) -> dict:
    """Generic archive: arrays ``y`` (T, D) and optional ``missing`` or NaNs.

    ``stays`` may be an object array of (T_i, D) series. ``dt`` is optional.
    """
    archive = np.load(path, allow_pickle=True)
    if "stays" in archive:
        series = [np.asarray(block, dtype=np.float64) for block in archive["stays"]]
    else:
        y = np.asarray(archive["y"], dtype=np.float64)
        series = [y if y.ndim == 2 else y.reshape(-1, 1)]
    missing = []
    for block in series:
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        missing.append(~np.isfinite(block))
    dt = float(archive["dt"]) if "dt" in archive else 0.01
    name = str(archive["dataset_name"]) if "dataset_name" in archive else "Lorenz63"
    process = (
        str(archive["observation_process"])
        if "observation_process" in archive
        else "only_x"
    )
    return {
        "series": series,
        "missing": missing,
        "dt": dt,
        "lorenz_params": None,
        "observation_process": process,
        "dataset_name": name,
        "requested_observation_process": process,
    }


def window_series(
    y: Array,
    missing: Array,
    seq_len: int,
    max_windows: int,
) -> List[Tuple[Array, Array]]:
    """Non-overlapping windows that lie inside one series."""
    y = np.asarray(y, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        missing = missing.reshape(-1, 1)
    windows = []
    for start in range(0, y.shape[0] - seq_len + 1, seq_len):
        windows.append((y[start : start + seq_len], missing[start : start + seq_len]))
        if len(windows) >= max_windows:
            break
    if not windows and y.shape[0] >= 2:
        windows.append((y, missing))
    return windows
