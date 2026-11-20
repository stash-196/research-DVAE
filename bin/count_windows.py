#!/usr/bin/env python3
"""Count train and validation windows the way training builds them.

Prints counts and durations only. Sample values are never written.
Loader logs go to stderr.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import-time path setup prints to stdout. Counts are the only stdout
# this script is allowed to emit; those notes belong on stderr.
_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    import numpy as np

    from dvae.dataset.dataset_builder import build_dataloader
    from dvae.dataset.xhro_proper_dataset import training_initial_sequence_len
    from dvae.eval.utils.local_drift_analysis import enforce_drift_span
    from dvae.eval.utils.spectrum_validation import (
        capture_contiguous_val_windows,
        capture_fixed_spectrum_windows,
    )
    from dvae.learning_algo import training_dataset_config
    from dvae.run_options import (
        DEFAULT_TF_MAX_GAP_FRAC,
        DEFAULT_TF_MIN_OBSERVED,
        legacy_defaults_from_config,
        parse_optional_int,
        parse_unit_interval,
        parse_val_drift_horizon,
        resolve_val_criteria,
        spectrum_monitoring_enabled,
    )
    from dvae.utils import load_device_paths, myconf
    from dvae.visualizers.spectrum_comparison import dataset_sampling_freq_hz
finally:
    sys.stdout = _stdout


def _seq_array(dataset) -> np.ndarray:
    seq = dataset.seq
    if hasattr(seq, "detach"):
        seq = seq.detach().cpu().numpy()
    arr = np.asarray(seq)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _finite_row_count(dataset, role: str) -> int:
    """Finite rows. Contiguous val does not count the shared head twice."""
    arr = _seq_array(dataset)
    if arr.size == 0:
        return 0
    finite = np.isfinite(arr).any(axis=-1)
    region = getattr(dataset, "_val_region_id", None)
    if region is not None and getattr(dataset, "split", None) != "test":
        region = np.asarray(region).reshape(-1)
        if region.shape[0] == finite.shape[0]:
            if role == "train":
                finite = finite & ((region % 2) == 0)
            elif role == "val":
                finite = finite & ((region % 2) == 1)
    return int(np.count_nonzero(finite))


def _finite_seconds(dataset, role: str):
    freq = dataset_sampling_freq_hz(dataset)
    if freq is None:
        return None
    return _finite_row_count(dataset, role) / float(freq)


def _missing_fraction(dataset) -> float | None:
    rate = getattr(dataset, "dataset_missing_rate", None)
    if rate is not None:
        try:
            value = float(rate)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            return value
    arr = _seq_array(dataset)
    if arr.size == 0:
        return None
    return float(np.mean(~np.isfinite(arr)))


def _spectrum_window_count(cfg, dataset):
    """How many fixed validation windows training would copy. None if off."""
    criteria = resolve_val_criteria(cfg)
    if not spectrum_monitoring_enabled(
        criteria, legacy=legacy_defaults_from_config(cfg)
    ):
        return None
    n_windows = parse_optional_int(
        cfg.get("Training", "val_spectrum_n_windows", fallback=None),
        10,
        "val_spectrum_n_windows",
    )
    window_len = parse_optional_int(
        cfg.get("Training", "val_spectrum_window_len", fallback=None),
        2000,
        "val_spectrum_window_len",
    )
    tf_steps = parse_optional_int(
        cfg.get("Training", "val_spectrum_tf_steps", fallback=None),
        1000,
        "val_spectrum_tf_steps",
    )
    if tf_steps < 0 or tf_steps >= window_len:
        raise ValueError(
            "val_spectrum_tf_steps must be in [0, val_spectrum_window_len), "
            f"got tf_steps={tf_steps}, window_len={window_len}."
        )
    horizon = parse_val_drift_horizon(
        cfg.get("Training", "val_drift_horizon", fallback=None)
    )
    enforce_drift_span(window_len, tf_steps, horizon)
    min_observed = parse_unit_interval(
        cfg.get("Training", "val_window_tf_min_observed", fallback=None),
        DEFAULT_TF_MIN_OBSERVED,
        "val_window_tf_min_observed",
    )
    max_gap_frac = parse_unit_interval(
        cfg.get("Training", "val_window_tf_max_gap_frac", fallback=None),
        DEFAULT_TF_MAX_GAP_FRAC,
        "val_window_tf_max_gap_frac",
    )
    contiguous = bool(getattr(dataset, "val_stretches", None))
    if contiguous:
        max_gap_steps = parse_optional_int(
            cfg.get("Training", "val_window_max_gap_steps", fallback=None),
            10,
            "val_window_max_gap_steps",
        )
        stride = parse_optional_int(
            cfg.get("Training", "val_window_stride", fallback=None),
            max(1, window_len // 2),
            "val_window_stride",
        )
        if tf_steps < 1:
            raise ValueError(
                "contiguous_val_split needs val_spectrum_tf_steps >= 1."
            )
        windows, _info = capture_contiguous_val_windows(
            dataset,
            n_windows=n_windows,
            window_len=window_len,
            tf_steps=tf_steps,
            min_observed=min_observed,
            max_gap_steps=max_gap_steps,
            stride=stride,
        )
    else:
        windows, _info = capture_fixed_spectrum_windows(
            dataset,
            n_windows=n_windows,
            window_len=window_len,
            tf_steps=tf_steps,
            min_observed=min_observed,
            max_gap_frac=max_gap_frac,
            return_info=True,
        )
    for window in windows:
        enforce_drift_span(np.asarray(window).shape[0], tf_steps, horizon)
    return int(len(windows))


def _format_seconds(value) -> str:
    if value is None or not np.isfinite(value):
        return "na"
    return f"{float(value):.6f}"


def _format_fraction(value) -> str:
    if value is None or not np.isfinite(value):
        return "na"
    return f"{float(value):.6f}"


def count_report(cfg_path: str, *, seq_len: int | None, data_dir: str | None) -> list[str]:
    """Build the training datasets and return the count lines."""
    cfg = myconf()
    cfg.read(cfg_path)
    if data_dir is None:
        device = load_device_paths(str(_ROOT / "config" / "device_paths.yaml"))
        data_dir = device["data_dir"]
    initial = training_initial_sequence_len(cfg)
    config = training_dataset_config(
        cfg,
        data_dir=data_dir,
        device="cpu",
        batch_size=cfg.getint("DataFrame", "batch_size"),
        shuffle=cfg.getboolean("DataFrame", "shuffle"),
        num_workers=cfg.getint("DataFrame", "num_workers"),
        dataset_label=cfg.get("DataFrame", "dataset_label", fallback="None"),
        mask_label=cfg.get("DataFrame", "mask_label", fallback="None"),
        seq_len=initial,
    )
    name = cfg.get("DataFrame", "dataset_name")
    train_loader, val_loader, _train_n, _val_n = build_dataloader(
        name, config, "train"
    )
    test_loader = build_dataloader(name, config, "test")
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    spectrum = _spectrum_window_count(cfg, val_ds)
    if seq_len is not None:
        for dataset in (train_ds, val_ds, test_ds):
            dataset.update_sequence_length(int(seq_len))
    spectrum_text = "na" if spectrum is None else str(int(spectrum))
    return [
        f"n_train_windows={len(train_ds)}",
        f"n_val_windows={len(val_ds)}",
        f"n_val_spectrum_windows={spectrum_text}",
        f"train_finite_sec={_format_seconds(_finite_seconds(train_ds, 'train'))}",
        f"val_finite_sec={_format_seconds(_finite_seconds(val_ds, 'val'))}",
        f"test_finite_sec={_format_seconds(_finite_seconds(test_ds, 'test'))}",
        f"missing_fraction={_format_fraction(_missing_fraction(train_ds))}",
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ini", help="Training config file")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Window length after the training initial-length construction",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the device-paths data_dir (tests and local trees)",
    )
    args = parser.parse_args(argv)
    saved = sys.stdout
    sys.stdout = sys.stderr
    try:
        lines = count_report(args.ini, seq_len=args.seq_len, data_dir=args.data_dir)
    finally:
        sys.stdout = saved
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
