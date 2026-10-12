"""Spec 1. Defaults when legacy_defaults is false.

Synthetic series only. These tests call the window rules, the block-mask
function, and a Lorenz loader built from a fake pickle. They do not read
wearable recordings.
"""

from __future__ import annotations

import inspect
import pickle
import re
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dvae.dataset.dataset_builder import DatasetConfig
from dvae.dataset.gap_grid import (
    PRESENT_COL,
    dataset_missing_rate,
    filter_window_starts,
    gap_length_quantile,
    gap_split_limit,
)
from dvae.dataset.lorenz63_dataset import Lorenz63
from dvae.dataset.utils.data_utils import find_indices
from dvae.dataset.xhro_dataset import Xhro
from dvae.dataset.xhro_proper_dataset import (
    apply_block_mask,
    resolve_windowing,
)
from dvae.run_options import (
    parse_optional_int,
    parse_val_criteria,
    parse_val_drift_horizon,
    resolve_val_criteria,
)

ROOT = Path(__file__).resolve().parents[2]


def _blank_cfg() -> ConfigParser:
    cfg = ConfigParser()
    cfg.add_section("Training")
    cfg.add_section("DataFrame")
    return cfg


def _trainer_int_default(option: str) -> int:
    text = (ROOT / "src/dvae/learning_algo.py").read_text()
    match = re.search(
        rf'parse_optional_int\(\s*'
        rf'self\.cfg\.get\("Training", "{option}", fallback=None\),\s*'
        rf"(\d+),",
        text,
    )
    assert match, f"{option} default is not passed to parse_optional_int"
    return int(match.group(1))


def _xhro(seq, **flags):
    dataset = Xhro.__new__(Xhro)
    dataset.seq = torch.as_tensor(np.asarray(seq, dtype=np.float32))
    dataset.seq_len = int(flags.get("seq_len", 10))
    dataset.split = "train"
    dataset.val_indices = 0.0
    dataset.shuffle = False
    dataset.overlap = False
    dataset.split_boundaries = None
    dataset.missing_run_spans = flags.get("spans", [])
    dataset.filled_gap_spans = None
    dataset.max_gap_window_frac = flags.get("frac", 0.25)
    dataset.dataset_gap_quantile = flags.get("quantile")
    dataset.strict_non_overlap = flags.get("strict", True)
    dataset.start_on_observed = flags.get("start_on_observed", True)
    dataset.observed_per_row = flags.get("observed")
    dataset.legacy_windowing = flags.get("legacy", False)
    dataset.max_window_nan_ratio = flags.get("ratio")
    dataset.max_window_nan_ratio_off = bool(flags.get("ratio_off", False))
    dataset.max_window_nan_excess = flags.get("excess", 0.2)
    dataset.dataset_missing_rate = flags.get("rate", 0.0)
    return dataset


def _nan_dataset(window, rate, excess=0.2, ratio=None):
    class _Series:
        pass

    dataset = _Series()
    dataset.seq = np.asarray(window, dtype=np.float64)
    dataset.seq_len = int(np.asarray(window).shape[0])
    dataset.dataset_missing_rate = float(rate)
    dataset.max_window_nan_excess = excess
    dataset.max_window_nan_ratio = ratio
    dataset.max_window_nan_ratio_off = False
    dataset.legacy_windowing = False
    return dataset


def test_blank_config_selects_recon_and_spectrum_and_horizon_60():
    cfg = _blank_cfg()
    assert resolve_val_criteria(cfg) == ("recon", "spectrum")
    assert parse_val_criteria(None) == ("recon", "spectrum")
    assert parse_val_criteria("") == ("recon", "spectrum")
    assert parse_val_drift_horizon(None) == 60
    assert parse_val_drift_horizon("") == 60


def test_spectrum_windows_default_to_10_by_2000_flipping_at_1000():
    assert inspect.signature(
        __import__(
            "dvae.eval.utils.spectrum_validation",
            fromlist=["capture_fixed_spectrum_windows"],
        ).capture_fixed_spectrum_windows
    ).parameters["n_windows"].default == 10
    assert _trainer_int_default("val_spectrum_n_windows") == 10
    assert _trainer_int_default("val_spectrum_window_len") == 2000
    assert _trainer_int_default("val_spectrum_tf_steps") == 1000
    assert parse_optional_int(None, 10, "val_spectrum_n_windows") == 10
    assert parse_optional_int(None, 2000, "val_spectrum_window_len") == 2000
    assert parse_optional_int(None, 1000, "val_spectrum_tf_steps") == 1000


def test_window_rule_defaults_match_the_spec():
    resolved = resolve_windowing()
    assert resolved["legacy_defaults"] is False
    assert resolved["start_on_observed"] is True
    assert resolved["strict_non_overlap"] is True
    assert resolved["max_gap_window_frac"] == pytest.approx(0.25)
    assert resolved["gap_split_quantile"] == pytest.approx(0.99)
    assert resolved["max_window_nan_ratio"] is None
    assert resolved["max_window_nan_excess"] == pytest.approx(0.2)
    assert DatasetConfig.__dataclass_fields__["strict_non_overlap"].default is True
    assert DatasetConfig.__dataclass_fields__["start_on_observed"].default is True
    assert DatasetConfig.__dataclass_fields__["max_gap_window_frac"].default == 0.25
    assert DatasetConfig.__dataclass_fields__["gap_split_quantile"].default == 0.99
    assert DatasetConfig.__dataclass_fields__["max_window_nan_excess"].default == 0.2


def test_strict_non_overlap_tiles_and_shifts_onto_the_first_observed_sample():
    series = np.ones(33, dtype=np.float32)
    series[:3] = np.nan
    observed = np.isfinite(series)
    dataset = _xhro(
        series,
        seq_len=10,
        strict=True,
        start_on_observed=True,
        observed=observed,
        frac=None,
    )
    dataset.update_sequence_length(10)
    starts = list(dataset.data_idx)
    assert starts[0] == 3
    assert starts == [3, 13, 23]
    for start in starts:
        assert observed[start]
        assert start + 10 <= len(series)
    for left, right in zip(starts, starts[1:]):
        assert right >= left + 10


def test_gaps_shorter_than_the_split_limit_stay_inside_a_window():
    series = np.ones(40, dtype=np.float32)
    series[10:12] = np.nan
    lengths = np.array([2], dtype=np.float64)
    quantile = gap_length_quantile(lengths, 0.99)
    limit = gap_split_limit(20, 0.25, quantile)
    assert limit == pytest.approx(max(0.25 * 20, quantile))
    assert 2 <= limit
    dataset = _xhro(
        series,
        seq_len=20,
        strict=True,
        start_on_observed=True,
        frac=0.25,
        quantile=quantile,
        spans=[np.array([10, 11])],
        rate=dataset_missing_rate(series),
    )
    dataset.update_sequence_length(20)
    assert 0 in list(dataset.data_idx)
    window = dataset.seq[0:20].detach().cpu().numpy()
    assert np.isnan(window[10:12]).all()


def test_gap_splits_only_when_strictly_longer_than_fraction_and_quantile():
    # Several short runs plus one long run. The long run is above the
    # 0.99 quantile, so it splits. A lone run equals its own quantile
    # and must not split, because the comparison is strict.
    short = [np.arange(2), np.arange(4, 6), np.arange(8, 10)]
    long = np.arange(20, 35)
    lengths = np.array([2, 2, 2, 15], dtype=np.float64)
    quantile = float(np.quantile(lengths, 0.99))
    window = 20
    limit = max(0.25 * window, quantile)
    assert 15 > limit
    assert not (2 > limit)
    series = np.ones(80, dtype=np.float32)
    for span in short + [long]:
        series[span] = np.nan
    dataset = _xhro(
        series,
        seq_len=window,
        strict=True,
        start_on_observed=True,
        frac=0.25,
        quantile=quantile,
        spans=short + [long],
        rate=0.0,
        excess=1.0,
    )
    dataset.update_sequence_length(window)
    assert int(dataset.window_recut_stats["gap_splits"]) > 0
    for start in dataset.data_idx:
        stop = int(start) + int(dataset.seq_len)
        assert not (start < int(long[0]) < stop and stop > int(long[-1]))

    # One missing run equals its own 0.99 quantile, so gap > max(...) is false.
    lone = np.ones(80, dtype=np.float32)
    lone[20:50] = np.nan
    lone_q = float(np.quantile(np.array([30.0]), 0.99))
    assert 30 <= max(0.25 * 20, lone_q)
    lone_ds = _xhro(
        lone,
        seq_len=20,
        strict=True,
        start_on_observed=True,
        frac=0.25,
        quantile=lone_q,
        spans=[np.arange(20, 50)],
        rate=0.0,
        excess=1.0,
    )
    lone_ds.update_sequence_length(20)
    assert int(lone_ds.window_recut_stats["gap_splits"]) == 0


def test_relative_nan_filter_drops_only_above_the_cap_and_a_ratio_overrides_it():
    rate = 0.10
    excess = 0.20
    threshold = rate + excess
    clear_keep = np.zeros((100, 1))
    clear_keep[:25] = np.nan
    clear_drop = np.zeros((100, 1))
    clear_drop[:40] = np.nan
    kept = filter_window_starts(_nan_dataset(clear_keep, rate, excess), [0])
    dropped = filter_window_starts(_nan_dataset(clear_drop, rate, excess), [0])
    assert float(np.isnan(clear_keep).mean()) < threshold
    assert float(np.isnan(clear_drop).mean()) > threshold
    assert list(kept) == [0]
    assert list(dropped) == []

    overridden = filter_window_starts(
        _nan_dataset(clear_drop, rate, excess, ratio=0.50),
        [0],
    )
    assert list(overridden) == [0]


def test_window_just_above_the_nan_cap_is_dropped():
    """A fraction strictly above the cap is dropped, including near a float16 boundary.

    4001/8000 = 0.500125 is above missing-rate 0.3 plus excess 0.2.
    """
    window = np.zeros((8000, 1))
    window[:4001] = np.nan
    true_fraction = float(np.isnan(window).mean())
    threshold = 0.3 + 0.2
    assert true_fraction > threshold
    kept = filter_window_starts(
        _nan_dataset(window, rate=0.3, excess=0.2),
        [0],
    )
    assert list(kept) == []


def test_window_at_the_nan_cap_is_kept():
    """Drop only when the NaN fraction is strictly above the cap.

    30/100 is exactly missing-rate 0.1 plus excess 0.2. The spec keeps it.
    """
    window = np.zeros((100, 1))
    window[:30] = np.nan
    true_fraction = float(np.isnan(window).mean())
    threshold = 0.1 + 0.2
    assert true_fraction == pytest.approx(threshold)
    kept = filter_window_starts(_nan_dataset(window, rate=0.1, excess=0.2), [0])
    assert list(kept) == [0]


def test_block_mask_not_good_writes_nan_without_resampling_or_cutting_time():
    n = 20
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="100ms"),
            "ch": np.arange(n, dtype=np.float64),
        }
    )
    original = frame.copy()
    path = ROOT / "tests/spec_review/_not_used.npz"
    # The npz lives next to the test so it is obvious synthetic data.
    path = Path(__file__).resolve().parent / "_block_mask_synthetic.npz"
    # good[0] is clean even though dirty[0] is set. not_good must ignore dirty.
    np.savez(
        path,
        t0=np.array([0.0, 1.0]),
        dirty=np.array([True, False]),
        lowamp=np.array([False, False]),
        good=np.array([True, False]),
    )
    try:
        masked = apply_block_mask(frame, path, None)
    finally:
        path.unlink(missing_ok=True)
    assert len(masked) == len(original)
    pd.testing.assert_series_equal(masked["datetime"], original["datetime"])
    np.testing.assert_array_equal(masked["ch"].to_numpy()[:10], np.arange(10))
    assert np.isnan(masked["ch"].to_numpy()[10:]).all()
    assert not bool(masked[PRESENT_COL].to_numpy()[10:].any())
    assert bool(masked[PRESENT_COL].to_numpy()[:10].all())


def _synthetic_lorenz(tmp_path, **window_kwargs):
    series = np.stack(
        [np.linspace(0.0, 1.0, 40), np.linspace(1.0, 2.0, 40), np.linspace(2.0, 3.0, 40)],
        axis=1,
    )
    series[:3, 0] = np.nan
    label = "synthetic_lorenz"
    folder = tmp_path / "lorenz63" / "data" / label
    folder.mkdir(parents=True)
    with (folder / "complete_dataset_train.pkl").open("wb") as handle:
        pickle.dump(series, handle)
    return Lorenz63(
        data_dir=str(tmp_path),
        dataset_label=label,
        mask_label=None,
        split="train",
        seq_len=10,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="only_x",
        device="cpu",
        overlap=False,
        with_nan=True,
        shuffle=False,
        eval_mode=True,
        **window_kwargs,
    )


def test_lorenz_starts_move_forward_to_the_first_observed_sample(tmp_path):
    dataset = _synthetic_lorenz(
        tmp_path,
        start_on_observed=True,
        strict_non_overlap=True,
        max_window_nan_excess=0.2,
        max_gap_window_frac=0.25,
    )
    values = np.asarray(dataset.seq, dtype=np.float64).reshape(-1)
    first_observed = int(np.flatnonzero(np.isfinite(values))[0])
    assert first_observed == 3
    assert int(dataset.data_idx[0]) == first_observed


def test_lorenz_drops_a_window_above_the_relative_nan_cap(tmp_path):
    dataset = _synthetic_lorenz(tmp_path, strict_non_overlap=True)
    values = np.asarray(dataset.seq, dtype=np.float64).reshape(-1)
    rate = float(np.isnan(values).mean())
    threshold = rate + 0.2
    offenders = []
    for start in dataset.data_idx:
        window = values[int(start) : int(start) + int(dataset.seq_len)]
        if float(np.isnan(window).mean()) > threshold:
            offenders.append(int(start))
    assert offenders == []


def test_compute_emd_rosenstein_defaults_to_true():
    template = (ROOT / "config/general_signal/cfg_base_template.ini").read_text()
    assert "compute_emd_rosenstein" in template
    from dvae.eval.utils.free_run_metrics import resolve_compute_emd_rosenstein

    assert resolve_compute_emd_rosenstein(None, None) is True
