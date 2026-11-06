"""Contiguous validation stretch and fixed val windows. Synthetic data only."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dvae.dataset.xhro_dataset import Xhro
from dvae.dataset.xhro_proper_dataset import XhroProper
from dvae.eval.utils.local_drift_analysis import fixed_window_recon
from dvae.eval.utils.spectrum_validation import (
    capture_contiguous_val_windows,
    free_run_spectrum_error,
    select_contiguous_val_windows,
)
from tests.test_xhro_proper_block_mask import _write_stage


def _session(root: Path, n: int, fs: float = 100.0, ch1=None) -> None:
    sess = root / "xhro_packet_loss" / "grok_output" / "REC1_realtime"
    extra = {} if ch1 is None else {"ch1": ch1}
    _write_stage(sess / "intermediates" / "stage05_ecg_filtered.npz", n, fs, **extra)


def _load(root: Path, **overrides) -> XhroProper:
    params = dict(
        data_dir=str(root),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=20,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.2,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=True,
    )
    params.update(overrides)
    return XhroProper(**params)


# ---------------------------------------------------------------- loader split


def test_contiguous_split_train_val_test_rows(tmp_path):
    n = 1000
    _session(tmp_path, n)
    train = _load(tmp_path, split="train", contiguous_val_split=True)
    valid = _load(tmp_path, split="valid", contiguous_val_split=True)
    test = _load(tmp_path, split="test", contiguous_val_split=True)
    head = len(np.asarray(train.seq))
    assert head == n - n // 5 == 800
    cut = head - head // 5
    assert cut == 640  # 0.64 n
    assert valid.val_stretches == [(640, 800, 0)]
    length = int(train.seq_len)
    tr = np.asarray(train.data_idx)
    va = np.asarray(valid.data_idx)
    assert tr.size and va.size
    assert (tr + length <= cut).all()
    assert (va >= cut).all() and (va + length <= head).all()
    # The stretch is contiguous and every non-overlapping window in it is used.
    assert sorted(va.tolist()) == list(range(cut, head - length + 1, length))
    # Test split untouched: same rows as with the option off.
    test_off = _load(tmp_path, split="test")
    assert test.val_stretches is None
    np.testing.assert_array_equal(np.asarray(test.seq), np.asarray(test_off.seq))
    assert test.data_idx == test_off.data_idx


def test_option_off_reproduces_shuffled_draw(tmp_path):
    _session(tmp_path, 1000)
    off_tr = _load(tmp_path, split="train")
    off_va = _load(tmp_path, split="valid")
    assert off_va.val_stretches is None
    starts = np.array(sorted(off_tr.data_idx + off_va.data_idx), dtype=np.int64)
    a, b = Xhro.split_dataset(off_tr, np.array(starts, copy=True), 0.2)
    assert sorted(a.tolist()) == sorted(off_tr.data_idx)
    assert sorted(b.tolist()) == sorted(off_va.data_idx)
    # The shuffled draw puts validation windows before 0.64 n; the new one cannot.
    assert min(off_va.data_idx) < 640


def test_curriculum_recut_keeps_split(tmp_path):
    _session(tmp_path, 1000)
    valid = _load(tmp_path, split="valid", contiguous_val_split=True)
    train = _load(tmp_path, split="train", contiguous_val_split=True)
    for length in (8, 50, 100):
        valid.update_sequence_length(length)
        train.update_sequence_length(length)
        assert valid.data_idx and train.data_idx
        assert all(s >= 640 and s + length <= 800 for s in valid.data_idx)
        assert all(s + length <= 640 for s in train.data_idx)


def test_split_dataset_drops_windows_spanning_the_cut():
    ds = object.__new__(XhroProper)
    ds.seq_len = 10
    region = np.zeros(100, dtype=np.int64)
    region[64:80] = 1
    ds._val_region_id = region
    tr, va = XhroProper.split_dataset(ds, np.array([0, 54, 60, 64, 70, 71]), 0.2)
    assert tr.tolist() == [0, 54]
    assert va.tolist() == [64, 70]


def test_per_segment_stretches_for_segment_loads():
    ds = object.__new__(XhroProper)
    ds.contiguous_val_split = True
    ds.split = "valid"
    ds.coarse_mode = False
    ds.seq = np.zeros((800, 1))
    ds.waveform_segments = [
        {"row_lo": 0, "row_hi": 500},
        {"row_lo": None, "row_hi": None},
        {"row_lo": 500, "row_hi": 800},
    ]
    ds.split_boundaries = np.array([500])
    XhroProper._setup_contiguous_val_split(ds)
    assert ds.val_stretches == [(400, 500, 0), (740, 800, 2)]
    assert ds.split_boundaries.tolist() == [400, 500, 740]


def test_coarse_mode_rejects_option():
    ds = object.__new__(XhroProper)
    ds.contiguous_val_split = True
    ds.split = "train"
    ds.coarse_mode = True
    ds.seq = np.zeros((10, 1))
    with pytest.raises(ValueError, match="coarse"):
        XhroProper._setup_contiguous_val_split(ds)


# ------------------------------------------------------------ window selection


def _series(n=400, nan_rows=()):
    x = np.sin(np.arange(n) / 5.0).reshape(-1, 1)
    x = np.repeat(x, 2, axis=1)
    for a, b in nan_rows:
        x[a:b] = np.nan
    return x


def test_both_halves_checked_and_refill_to_n():
    # Stretch [100, 400): window 40, tf 20, stride 20 -> 14 candidates.
    series = _series(nan_rows=[(130, 145)])
    rec, counts = select_contiguous_val_windows(
        series, [(100, 400, 0)], 8, 40, 20, max_gap_steps=10, seed=3
    )
    assert counts["n_candidates"] == 14
    starts = [r["start"] for r in rec]
    assert len(starts) == 8
    for s in starts:
        assert 100 <= s and s + 40 <= 400
        assert not (s < 145 and s + 40 > 130)
        assert np.isfinite(series[s : s + 40]).all()
    assert all(r["qualified"] for r in rec)
    assert starts == sorted(starts)
    assert 100 not in starts  # TF half clean, free-run half has the gap
    rec2, _ = select_contiguous_val_windows(
        series, [(100, 400, 0)], 8, 40, 20, max_gap_steps=10, seed=3
    )
    assert [r["start"] for r in rec2] == starts


def test_free_run_half_alone_fails_window():
    series = _series(nan_rows=[(125, 140)])  # only the FR half of start 100
    rec, c = select_contiguous_val_windows(
        series, [(100, 140, 0), (200, 240, 1)], 2, 40, 20, max_gap_steps=10
    )
    assert [r["start"] for r in rec] == [200]
    assert c["n_passing"] == 1


def test_fewer_than_n_uses_only_passing_and_warns():
    series = _series(nan_rows=[(100, 330)])
    with pytest.warns(UserWarning, match="only"):
        rec, counts = select_contiguous_val_windows(
            series, [(100, 400, 0)], 10, 40, 20, max_gap_steps=10
        )
    assert counts["n_passing"] == len(rec) < 10
    assert all(np.isfinite(series[r["start"] : r["start"] + 40]).all() for r in rec)


def test_short_gap_within_limit_passes_long_gap_fails():
    ok = _series(nan_rows=[(125, 126)])
    rec, c = select_contiguous_val_windows(ok, [(100, 140, 0)], 1, 40, 20, max_gap_steps=10)
    assert [r["start"] for r in rec] == [100]
    bad = _series(nan_rows=[(122, 134)])
    with pytest.raises(ValueError, match="No validation window"):
        select_contiguous_val_windows(bad, [(100, 140, 0)], 1, 40, 20, max_gap_steps=10)


def test_capture_reports_segment_ids_and_counts():
    class D:
        pass

    d = D()
    d.seq = torch.tensor(_series(600), dtype=torch.float32)
    d.val_stretches = [(100, 200, 0), (400, 520, 3)]
    windows, info = capture_contiguous_val_windows(d, 6, 40, 20, stride=20)
    assert info["mode"] == "contiguous"
    assert info["n_candidates"] == 4 + 5
    assert info["n_used"] == len(windows) == 6
    for s, sid in zip(info["starts"], info["segment_ids"]):
        lo, hi = (100, 200) if sid == 0 else (400, 520)
        assert lo <= s and s + 40 <= hi
    assert info["relaxed"] is False


# ------------------------------------------------------------- metric pieces


def test_strict_spectrum_is_nan_instead_of_subset():
    rng = np.random.RandomState(0)
    gt = [rng.randn(400, 1) for _ in range(3)]
    pred = [rng.randn(400, 1) for _ in range(3)]
    pred[1][200:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = free_run_spectrum_error(pred, gt, 200)
        strict, skipped = free_run_spectrum_error(
            pred, gt, 200, return_skipped=True, strict=True
        )
    assert np.isfinite(loose)
    assert np.isnan(strict) and skipped == 1


def test_fixed_window_recon_ignores_missing_truth():
    truth = [np.array([[1.0], [np.nan], [3.0]]), np.array([[0.0], [0.0], [0.0]])]
    recon = [np.array([[2.0], [100.0], [3.0]]), np.array([[1.0], [1.0], [1.0]])]
    assert fixed_window_recon(truth, recon) == pytest.approx(0.8)


def test_selection_recon_uses_fixed_window_value():
    from dvae.learning_algo import LearningAlgorithm

    algo = object.__new__(LearningAlgorithm)
    algo._contiguous_val = True
    algo._val_fixed_recon = np.array([0.5, 0.25])
    loader = np.array([9.0, 9.0])
    assert algo._selection_recon(1, loader) == 0.25
    algo._contiguous_val = False
    assert algo._selection_recon(1, loader) == 9.0


# ------------------------------------------------------------------ end to end


def test_train_contiguous_val_end_to_end(tmp_path):
    from dvae.learning_algo import LearningAlgorithm
    from tests.test_train_smoke import _base_config

    n = 2000
    ch1 = np.sin(np.arange(n) / 7.0) + 0.1 * np.cos(np.arange(n) / 3.0)
    ch1[1330:1345] = np.nan
    data_root = tmp_path / "data"
    _session(data_root, n, ch1=ch1)
    text = _base_config(
        dataset_name="XhroProper",
        dataset_label="REC1",
        mask_label="realtime",
        observation_process="raw_ch1",
        val_indices="0.2",
        val_spectrum_n_windows="6",
        val_spectrum_window_len="40",
        val_spectrum_tf_steps="20",
        val_drift_horizon="5",
        epochs="3",
    )
    text = text.replace(
        "[DataFrame]\n", "[DataFrame]\ncontiguous_val_split = true\n"
    ).replace(
        "[Training]\n",
        "[Training]\nval_window_stride = 20\nval_window_max_gap_steps = 10\n",
    )
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(text)
    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "contig",
            "reload": False,
            "device_config": {
                "saved_root": str(tmp_path / "saved"),
                "data_dir": str(data_root),
                "device_name": "cpu",
            },
        }
    )
    algo.build_model()
    algo.train()
    run = list((tmp_path / "saved").rglob("loss_model.pckl"))
    assert len(run) == 1
    with open(run[0], "rb") as handle:
        h = pickle.load(handle)
    head = n - n // 5
    cut = head - head // 5
    assert h["val_window_mode"] == "contiguous"
    assert h["val_window_n_used"] == 6 == len(h["val_window_starts"])
    assert h["val_window_len"] == 40 and h["val_window_stride"] == 20
    assert h["val_window_segment_ids"].tolist() == [0] * 6
    assert h["val_stretches"].tolist() == [[cut, head, 0]]
    for s in h["val_window_starts"]:
        assert cut <= s and s + 40 <= head
        assert not (s < 1345 and s + 40 > 1330)
    recon = np.asarray(h["val_fixed_recon"], dtype=float)
    assert recon.size >= 1 and np.isfinite(recon).all()
    assert np.isfinite(np.asarray(h["val_spectrum"], dtype=float)).all()
    assert np.isfinite(np.asarray(h["val_loss"], dtype=float)).all()
    assert algo._metric_reference["recon"] == pytest.approx(recon[0])
