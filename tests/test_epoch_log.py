"""Structured epoch log lines, bump rows, and CUDA wall-clock sync.

Synthetic values only. No training loop and no GPU.
"""

from __future__ import annotations

import torch

from dvae.epoch_log import (
    format_bump_line,
    format_early_stop_line,
    format_epoch_budget_line,
    format_new_best_line,
    format_selection_line,
    format_timing_line,
    make_bump_row,
    vertical_line_epochs,
)
import numpy as np

from dvae.training_history import warmup_change_epochs
from dvae.wall_clock import WallClock, sync_cuda

import pytest


def test_selection_and_new_best_lines():
    raw = {"recon": 1.5, "spectrum": 0.3}
    norm = {"recon": 1.0, "spectrum": 0.5}
    line = format_selection_line(4, 1.5, raw, norm, True)
    assert line == (
        "Selection epoch 4: score=1.500000 improved=yes "
        "components=recon raw=1.500000 norm=1.000000, "
        "spectrum raw=0.300000 norm=0.500000"
    )
    best = format_new_best_line(4, 1.5, raw, norm)
    assert best == (
        "New best epoch 4: score=1.500000, components="
        "recon raw=1.500000 norm=1.000000, "
        "spectrum raw=0.300000 norm=0.500000"
    )
    single = format_selection_line(2, 0.25, {"recon": 0.25}, {}, False)
    assert single == (
        "Selection epoch 2: score=0.250000 improved=no "
        "components=recon raw=0.250000 norm=off"
    )


def test_bump_early_stop_and_budget_lines():
    row = make_bump_row(3, "window", 80, 100, 2, 2, 5)
    assert row == {
        "epoch": 3,
        "type": "window",
        "old": 80.0,
        "new": 100.0,
        "patience": 2.0,
        "patience_after": 2.0,
        "cpt_patience": 5,
    }
    assert format_bump_line(row) == (
        "Warm-up bump epoch=3 type=window old=80 new=100 "
        "patience_before=2 patience_after=2 cpt_patience=5"
    )
    with pytest.raises(ValueError, match="bump type"):
        make_bump_row(1, "lr", 0.1, 0.2, 2, 2, 1)
    scale = make_bump_row(1, "subband", 0.0, 0.5, 1, 2, 1)
    assert scale["type"] == "subband"
    assert format_bump_line(scale).startswith(
        "Warm-up bump epoch=1 type=subband old=0 new=0.5 "
    )
    assert vertical_line_epochs([scale], "subband") == [0, 2]
    assert vertical_line_epochs([scale], "subband") == list(
        warmup_change_epochs(np.array([0.0, 0.0, 0.5]))
    )
    assert format_early_stop_line(9, 10, 10) == (
        "Early stop epoch=9: reason=patience cpt_patience=10 "
        "active_patience=10 warmups_finished=true"
    )
    assert format_epoch_budget_line(19) == (
        "Training finished epoch=19: reason=epoch_budget"
    )


def test_timing_line_names_every_stage():
    line = format_timing_line(
        1,
        {
            "time_train": 0.4,
            "time_val": 0.2,
            "time_val_forward": 0.05,
            "time_spectrum": 0.01,
            "time_drift": 0.02,
            "time_rosenstein": 0.1,
            "time_plot": 0.3,
            "time_eval": 0.0,
        },
    )
    assert line == (
        "Epoch: 1 wall_s train=0.4000 val=0.2000 metrics_forward=0.0500 "
        "spectrum=0.0100 drift=0.0200 rosenstein=0.1000 plot=0.3000 eval=0.0000"
    )
    assert "train=" in format_timing_line(0, {})
    assert "eval=0.0000" in format_timing_line(0, {})


def test_bump_table_rebuilds_warmup_change_epochs():
    series = [80.0, 80.0, 100.0, 100.0]
    events = [make_bump_row(1, "window", 80, 100, 2, 2, 2)]
    assert list(warmup_change_epochs(series)) == [0, 2]
    assert vertical_line_epochs(events, "window") == [0, 2]
    assert vertical_line_epochs(events, "KL") == [0]


def test_sync_cuda_waits_only_for_a_cuda_device(monkeypatch):
    calls = []

    def _sync(dev=None):
        calls.append(dev)

    monkeypatch.setattr(torch.cuda, "synchronize", _sync)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sync_cuda("cpu")
    sync_cuda("cuda:0")
    sync_cuda(None)
    assert calls == []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sync_cuda("cpu")
    sync_cuda(torch.device("cpu"))
    assert calls == []
    sync_cuda("cuda:0")
    sync_cuda(torch.device("cuda:0"))
    assert len(calls) == 2
    assert all(torch.device(dev).type == "cuda" for dev in calls)

    with WallClock("cpu") as clock:
        pass
    assert clock.elapsed >= 0.0
