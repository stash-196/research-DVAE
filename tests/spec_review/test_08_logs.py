"""Spec 8. Epoch logs and the bump_events table.

The timing line, the selection line, the New best line, and one warm-up
bump line per changed knob. bump_events uses the decision epoch. The
value series stores the new value on the following epoch, so the figure
line is at epoch + 1.
"""

from __future__ import annotations

import inspect

import numpy as np

from dvae.epoch_log import (
    format_bump_line,
    format_new_best_line,
    format_selection_line,
    format_timing_line,
    make_bump_row,
    vertical_line_epochs,
)
from dvae.learning_algo import LearningAlgorithm
from dvae.training_history import warmup_change_epochs
from dvae.warmup_curriculum import WarmupCurriculum, simulate_recon_warmup


def test_log_lines_name_timing_selection_best_and_bumps():
    timing = format_timing_line(
        4,
        {
            "time_train": 0.1,
            "time_val": 0.2,
            "time_val_forward": 0.3,
            "time_spectrum": 0.4,
            "time_drift": 0.5,
            "time_rosenstein": 0.0,
            "time_plot": 0.6,
            "time_eval": 0.0,
        },
    )
    assert timing.startswith("Epoch: 4 wall_s ")
    for label in (
        "train=0.1000",
        "val=0.2000",
        "metrics_forward=0.3000",
        "spectrum=0.4000",
        "drift=0.5000",
        "rosenstein=0.0000",
        "plot=0.6000",
        "eval=0.0000",
    ):
        assert label in timing

    selection = format_selection_line(
        4,
        1.5,
        {"ftle": 0.2, "error_drift_cross": 0.4},
        {"ftle": 0.5, "error_drift_cross": 1.0},
        True,
    )
    assert selection.startswith("Selection epoch 4: ")
    assert "score=1.500000" in selection
    assert "improved=yes" in selection
    assert "ftle raw=0.200000 norm=0.500000" in selection

    best = format_new_best_line(
        4, 1.5, {"recon": 0.2}, {}
    )
    assert best.startswith("New best epoch 4: ")

    row = make_bump_row(3, "window", 10, 12, 40, 48, 41)
    line = format_bump_line(row)
    assert line.startswith("Warm-up bump epoch=3 type=window ")
    assert "old=10" in line and "new=12" in line
    assert "patience_before=40" in line
    assert "patience_after=48" in line
    assert "cpt_patience=41" in line

    source = inspect.getsource(LearningAlgorithm.train)
    for name in (
        "format_timing_line",
        "format_selection_line",
        "format_new_best_line",
        "format_bump_line",
    ):
        assert name in source


def test_bump_events_match_the_curriculum_and_the_next_epoch_line():
    curriculum = WarmupCurriculum(
        order="window_first",
        patience_start=1,
        early_stop_patience=4,
        initial_sequence_len=10,
        sequence_len=14,
        enable_autonomous=True,
        auto_start=0.0,
        auto_limit=0.2,
        enable_noise=False,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=False,
        enable_kl=False,
    )
    simulated = simulate_recon_warmup(
        [1.0] * 60,
        curriculum,
        auto=0.0,
        noise=0.0,
        kl=1.0,
        sequence_len=10,
    )
    bumps = [event for event in simulated.events if event["kind"] != "stop"]
    assert bumps
    rows = []
    seq_values = []
    auto_values = []
    auto, kl, seq = 0.0, 1.0, 10
    bump_at = {event["epoch"]: event for event in bumps}
    for epoch in range(max(event["epoch"] for event in simulated.events) + 1):
        seq_values.append(seq)
        auto_values.append(auto)
        event = bump_at.get(epoch)
        if event is None:
            continue
        if event["kind"] == "window":
            rows.append(
                make_bump_row(
                    epoch,
                    "window",
                    event["sequence_before"],
                    event["sequence_len"],
                    event["active"],
                    event["active"],
                    event["patience"],
                )
            )
            assert event["reset_warmups"] is False
            assert event["auto"] == event["auto_before"]
            assert event["kl"] == event["kl_before"]
        elif event["kind"] == "auto":
            rows.append(
                make_bump_row(
                    epoch,
                    "auto",
                    event["auto_before"],
                    event["auto"],
                    event["active"],
                    event["active"],
                    event["patience"],
                )
            )
        auto, kl, seq = event["auto"], event["kl"], event["sequence_len"]
    assert [row["epoch"] for row in rows] == [event["epoch"] for event in bumps]
    assert [row["type"] for row in rows] == [event["kind"] for event in bumps]
    assert vertical_line_epochs(rows, "window") == list(
        warmup_change_epochs(np.asarray(seq_values, dtype=float))
    )
    assert vertical_line_epochs(rows, "auto") == list(
        warmup_change_epochs(np.asarray(auto_values, dtype=float))
    )
