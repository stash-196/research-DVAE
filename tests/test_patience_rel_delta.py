"""Relative patience threshold. Rel 0 is the absolute 1e-4 floor."""

from __future__ import annotations

import configparser

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dvae.visualizers.visualize_training_metrics import visualize_combined_metrics
from dvae.warmup_curriculum import (
    ABS_IMPROVEMENT_FLOOR,
    WarmupCurriculum,
    improvement_threshold,
    patience_rel_delta_from_config,
    simulate_recon_warmup,
)


def _cfg(text: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


def _curriculum():
    return WarmupCurriculum(
        order="auto_first",
        patience_start=None,
        early_stop_patience=100,
        initial_sequence_len=100,
        sequence_len=100,
        enable_autonomous=False,
        auto_start=1.0,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=1.0,
        noise_limit=1.0,
        noise_reset_on_window=False,
        enable_kl=False,
    )


def _replay(losses, **kwargs):
    return simulate_recon_warmup(
        losses,
        _curriculum(),
        auto=1.0,
        noise=1.0,
        kl=1.0,
        sequence_len=100,
        **kwargs,
    )


def test_rel_zero_matches_the_absolute_floor():
    assert improvement_threshold(10.0, 0.0) == ABS_IMPROVEMENT_FLOOR
    assert improvement_threshold(10.0, 0.0, abs_floor=0.0) == 0.0
    assert improvement_threshold(float("inf"), 1e-3) == ABS_IMPROVEMENT_FLOOR
    # Below the floor, the relative term does not lower the threshold.
    assert improvement_threshold(1e-6, 1e-3) == ABS_IMPROVEMENT_FLOOR
    losses = [10.0, 9.995, 9.98, 10.0, 9.0]
    plain = _replay(losses)
    rel0 = _replay(losses, patience_rel_delta=0.0)
    assert rel0.events == plain.events
    assert rel0.improved == plain.improved
    assert rel0.stopped is False


def test_rel_1e_3_needs_more_than_one_percent_of_best():
    # best=10, rel=1e-3 → threshold 1e-2. A drop of 0.005 does not count.
    assert improvement_threshold(10.0, 1e-3) == pytest.approx(1e-2)
    losses = [10.0, 9.995, 9.98, 10.0, 9.0]
    sim = _replay(losses, patience_rel_delta=1e-3)
    assert sim.improved == [True, False, True, False, True]
    assert sim.stopped is False
    # Same decision the training loop makes: threshold from the best
    # after any warm-up reset, then delta < -threshold.
    best = float("inf")
    flags = []
    for loss in losses:
        thr = improvement_threshold(best, 1e-3)
        delta = loss - best
        improved = delta < -thr
        flags.append(improved)
        if improved:
            best = loss
    assert sim.improved == flags


def test_config_without_the_key_blank_and_none_are_zero():
    assert patience_rel_delta_from_config(_cfg("[Training]\nepochs = 1\n")) == 0.0
    assert (
        patience_rel_delta_from_config(
            _cfg("[Training]\npatience_rel_delta =\n")
        )
        == 0.0
    )
    assert (
        patience_rel_delta_from_config(
            _cfg("[Training]\npatience_rel_delta = None\n")
        )
        == 0.0
    )
    assert (
        patience_rel_delta_from_config(
            _cfg("[Training]\npatience_rel_delta = null\n")
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="patience_rel_delta"):
        patience_rel_delta_from_config(
            _cfg("[Training]\npatience_rel_delta = -0.1\n")
        )


def test_varying_threshold_splits_the_delta_panel(tmp_path, monkeypatch):
    captured = {}
    real_savefig = plt.savefig

    def _save(path, *args, **kwargs):
        ax = plt.gcf().axes[0]
        captured["labels"] = [text.get_text() for text in ax.get_legend().get_texts()]
        captured["lines"] = [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
        return real_savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", _save)
    n = 3
    delta = np.array([0.0, 0.2, 0.05])
    thr = np.array([0.1, 0.1, 0.01])
    visualize_combined_metrics(
        delta,
        1e-4,
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.full(n, 10),
        np.zeros(n),
        np.zeros(n),
        "PLRNN",
        "ss",
        "none",
        str(tmp_path),
        "rel",
        improvement_threshold_epochs=thr,
    )
    assert any("per-epoch threshold" in label for label in captured["labels"])
    below, above = captured["lines"][0], captured["lines"][1]
    # 0 < 0.1, 0.2 >= 0.1, 0.05 >= 0.01.
    assert np.isnan(below[1]) and np.isnan(below[2])
    assert below[0] == pytest.approx(0.0)
    assert np.isnan(above[0])
    assert above[1] == pytest.approx(0.2)
    assert above[2] == pytest.approx(0.05)


def test_floor_series_keeps_the_scalar_legend(tmp_path, monkeypatch):
    captured = {}
    real_savefig = plt.savefig

    def _save(path, *args, **kwargs):
        ax = plt.gcf().axes[0]
        captured["labels"] = [text.get_text() for text in ax.get_legend().get_texts()]
        return real_savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", _save)
    n = 3
    visualize_combined_metrics(
        np.zeros(n),
        1e-4,
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.zeros(n),
        np.array([0]),
        np.full(n, 10),
        np.zeros(n),
        np.zeros(n),
        "PLRNN",
        "ss",
        "none",
        str(tmp_path),
        "floor",
        improvement_threshold_epochs=np.full(n, 1e-4),
    )
    joined = " ".join(captured["labels"])
    assert "per-epoch" not in joined
    assert "0.0001" in joined
