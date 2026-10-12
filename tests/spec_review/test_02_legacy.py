"""Spec 2. legacy_defaults / legacy_windowing restore the base-branch behaviour.

The base branch is origin/master. Window starts are compared on the same
synthetic series. The warm-up comparison replays the master bump loop
(auto, then KL, then a window bump that resets auto and KL) against
WarmupCurriculum(order auto_first) and a single early_stop_patience.
"""

from __future__ import annotations

import subprocess
from configparser import ConfigParser

import numpy as np
import pytest

from dvae.dataset.utils.data_utils import find_indices
from dvae.dataset.xhro_proper_dataset import resolve_windowing
from dvae.run_options import parse_val_criteria, resolve_val_criteria
from dvae.warmup_curriculum import WarmupCurriculum, simulate_recon_warmup


def _master_find_indices():
    source = subprocess.check_output(
        ["git", "show", "origin/master:src/dvae/dataset/utils/data_utils.py"],
        text=True,
    )
    # find_indices calls helpers defined above it. Slice from the first
    # of those so exec on origin/master does not miss _boundary_cuts.
    start = source.index("def _boundary_cuts(")
    end = source.index("\nif __name__", start)
    namespace = {"np": np}
    exec(source[start:end], namespace)
    return namespace["find_indices"]


def _master_warmup(
    losses,
    *,
    patience,
    auto_start,
    auto_limit,
    seq0,
    seq_target,
    enable_kl=True,
):
    """Control flow of origin/master learning_algo warm-up, on a loss list.

    Priority is autonomous, then KL, then sequence length. A sequence bump
    resets KL and the autonomous fraction. Patience is one fixed integer.
    """
    best = float("inf")
    cpt = 0
    warm = False
    auto = float(auto_start)
    kl = 0.0 if enable_kl else 1.0
    seq = int(seq0)
    bump = 0.1
    events = []
    for epoch, loss in enumerate(losses):
        loss = float(loss)
        if warm:
            best = loss
            warm = False
        delta = loss - best
        if delta < -1e-4:
            cpt = 0
            best = loss
        else:
            cpt += 1
        if cpt > patience:
            auto_ready = auto < auto_limit
            kl_ready = enable_kl and kl < 1.0
            seq_ready = seq < seq_target
            if auto_ready or kl_ready or seq_ready:
                cpt = 0
                warm = True
                if auto_ready:
                    auto = min(auto_limit, auto + bump * auto_limit)
                    kind = "auto"
                elif kl_ready:
                    kl = kl + bump
                    kind = "kl"
                else:
                    seq = min(seq_target, seq + int(bump * seq_target))
                    if enable_kl:
                        kl = 0.0
                    auto = float(auto_start)
                    kind = "window"
                events.append(
                    {
                        "epoch": epoch,
                        "kind": kind,
                        "auto": float(auto),
                        "kl": float(kl),
                        "sequence_len": int(seq),
                    }
                )
            else:
                events.append({"epoch": epoch, "kind": "stop"})
                break
    return events


def test_legacy_switches_restore_old_windowing_and_recon_only_selection():
    for key in ("legacy_defaults", "legacy_windowing"):
        resolved = resolve_windowing(**{key: True})
        assert resolved["start_on_observed"] is False
        assert resolved["strict_non_overlap"] is False
        assert resolved["max_gap_window_frac"] is None
        assert resolved["max_window_nan_excess"] is None
        assert resolved["max_window_nan_ratio"] is None
        assert parse_val_criteria(None, legacy_windowing=True) == ("recon",)
        assert parse_val_criteria("", legacy_windowing=True) == ("recon",)

    cfg = ConfigParser()
    cfg.add_section("Training")
    cfg.add_section("DataFrame")
    cfg.set("DataFrame", "legacy_defaults", "true")
    assert resolve_val_criteria(cfg) == ("recon",)
    cfg.remove_option("DataFrame", "legacy_defaults")
    cfg.set("DataFrame", "legacy_windowing", "true")
    assert resolve_val_criteria(cfg) == ("recon",)


def test_legacy_find_indices_matches_origin_master_on_synthetic_series():
    master = _master_find_indices()
    cases = ((40, 10), (80, 16), (150, 50), (200, 25), (100, 10))
    for num_frames, seq_len in cases:
        num_indices = num_frames // seq_len
        legacy = np.asarray(
            find_indices(
                num_frames,
                seq_len,
                num_indices,
                strict_non_overlap=False,
            )
        )
        old = np.asarray(master(num_frames, seq_len, num_indices))
        np.testing.assert_array_equal(legacy, old)
        strict = np.asarray(
            find_indices(
                num_frames,
                seq_len,
                num_indices,
                strict_non_overlap=True,
            )
        )
        if num_frames != 100:
            assert not np.array_equal(strict, old)


def test_legacy_warmup_matches_the_master_bump_loop():
    losses = [1.0] * 150
    patience = 2
    curriculum = WarmupCurriculum(
        order="auto_first",
        patience_start=None,
        early_stop_patience=patience,
        initial_sequence_len=36,
        sequence_len=40,
        enable_autonomous=True,
        auto_start=0.8,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=True,
        enable_kl=True,
    )
    simulated = simulate_recon_warmup(
        losses,
        curriculum,
        auto=0.8,
        noise=0.0,
        kl=0.0,
        sequence_len=36,
    )
    master = _master_warmup(
        losses,
        patience=patience,
        auto_start=0.8,
        auto_limit=1.0,
        seq0=36,
        seq_target=40,
        enable_kl=True,
    )
    assert simulated.stopped is True
    assert [event["kind"] for event in simulated.events] == [
        event["kind"] for event in master
    ]
    assert [event["epoch"] for event in simulated.events] == [
        event["epoch"] for event in master
    ]
    for new, old in zip(simulated.events, master):
        if new["kind"] == "stop":
            continue
        assert new["auto"] == pytest.approx(old["auto"])
        assert new["kl"] == pytest.approx(old["kl"])
        assert new["sequence_len"] == old["sequence_len"]
        if new["kind"] == "window":
            assert new["reset_warmups"] is True
            assert new["auto"] == pytest.approx(0.8)
            assert new["kl"] == pytest.approx(0.0)
    assert any(event["kind"] == "window" for event in simulated.events)
    assert {event["active"] for event in simulated.events} == {float(patience)}
