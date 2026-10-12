"""Spec 4. Adaptive patience.

Held at round(0.2 * early_stop_patience) for every window bump (40 when
patience is 200, 100 when it is 500). Rises over the autonomous and KL
bumps. The final stage, after those bumps, uses the full patience. An
explicit warmup_patience_start replaces the fraction.
"""

from __future__ import annotations

from configparser import ConfigParser

import pytest

from dvae.warmup_curriculum import (
    WarmupCurriculum,
    patience_start_from_fraction,
    warmup_settings_from_config,
)


def _curriculum(early, start, **overrides):
    params = dict(
        order="window_first",
        patience_start=start,
        early_stop_patience=early,
        initial_sequence_len=10,
        sequence_len=16,
        enable_autonomous=True,
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=False,
        enable_kl=True,
    )
    params.update(overrides)
    return WarmupCurriculum(**params)


def test_start_fraction_is_40_for_200_and_100_for_500():
    assert patience_start_from_fraction(0.2, 200) == 40
    assert patience_start_from_fraction(0.2, 500) == 100
    for early, expected in ((200, 40), (500, 100)):
        cfg = ConfigParser()
        cfg.add_section("Training")
        cfg.set("Training", "early_stop_patience", str(early))
        order, start = warmup_settings_from_config(cfg)
        assert order == "window_first"
        assert start == pytest.approx(expected)


def test_patience_holds_then_rises_and_the_final_stage_is_full():
    early = 200
    start = 40
    curriculum = _curriculum(early, start)
    auto_kl = curriculum.autonomous_steps + curriculum.kl_steps
    assert auto_kl > 1

    # Every sequence length short of the target is still a window bump.
    seq = curriculum.initial_sequence_len
    while seq < curriculum.sequence_len:
        assert curriculum.bump_kind(0.0, 0.0, 0.0, seq) == "window"
        assert curriculum.active_patience(0, 0.0, 0.0, 0.0, seq) == pytest.approx(
            start
        )
        seq = curriculum.apply_bump(0.0, 0.0, 0.0, seq).sequence_len

    auto = 0.0
    kl = 0.0
    for done in range(auto_kl):
        expected = start + (early - start) * (done / auto_kl)
        assert curriculum.active_patience(
            done, auto, 0.0, kl, curriculum.sequence_len
        ) == pytest.approx(expected)
        bumped = curriculum.apply_bump(auto, 0.0, kl, curriculum.sequence_len)
        assert bumped.kind in ("auto", "kl")
        auto, kl = bumped.auto, bumped.kl

    assert curriculum.finished(auto, 0.0, kl, curriculum.sequence_len)
    assert curriculum.active_patience(
        auto_kl, auto, 0.0, kl, curriculum.sequence_len
    ) == pytest.approx(early)
    # The last autonomous/KL bump itself is one step short of the full patience.
    last_bump = start + (early - start) * ((auto_kl - 1) / auto_kl)
    assert last_bump < early


def test_explicit_patience_start_overrides_the_fraction():
    cfg = ConfigParser()
    cfg.add_section("Training")
    cfg.set("Training", "early_stop_patience", "200")
    cfg.set("Training", "warmup_patience_start", "7")
    _order, start = warmup_settings_from_config(cfg)
    assert start == pytest.approx(7)
    curriculum = _curriculum(200, 7)
    assert curriculum.active_patience(
        0, 0.0, 0.0, 0.0, curriculum.initial_sequence_len
    ) == pytest.approx(7)
