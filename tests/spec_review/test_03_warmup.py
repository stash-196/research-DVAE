"""Spec 3. window_first grows the window at full teacher forcing, then auto/KL.

Bump counts come from the same 0.1 arithmetic the curriculum stores.
No model and no dataset.
"""

from __future__ import annotations

from configparser import ConfigParser

from dvae.warmup_curriculum import (
    WarmupCurriculum,
    autonomous_step_count,
    kl_step_count,
    planned_bump_count,
    warmup_settings_from_config,
    window_step_count,
)


def _curriculum(**overrides):
    params = dict(
        order="window_first",
        patience_start=2,
        early_stop_patience=10,
        initial_sequence_len=10,
        sequence_len=20,
        enable_autonomous=True,
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=True,
        enable_kl=True,
    )
    params.update(overrides)
    return WarmupCurriculum(**params)


def test_blank_config_is_window_first():
    cfg = ConfigParser()
    cfg.add_section("Training")
    cfg.set("Training", "early_stop_patience", "200")
    order, _patience = warmup_settings_from_config(cfg)
    assert order == "window_first"


def test_window_grows_at_full_teacher_forcing_before_auto_and_kl():
    curriculum = _curriculum()
    auto, kl, seq = 0.0, 0.0, 10
    kinds = []
    while not curriculum.finished(auto, 0.0, kl, seq):
        bumped = curriculum.apply_bump(auto, 0.0, kl, seq)
        kinds.append(bumped.kind)
        if bumped.kind == "window":
            assert bumped.auto == 0.0
            assert bumped.kl == 0.0
            assert bumped.reset_warmups is False
            assert bumped.sequence_len > seq
        auto, kl, seq = bumped.auto, bumped.kl, bumped.sequence_len
    assert kinds[0] == "window"
    assert "auto" in kinds and "kl" in kinds
    assert kinds.index("auto") > kinds.index("window")
    assert kinds.index("kl") > kinds.index("auto")
    assert seq == 20
    assert auto == 1.0
    assert kl >= 1.0


def test_window_bump_does_not_reset_auto_or_kl_even_if_they_already_moved():
    curriculum = _curriculum()
    bumped = curriculum.apply_bump(auto=0.4, noise=0.2, kl=0.3, sequence_len=10)
    assert bumped.kind == "window"
    assert bumped.reset_warmups is False
    assert bumped.auto == 0.4
    assert bumped.noise == 0.2
    assert bumped.kl == 0.3


def test_auto_first_window_bump_resets_auto_and_kl():
    curriculum = _curriculum(order="auto_first", patience_start=None)
    # Autonomous and KL are already finished, so the next bump is the window.
    bumped = curriculum.apply_bump(auto=1.0, noise=0.0, kl=1.0, sequence_len=10)
    assert bumped.kind == "window"
    assert bumped.reset_warmups is True
    assert bumped.auto == 0.0
    assert bumped.kl == 0.0


def test_bump_counts_are_derived_from_the_configured_lengths():
    curriculum = _curriculum(
        initial_sequence_len=20,
        sequence_len=50,
        auto_start=0.0,
        auto_limit=1.0,
    )
    windows = window_step_count(20, 50)
    autos = autonomous_step_count(
        enable_autonomous=True,
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=0.0,
        noise_limit=1.0,
    )
    kls = kl_step_count(True)
    assert curriculum.window_steps == windows
    assert curriculum.autonomous_steps == autos
    assert curriculum.kl_steps == kls
    assert curriculum.n_planned == planned_bump_count(
        "window_first", windows, autos, kls
    )
    assert curriculum.n_planned == windows + autos + kls

    # The live addition `x += 0.1` while `x < 1` takes 11 steps: ten
    # additions of the float 0.1 are still just under 1.
    value = 0.0
    live_steps = 0
    while value < 1.0:
        value += 0.1
        live_steps += 1
    assert kls == live_steps
    assert kls == 11
