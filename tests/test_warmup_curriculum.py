"""Warm-up order and adaptive patience. Stubbed validation losses only."""

from __future__ import annotations

import configparser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from dvae.visualizers.visualize_training_metrics import visualize_combined_metrics
from dvae.warmup_curriculum import (
    WarmupCurriculum,
    autonomous_step_count,
    format_patience,
    kl_step_count,
    patience_at_bump,
    planned_bump_count,
    simulate_recon_warmup,
    warmup_action,
    warmup_settings_from_config,
    window_step_count,
)


def _flat(n=80):
    return [1.0] * n


def _curriculum(**overrides):
    params = dict(
        order="auto_first",
        patience_start=None,
        early_stop_patience=0,
        initial_sequence_len=80,
        sequence_len=100,
        enable_autonomous=True,
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=False,
        noise_start=1.0,
        noise_limit=1.0,
        noise_reset_on_window=True,
        enable_kl=True,
    )
    params.update(overrides)
    return WarmupCurriculum(**params)


def _kinds(sim):
    return [event["kind"] for event in sim.events]


def test_blank_config_uses_window_first_and_patience_2():
    cfg = configparser.ConfigParser()
    cfg.read_string("[Training]\nwarmup_order =\nwarmup_patience_start =\n")
    order, start = warmup_settings_from_config(cfg)
    assert order == "window_first"
    assert start == 2.0

    cfg = configparser.ConfigParser()
    cfg.read_string("[Training]\nepochs = 3\n")
    order, start = warmup_settings_from_config(cfg)
    assert order == "window_first"
    assert start == 2.0

    cfg = configparser.ConfigParser()
    cfg.read_string(
        "[Training]\n"
        "warmup_order = auto_first\n"
        "warmup_patience_start = off\n"
    )
    order, start = warmup_settings_from_config(cfg)
    assert order == "auto_first"
    assert start is None

    cfg = configparser.ConfigParser()
    cfg.read_string("[Training]\nwarmup_patience_start = none\n")
    assert warmup_settings_from_config(cfg) == ("window_first", 2.0)
    cfg = configparser.ConfigParser()
    cfg.read_string("[Training]\nwarmup_patience_start = null\n")
    assert warmup_settings_from_config(cfg)[1] == 2.0
    cfg = configparser.ConfigParser()
    cfg.read_string("[Training]\nwarmup_patience_start = 4\n")
    assert warmup_settings_from_config(cfg)[1] == 4.0


def test_legacy_switch_restores_auto_first_and_a_single_patience():
    for name in ("legacy_defaults", "legacy_windowing"):
        cfg = _cfg(
            "[Training]\n"
            "warmup_order =\n"
            "warmup_patience_start =\n"
            "[DataFrame]\n"
            f"{name} = true\n"
        )
        assert warmup_settings_from_config(cfg) == ("auto_first", None)

    # Either name is enough, and an explicit warm-up value still wins.
    cfg = _cfg(
        "[Training]\n"
        "warmup_order = window_first\n"
        "warmup_patience_start = 4\n"
        "[DataFrame]\n"
        "legacy_defaults = false\n"
        "legacy_windowing = true\n"
    )
    assert warmup_settings_from_config(cfg) == ("window_first", 4.0)

    cfg = _cfg(
        "[Training]\n"
        "warmup_patience_start = off\n"
        "[DataFrame]\n"
        "legacy_defaults = true\n"
    )
    assert warmup_settings_from_config(cfg) == ("auto_first", None)


def test_template_documents_window_first_and_patience_2():
    text = (
        Path(__file__).resolve().parents[1]
        / "config/general_signal/cfg_base_template.ini"
    ).read_text()
    assert "warmup_order = {warmup_order}" in text
    assert "warmup_patience_start = {warmup_patience_start}" in text
    assert "Blank = window_first" in text
    assert "Blank = 2." in text
    assert "legacy_defaults = {legacy_defaults}" in text
    assert "legacy_windowing = {legacy_windowing}" in text


def test_bad_order_and_negative_patience_are_rejected():
    with pytest.raises(ValueError):
        warmup_settings_from_config(
            _cfg("[Training]\nwarmup_order = kl_first\n")
        )
    with pytest.raises(ValueError):
        warmup_settings_from_config(
            _cfg("[Training]\nwarmup_patience_start = -1\n")
        )


def _cfg(text):
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


def test_step_counts_follow_the_live_bump_arithmetic():
    assert window_step_count(80, 100) == 2
    assert window_step_count(70, 100) == 3
    assert window_step_count(100, 100) == 0
    assert window_step_count(50, 100) == 5

    kl = 0.0
    steps = 0
    while kl < 1.0:
        kl += 0.1
        steps += 1
    assert kl_step_count(True) == steps == 11
    assert kl_step_count(False) == 0

    assert (
        autonomous_step_count(
            enable_autonomous=True,
            auto_start=0.8,
            auto_limit=1.0,
            enable_noise=False,
            noise_start=1.0,
            noise_limit=1.0,
        )
        == 2
    )
    # Shared trigger: noise from 0 needs the full 0.1 ramp, auto from 0.8
    # finishes first, and the branch keeps running until noise is done.
    assert (
        autonomous_step_count(
            enable_autonomous=True,
            auto_start=0.8,
            auto_limit=1.0,
            enable_noise=True,
            noise_start=0.0,
            noise_limit=1.0,
        )
        == 11
    )
    assert (
        autonomous_step_count(
            enable_autonomous=False,
            auto_start=0.0,
            auto_limit=1.0,
            enable_noise=True,
            noise_start=0.0,
            noise_limit=1.0,
        )
        == 0
    )


def test_patience_schedule_endpoints():
    # Five planned bumps: index 0 is the start, index 4 is the full patience.
    values = [patience_at_bump(i, 5, 2, 10) for i in range(5)]
    assert values == pytest.approx([2, 4, 6, 8, 10])
    assert patience_at_bump(9, 5, 2, 10) == 10
    # A single planned bump uses the start. The slot after it is the stop.
    assert patience_at_bump(0, 1, 1, 5) == 1
    assert patience_at_bump(1, 1, 1, 5) == 5
    assert format_patience(2.0) == "2"
    assert format_patience(2.5) == "2.5"
    assert warmup_action(2, 2, False) == "continue"
    assert warmup_action(3, 2, False) == "bump"
    assert warmup_action(3, 2, True) == "stop"


def test_auto_first_order_resets_after_each_window():
    curriculum = _curriculum(
        order="auto_first",
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=True,
        noise_start=0.0,
        noise_limit=1.0,
        enable_kl=True,
        initial_sequence_len=80,
        sequence_len=100,
        early_stop_patience=0,
    )
    assert curriculum.window_steps == 2
    assert curriculum.autonomous_steps == 11
    assert curriculum.kl_steps == 11
    # Nested: a full inner ramp at each length, plus the window bumps.
    inner = 11 + 11
    assert curriculum.n_planned == 2 * inner + 2 + inner
    sim = simulate_recon_warmup(
        _flat(90),
        curriculum,
        auto=0.0,
        noise=0.0,
        kl=0.0,
        sequence_len=80,
    )
    kinds = _kinds(sim)
    assert kinds[-1] == "stop"
    body = kinds[:-1]
    phase = ["auto"] * 11 + ["kl"] * 11 + ["window"]
    assert body == phase * 2 + ["auto"] * 11 + ["kl"] * 11
    assert len(body) == curriculum.n_planned
    for event in sim.events:
        if event["kind"] == "window":
            assert event["auto"] == 0.0
            assert event["noise"] == 0.0
            assert event["kl"] == 0.0
            assert event["reset_warmups"] is True
            assert event["sequence_len"] > event["sequence_before"]


def test_window_first_holds_teacher_forcing_until_sequence_len():
    curriculum = _curriculum(
        order="window_first",
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=True,
        noise_start=0.0,
        noise_limit=1.0,
        enable_kl=True,
        initial_sequence_len=80,
        sequence_len=100,
        early_stop_patience=0,
    )
    sim = simulate_recon_warmup(
        _flat(90),
        curriculum,
        auto=0.0,
        noise=0.0,
        kl=0.0,
        sequence_len=80,
    )
    kinds = _kinds(sim)
    assert kinds[-1] == "stop"
    body = kinds[:-1]
    w, a, k = (
        curriculum.window_steps,
        curriculum.autonomous_steps,
        curriculum.kl_steps,
    )
    assert body == ["window"] * w + ["auto"] * a + ["kl"] * k
    assert len(body) == curriculum.n_planned
    windows = [event for event in sim.events if event["kind"] == "window"]
    assert windows[-1]["sequence_len"] == 100
    for event in windows:
        assert event["auto_before"] == 0.0
        assert event["auto"] == 0.0
        assert event["noise_before"] == 0.0
        assert event["noise"] == 0.0
        assert event["kl"] == 0.0
    first_auto = sim.events[w]
    assert first_auto["kind"] == "auto"
    assert first_auto["auto_bumped"] and first_auto["noise_bumped"]
    assert first_auto["auto"] > 0.0
    assert sim.events[w + a]["kind"] == "kl"
    assert all(not event["reset_warmups"] for event in windows)


def test_window_first_never_resets_auto_on_a_window_bump():
    curriculum = _curriculum(
        order="window_first",
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=True,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=True,
        enable_kl=True,
        initial_sequence_len=80,
        sequence_len=100,
    )
    bumped = curriculum.apply_bump(auto=0.4, noise=0.3, kl=0.5, sequence_len=80)
    assert bumped.kind == "window"
    assert bumped.sequence_len == 90
    assert bumped.auto == pytest.approx(0.4)
    assert bumped.noise == pytest.approx(0.3)
    assert bumped.kl == pytest.approx(0.5)
    assert bumped.reset_warmups is False

    # Same inputs under auto_first still take the autonomous branch first.
    # Force the window branch by sitting at the ramp limits.
    legacy = _curriculum(
        order="auto_first",
        auto_start=0.0,
        auto_limit=1.0,
        enable_noise=True,
        noise_start=0.0,
        noise_limit=1.0,
        noise_reset_on_window=True,
        enable_kl=True,
        initial_sequence_len=80,
        sequence_len=100,
    )
    reset = legacy.apply_bump(auto=1.0, noise=1.0, kl=1.0, sequence_len=80)
    assert reset.kind == "window"
    assert reset.auto == pytest.approx(0.0)
    assert reset.noise == pytest.approx(0.0)
    assert reset.kl == pytest.approx(0.0)
    assert reset.reset_warmups is True
    held = curriculum.apply_bump(auto=1.0, noise=1.0, kl=1.0, sequence_len=80)
    assert held.kind == "window"
    assert held.auto == pytest.approx(1.0)
    assert held.noise == pytest.approx(1.0)
    assert held.kl == pytest.approx(1.0)
    assert held.reset_warmups is False


def test_patience_length_is_nested_only_for_auto_first():
    assert planned_bump_count("window_first", 3, 2, 0) == 3 + 2
    assert planned_bump_count("auto_first", 3, 2, 0) == 3 * 2 + 3 + 2
    assert planned_bump_count("window_first", 3, 2, 4) == 3 + 2 + 4
    assert planned_bump_count("auto_first", 3, 2, 4) == 3 * (2 + 4) + 3 + (2 + 4)
    assert planned_bump_count("auto_first", 0, 2, 4) == 2 + 4


def test_disabled_phases_are_skipped_in_both_orders():
    for order in ("auto_first", "window_first"):
        curriculum = _curriculum(
            order=order,
            enable_autonomous=False,
            enable_noise=False,
            enable_kl=False,
            auto_start=0.0,
            auto_limit=1.0,
            initial_sequence_len=70,
            sequence_len=100,
            early_stop_patience=0,
        )
        assert curriculum.autonomous_steps == 0
        assert curriculum.kl_steps == 0
        assert curriculum.window_steps == 3
        sim = simulate_recon_warmup(
            _flat(20),
            curriculum,
            auto=curriculum.auto_limit,
            noise=curriculum.noise_limit,
            kl=1.0,
            sequence_len=70,
        )
        assert _kinds(sim) == ["window", "window", "window", "stop"]
        assert [event["sequence_len"] for event in sim.events if event["kind"] == "window"] == [
            80,
            90,
            100,
        ]


def test_one_trigger_bumps_auto_and_noise_together():
    curriculum = _curriculum(
        order="auto_first",
        enable_kl=False,
        enable_noise=True,
        auto_start=0.8,
        auto_limit=1.0,
        noise_start=0.5,
        noise_limit=1.0,
        initial_sequence_len=100,
        sequence_len=100,
        early_stop_patience=0,
    )
    # 0.5 + 0.1 five times is still just under 1, so the ramp takes 6 steps.
    assert curriculum.autonomous_steps == 6
    assert curriculum.window_steps == 0
    assert curriculum.n_planned == 6
    sim = simulate_recon_warmup(
        _flat(20),
        curriculum,
        auto=0.8,
        noise=0.5,
        kl=1.0,
        sequence_len=100,
    )
    bumps = [event for event in sim.events if event["kind"] != "stop"]
    assert [event["kind"] for event in bumps] == ["auto"] * 6
    assert all(event["auto_bumped"] and event["noise_bumped"] for event in bumps[:2])
    assert all(
        (not event["auto_bumped"]) and event["noise_bumped"] for event in bumps[2:]
    )
    assert bumps[-1]["auto"] == pytest.approx(1.0)
    assert bumps[-1]["noise"] == pytest.approx(1.0)
    assert sim.events[-1]["kind"] == "stop"


def test_window_bump_can_keep_noise_when_reset_is_off():
    curriculum = _curriculum(
        order="auto_first",
        enable_kl=False,
        enable_noise=True,
        auto_start=0.8,
        auto_limit=1.0,
        noise_start=0.8,
        noise_limit=1.0,
        noise_reset_on_window=False,
        initial_sequence_len=90,
        sequence_len=100,
        early_stop_patience=0,
    )
    sim = simulate_recon_warmup(
        _flat(20),
        curriculum,
        auto=0.8,
        noise=0.8,
        kl=1.0,
        sequence_len=90,
    )
    windows = [event for event in sim.events if event["kind"] == "window"]
    assert len(windows) == 1
    assert windows[0]["auto"] == pytest.approx(0.8)
    assert windows[0]["noise"] == pytest.approx(1.0)
    assert windows[0]["noise_before"] == pytest.approx(1.0)


def test_unset_patience_uses_early_stop_for_every_bump_and_the_stop():
    curriculum = _curriculum(
        order="window_first",
        patience_start=None,
        early_stop_patience=1,
        enable_autonomous=False,
        enable_noise=False,
        enable_kl=False,
        initial_sequence_len=70,
        sequence_len=100,
    )
    sim = simulate_recon_warmup(
        _flat(20),
        curriculum,
        auto=1.0,
        noise=1.0,
        kl=1.0,
        sequence_len=70,
    )
    bumps = [event for event in sim.events if event["kind"] == "window"]
    assert [event["epoch"] for event in bumps] == [2, 4, 6]
    assert [event["active"] for event in bumps] == [1, 1, 1]
    stop = sim.events[-1]
    assert stop["kind"] == "stop"
    assert stop["epoch"] == 8
    assert stop["active"] == 1
    assert sim.active_patience == [1] * 9


def test_patience_grows_across_planned_bumps_and_the_stop_stays_full():
    curriculum = _curriculum(
        order="window_first",
        patience_start=1,
        early_stop_patience=3,
        enable_autonomous=False,
        enable_noise=False,
        enable_kl=False,
        initial_sequence_len=70,
        sequence_len=100,
    )
    assert curriculum.n_planned == 3
    sim = simulate_recon_warmup(
        _flat(30),
        curriculum,
        auto=1.0,
        noise=1.0,
        kl=1.0,
        sequence_len=70,
    )
    bumps = [event for event in sim.events if event["kind"] == "window"]
    assert [event["epoch"] for event in bumps] == [2, 5, 9]
    assert [event["active"] for event in bumps] == pytest.approx([1, 2, 3])
    assert [event["sequence_len"] for event in bumps] == [80, 90, 100]
    stop = sim.events[-1]
    assert stop["kind"] == "stop"
    assert stop["epoch"] == 13
    assert stop["active"] == 3


def test_single_bump_uses_the_start_and_the_final_stop_uses_full_patience():
    curriculum = _curriculum(
        order="window_first",
        patience_start=1,
        early_stop_patience=5,
        enable_autonomous=False,
        enable_noise=False,
        enable_kl=False,
        initial_sequence_len=90,
        sequence_len=100,
    )
    assert curriculum.n_planned == 1
    sim = simulate_recon_warmup(
        _flat(20),
        curriculum,
        auto=1.0,
        noise=1.0,
        kl=1.0,
        sequence_len=90,
    )
    bumps = [event for event in sim.events if event["kind"] == "window"]
    assert len(bumps) == 1
    assert bumps[0]["epoch"] == 2
    assert bumps[0]["active"] == 1
    assert bumps[0]["sequence_len"] == 100
    assert sim.active_patience[:3] == [1, 1, 1]
    assert sim.active_patience[3:] == [5, 5, 5, 5, 5, 5]
    stop = sim.events[-1]
    assert stop["epoch"] == 8
    assert stop["active"] == 5


def test_no_planned_bumps_still_stops_on_the_full_patience():
    curriculum = _curriculum(
        order="auto_first",
        patience_start=1,
        early_stop_patience=5,
        enable_autonomous=False,
        enable_noise=False,
        enable_kl=False,
        initial_sequence_len=100,
        sequence_len=100,
    )
    assert curriculum.n_planned == 0
    sim = simulate_recon_warmup(
        _flat(20),
        curriculum,
        auto=1.0,
        noise=1.0,
        kl=1.0,
        sequence_len=100,
    )
    assert _kinds(sim) == ["stop"]
    assert sim.events[0]["epoch"] == 6
    assert sim.events[0]["active"] == 5
    assert sim.active_patience == [5] * 7


def test_cpt_panel_draws_the_patience_threshold_dashed(tmp_path, monkeypatch):
    captured = {}
    real_savefig = plt.savefig

    def _save(path, *args, **kwargs):
        fig = plt.gcf()
        ax = fig.axes[3]
        dashed = [
            line
            for line in ax.get_lines()
            if line.get_linestyle() == "--" and len(np.atleast_1d(line.get_ydata())) > 2
        ]
        captured["dashed"] = dashed
        captured["labels"] = [text.get_text() for text in ax.get_legend().get_texts()]
        return real_savefig(path, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", _save)
    n = 4
    threshold = np.array([1.0, 1.0, 2.5, 4.0])
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
        np.array([0, 1, 0, 2]),
        np.zeros(n),
        "PLRNN",
        "ss",
        "none",
        str(tmp_path),
        "unit",
        threshold,
    )
    assert len(captured["dashed"]) == 1
    assert to_rgb(captured["dashed"][0].get_color()) == to_rgb("black")
    np.testing.assert_allclose(captured["dashed"][0].get_ydata(), threshold)
    assert "cpt_patience" in captured["labels"]
    assert "patience threshold" in captured["labels"]
    assert (tmp_path / "vis_training_delta_kl_cpt_best_state_unit.png").is_file()


def test_cpt_panel_omits_the_threshold_when_it_is_not_passed(tmp_path, monkeypatch):
    captured = {}
    real_savefig = plt.savefig

    def _save(path, *args, **kwargs):
        ax = plt.gcf().axes[3]
        captured["labels"] = [text.get_text() for text in ax.get_legend().get_texts()]
        captured["dashed"] = [
            line for line in ax.get_lines() if line.get_linestyle() == "--"
        ]
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
        np.arange(n),
        np.zeros(n),
        np.zeros(n),
        "PLRNN",
        "ss",
        "none",
        str(tmp_path),
        "bare",
    )
    assert captured["dashed"] == []
    assert captured["labels"] == ["cpt_patience"]
