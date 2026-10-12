"""Default-off training switches, spectrum windows, and eval output suffix."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from dvae.eval.utils.spectrum_validation import (
    capture_fixed_spectrum_windows,
    score_spectrum_windows,
)
import configparser

from dvae.run_options import (
    criteria_log_fields,
    eval_output_dir,
    normalised_criterion_sum,
    parse_checkpoint_every_n_epochs,
    parse_val_criteria,
    resolve_mask_label_override,
    resolve_val_criteria,
    spectrum_monitoring_enabled,
)


class _Series:
    def __init__(self, seq, split="valid", overlap=False, shuffle=False, val_indices=0.0):
        self.seq = np.asarray(seq, dtype=np.float64)
        self.split = split
        self.overlap = overlap
        self.shuffle = shuffle
        self.val_indices = val_indices
        self.coarse_mode = False

    def update_sequence_length(self, *args, **kwargs):
        raise AssertionError("spectrum windows must not call update_sequence_length")


class _Subjects(_Series):
    def __init__(self, blocks, **kwargs):
        parts = [np.asarray(block, dtype=np.float64) for block in blocks]
        seq = np.concatenate(parts, axis=0)
        super().__init__(seq, **kwargs)
        index = []
        for i, block in enumerate(parts):
            index.extend([i] * len(block))
        self.subject_index_per_row = np.asarray(index, dtype=np.int64)


def test_val_criteria_default_is_normalised_recon_and_spectrum():
    assert parse_val_criteria(None) == ("recon", "spectrum")
    assert parse_val_criteria("") == ("recon", "spectrum")
    assert parse_val_criteria("None") == ("recon", "spectrum")
    assert parse_val_criteria("recon") == ("recon",)
    assert parse_val_criteria("off") == ("recon",)
    assert parse_val_criteria("", legacy_windowing=True) == ("recon",)
    assert parse_val_criteria(None, legacy_windowing=True) == ("recon",)
    assert parse_val_criteria("recon,spectrum", legacy_windowing=True) == (
        "recon",
        "spectrum",
    )
    assert parse_val_criteria("spectrum") == ("spectrum",)
    assert parse_val_criteria("recon,spectrum") == ("recon", "spectrum")
    assert parse_val_criteria("spectrum, recon") == ("spectrum", "recon")
    cfg = configparser.ConfigParser()
    cfg.add_section("Training")
    cfg.add_section("DataFrame")
    assert resolve_val_criteria(cfg) == ("recon", "spectrum")
    cfg.set("DataFrame", "legacy_windowing", "true")
    assert resolve_val_criteria(cfg) == ("recon",)
    cfg.remove_option("DataFrame", "legacy_windowing")
    cfg.set("DataFrame", "legacy_defaults", "true")
    assert resolve_val_criteria(cfg) == ("recon",)
    cfg.set("DataFrame", "legacy_defaults", "false")
    cfg.set("DataFrame", "legacy_windowing", "true")
    assert resolve_val_criteria(cfg) == ("recon",)
    cfg.set("Training", "val_criteria", "recon")
    assert resolve_val_criteria(cfg) == ("recon",)
    cfg.set("Training", "val_criteria", "recon,spectrum")
    assert resolve_val_criteria(cfg) == ("recon", "spectrum")
    cfg.remove_option("Training", "val_criteria")
    cfg.set("Training", "val_criterion", "recon")
    assert resolve_val_criteria(cfg) == ("recon",)
    with pytest.raises(ValueError):
        parse_val_criteria("both")
    with pytest.raises(ValueError):
        parse_val_criteria("recon,recon")
    reference = {"recon": 2.0, "spectrum": 4.0}
    score = normalised_criterion_sum(
        {"recon": 1.0, "spectrum": 2.0}, reference
    )
    assert score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-zero"):
        normalised_criterion_sum({"recon": 1.0, "spectrum": 1.0}, {"recon": 0.0, "spectrum": 1.0})
    assert criteria_log_fields(("recon",), None, None, None, 1) == {}
    fields = criteria_log_fields(
        ("recon", "spectrum"),
        np.array([4.0, 2.0]),
        np.array([2.0, 1.0]),
        reference,
        2,
    )
    assert fields["val_criteria"] == "recon,spectrum"
    assert list(fields["val_criteria_score"]) == [2.0, 1.0]
    assert fields["val_criteria_reference"]["recon"] == pytest.approx(2.0)
    assert parse_checkpoint_every_n_epochs(None) == 0
    assert parse_checkpoint_every_n_epochs("") == 0
    assert parse_checkpoint_every_n_epochs("off") == 0
    assert parse_checkpoint_every_n_epochs("0") == 0
    assert parse_checkpoint_every_n_epochs("5") == 5
    assert spectrum_monitoring_enabled(("recon",), legacy=False) is True
    assert spectrum_monitoring_enabled(("recon",), legacy=True) is False
    assert spectrum_monitoring_enabled(("recon", "spectrum"), legacy=True) is True
    assert spectrum_monitoring_enabled(("spectrum",), legacy=False) is True


def test_mask_label_suffix_only_when_override_differs():
    label, suffix = resolve_mask_label_override("realtime", None, None)
    assert label == "realtime"
    assert suffix is None
    label, suffix = resolve_mask_label_override("realtime", "realtime", "custom")
    assert suffix is None
    label, suffix = resolve_mask_label_override("realtime", "retrans", None)
    assert label == "retrans"
    assert suffix == "retrans"
    label, suffix = resolve_mask_label_override("realtime", "retrans", "cross")
    assert suffix == "cross"
    assert eval_output_dir("/runs/exp", None) == "/runs/exp"
    assert eval_output_dir("/runs/exp", "retrans") == "/runs/exp/eval_mask_retrans"


def test_spectrum_windows_are_copies_and_do_not_resize():
    series = np.arange(80, dtype=np.float64).reshape(80, 1)
    dataset = _Series(series, split="valid", overlap=False, val_indices=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        windows = capture_fixed_spectrum_windows(dataset, n_windows=10, window_len=20)
    assert len(windows) == 4  # non-overlap, full series is the validation split
    assert any(issubclass(item.category, UserWarning) for item in caught)
    windows[0][0, 0] = -999.0
    assert dataset.seq[0, 0] == 0.0
    assert windows[0].shape == (20, 1)


def test_tf_clean_windows_are_deterministic_and_relax_when_short():
    clean = np.ones((4000, 1), dtype=np.float64)
    dataset = _Series(clean, split="valid", overlap=True, val_indices=1.0)
    first, info = capture_fixed_spectrum_windows(
        dataset,
        n_windows=10,
        window_len=200,
        tf_steps=100,
        min_observed=0.95,
        max_gap_frac=0.01,
        return_info=True,
    )
    second, again = capture_fixed_spectrum_windows(
        dataset,
        n_windows=10,
        window_len=200,
        tf_steps=100,
        min_observed=0.95,
        max_gap_frac=0.01,
        return_info=True,
    )
    assert len(first) == 10
    assert info["relaxed"] is False
    assert info["qualified"].all()
    assert info["tf_observed"].min() >= 0.95
    assert info["tf_max_gap"].max() <= 1
    assert info["seed"] == 1234567890
    np.testing.assert_array_equal(info["starts"], again["starts"])
    assert np.array_equal(first[0], second[0])
    for start, window in zip(info["starts"], first):
        assert np.isfinite(clean[int(start), 0])
        assert window.shape == (200, 1)

    gappy = np.ones((2000, 1), dtype=np.float64)
    for index in range(8):
        begin = index * 200
        gappy[begin + 10 : begin + 30] = np.nan
    gappy[0, 0] = np.nan
    gappy_set = _Series(gappy, split="valid", overlap=False, val_indices=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _windows, relaxed = capture_fixed_spectrum_windows(
            gappy_set,
            n_windows=10,
            window_len=200,
            tf_steps=100,
            min_observed=0.95,
            max_gap_frac=0.01,
            return_info=True,
        )
    assert relaxed["relaxed"] is True
    assert any(issubclass(item.category, UserWarning) for item in caught)
    assert "observed=" in str(caught[0].message)
    assert int(relaxed["starts"][0]) != 0
    assert relaxed["qualified"].sum() == 2
    assert relaxed["qualified"].size < 10
    for start in relaxed["starts"]:
        assert np.isfinite(gappy[int(start), 0])


def test_spectrum_windows_do_not_cross_subjects():
    dataset = _Subjects(
        [np.ones((30, 1)), np.full((30, 1), 2.0)],
        split="valid",
        overlap=True,
        val_indices=1.0,
    )
    windows = capture_fixed_spectrum_windows(dataset, n_windows=4, window_len=20)
    assert windows
    for window in windows:
        assert set(np.unique(window[:, 0])).issubset({1.0, 2.0})
        assert len(set(np.unique(window[:, 0]).tolist())) == 1
    with pytest.raises(ValueError):
        capture_fixed_spectrum_windows(dataset, n_windows=2, window_len=40)


class _Identity:
    device = "cpu"

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        return batch.clone()


class _ReplaceTail:
    def __init__(self, tf_steps):
        self.device = "cpu"
        self.tf_steps = tf_steps

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        out = batch.clone()
        steps = out.shape[0] - self.tf_steps
        t = torch.arange(steps, dtype=out.dtype)
        out[self.tf_steps :, :, 0] = torch.sin(t / 3.0).unsqueeze(-1)
        if out.shape[-1] > 1:
            out[self.tf_steps :, :, 1] = (0.25 * torch.sin(t / 7.0)).unsqueeze(-1)
        return out


class _ZeroHead:
    def __init__(self, tf_steps):
        self.device = "cpu"
        self.tf_steps = tf_steps

    def __call__(self, batch, mode_selector=None, inference_mode=None):
        out = batch.clone()
        out[: self.tf_steps] = 0.0
        return out


def _sine_windows():
    t = np.arange(64, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * t / 16.0)
    window = np.column_stack([wave, 0.5 * wave])
    return [window.copy(), window.copy()]


def test_spectrum_score_ignores_teacher_forced_half():
    windows = _sine_windows()
    tf_steps = 32
    same = score_spectrum_windows(_Identity(), windows, tf_steps, None, "cpu")
    head = score_spectrum_windows(_ZeroHead(tf_steps), windows, tf_steps, None, "cpu")
    tail = score_spectrum_windows(
        _ReplaceTail(tf_steps), windows, tf_steps, None, "cpu"
    )
    # Identical free-run halves score the same. Hellinger on a perfect match
    # is NaN in power_spectrum_error (the +1e-10 sits outside the normalizer).
    assert np.isnan(same) and np.isnan(head)
    assert np.isfinite(tail) and tail > 0.02


def _vertical_line_colors(ax):
    colors = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        if x.size >= 2 and np.allclose(x, x[0]):
            colors.append(line.get_color())
    return colors


def _panel_titles(fig):
    return [ax.get_legend().get_title().get_text() for ax in fig.axes]


def _best_epoch_ys(ax):
    ys = []
    for line in ax.lines:
        if line.get_label() == "best epoch":
            ys.append(float(np.asarray(line.get_ydata(), dtype=float).reshape(-1)[0]))
    return ys


def test_val_criteria_figure_matches_loss_plot_markers(tmp_path):
    from dvae.visualizers.visualize_training_metrics import (
        build_val_criteria_figure,
        val_criteria_plot_series,
        visualize_val_criteria,
    )

    monitored = val_criteria_plot_series(
        ("recon",), [1.0, 0.5], [9.0, 4.0], [3.0, 1.0]
    )
    assert [(name, selected) for name, _, selected in monitored] == [
        ("recon", True),
        ("spectrum", False),
    ]
    legacy = val_criteria_plot_series(("recon",), [1.0, 0.5], None, None)
    assert [name for name, _, _ in legacy] == ["recon"]
    forced_off = val_criteria_plot_series(
        ("recon",),
        [1.0, 0.5],
        [9.0, 4.0],
        None,
        include_spectrum=False,
    )
    assert [name for name, _, _ in forced_off] == ["recon"]
    with pytest.raises(ValueError, match="val_spectrum is missing"):
        val_criteria_plot_series(
            ("recon",), [1.0], None, None, include_spectrum=True
        )
    both = val_criteria_plot_series(
        ("recon", "spectrum"),
        [1.0, 0.5],
        [4.0, 2.0],
        [2.0, 1.0],
    )
    assert [(name, selected) for name, _, selected in both] == [
        ("recon", True),
        ("spectrum", True),
        ("combined", True),
    ]

    kl = [0, 3]
    auto = [0, 4]
    noise = [0, 5]
    seq = [0, 6]
    fig = build_val_criteria_figure(
        ("recon", "spectrum"),
        [1.0, 0.5],
        [4.0, 2.0],
        [2.0, 1.0],
        kl,
        auto,
        noise,
        seq,
        "VRNN",
        "ss",
        "ss",
        best_epoch=1,
    )
    assert len(fig.axes) == 3
    assert [ax.get_ylabel() for ax in fig.axes] == [
        "loss",
        "spectrum error",
        "normalised sum",
    ]
    assert _panel_titles(fig) == [
        "VRNN: recon (selected)",
        "VRNN: spectrum error (selected)",
        "VRNN: selection score",
    ]
    assert [_best_epoch_ys(ax) for ax in fig.axes] == [[0.5], [2.0], [1.0]]
    for ax in fig.axes:
        colors = _vertical_line_colors(ax)
        assert colors.count("c") == len(kl)
        assert colors.count("r") == len(auto)
        assert colors.count("g") == len(noise)
        assert colors.count("m") == len(seq)
        assert ax.get_xlabel() == "epochs"
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["validation", "best epoch"]
    rnn = build_val_criteria_figure(
        ("recon",),
        [1.0, 0.4],
        None,
        None,
        kl,
        auto,
        noise,
        seq,
        "RNN",
        "none",
        "none",
    )
    assert len(rnn.axes) == 1
    assert _panel_titles(rnn) == ["RNN: recon (selected)"]
    assert _best_epoch_ys(rnn.axes[0]) == []
    rnn_colors = _vertical_line_colors(rnn.axes[0])
    assert "c" not in rnn_colors
    assert "r" not in rnn_colors
    assert rnn_colors.count("m") == len(seq)
    recorded = build_val_criteria_figure(
        ("recon",),
        [1.0, 0.4],
        [8.0, 3.0],
        None,
        kl,
        auto,
        noise,
        seq,
        "RNN",
        "none",
        "none",
        best_epoch=1,
    )
    assert len(recorded.axes) == 2
    assert _panel_titles(recorded) == [
        "RNN: recon (selected)",
        "RNN: spectrum error",
    ]
    assert [_best_epoch_ys(ax) for ax in recorded.axes] == [[0.4], [3.0]]
    for ax in recorded.axes:
        colors = _vertical_line_colors(ax)
        assert colors.count("m") == len(seq)
    path = visualize_val_criteria(
        ("recon",),
        [1.0, 0.4],
        [8.0, 3.0],
        None,
        kl,
        [],
        [],
        seq,
        "RNN",
        "none",
        "none",
        str(tmp_path),
        "unit",
        best_epoch=0,
        include_spectrum=True,
    )
    assert path.endswith("vis_training_val_criteria_unit.png")
    assert (tmp_path / "vis_training_val_criteria_unit.png").stat().st_size > 0
