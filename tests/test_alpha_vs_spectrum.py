"""Eval alpha-vs-spectrum figure for MT sigmas and PLRNN diagonal A."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dvae.eval.alpha_vs_spectrum import (
    eval_timescale_tracks,
    gt_power_spectra,
    save_alpha_vs_spectrum_figures,
)
from dvae.timescale_history import diagonal_A_from_sigmas
from dvae.visualizers.visualizers import visualize_alpha_history_and_spectrums

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG"
EVAL_SIGNAL = REPO_ROOT / "src" / "dvae" / "eval" / "eval_signal.py"
PLOTTER = REPO_ROOT / "src" / "dvae" / "visualizers" / "visualizers.py"


def _signal(freq_hz, dt=0.01, n=256):
    t = np.arange(n) * dt
    return np.sin(2 * np.pi * freq_hz * t)


def _benchmarks():
    return {
        "dt": 0.01,
        "channels": [
            {
                "key": "x",
                "name": "x",
                "color": "blue",
                "gt_full": _signal(4.0),
            },
            {
                "key": "y",
                "name": "y",
                "color": "orange",
                "gt_full": _signal(1.5),
            },
        ],
    }


def _history():
    return np.array(
        [
            [-1.0, -0.4, 0.0, 0.3, 0.6],
            [0.0, 0.2, 0.4, 0.5, 0.7],
            [1.0, 0.8, 0.5, 0.2, 0.0],
        ],
        dtype=np.float64,
    )


class _Dataset:
    dataset_name = "Lorenz63"
    true_alphas = [0.02, 0.05]


def _assert_png(path):
    raw = Path(path).read_bytes()
    assert raw.startswith(PNG_MAGIC)
    assert len(raw) > 1000


def test_eval_signal_calls_alpha_vs_spectrum_on_first_window():
    text = EVAL_SIGNAL.read_text()
    assert "save_alpha_vs_spectrum_figures(" in text
    assert "if save_metric_figures and i == 0:" in text
    plotter = PLOTTER.read_text()
    assert "1 / (1 + np.exp(-np.max(sigmas_history))) / 10" in plotter
    assert "alphas = 1 / (1 + np.exp(-sigmas_history[i]))" in plotter


def test_tracks_keep_mt_sigmas_separate_from_diagonal_a():
    history = _history()
    assert eval_timescale_tracks({}, "MT_RNN") == []
    assert eval_timescale_tracks({}, "RNN") == []

    mt_only = eval_timescale_tracks({"sigmas_history": history}, "MT_RNN")
    assert [track["kind"] for track in mt_only] == ["mt"]
    assert mt_only[0]["explain"] == ""

    plrnn_only = eval_timescale_tracks(
        {"A_sigmas_history": history, "sigmas_history": history},
        "RNN",
    )
    assert [track["kind"] for track in plrnn_only] == ["plrnn"]
    assert plrnn_only[0]["explain"] == ""

    both = eval_timescale_tracks(
        {"sigmas_history": history, "A_sigmas_history": history * 0.5},
        "MT_RNN",
    )
    assert [track["kind"] for track in both] == ["mt", "plrnn"]
    assert both[1]["explain"] == "diagA"
    np.testing.assert_allclose(both[0]["history"], history)
    np.testing.assert_allclose(both[1]["history"], history * 0.5)


def test_mt_plot_call_uses_historical_kwargs_only():
    captured = []

    def _record(**kwargs):
        captured.append(kwargs)

    paths = save_alpha_vs_spectrum_figures(
        loaded_data={
            "sigmas_history": _history(),
            "kl_warm_epochs": np.array([0, 2]),
        },
        model_name="MT_RNN",
        channel_benchmarks=_benchmarks(),
        dataset=_Dataset(),
        save_dir="/tmp/unused-alpha-vs-spectrum",
        visualize_alpha_history_and_spectrums=_record,
        dataset_name="Lorenz63",
    )
    assert len(captured) == 1
    assert "alpha_from_sigma" not in captured[0]
    assert "max_curves" not in captured[0]
    assert captured[0]["spectrum_title"] == "Lorenz63\nPower Spectrum"
    assert captured[0]["explain"] == ""
    assert captured[0]["true_alphas"] == [0.02, 0.05]
    assert paths == [
        "/tmp/unused-alpha-vs-spectrum/vis_alpha_vs_power_spectrum_.png"
    ]


def test_plrnn_alpha_vs_spectrum_figure_uses_diagonal_a(tmp_path):
    captured = {}

    def _wrap(**kwargs):
        captured.update(kwargs)
        return visualize_alpha_history_and_spectrums(**kwargs)

    paths = save_alpha_vs_spectrum_figures(
        loaded_data={"A_sigmas_history": _history()},
        model_name="RNN",
        channel_benchmarks=_benchmarks(),
        dataset=_Dataset(),
        save_dir=tmp_path,
        visualize_alpha_history_and_spectrums=_wrap,
        dataset_name="Lorenz63",
    )
    assert len(paths) == 1
    _assert_png(paths[0])
    assert captured["alpha_from_sigma"] is diagonal_A_from_sigmas
    assert captured["max_curves"] == 3
    assert captured["spectrum_title"] == "Lorenz63\nPower Spectrum"
    assert captured["alpha_from_sigma"](np.array([1.0]))[0] == pytest.approx(
        1.0 / (1.0 + 10.0 ** (-1.0))
    )
    spectra = gt_power_spectra(_benchmarks())
    assert spectra is not None
    power, frequencies, colors, names = spectra
    assert len(power) == 2
    assert power[0].shape == frequencies.shape
    assert colors == ["blue", "orange"]
    assert names == ["x", "y"]


def test_xhro_channels_are_not_lorenz_gated(tmp_path):
    """Four wearable channels at 250 Hz still produce the eval figure."""
    dt = 1.0 / 250.0
    n = 500
    t = np.arange(n) * dt
    channels = []
    for idx, freq in enumerate((1.0, 4.0, 8.0, 12.0), start=1):
        channels.append(
            {
                "key": f"ch{idx}",
                "name": f"ch{idx}",
                "color": "cyan",
                "gt_full": np.sin(2 * np.pi * freq * t),
            }
        )
    benchmarks = {"dt": dt, "channels": channels}

    class _Xhro:
        dataset_name = "Xhro"
        true_alphas = None

    captured = {}

    def _wrap(**kwargs):
        captured.update(kwargs)
        return visualize_alpha_history_and_spectrums(**kwargs)

    paths = save_alpha_vs_spectrum_figures(
        loaded_data={"A_sigmas_history": _history()},
        model_name="RNN",
        channel_benchmarks=benchmarks,
        dataset=_Xhro(),
        save_dir=tmp_path,
        visualize_alpha_history_and_spectrums=_wrap,
        dataset_name="Xhro",
    )
    assert len(paths) == 1
    _assert_png(paths[0])
    assert captured["spectrum_title"] == "Xhro\nPower Spectrum"
    assert captured["dt"] == pytest.approx(dt)
    assert len(captured["power_spectrum_lst"]) == 4
    assert [name for name in captured["spectrum_name_lst"]] == [
        "ch1",
        "ch2",
        "ch3",
        "ch4",
    ]
    helper = (REPO_ROOT / "src" / "dvae" / "eval" / "alpha_vs_spectrum.py").read_text()
    assert '== "Lorenz63"' not in helper

    def _accept(**_kwargs):
        return None

    for name in ("Xhro", "XhroPacketLoss", "XhroProper", "PhysioNet2012"):
        wrote = save_alpha_vs_spectrum_figures(
            loaded_data={"sigmas_history": _history()},
            model_name="MT_RNN",
            channel_benchmarks=benchmarks,
            dataset=_Xhro(),
            save_dir=tmp_path,
            visualize_alpha_history_and_spectrums=_accept,
            dataset_name=name,
        )
        assert wrote, name
        assert wrote[0].endswith("vis_alpha_vs_power_spectrum_.png")


def test_default_mt_figure_still_renders(tmp_path):
    benchmarks = _benchmarks()
    power, frequencies, colors, names = gt_power_spectra(benchmarks)
    visualize_alpha_history_and_spectrums(
        sigmas_history=_history(),
        power_spectrum_lst=power,
        spectrum_color_lst=colors,
        spectrum_name_lst=names,
        frequencies=frequencies,
        save_dir=tmp_path,
        dt=0.01,
        kl_warm_epochs=[0],
        true_alphas=[0.02],
    )
    _assert_png(tmp_path / "vis_alpha_vs_power_spectrum_.png")
