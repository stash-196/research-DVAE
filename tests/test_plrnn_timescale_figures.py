"""Training-time sigma/alpha figures for PLRNN and shPLRNN.

MT_RNN writes vis_training_history_of_{sigma,alpha}_{tag}.png from
model.sigmas. PLRNN-family cells store the analogous timescale as
rnn.A_sigmas (diagonal A = sigmoid_10(A_sigmas)). These tests generate
those figures without a dataset or GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dvae.timescale_history import (
    attach_timescale_history,
    diagonal_A_from_sigmas,
    init_A_sigmas_history,
    log_plrnn_timescales,
    save_plrnn_timescale_figures,
    update_A_sigmas_history,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG"


class _ListLogger:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(message)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics():
    return _load_module(
        "visualize_training_metrics_under_test",
        REPO_ROOT / "src" / "dvae" / "visualizers" / "visualize_training_metrics.py",
    )


def _plrnn_module():
    pytest.importorskip("torch")
    return _load_module(
        "plrnn_cells_under_test",
        REPO_ROOT / "src" / "dvae" / "model" / "plrnn.py",
    )


def _assert_png(path):
    raw = Path(path).read_bytes()
    assert raw.startswith(PNG_MAGIC)
    assert len(raw) > 1000


def test_learning_algo_still_logs_mt_alphas_and_plots_plrnn():
    text = (REPO_ROOT / "src" / "dvae" / "learning_algo.py").read_text()
    assert "save_plrnn_timescale_figures(" in text
    assert "update_A_sigmas_history(" in text
    assert "init_A_sigmas_history(" in text
    assert "log_plrnn_timescales(" in text
    assert "visualize_sigma_history(" in text
    assert "visualize_alpha_history(" in text
    assert "1 / (1 + np.exp(-sigmas_history[:, epoch]))" in text
    assert "1 / (1 + np.exp(-sigmas_history[:, -1]))" in text


def test_diagonal_A_matches_plrnn_and_shplrnn_cells():
    torch = pytest.importorskip("torch")
    plrnn = _plrnn_module()
    logits = torch.tensor([1.0, 0.0, -1.0, 0.5])

    for cell in (
        plrnn.PLRNN(input_size=2, hidden_size=4),
        plrnn.shPLRNN(input_size=2, hidden_size=4, hidden_sh_size=6),
    ):
        with torch.no_grad():
            cell.A_sigmas.copy_(logits)
        np.testing.assert_allclose(
            diagonal_A_from_sigmas(cell.A_sigmas.detach().numpy()),
            cell.A.detach().numpy(),
        )
        model = type("Model", (), {"rnn": cell})()
        history = init_A_sigmas_history(model, epochs=3)
        assert history.shape == (4, 3)
        np.testing.assert_allclose(history[:, 0], logits.numpy())

        with torch.no_grad():
            cell.A_sigmas.add_(0.25)
        update_A_sigmas_history(history, model, epoch=1)
        np.testing.assert_allclose(history[:, 1], cell.A_sigmas.detach().numpy())
        np.testing.assert_allclose(history[:, 0], logits.numpy())

    bare = plrnn.shPLRNN_wo_A(input_size=2, hidden_size=4, hidden_sh_size=6)
    assert init_A_sigmas_history(type("Model", (), {"rnn": bare})(), 3) is None
    assert init_A_sigmas_history(object(), 3) is None


def test_log_lines_match_mt_format_and_use_base10_sigmoid():
    history = np.array([[1.0, 0.5], [0.0, -1.0]], dtype=np.float64)
    logger = _ListLogger()
    log_plrnn_timescales(logger, history, epoch=0)
    assert logger.lines == [
        "alphas: ['0.90909', '0.50000']",
        "A_sigmas: ['1.00000', '0.00000']",
    ]
    # Natural sigmoid of 1 is ~0.73106; the log must not use that.
    assert "0.73106" not in logger.lines[0]

    mt_logger = _ListLogger()
    log_plrnn_timescales(mt_logger, history, epoch=0, alongside_mt=True, final=True)
    assert mt_logger.lines[0].startswith("Final diagonal A:")
    assert mt_logger.lines[1].startswith("Final A_sigmas:")
    log_plrnn_timescales(mt_logger, None, epoch=0)
    assert len(mt_logger.lines) == 2


def test_pickle_keeps_mt_sigmas_and_adds_plrnn_A_sigmas():
    mt = np.array([[0.1, 0.2]])
    plrnn = np.array([[0.3, 0.4]])
    both = attach_timescale_history(
        {}, sigmas_history=mt, a_sigmas_history=plrnn
    )
    assert both["sigmas_history"] is mt
    assert both["A_sigmas_history"] is plrnn

    only_plrnn = attach_timescale_history({}, a_sigmas_history=plrnn)
    assert only_plrnn["sigmas_history"] is plrnn
    assert only_plrnn["A_sigmas_history"] is plrnn

    assert attach_timescale_history({}) == {}


def test_default_alpha_figure_still_uses_natural_sigmoid(tmp_path):
    metrics = _metrics()
    seen = {}
    original = metrics.mt_alpha_from_sigma

    def _wrapped(sigmas):
        seen["value"] = original(sigmas)
        return seen["value"]

    metrics.mt_alpha_from_sigma = _wrapped
    metrics.visualize_alpha_history(
        np.array([[1.0, 1.0]]),
        "MT_RNN",
        tmp_path,
        "MT_RNN",
    )
    assert seen["value"][0] == pytest.approx(1 / (1 + np.exp(-1)))
    alpha_path = tmp_path / "vis_training_history_of_alpha_MT_RNN.png"
    sigma_called = tmp_path / "vis_training_history_of_sigma_MT_RNN.png"
    _assert_png(alpha_path)
    assert not sigma_called.exists()
    metrics.visualize_sigma_history(
        np.array([[1.0, 0.2]]),
        "MT_RNN",
        tmp_path,
        "MT_RNN",
    )
    _assert_png(sigma_called)


def test_shplrnn_and_plrnn_write_the_same_history_figures(tmp_path):
    metrics = _metrics()
    history = np.zeros((3, 4), dtype=np.float64)
    history[:, 0] = [1.0, 0.0, -1.0]
    history[:, 1] = [0.5, -0.5, 0.2]
    captured = {}

    def _alpha(*args, **kwargs):
        captured["alpha_from_sigma"] = kwargs["alpha_from_sigma"]
        captured["series_name"] = kwargs["series_name"]
        captured["file_tag"] = kwargs["file_tag"]
        return metrics.visualize_alpha_history(*args, **kwargs)

    def _sigma(*args, **kwargs):
        captured["sigma_series"] = kwargs["series_name"]
        captured["sigma_tag"] = kwargs["file_tag"]
        return metrics.visualize_sigma_history(*args, **kwargs)

    tag = save_plrnn_timescale_figures(
        history[:, :2],
        "RNN",
        tmp_path,
        "shPLRNN",
        auto_warm_epochs=[0],
        sequence_len_epochs=[0],
        visualize_sigma_history=_sigma,
        visualize_alpha_history=_alpha,
    )
    assert tag == "shPLRNN"
    assert captured["file_tag"] == "shPLRNN"
    assert captured["series_name"] == "A"
    assert captured["sigma_series"] == "A_sigma"
    assert captured["alpha_from_sigma"](np.array([1.0]))[0] == pytest.approx(
        1.0 / (1.0 + 10.0 ** (-1.0))
    )
    _assert_png(tmp_path / "vis_training_history_of_sigma_shPLRNN.png")
    _assert_png(tmp_path / "vis_training_history_of_alpha_shPLRNN.png")

    save_plrnn_timescale_figures(
        history[:, :2],
        "RNN",
        tmp_path,
        "PLRNN",
        visualize_sigma_history=metrics.visualize_sigma_history,
        visualize_alpha_history=metrics.visualize_alpha_history,
    )
    _assert_png(tmp_path / "vis_training_history_of_sigma_PLRNN.png")
    _assert_png(tmp_path / "vis_training_history_of_alpha_PLRNN.png")

    # MT mixing-alpha filenames stay available for the unchanged MT call.
    save_plrnn_timescale_figures(
        history[:, :2],
        "MT_RNN",
        tmp_path,
        "MT_RNN",
        visualize_sigma_history=metrics.visualize_sigma_history,
        visualize_alpha_history=metrics.visualize_alpha_history,
    )
    _assert_png(tmp_path / "vis_training_history_of_sigma_MT_RNN_diagA.png")
    _assert_png(tmp_path / "vis_training_history_of_alpha_MT_RNN_diagA.png")
    # Suffix keeps these off the filenames the unchanged MT plot call uses.
    assert not (tmp_path / "vis_training_history_of_alpha_MT_RNN.png").exists()
    assert not (tmp_path / "vis_training_history_of_sigma_MT_RNN.png").exists()

    called = {"n": 0}

    def _boom(*_args, **_kwargs):
        called["n"] += 1

    assert (
        save_plrnn_timescale_figures(
            None,
            "RNN",
            tmp_path,
            "shPLRNN",
            visualize_sigma_history=_boom,
            visualize_alpha_history=_boom,
        )
        is None
    )
    assert called["n"] == 0
