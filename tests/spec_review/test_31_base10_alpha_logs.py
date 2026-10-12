"""Pin 668181d: every sigma-to-alpha log and figure uses the base-10 sigmoid.

Synthetic logits only. A grep of src/dvae rejects a natural sigmoid of sigmas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from dvae.eval.alpha_vs_spectrum import save_alpha_vs_spectrum_figures
from dvae.learning_algo import sigmoid_10
from dvae.model.mt_rnn import MT_RNN
from dvae.model.mt_vrnn import MT_VRNN
from dvae.timescale_history import diagonal_A_from_sigmas, log_plrnn_timescales
from dvae.visualizers.visualize_training_metrics import (
    mt_alpha_from_sigma,
    visualize_alpha_history,
)
from dvae.visualizers.visualizers import visualize_alpha_history_and_spectrums


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "dvae"
SIGMAS = np.array([1.0, 0.0, -1.0], dtype=np.float64)


def _base10(sigmas):
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(sigmas, dtype=np.float64)))


def _natural(sigmas):
    return 1.0 / (1.0 + np.exp(-np.asarray(sigmas, dtype=np.float64)))


def _log_lines(sigmas):
    logged = sigmoid_10(sigmas)
    epoch = "alphas: {}".format([f"{alpha:.5f}" for alpha in logged])
    final = "Final alphas: {}".format([f"{alpha:.5f}" for alpha in logged])
    return logged, epoch, final


def test_epoch_and_final_logs_match_the_model_base10_sigmoid():
    logged, epoch, final = _log_lines(SIGMAS)
    model = MT_RNN.sigmoid_10(torch.tensor(SIGMAS, dtype=torch.float64)).numpy()
    vrnn = MT_VRNN.sigmoid_10(torch.tensor(SIGMAS, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(logged, _base10(SIGMAS))
    np.testing.assert_allclose(logged, model)
    np.testing.assert_allclose(logged, vrnn)
    np.testing.assert_allclose(logged, mt_alpha_from_sigma(SIGMAS))
    np.testing.assert_allclose(logged, diagonal_A_from_sigmas(SIGMAS))
    assert not np.allclose(logged, _natural(SIGMAS))

    # sigma = 1 -> 0.90909; sigma = -1 -> 0.09091. Natural would print 0.73106 and 0.26894.
    assert "0.90909" in epoch and "0.09091" in epoch
    assert "0.90909" in final and "0.09091" in final
    assert "0.73106" not in epoch and "0.26894" not in epoch
    assert "0.73106" not in final and "0.26894" not in final

    text = (SRC / "learning_algo.py").read_text()
    assert "alphas = sigmoid_10(sigmas_history[:, epoch])" in text
    assert "alphas = sigmoid_10(sigmas_history[:, -1])" in text
    assert "return 1 / (1 + 10 ** (-x))" in text


def test_alpha_figures_draw_the_base10_sigmoid(tmp_path):
    history = np.vstack([SIGMAS, SIGMAS + 0.25])
    drawn = []
    original = plt.plot

    def _capture(*args, **kwargs):
        drawn.append(np.asarray(args[0], dtype=np.float64).copy())
        return original(*args, **kwargs)

    plt.plot = _capture
    try:
        visualize_alpha_history(
            history, "MT_RNN", str(tmp_path), "pin", file_tag="pin"
        )
    finally:
        plt.plot = original
    assert len(drawn) == history.shape[0]
    for row, line in zip(history, drawn):
        np.testing.assert_allclose(line, _base10(row))
        assert not np.allclose(line, _natural(row))

    captured = {}
    real_savefig = plt.savefig

    def _save(*args, **kwargs):
        ax = plt.gcf().axes[0]
        captured["texts"] = [text.get_text() for text in ax.texts]
        captured["labels"] = [
            line.get_label() for line in ax.get_lines() if line.get_label().startswith("α ")
        ]
        captured["ylim"] = ax.get_ylim()
        return real_savefig(*args, **kwargs)

    plt.savefig = _save
    try:
        visualize_alpha_history_and_spectrums(
            sigmas_history=np.array([[1.0, -1.0]], dtype=np.float64),
            power_spectrum_lst=[np.array([1.0, 0.5])],
            spectrum_color_lst=["blue"],
            spectrum_name_lst=["x"],
            frequencies=np.array([1.0, 10.0]),
            save_dir=tmp_path,
            dt=0.01,
            explain="pin",
        )
    finally:
        plt.savefig = real_savefig
    joined = "\n".join(captured["texts"])
    assert f"{_base10(1.0):.4g}" in joined
    assert f"{_base10(-1.0):.4g}" in joined
    assert f"{_natural(1.0):.4g}" not in joined
    assert f"{_natural(-1.0):.4g}" not in joined
    assert captured["labels"] == ["α 1"]
    assert (tmp_path / "vis_alpha_vs_power_spectrum_pin.png").is_file()


def test_eight_mt_curves_still_stop_at_seven_and_the_tiny_axis_uses_the_floor(tmp_path):
    captured = {}
    real_savefig = plt.savefig

    def _save(*args, **kwargs):
        ax = plt.gcf().axes[0]
        captured["n"] = sum(
            line.get_label().startswith("α ") for line in ax.get_lines()
        )
        captured["ylim"] = ax.get_ylim()
        return real_savefig(*args, **kwargs)

    wide = np.ones((8, 2), dtype=np.float64)
    plt.savefig = _save
    try:
        visualize_alpha_history_and_spectrums(
            sigmas_history=wide,
            power_spectrum_lst=[np.ones(2)],
            spectrum_color_lst=["blue"],
            spectrum_name_lst=["x"],
            frequencies=np.array([1.0, 10.0]),
            save_dir=tmp_path,
            dt=0.01,
        )
        assert captured["n"] == 7

        visualize_alpha_history_and_spectrums(
            sigmas_history=np.full((1, 2), -20.0),
            power_spectrum_lst=[np.ones(2)],
            spectrum_color_lst=["blue"],
            spectrum_name_lst=["x"],
            frequencies=np.array([1.0, 10.0]),
            save_dir=tmp_path,
            dt=0.01,
            explain="floor",
        )
    finally:
        plt.savefig = real_savefig
    # max(sigmoid_10(-20), 1e-8) / 10 == 1e-9, so the period axis tops out at dt / 1e-9.
    assert captured["ylim"][1] == np.float64(0.01 / 1e-9)


def test_plrnn_log_and_mt_eval_figure_stay_on_base10(tmp_path):
    messages = []

    class _Logger:
        def info(self, message):
            messages.append(message)

    log_plrnn_timescales(_Logger(), SIGMAS.reshape(-1, 1), 0)
    assert messages[0].startswith("alphas: ")
    assert "0.90909" in messages[0]
    assert "0.73106" not in messages[0]

    class _Dataset:
        dataset_name = "Lorenz63"
        true_alphas = [0.02]

    texts = []
    real_close = plt.close

    def _close(*args, **kwargs):
        texts.extend(text.get_text() for text in plt.gcf().axes[0].texts)
        return real_close(*args, **kwargs)

    plt.close = _close
    try:
        saved = save_alpha_vs_spectrum_figures(
            {"sigmas_history": np.array([[1.0, -1.0]], dtype=np.float64)},
            "MT_RNN",
            {
                "dt": 0.01,
                "channels": [
                    {
                        "key": "x",
                        "name": "x",
                        "color": "blue",
                        "gt_full": np.sin(np.linspace(0, 8, 64)),
                    }
                ],
            },
            _Dataset(),
            tmp_path,
            visualize_alpha_history_and_spectrums,
        )
    finally:
        plt.close = real_close
    assert saved == [str(tmp_path / "vis_alpha_vs_power_spectrum_.png")]
    joined = "\n".join(texts)
    assert f"{_base10(1.0):.4g}" in joined
    assert f"{_natural(1.0):.4g}" not in joined


def test_src_dvae_has_no_natural_sigmoid_of_sigmas():
    forbidden = (
        "np.exp(-",
        "1 / (1 + np.exp",
        "1/(1+np.exp",
        "torch.sigmoid(sigmas",
        "torch.sigmoid(self.sigmas",
        "expit(",
    )
    hits = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        rel = path.relative_to(ROOT)
        for token in forbidden:
            if token in text:
                hits.append(f"{rel}: {token}")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "sigma" not in line.lower():
                continue
            if any(
                token in line
                for token in ("np.exp(", "torch.exp(", "math.exp(", "torch.sigmoid(")
            ):
                hits.append(f"{rel}:{lineno}: {line.strip()}")
    assert hits == []
