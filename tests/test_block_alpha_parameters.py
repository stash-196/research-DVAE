"""Block alpha lists and ranges are the initial values of shared sigmas.

Synthetic tensors only. There is no extra timescale parameter. After
init, sigmas trains, logs, and checkpoints like any other MT-RNN.
optimize_alphas does not freeze those values.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dvae.model.mt_rnn import MT_RNN
from dvae.visualizers.visualize_training_metrics import (
    mt_alpha_from_sigma,
    visualize_alpha_history,
    visualize_sigma_history,
)


_HEART = (0.11, 0.18, 0.25, 0.32, 0.40, 0.48, 0.55, 0.63, 0.72)
_BRAIN = (0.16, 0.22, 0.29, 0.36, 0.44, 0.51, 0.58, 0.66, 0.74)
_REPO = Path(__file__).resolve().parents[1]


def _cfg(optimize, **network):
    parser = configparser.ConfigParser()
    base = {
        "x_dim": "4",
        "dim_rnn": "36",
        "hidden_sh_size": "6",
        "type_rnn": "RNN",
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "18,18",
        "ecg_channels": "0,1",
        "eeg_channels": "2,3",
        "heart_alphas": ",".join(str(value) for value in _HEART),
        "brain_alphas": ",".join(str(value) for value in _BRAIN),
    }
    base.update(network)
    parser["Network"] = {key: str(value) for key, value in base.items()}
    parser["Training"] = {"optimize_alphas": "true" if optimize else "false"}
    return parser


def _model(optimize, **network):
    dim = int(network.get("dim_rnn", 36))
    return MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=4,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=dim,
        cfg=_cfg(optimize, **network),
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )


def _tiled(values, width):
    src = torch.tensor(values, dtype=torch.float32)
    repeats = width // src.numel()
    remainder = width % src.numel()
    parts = [src for _ in range(repeats)]
    if remainder:
        parts.append(src[:remainder])
    return torch.cat(parts)


def _alpha_group(model, alpha_lr=0.05, base_lr=1e-4):
    """Same two groups ``LearningAlgo.init_optimizer`` builds for MT-RNN."""
    base = list(model.base_parameters())
    alpha = [model.sigmas]
    assert set(map(id, base)).isdisjoint(set(map(id, alpha)))
    return torch.optim.Adam(
        [
            {"params": base, "lr": base_lr},
            {"params": alpha, "lr": alpha_lr},
        ]
    )


def _group_of(opt, param):
    target = id(param)
    for group in opt.param_groups:
        if any(id(item) == target for item in group["params"]):
            return group
    raise AssertionError("parameter is not in the optimizer")


def test_block_specs_only_initialize_the_shared_sigmas():
    model = _model(True)
    names = [name for name, _ in model.named_parameters()]
    assert "sigmas" in names
    assert "heart_sigmas" not in names
    assert "brain_sigmas" not in names
    assert "heart_sigmas" not in model.state_dict()
    assert "brain_sigmas" not in model.state_dict()
    assert model.sigmas.shape == (36,)
    got = model.alphas_per_unit().detach()
    torch.testing.assert_close(got[:18], _tiled(_HEART, 18))
    torch.testing.assert_close(got[18:], _tiled(_BRAIN, 18))
    # The list is written once per unit. The copies are not tied:
    # a step on the first nine leaves the repeated nine in place.
    assert torch.allclose(got[:9], got[9:18])
    opt = _alpha_group(model, alpha_lr=0.2, base_lr=1e-4)
    seeded = model.sigmas.detach().clone()
    opt.zero_grad()
    (model.alphas_per_unit()[:9] - 0.95).pow(2).sum().backward()
    opt.step()
    assert not torch.allclose(model.sigmas[:9], seeded[:9])
    torch.testing.assert_close(model.sigmas[9:], seeded[9:])
    with torch.no_grad():
        model.sigmas.copy_(seeded)

    opt = _alpha_group(model, alpha_lr=0.2, base_lr=1e-4)
    group = _group_of(opt, model.sigmas)
    assert group["lr"] == 0.2
    other = next(item for item in opt.param_groups if item is not group)
    assert other["lr"] == 1e-4
    assert _group_of(opt, model.rnn.weight_hh_l0) is other

    model.zero_grad()
    model(torch.randn(4, 2, 4)).sum().backward()
    assert model.sigmas.grad.abs().sum().item() > 0.0

    before = model.sigmas.detach().clone()
    for _ in range(4):
        opt.zero_grad()
        alphas = model.alphas_per_unit()
        loss = (alphas[:18] - 0.9).pow(2).mean() + (alphas[18:] - 0.15).pow(2).mean()
        loss.backward()
        assert model.sigmas.grad.abs().sum().item() > 0.0
        opt.step()
    assert not torch.allclose(model.sigmas, before)
    moved = model.alphas_per_unit().detach()
    assert not torch.allclose(moved[:18], _tiled(_HEART, 18))


def test_a_range_is_one_initial_value_per_unit():
    model = _model(
        False,
        dim_rnn="8",
        hidden_block_split="4,4",
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alpha_range="0.3,0.5",
    )
    assert model.sigmas.shape == (8,)
    got = model.alphas_per_unit().detach()
    torch.testing.assert_close(got[:4], torch.linspace(0.2, 0.8, 4))
    torch.testing.assert_close(got[4:], torch.linspace(0.3, 0.5, 4))
    opt = _alpha_group(model, alpha_lr=0.3)
    before = model.sigmas.detach().clone()
    opt.zero_grad()
    (model.alphas_per_unit() - 0.95).pow(2).sum().backward()
    assert torch.count_nonzero(model.sigmas.grad).item() == 8
    opt.step()
    assert not torch.allclose(model.sigmas, before)


def test_optimize_alphas_does_not_change_block_initialization():
    trained = _model(True)
    frozen_flag = _model(False)
    torch.testing.assert_close(trained.sigmas, frozen_flag.sigmas)
    assert trained.sigmas.requires_grad
    assert frozen_flag.sigmas.requires_grad
    expected = torch.cat((_tiled(_HEART, 18), _tiled(_BRAIN, 18)))
    torch.testing.assert_close(frozen_flag.alphas_per_unit().detach(), expected)
    opt = _alpha_group(frozen_flag, alpha_lr=0.5)
    before = frozen_flag.sigmas.detach().clone()
    opt.zero_grad()
    (frozen_flag.alphas_per_unit() - 0.4).pow(2).sum().backward()
    opt.step()
    assert not torch.allclose(frozen_flag.sigmas, before)


def test_a_split_without_a_block_spec_keeps_the_shared_tile():
    model = _model(
        True,
        dim_rnn="4",
        hidden_block_split="2,2",
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="",
        brain_alpha_range="",
    )
    assert model.sigmas.shape == (2,)
    torch.testing.assert_close(
        model.alphas_per_unit().detach(),
        torch.tensor([0.2, 0.8, 0.2, 0.8]),
    )
    heart = model.hidden_block.heart
    assert model.block_hh_mask.shape == (4, 4)
    assert model.block_hh_mask[:heart, heart:].max().item() == 0.0
    assert model.block_hh_mask[:heart, :heart].min().item() == 1.0


def test_a_blank_side_keeps_the_tiled_shared_slice():
    model = _model(
        True,
        dim_rnn="4",
        hidden_block_split="1,3",
        heart_alphas="0.1",
        brain_alphas="",
        brain_alpha_range="",
    )
    assert model.sigmas.shape == (4,)
    torch.testing.assert_close(
        model.alphas_per_unit().detach(),
        torch.tensor([0.1, 0.8, 0.2, 0.8]),
    )


def test_checkpoint_round_trip_restores_sigmas_and_adam_state(tmp_path):
    model = _model(True)
    opt = _alpha_group(model, alpha_lr=0.2)
    for _ in range(3):
        opt.zero_grad()
        (model.alphas_per_unit() - 0.25).pow(2).sum().backward()
        opt.step()
    path = tmp_path / "MT_RNN_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": opt.state_dict(),
        },
        path,
    )
    fresh = _model(True)
    assert not torch.allclose(fresh.sigmas, model.sigmas)
    blob = torch.load(path, weights_only=False)
    fresh.load_state_dict(blob["model_state_dict"])
    fresh_opt = _alpha_group(fresh, alpha_lr=0.2)
    fresh_opt.load_state_dict(blob["optim_state_dict"])
    torch.testing.assert_close(fresh.sigmas, model.sigmas)
    torch.testing.assert_close(fresh.alphas_per_unit(), model.alphas_per_unit())
    group = _group_of(fresh_opt, fresh.sigmas)
    assert group["lr"] == 0.2
    saved_step = opt.state[model.sigmas]["step"]
    loaded_step = fresh_opt.state[fresh.sigmas]["step"]
    assert int(loaded_step) == int(saved_step)
    torch.testing.assert_close(
        fresh_opt.state[fresh.sigmas]["exp_avg"],
        opt.state[model.sigmas]["exp_avg"],
    )


def test_sigmas_history_has_one_row_per_sigma_and_uses_base10(tmp_path):
    model = _model(True)
    logged = model.sigmas.detach()
    assert logged.shape == (36,)
    history = torch.stack([logged, logged + 0.1, logged - 0.05], dim=1).numpy()
    assert history.shape == (36, 3)
    np.testing.assert_allclose(
        mt_alpha_from_sigma(logged.numpy()),
        model.alphas_per_unit().detach().numpy(),
    )
    assert mt_alpha_from_sigma(np.array([1.0]))[0] == pytest.approx(
        1 / (1 + 10 ** (-1))
    )

    calls = {"n": 0}
    original = plt.plot

    def _count(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    plt.plot = _count
    try:
        visualize_sigma_history(
            history, "MT_RNN", str(tmp_path), "block", file_tag="sigma"
        )
        sigma_lines = calls["n"]
        calls["n"] = 0
        visualize_alpha_history(
            history, "MT_RNN", str(tmp_path), "block", file_tag="alpha"
        )
        alpha_lines = calls["n"]
    finally:
        plt.plot = original
    assert sigma_lines == 36
    assert alpha_lines == 36
    assert os.path.isfile(tmp_path / "vis_training_history_of_sigma_sigma.png")
    assert os.path.isfile(tmp_path / "vis_training_history_of_alpha_alpha.png")


def test_the_block_weight_mask_stays_on():
    model = _model(True)
    heart = model.hidden_block.heart
    assert model.block_hh_mask.shape == (36, 36)
    assert model.block_hh_mask[:heart, heart:].max().item() == 0.0
    assert model.block_hh_mask[heart:, :heart].max().item() == 0.0
    assert model.block_hh_mask[:heart, :heart].min().item() == 1.0
    assert model.decoder_mask.shape == (4, 36)


def test_training_loop_still_uses_only_the_shared_sigmas():
    text = (_REPO / "src" / "dvae" / "learning_algo.py").read_text()
    model_text = (_REPO / "src" / "dvae" / "model" / "mt_rnn.py").read_text()
    assert '{"params": [self.model.sigmas], "lr": self.alpha_lr}' in text
    assert "alpha_parameters" not in text
    assert "logged_sigmas" not in text
    assert "heart_sigmas" not in model_text
    assert "brain_sigmas" not in model_text
    assert "self.model.sigmas.detach().cpu().numpy()" in text
