"""Trainable per-block MT-RNN alphas when optimize_alphas is on.

Synthetic tensors only. optimize_alphas false keeps the constant tile
and the constant range. The learnable values use the same base-10
sigmoid as the shared sigmas, sit in the alpha_lr group, and are
rows of the sigma history that the existing alpha figure draws.
"""

from __future__ import annotations

import configparser
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from dvae.model.mt_rnn import MT_RNN
from dvae.visualizers.visualize_training_metrics import (
    visualize_alpha_history,
    visualize_sigma_history,
)


_HEART = (0.11, 0.18, 0.25, 0.32, 0.40, 0.48, 0.55, 0.63, 0.72)
_BRAIN = (0.16, 0.22, 0.29, 0.36, 0.44, 0.51, 0.58, 0.66, 0.74)


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
    base = list(model.base_parameters())
    alpha = list(model.alpha_parameters())
    assert set(map(id, base)).isdisjoint(set(map(id, alpha)))
    opt = torch.optim.Adam(
        [
            {"params": base, "lr": base_lr},
            {"params": alpha, "lr": alpha_lr},
        ]
    )
    return opt


def _group_of(opt, param):
    target = id(param)
    for group in opt.param_groups:
        if any(id(item) == target for item in group["params"]):
            return group
    raise AssertionError("parameter is not in the optimizer")


def test_trainable_block_alphas_get_grads_and_move():
    model = _model(True)
    assert model.heart_sigmas.shape == (9,)
    assert model.brain_sigmas.shape == (9,)
    assert "heart_sigmas" in model.state_dict()
    assert "brain_sigmas" in model.state_dict()
    got = model.alphas_per_unit()
    torch.testing.assert_close(got[:18], _tiled(_HEART, 18))
    torch.testing.assert_close(got[18:], _tiled(_BRAIN, 18))
    # Nine values, each written twice. The two copies share one logit.
    assert torch.allclose(got[:9], got[9:18])

    opt = _alpha_group(model, alpha_lr=0.2)
    group = _group_of(opt, model.heart_sigmas)
    assert group["lr"] == 0.2
    assert _group_of(opt, model.brain_sigmas) is group
    assert _group_of(opt, model.sigmas) is group
    other = next(item for item in opt.param_groups if item is not group)
    assert _group_of(opt, model.rnn.weight_hh_l0) is other

    model.zero_grad()
    model(torch.randn(4, 2, 4)).sum().backward()
    assert model.heart_sigmas.grad.abs().sum().item() > 0.0
    assert model.brain_sigmas.grad.abs().sum().item() > 0.0

    before_heart = model.heart_sigmas.detach().clone()
    before_brain = model.brain_sigmas.detach().clone()
    for _ in range(4):
        opt.zero_grad()
        alphas = model.alphas_per_unit()
        loss = (alphas[:18] - 0.9).pow(2).mean() + (alphas[18:] - 0.15).pow(2).mean()
        loss.backward()
        assert model.heart_sigmas.grad.abs().sum().item() > 0.0
        assert model.brain_sigmas.grad.abs().sum().item() > 0.0
        opt.step()
    assert not torch.allclose(model.heart_sigmas, before_heart)
    assert not torch.allclose(model.brain_sigmas, before_brain)
    moved = model.alphas_per_unit().detach()
    assert not torch.allclose(moved[:18], _tiled(_HEART, 18))
    assert torch.allclose(moved[:9], moved[9:18])


def test_a_range_is_one_learnable_value_per_unit():
    model = _model(
        True,
        dim_rnn="8",
        hidden_block_split="4,4",
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alpha_range="0.3,0.5",
    )
    assert model.heart_sigmas.shape == (4,)
    assert model.brain_sigmas.shape == (4,)
    got = model.alphas_per_unit().detach()
    torch.testing.assert_close(got[:4], torch.linspace(0.2, 0.8, 4))
    torch.testing.assert_close(got[4:], torch.linspace(0.3, 0.5, 4))
    opt = _alpha_group(model, alpha_lr=0.3)
    before = model.heart_sigmas.detach().clone()
    opt.zero_grad()
    (model.alphas_per_unit() - 0.95).pow(2).sum().backward()
    assert torch.count_nonzero(model.heart_sigmas.grad).item() == 4
    assert torch.count_nonzero(model.brain_sigmas.grad).item() == 4
    opt.step()
    assert not torch.allclose(model.heart_sigmas, before)


def test_frozen_block_alphas_stay_the_constant_tile():
    model = _model(False)
    assert model.heart_sigmas is None
    assert model.brain_sigmas is None
    assert "heart_sigmas" not in model.state_dict()
    assert "brain_sigmas" not in model.state_dict()
    expected = torch.cat((_tiled(_HEART, 18), _tiled(_BRAIN, 18)))
    before = model.alphas_per_unit().detach().clone()
    torch.testing.assert_close(before, expected)
    # The shared logits are still a parameter. Stepping them must not
    # move a block that was replaced by a constant list.
    opt = torch.optim.Adam([model.sigmas], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        model.sigmas.sum().backward()
        opt.step()
    torch.testing.assert_close(model.alphas_per_unit().detach(), before)
    assert not torch.allclose(model.sigmas.detach(), MT_RNN.inverse_sigmoid_10(
        torch.tensor([0.2, 0.8])
    ))


def test_checkpoint_round_trip_restores_block_logits_and_adam_state(tmp_path):
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
    assert not torch.allclose(fresh.heart_sigmas, model.heart_sigmas)
    blob = torch.load(path, weights_only=False)
    fresh.load_state_dict(blob["model_state_dict"])
    fresh_opt = _alpha_group(fresh, alpha_lr=0.2)
    fresh_opt.load_state_dict(blob["optim_state_dict"])
    torch.testing.assert_close(fresh.heart_sigmas, model.heart_sigmas)
    torch.testing.assert_close(fresh.brain_sigmas, model.brain_sigmas)
    torch.testing.assert_close(fresh.alphas_per_unit(), model.alphas_per_unit())
    group = _group_of(fresh_opt, fresh.heart_sigmas)
    assert group["lr"] == 0.2
    assert _group_of(fresh_opt, fresh.brain_sigmas) is group
    saved_step = opt.state[model.heart_sigmas]["step"]
    loaded_step = fresh_opt.state[fresh.heart_sigmas]["step"]
    assert int(loaded_step) == int(saved_step)
    torch.testing.assert_close(
        fresh_opt.state[fresh.heart_sigmas]["exp_avg"],
        opt.state[model.heart_sigmas]["exp_avg"],
    )


def test_block_logits_are_rows_of_the_existing_alpha_figure(tmp_path):
    model = _model(True)
    logged = model.logged_sigmas().detach()
    assert logged.shape == (2 + 9 + 9,)
    torch.testing.assert_close(logged[:2], model.sigmas.detach())
    torch.testing.assert_close(logged[2:11], model.heart_sigmas.detach())
    torch.testing.assert_close(logged[11:], model.brain_sigmas.detach())
    history = torch.stack([logged, logged + 0.1, logged - 0.05], dim=1).numpy()
    assert history.shape == (20, 3)

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
    assert sigma_lines == 20
    assert alpha_lines == 20
    assert os.path.isfile(tmp_path / "vis_training_history_of_sigma_sigma.png")
    assert os.path.isfile(tmp_path / "vis_training_history_of_alpha_alpha.png")
