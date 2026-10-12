"""Pin bf59f0b: block lists and ranges only seed the shared sigmas parameter.

Synthetic config and tensors. No extra heart or brain parameters.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dvae.eval.maybe_alphas import alphas_to_metric_list, maybe_alphas_per_unit
from dvae.eval.utils.jacobian_lyapunov import jacobian_alpha_mix
from dvae.learning_algo import LearningAlgorithm
from dvae.model.mt_rnn import MT_RNN
from dvae.training_history import (
    EPOCH_SERIES_KEYS,
    build_training_history,
    load_training_history,
    save_training_history,
)
from dvae.visualizers.visualize_training_metrics import (
    mt_alpha_from_sigma,
    visualize_alpha_history,
)
from dvae.visualizers.visualizers import visualize_variable_evolution


ROOT = Path(__file__).resolve().parents[2]


def _cfg(*, optimize="true", include_optimize=True, **network):
    parser = configparser.ConfigParser()
    base = {
        "x_dim": "4",
        "dim_rnn": "6",
        "type_rnn": "RNN",
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "off",
        "ecg_channels": "0,1",
        "eeg_channels": "2,3",
        "heart_alphas": "",
        "brain_alphas": "",
        "heart_alpha_range": "",
        "brain_alpha_range": "",
    }
    base.update(network)
    parser["Network"] = {key: str(value) for key, value in base.items()}
    training = {}
    if include_optimize:
        training["optimize_alphas"] = str(optimize)
    parser["Training"] = training
    return parser


def _model(*, seed=0, optimize="true", include_optimize=True, **network):
    torch.manual_seed(seed)
    cfg = _cfg(optimize=optimize, include_optimize=include_optimize, **network)
    alphas = [
        float(part)
        for part in cfg.get("Network", "alphas").split(",")
        if part.strip()
    ]
    return MT_RNN(
        alphas=alphas,
        x_dim=int(cfg.get("Network", "x_dim")),
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=int(cfg.get("Network", "dim_rnn")),
        cfg=cfg,
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )


def _legacy_sigmas(alphas):
    return torch.tensor(
        [MT_RNN.inverse_sigmoid_10(alpha) for alpha in alphas],
        dtype=torch.float32,
    )


def _tile(values, n):
    src = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if n <= src.numel():
        return src[:n]
    repeats = n // int(src.numel())
    remainder = n % int(src.numel())
    parts = [src for _ in range(repeats)]
    if remainder:
        parts.append(src[:remainder])
    return torch.cat(parts)


def _init_optimizer(model, base_lr="1e-4", alpha_lr="0.05"):
    model.cfg.set("Training", "optimization", "adam")
    model.cfg.set("Training", "lr", base_lr)
    model.cfg.set("Training", "alpha_lr", alpha_lr)
    holder = SimpleNamespace(model_name="MT_RNN", model=model, cfg=model.cfg)
    return LearningAlgorithm.init_optimizer(holder), holder


def test_block_lists_and_ranges_only_seed_the_shared_sigmas():
    model = _model(
        hidden_block_split="4,2",
        heart_alphas="0.2,0.3",
        brain_alphas="0.7",
    )
    names = [name for name, _ in model.named_parameters()]
    assert "heart_sigmas" not in names
    assert "brain_sigmas" not in names
    assert not hasattr(model, "heart_sigmas")
    assert not hasattr(model, "brain_sigmas")
    assert "heart_sigmas" not in model.state_dict()
    assert model.sigmas.shape == (6,)
    assert model.sigmas.requires_grad

    seeded = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.7, 0.7])
    assert torch.equal(
        model.sigmas.detach(),
        MT_RNN.inverse_sigmoid_10(seeded).to(dtype=torch.float32),
    )
    assert torch.equal(model.alphas_per_unit(), MT_RNN.sigmoid_10(model.sigmas))

    # One logit per unit: a loss on unit 0 does not move the repeated copy.
    model.zero_grad(set_to_none=True)
    model.alphas_per_unit()[0].backward()
    assert model.sigmas.grad[0].abs().item() > 0.0
    assert torch.equal(model.sigmas.grad[1:], torch.zeros(5))

    ranged = _model(
        dim_rnn="6",
        hidden_block_split="4,2",
        heart_alpha_range="0.2,0.8",
        brain_alphas="0.4,0.6",
    )
    heart = torch.linspace(0.2, 0.8, 4)
    brain = _tile((0.4, 0.6), 2)
    expected = torch.cat((heart, brain))
    assert ranged.sigmas.shape == (6,)
    assert torch.equal(
        ranged.sigmas.detach(),
        MT_RNN.inverse_sigmoid_10(expected).to(dtype=torch.float32),
    )
    ranged.zero_grad(set_to_none=True)
    ranged.alphas_per_unit()[1].backward()
    assert ranged.sigmas.grad[1].abs().item() > 0.0
    assert torch.count_nonzero(ranged.sigmas.grad).item() == 1

    blank_brain = _model(
        dim_rnn="6",
        hidden_block_split="4,2",
        heart_alphas="0.25",
    )
    shared = _tile((0.2, 0.8), 6)
    shared = shared.clone()
    shared[:4] = 0.25
    assert torch.equal(
        blank_brain.sigmas.detach(),
        MT_RNN.inverse_sigmoid_10(shared).to(dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="not both"):
        _model(
            hidden_block_split="4,2",
            heart_alphas="0.2,0.3",
            heart_alpha_range="0.2,0.8",
        )

    src = (ROOT / "src" / "dvae" / "model" / "mt_rnn.py").read_text()
    algo = (ROOT / "src" / "dvae" / "learning_algo.py").read_text()
    assert "heart_sigmas" not in src
    assert "brain_sigmas" not in src
    assert "heart_sigmas" not in algo
    assert "brain_sigmas" not in algo


def test_no_block_spec_matches_the_legacy_shared_sigmas():
    legacy = _model(seed=2, hidden_block_split="off")
    missing = _model(seed=2, include_optimize=False, hidden_block_split="")
    assert legacy.hidden_block is None
    assert missing.hidden_block is None
    assert legacy.sigmas.shape == (2,)
    assert torch.equal(legacy.sigmas.detach(), _legacy_sigmas((0.2, 0.8)))
    assert torch.equal(missing.sigmas.detach(), legacy.sigmas.detach())
    tiled = _tile(MT_RNN.sigmoid_10(legacy.sigmas).detach(), legacy.dim_rnn)
    assert torch.equal(legacy.alphas_per_unit(), tiled)
    assert list(legacy.state_dict()) == list(missing.state_dict())
    for key in legacy.state_dict():
        assert torch.equal(legacy.state_dict()[key], missing.state_dict()[key])
    torch.manual_seed(3)
    y_legacy = legacy(torch.ones(3, 2, 4))
    torch.manual_seed(3)
    y_missing = missing(torch.ones(3, 2, 4))
    assert torch.equal(y_legacy, y_missing)

    flagged = _model(seed=2, optimize="false", hidden_block_split="off")
    assert torch.equal(flagged.sigmas.detach(), legacy.sigmas.detach())
    for key in legacy.state_dict():
        assert torch.equal(flagged.state_dict()[key], legacy.state_dict()[key])

    # A split with no list and no range keeps the short shared parameter.
    split_only = _model(
        seed=2,
        hidden_block_split="4,2",
        ecg_channels="0,1",
        eeg_channels="2,3",
    )
    assert split_only.sigmas.shape == (2,)
    assert torch.equal(split_only.sigmas.detach(), legacy.sigmas.detach())
    assert torch.equal(split_only.alphas_per_unit(), tiled)


def test_per_unit_sigmas_still_train_log_and_round_trip(tmp_path):
    model = _model(
        hidden_block_split="4,2",
        heart_alphas="0.2,0.3",
        brain_alphas="0.7",
    )
    opt, holder = _init_optimizer(model, base_lr="1e-4", alpha_lr="0.05")
    assert holder.alpha_lr == 0.05
    base_ids = {id(param) for param in model.base_parameters()}
    assert id(model.sigmas) not in base_ids
    alpha_group = next(
        group for group in opt.param_groups if any(param is model.sigmas for param in group["params"])
    )
    assert alpha_group["lr"] == 0.05
    assert alpha_group["params"] == [model.sigmas]
    other = next(group for group in opt.param_groups if group is not alpha_group)
    assert other["lr"] == 1e-4
    assert id(model.rnn.weight_hh_l0) in {id(param) for param in other["params"]}

    turned_off = _model(
        optimize="false",
        hidden_block_split="4,2",
        heart_alphas="0.2,0.3",
        brain_alphas="0.7",
    )
    assert torch.equal(turned_off.sigmas.detach(), model.sigmas.detach())
    assert turned_off.sigmas.requires_grad
    off_opt, _ = _init_optimizer(turned_off, alpha_lr="0.05")
    off_group = next(
        group
        for group in off_opt.param_groups
        if any(param is turned_off.sigmas for param in group["params"])
    )
    assert off_group["lr"] == 0.05

    for _ in range(3):
        opt.zero_grad()
        (model.alphas_per_unit() - 0.4).pow(2).sum().backward()
        opt.step()
    fresh = _model(
        seed=1,
        hidden_block_split="4,2",
        heart_alphas="0.2,0.3",
        brain_alphas="0.7",
    )
    path = tmp_path / "MT_RNN_checkpoint.pt"
    history = np.zeros((model.sigmas.numel(), 2), dtype=np.float64)
    history[:, 0] = model.sigmas.detach().cpu().numpy().reshape(-1)
    torch.save(
        {
            "epoch": 2,
            "best_val_loss": 1.0,
            "cpt_patience": 0,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": opt.state_dict(),
            "loss_log": {"sigmas_history": history},
        },
        path,
    )
    blob = torch.load(path, weights_only=False)
    fresh.load_state_dict(blob["model_state_dict"])
    fresh_opt, _ = _init_optimizer(fresh, alpha_lr="0.05")
    fresh_opt.load_state_dict(blob["optim_state_dict"])
    assert torch.equal(fresh.sigmas.detach(), model.sigmas.detach())
    assert torch.equal(fresh.alphas_per_unit(), model.alphas_per_unit())
    assert int(fresh_opt.state[fresh.sigmas]["step"]) == int(opt.state[model.sigmas]["step"])
    assert torch.equal(
        fresh_opt.state[fresh.sigmas]["exp_avg"],
        opt.state[model.sigmas]["exp_avg"],
    )
    assert torch.equal(
        fresh_opt.state[fresh.sigmas]["exp_avg_sq"],
        opt.state[model.sigmas]["exp_avg_sq"],
    )

    series = {key: np.zeros(2, dtype=np.float64) for key in EPOCH_SERIES_KEYS}
    built = build_training_history(
        end=1,
        model_name="MT_RNN",
        tag="seed",
        auto_sampling_method="none",
        noise_sampling_method="none",
        delta_threshold=0.0,
        val_criteria=(),
        val_reference=None,
        series=series,
        dataset_stats={},
        sigmas_history=history,
        spectrum_monitored=False,
    )
    pickle_path = tmp_path / "loss_model.pckl"
    save_training_history(pickle_path, built)
    loaded = load_training_history(pickle_path)
    assert loaded["sigmas_history"].shape == (6, 1)
    assert np.array_equal(loaded["sigmas_history"][:, 0], history[:, 0])

    recorded = maybe_alphas_per_unit(model)
    assert recorded.shape == (model.dim_rnn,)
    assert torch.equal(recorded, model.alphas_per_unit())
    assert len(alphas_to_metric_list(recorded)) == model.dim_rnn
    mixed = jacobian_alpha_mix(torch.eye(model.dim_rnn), recorded.detach())
    assert mixed.shape == (model.dim_rnn, model.dim_rnn)

    states = torch.zeros(3, 1, model.dim_rnn)
    visualize_variable_evolution(
        states,
        torch.zeros(3, 1, 4),
        str(tmp_path),
        variable_name="hidden",
        alphas=recorded.detach(),
    )
    plt.close("all")


def test_alpha_figure_uses_base10_and_the_epoch_log_does_not(tmp_path):
    sigma = np.array([1.0, -0.5], dtype=np.float64)
    base10 = 1.0 / (1.0 + np.power(10.0, -sigma))
    natural = 1.0 / (1.0 + np.exp(-sigma))
    assert np.allclose(mt_alpha_from_sigma(sigma), base10)
    assert not np.allclose(base10, natural)

    history = np.vstack([sigma, sigma + 0.2])
    drawn = []
    original = plt.plot

    def _capture(*args, **kwargs):
        drawn.append(np.asarray(args[0], dtype=np.float64).copy())
        return original(*args, **kwargs)

    plt.plot = _capture
    try:
        visualize_alpha_history(history, "MT_RNN", str(tmp_path), "seed", file_tag="seed")
    finally:
        plt.plot = original
    assert len(drawn) == history.shape[0]
    for row, line in zip(history, drawn):
        assert np.allclose(line, mt_alpha_from_sigma(row))
        assert not np.allclose(line, 1.0 / (1.0 + np.exp(-row)))

    text = (ROOT / "src" / "dvae" / "learning_algo.py").read_text()
    assert "alphas = 1 / (1 + np.exp(-sigmas_history[:, epoch]))" in text
    assert "alphas = 1 / (1 + np.exp(-sigmas_history[:, -1]))" in text


def test_block_weight_masks_are_unchanged_by_the_sigma_seed():
    seeded = _model(
        seed=4,
        hidden_block_split="4,2",
        heart_alphas="0.2,0.3",
        brain_alphas="0.7",
    )
    plain = _model(seed=4, hidden_block_split="4,2")
    assert torch.equal(seeded.block_hh_mask, plain.block_hh_mask)
    assert torch.equal(seeded.decoder_mask, plain.decoder_mask)
    heart = 4
    assert torch.equal(seeded.block_hh_mask[:heart, heart:], torch.zeros(heart, 2))
    assert torch.equal(seeded.block_hh_mask[heart:, :heart], torch.zeros(2, heart))
    assert torch.equal(seeded.block_hh_mask[:heart, :heart], torch.ones(heart, heart))
    assert torch.equal(seeded.decoder_mask[:2, heart:], torch.zeros(2, 2))
    assert torch.equal(seeded.decoder_mask[2:, :heart], torch.zeros(2, heart))
    assert torch.equal(seeded.rnn.weight_hh_l0[:heart, heart:], torch.zeros(heart, 2))
    assert torch.equal(seeded.gen_out.weight[:2, heart:], torch.zeros(2, 2))

    opt, _ = _init_optimizer(seeded, base_lr="0.1", alpha_lr="0.1")
    opt.zero_grad()
    torch.manual_seed(5)
    seeded(torch.ones(3, 2, 4)).sum().backward()
    opt.step()
    assert torch.equal(seeded.rnn.weight_hh_l0[:heart, heart:], torch.zeros(heart, 2))
    assert torch.equal(seeded.rnn.weight_hh_l0[heart:, :heart], torch.zeros(2, heart))
    assert torch.equal(seeded.gen_out.weight[:2, heart:], torch.zeros(2, 2))
    assert torch.equal(seeded.gen_out.weight[2:, :heart], torch.zeros(2, heart))
