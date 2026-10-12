"""Pin trainable per-block MT-RNN alphas (commit 1b78ff1).

Synthetic config and tensors only. These tests call the model, the
real ``LearningAlgorithm.init_optimizer``, the history builder, and the
existing sigma/alpha figures. They do not change production code.
"""

from __future__ import annotations

import configparser
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dvae.learning_algo import LearningAlgorithm
from dvae.model.mt_rnn import MT_RNN
from dvae.training_history import (
    EPOCH_SERIES_KEYS,
    build_training_history,
    load_training_history,
    restore_matrix,
    save_training_history,
)
from dvae.visualizers.visualize_training_metrics import (
    mt_alpha_from_sigma,
    visualize_alpha_history,
    visualize_sigma_history,
)


_SHARED = (0.2, 0.8)


def _tile(values, n):
    """Independent copy of the block tile: repeat, then a prefix remainder."""
    src = torch.tensor(list(values), dtype=torch.float32)
    width = int(src.numel())
    if n <= width:
        return src[:n]
    repeats = n // width
    remainder = n % width
    parts = [src for _ in range(repeats)]
    if remainder:
        parts.append(src[:remainder])
    return torch.cat(parts)


def _cfg(optimize=True, *, include_optimize=True, **network):
    parser = configparser.ConfigParser()
    base = {
        "x_dim": "4",
        "dim_rnn": "6",
        "type_rnn": "RNN",
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "4,2",
        "ecg_channels": "0,1",
        "eeg_channels": "2,3",
        "heart_alphas": "0.2,0.3",
        "brain_alphas": "0.7",
    }
    base.update(network)
    parser["Network"] = {key: str(value) for key, value in base.items()}
    training = {}
    if include_optimize:
        if optimize is True:
            training["optimize_alphas"] = "true"
        elif optimize is False:
            training["optimize_alphas"] = "false"
        else:
            training["optimize_alphas"] = str(optimize)
    parser["Training"] = training
    return parser


def _model(optimize=True, *, include_optimize=True, seed=0, **network):
    torch.manual_seed(seed)
    cfg = _cfg(optimize, include_optimize=include_optimize, **network)
    dim = int(cfg.get("Network", "dim_rnn"))
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
        dim_rnn=dim,
        cfg=cfg,
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )


def _optim_cfg(model, base_lr="1e-4", alpha_lr="0.05"):
    cfg = model.cfg
    cfg.set("Training", "optimization", "adam")
    cfg.set("Training", "lr", base_lr)
    cfg.set("Training", "alpha_lr", alpha_lr)
    return cfg


def _init_optimizer(model, base_lr="1e-4", alpha_lr="0.05"):
    holder = SimpleNamespace(
        model_name="MT_RNN",
        model=model,
        cfg=_optim_cfg(model, base_lr, alpha_lr),
    )
    return LearningAlgorithm.init_optimizer(holder), holder


def _group_of(opt, param):
    target = id(param)
    found = [group for group in opt.param_groups if any(id(item) == target for item in group["params"])]
    assert len(found) == 1
    return found[0]


def _shared_per_unit(model):
    alphas = MT_RNN.sigmoid_10(model.sigmas)
    return _tile(alphas.detach(), model.dim_rnn)


def _grad_abs(param):
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().abs().sum())


def _backward_channels(model, channels):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    y = model(torch.ones(3, 2, model.x_dim))
    y[:, :, list(channels)].sum().backward()


def _capture_plots(fn, history, tmp_path, tag):
    drawn = []
    original = plt.plot

    def _capture(*args, **kwargs):
        drawn.append(np.asarray(args[0], dtype=np.float64).copy())
        return original(*args, **kwargs)

    plt.plot = _capture
    try:
        fn(history, "MT_RNN", str(tmp_path), tag, file_tag=tag)
    finally:
        plt.plot = original
    return drawn


def test_trainable_lists_use_the_shared_base10_map_and_tile():
    model = _model(True)
    assert isinstance(model.heart_sigmas, torch.nn.Parameter)
    assert isinstance(model.brain_sigmas, torch.nn.Parameter)
    assert model.heart_sigmas.requires_grad and model.brain_sigmas.requires_grad

    heart_logits = MT_RNN.inverse_sigmoid_10(torch.tensor([0.2, 0.3])).float()
    brain_logits = MT_RNN.inverse_sigmoid_10(torch.tensor([0.7])).float()
    shared_logits = torch.tensor(
        [MT_RNN.inverse_sigmoid_10(alpha) for alpha in _SHARED],
        dtype=torch.float32,
    )
    assert torch.equal(model.heart_sigmas.detach(), heart_logits)
    assert torch.equal(model.brain_sigmas.detach(), brain_logits)
    assert torch.equal(model.sigmas.detach(), shared_logits)
    # Same numeric alpha, same logit as the shared MT-RNN parameter.
    assert torch.equal(model.heart_sigmas.detach()[0], model.sigmas.detach()[0])

    got = model.alphas_per_unit()
    heart = _tile(MT_RNN.sigmoid_10(model.heart_sigmas).detach(), 4)
    brain = _tile(MT_RNN.sigmoid_10(model.brain_sigmas).detach(), 2)
    assert torch.equal(got[:4], heart)
    assert torch.equal(got[4:], brain)
    assert torch.equal(got[:2], got[2:4])

    remainder = _model(
        True,
        dim_rnn="8",
        hidden_block_split="5,3",
        heart_alphas="0.2,0.3",
        brain_alphas="0.4,0.5,0.6,0.7",
    )
    rem = remainder.alphas_per_unit()
    assert torch.equal(
        rem[:5],
        _tile(MT_RNN.sigmoid_10(remainder.heart_sigmas).detach(), 5),
    )
    # A list longer than the block is the prefix, same as the shared tile.
    assert remainder.brain_sigmas.shape == (4,)
    assert torch.equal(
        rem[5:],
        MT_RNN.sigmoid_10(remainder.brain_sigmas).detach()[:3],
    )


def test_block_logits_sit_only_in_the_alpha_lr_group():
    model = _model(True)
    opt, holder = _init_optimizer(model, base_lr="1e-4", alpha_lr="0.05")
    assert holder.lr == 1e-4
    assert holder.alpha_lr == 0.05
    assert len(opt.param_groups) == 2

    base_ids = [id(param) for param in model.base_parameters()]
    alpha_ids = [id(param) for param in model.alpha_parameters()]
    all_ids = [id(param) for param in model.parameters()]
    assert len(base_ids) == len(set(base_ids))
    assert len(alpha_ids) == len(set(alpha_ids))
    assert set(base_ids).isdisjoint(alpha_ids)
    assert sorted(base_ids + alpha_ids) == sorted(all_ids)

    alpha_group = _group_of(opt, model.heart_sigmas)
    assert alpha_group["lr"] == 0.05
    assert _group_of(opt, model.brain_sigmas) is alpha_group
    assert _group_of(opt, model.sigmas) is alpha_group
    assert [id(param) for param in alpha_group["params"]] == alpha_ids

    base_group = _group_of(opt, model.rnn.weight_hh_l0)
    assert base_group is not alpha_group
    assert base_group["lr"] == 1e-4
    assert id(model.heart_sigmas) not in {id(param) for param in base_group["params"]}
    assert id(model.brain_sigmas) not in {id(param) for param in base_group["params"]}
    sigma_names = [name for name, _ in model.named_parameters() if "sigmas" in name]
    assert sigma_names == ["sigmas", "heart_sigmas", "brain_sigmas"]


def test_optimize_false_matches_the_constant_replacement():
    frozen = _model(False)
    assert frozen.heart_sigmas is None
    assert frozen.brain_sigmas is None
    assert "heart_sigmas" not in frozen.state_dict()
    assert "brain_sigmas" not in frozen.state_dict()
    expected = torch.cat((_tile((0.2, 0.3), 4), _tile((0.7,), 2)))
    assert torch.equal(frozen.alphas_per_unit(), expected)
    assert torch.equal(frozen.logged_sigmas().detach(), frozen.sigmas.detach())

    omitted = _model(include_optimize=False, seed=0)
    assert omitted.heart_sigmas is None
    assert torch.equal(omitted.alphas_per_unit(), expected)
    assert list(frozen.state_dict()) == list(omitted.state_dict())
    for key in frozen.state_dict():
        assert torch.equal(frozen.state_dict()[key], omitted.state_dict()[key])

    before = frozen.alphas_per_unit().detach().clone()
    opt = torch.optim.Adam([frozen.sigmas], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        frozen.sigmas.sum().backward()
        opt.step()
    assert torch.equal(frozen.alphas_per_unit().detach(), before)
    torch.manual_seed(3)
    y_frozen = frozen(torch.ones(3, 2, 4))
    torch.manual_seed(3)
    y_omitted = omitted(torch.ones(3, 2, 4))
    assert torch.equal(y_frozen, y_omitted)

    # Installing the trainable logits does not consume the init RNG, so a
    # frozen model and a trainable model from the same seed share weights.
    trainable = _model(True, seed=0)
    assert torch.equal(frozen.rnn.weight_hh_l0, trainable.rnn.weight_hh_l0)
    assert torch.equal(frozen.rnn.weight_ih_l0, trainable.rnn.weight_ih_l0)
    assert torch.equal(frozen.gen_out.weight, trainable.gen_out.weight)

    ranged = _model(
        False,
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alpha_range="0.3,0.5",
    )
    assert torch.equal(ranged.alphas_per_unit()[:4], torch.linspace(0.2, 0.8, 4))
    assert torch.equal(ranged.alphas_per_unit()[4:], torch.linspace(0.3, 0.5, 2))


def test_checkpoint_restores_block_logits_and_adam_state(tmp_path):
    model = _model(True)
    opt, _ = _init_optimizer(model, alpha_lr="0.2")
    for _ in range(3):
        opt.zero_grad()
        (model.alphas_per_unit() - 0.25).pow(2).sum().backward()
        opt.step()
    loss_log = {"sigmas_history": model.logged_sigmas().detach().cpu().numpy().reshape(-1, 1)}
    path = tmp_path / "MT_RNN_checkpoint.pt"
    torch.save(
        {
            "epoch": 2,
            "best_val_loss": 1.5,
            "cpt_patience": 0,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": opt.state_dict(),
            "loss_log": loss_log,
        },
        path,
    )

    fresh = _model(True, seed=1)
    assert not torch.equal(fresh.heart_sigmas.detach(), model.heart_sigmas.detach())
    blob = torch.load(path, weights_only=False)
    fresh.load_state_dict(blob["model_state_dict"])
    fresh_opt, _ = _init_optimizer(fresh, alpha_lr="0.2")
    fresh_opt.load_state_dict(blob["optim_state_dict"])

    assert torch.equal(fresh.heart_sigmas.detach(), model.heart_sigmas.detach())
    assert torch.equal(fresh.brain_sigmas.detach(), model.brain_sigmas.detach())
    assert torch.equal(fresh.alphas_per_unit(), model.alphas_per_unit())
    assert _group_of(fresh_opt, fresh.heart_sigmas)["lr"] == 0.2
    assert _group_of(fresh_opt, fresh.brain_sigmas)["lr"] == 0.2
    # Both blocks replace the shared list, so the shared logits are unused
    # and Adam has no moment state for them. Heart and brain do.
    assert model.sigmas not in opt.state
    assert fresh.sigmas not in fresh_opt.state
    for source, loaded in (
        (model.heart_sigmas, fresh.heart_sigmas),
        (model.brain_sigmas, fresh.brain_sigmas),
    ):
        assert int(fresh_opt.state[loaded]["step"]) == int(opt.state[source]["step"])
        assert int(fresh_opt.state[loaded]["step"]) == 3
        assert torch.equal(fresh_opt.state[loaded]["exp_avg"], opt.state[source]["exp_avg"])
        assert torch.equal(
            fresh_opt.state[loaded]["exp_avg_sq"], opt.state[source]["exp_avg_sq"]
        )
    restored = load_training_history(path)
    assert np.array_equal(restored["sigmas_history"][2:4, 0], model.heart_sigmas.detach().numpy())


def test_block_logits_are_logged_plotted_and_saved(tmp_path):
    model = _model(True)
    logged0 = model.logged_sigmas().detach()
    assert logged0.shape == (2 + 2 + 1,)
    assert torch.equal(logged0[:2], model.sigmas.detach().reshape(-1))
    assert torch.equal(logged0[2:4], model.heart_sigmas.detach().reshape(-1))
    assert torch.equal(logged0[4:], model.brain_sigmas.detach().reshape(-1))

    epochs = 3
    history = np.zeros((logged0.numel(), epochs), dtype=np.float64)
    history[:, 0] = logged0.cpu().numpy().reshape(-1)
    with torch.no_grad():
        model.heart_sigmas.add_(0.25)
        model.brain_sigmas.add_(-0.1)
    history[:, 1] = model.logged_sigmas().detach().cpu().numpy().reshape(-1)
    assert not np.array_equal(history[2:4, 0], history[2:4, 1])

    series = {key: np.zeros(epochs, dtype=np.float64) for key in EPOCH_SERIES_KEYS}
    built = build_training_history(
        end=2,
        model_name="MT_RNN",
        tag="block",
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
    assert loaded["sigmas_history"].shape == (5, 2)
    assert np.array_equal(loaded["sigmas_history"][2:4, 1], history[2:4, 1])
    assert np.array_equal(loaded["sigmas_history"][4, 1], history[4, 1])
    resumed = restore_matrix(loaded["sigmas_history"], (5, epochs))
    assert np.array_equal(resumed[:, :2], history[:, :2])

    shown = history[:, :2]
    sigma_lines = _capture_plots(visualize_sigma_history, shown, tmp_path, "sigma")
    alpha_lines = _capture_plots(visualize_alpha_history, shown, tmp_path, "alpha")
    assert len(sigma_lines) == shown.shape[0]
    assert len(alpha_lines) == shown.shape[0]
    for row in range(shown.shape[0]):
        assert np.array_equal(sigma_lines[row], shown[row])
        assert np.array_equal(alpha_lines[row], mt_alpha_from_sigma(shown[row]))
    assert (tmp_path / "vis_training_history_of_sigma_sigma.png").is_file()
    assert (tmp_path / "vis_training_history_of_alpha_alpha.png").is_file()


def test_masks_hold_and_alpha_grads_do_not_cross_blocks():
    model = _model(True)
    heart, brain = 4, 2
    mask = model.block_hh_mask
    assert torch.equal(mask[:heart, :heart], torch.ones(heart, heart))
    assert torch.equal(mask[heart:, heart:], torch.ones(brain, brain))
    assert torch.equal(mask[:heart, heart:], torch.zeros(heart, brain))
    assert torch.equal(mask[heart:, :heart], torch.zeros(brain, heart))
    assert float(model.hidden_block.coupling) == 0.0

    hh = model.rnn.weight_hh_l0
    decoder = model.gen_out.weight
    assert torch.equal(hh[:heart, heart:], torch.zeros(heart, brain))
    assert torch.equal(hh[heart:, :heart], torch.zeros(brain, heart))
    assert torch.equal(decoder[:2, heart:], torch.zeros(2, brain))
    assert torch.equal(decoder[2:, :heart], torch.zeros(2, heart))

    opt, _ = _init_optimizer(model, base_lr="0.1", alpha_lr="0.1")
    opt.zero_grad()
    torch.manual_seed(5)
    model(torch.ones(3, 2, 4)).sum().backward()
    opt.step()
    assert torch.equal(hh[:heart, heart:], torch.zeros(heart, brain))
    assert torch.equal(hh[heart:, :heart], torch.zeros(brain, heart))
    assert torch.equal(decoder[:2, heart:], torch.zeros(2, brain))
    assert torch.equal(decoder[2:, :heart], torch.zeros(2, heart))

    x = torch.ones(3, 2, 4)
    with torch.no_grad():
        hh[0, heart] = 7.0
        decoder[0, heart] = 7.0
    torch.manual_seed(9)
    poked = model(x).detach()
    with torch.no_grad():
        hh[0, heart] = 0.0
        decoder[0, heart] = 0.0
    torch.manual_seed(9)
    clean = model(x).detach()
    assert torch.equal(poked, clean)

    _backward_channels(model, model.hidden_block.ecg)
    assert _grad_abs(model.heart_sigmas) > 0.0
    assert _grad_abs(model.brain_sigmas) == 0.0
    assert _grad_abs(model.sigmas) == 0.0

    _backward_channels(model, model.hidden_block.eeg)
    assert _grad_abs(model.brain_sigmas) > 0.0
    assert _grad_abs(model.heart_sigmas) == 0.0
    assert _grad_abs(model.sigmas) == 0.0

    one_sided = _model(True, brain_alphas="", brain_alpha_range="")
    assert one_sided.brain_sigmas is None
    shared_brain = _shared_per_unit(one_sided)[4:]
    assert torch.equal(one_sided.alphas_per_unit()[4:], shared_brain)
    _backward_channels(one_sided, one_sided.hidden_block.ecg)
    assert _grad_abs(one_sided.heart_sigmas) > 0.0
    assert _grad_abs(one_sided.sigmas) == 0.0
    _backward_channels(one_sided, one_sided.hidden_block.eeg)
    assert _grad_abs(one_sided.sigmas) > 0.0
    assert _grad_abs(one_sided.heart_sigmas) == 0.0


def test_range_form_uses_the_same_base10_map():
    frozen = _model(
        False,
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alpha_range="0.3,0.5",
    )
    heart_lin = torch.linspace(0.2, 0.8, 4)
    brain_lin = torch.linspace(0.3, 0.5, 2)
    assert torch.equal(frozen.alphas_per_unit()[:4], heart_lin)
    assert torch.equal(frozen.alphas_per_unit()[4:], brain_lin)

    trainable = _model(
        True,
        heart_alphas="",
        brain_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alpha_range="0.3,0.5",
    )
    assert trainable.heart_sigmas.shape == (4,)
    assert trainable.brain_sigmas.shape == (2,)
    assert torch.equal(
        trainable.heart_sigmas.detach(),
        MT_RNN.inverse_sigmoid_10(heart_lin).float(),
    )
    assert torch.equal(
        trainable.alphas_per_unit()[:4],
        MT_RNN.sigmoid_10(trainable.heart_sigmas),
    )
    assert torch.equal(
        trainable.alphas_per_unit()[4:],
        MT_RNN.sigmoid_10(trainable.brain_sigmas),
    )
    # A shared alpha of 0.2 is stored with this same map.
    assert torch.equal(trainable.heart_sigmas.detach()[0], trainable.sigmas.detach()[0])

    trainable.zero_grad(set_to_none=True)
    trainable.alphas_per_unit()[0].backward()
    assert trainable.heart_sigmas.grad[0].abs().item() > 0.0
    assert torch.equal(trainable.heart_sigmas.grad[1:], torch.zeros(3))
    assert _grad_abs(trainable.brain_sigmas) == 0.0

    with pytest.raises(ValueError, match="not both"):
        _model(True, heart_alphas="0.2,0.3", heart_alpha_range="0.2,0.8")

    mixed = _model(
        True,
        heart_alphas="",
        heart_alpha_range="0.2,0.8",
        brain_alphas="0.4,0.6",
    )
    assert torch.equal(
        mixed.alphas_per_unit()[:4],
        MT_RNN.sigmoid_10(mixed.heart_sigmas),
    )
    assert torch.equal(
        mixed.alphas_per_unit()[4:],
        _tile(MT_RNN.sigmoid_10(mixed.brain_sigmas).detach(), 2),
    )


def test_a_model_without_a_block_split_is_unchanged():
    variants = []
    for optimize, include in ((True, True), (False, True), (False, False)):
        model = _model(
            optimize,
            include_optimize=include,
            seed=4,
            hidden_block_split="off",
            heart_alphas="",
            brain_alphas="",
            ecg_channels="",
            eeg_channels="",
        )
        assert model.hidden_block is None
        assert model.heart_sigmas is None and model.brain_sigmas is None
        assert "heart_sigmas" not in model.state_dict()
        assert model.logged_sigmas().shape == model.sigmas.reshape(-1).shape
        assert list(model.alpha_parameters()) == [model.sigmas]
        per = model.alphas_per_unit()
        assert torch.equal(per, _tile(MT_RNN.sigmoid_10(model.sigmas).detach(), model.dim_rnn))
        variants.append(model)

    keys = [list(model.state_dict()) for model in variants]
    assert keys[0] == keys[1] == keys[2]
    reference = variants[0]
    for other in variants[1:]:
        other.load_state_dict(reference.state_dict())
        assert torch.equal(other.alphas_per_unit(), reference.alphas_per_unit())
        torch.manual_seed(8)
        y_ref = reference(torch.ones(3, 2, 4))
        torch.manual_seed(8)
        y_other = other(torch.ones(3, 2, 4))
        assert torch.equal(y_ref, y_other)

    missing = _model(
        include_optimize=False,
        seed=4,
        hidden_block_split="",
        heart_alphas="",
        brain_alphas="",
        ecg_channels="",
        eeg_channels="",
    )
    assert missing.hidden_block is None
    assert list(missing.state_dict()) == list(reference.state_dict())
