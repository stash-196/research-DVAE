"""Heart/brain split, subband loss, and the decorrelation penalty.

Synthetic tensors only. The split is off unless the config asks for it.
"""

from __future__ import annotations

import configparser

import pytest
import torch

from dvae.model.base_rnn import BaseRNN
from dvae.model.hidden_block import parse_hidden_block_split
from dvae.model.mt_rnn import MT_RNN
from dvae.regularizers import (
    decorr_hidden_loss,
    decorr_output_loss,
    parse_regularizer_settings,
    subband_power_loss,
    warmup_progress,
)


def _cfg(type_rnn="PLRNN", dim="4", x_dim="4", **network):
    parser = configparser.ConfigParser()
    base = {
        "x_dim": x_dim,
        "dim_rnn": dim,
        "hidden_sh_size": "6",
        "type_rnn": type_rnn,
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "",
    }
    base.update(network)
    parser["Network"] = {key: str(value) for key, value in base.items()}
    parser["Training"] = {}
    return parser


def _model(type_rnn="PLRNN", dim=4, x_dim=4, **network):
    cfg = _cfg(type_rnn, str(dim), str(x_dim), **network)
    return BaseRNN(
        x_dim=x_dim,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=dim,
        num_rnn=1,
        type_rnn=type_rnn,
        dropout_p=0,
        device="cpu",
        cfg=cfg,
    )


def _split_on(**network):
    network.setdefault("hidden_block_split", "half")
    network.setdefault("ecg_channels", "0,1")
    network.setdefault("eeg_channels", "2,3")
    return network


def test_blank_split_leaves_the_cell_and_decoder_alone():
    model = _model()
    assert model.hidden_block is None
    assert getattr(model, "decoder_mask", None) is None
    assert getattr(model.rnn, "block_hh_mask", None) is None
    x = torch.randn(5, 2, 4)
    y = model(x)
    assert y.shape == (5, 2, 4)


def test_half_and_explicit_sizes():
    even = parse_hidden_block_split(
        _cfg(**_split_on(hidden_block_split="half")), 4, 4
    )
    assert (even.heart, even.brain) == (2, 2)
    odd = parse_hidden_block_split(
        _cfg(dim="5", **_split_on(hidden_block_split="true")), 5, 4
    )
    assert (odd.heart, odd.brain) == (2, 3)
    sized = parse_hidden_block_split(
        _cfg(**_split_on(hidden_block_split="1,3")), 4, 4
    )
    assert (sized.heart, sized.brain) == (1, 3)
    with pytest.raises(ValueError, match="sum to dim_rnn"):
        parse_hidden_block_split(
            _cfg(**_split_on(hidden_block_split="2,1")), 4, 4
        )


def test_plrnn_keeps_a_zero_diagonal_and_a_dense_input_map():
    model = _model(**_split_on())
    heart = model.hidden_block.heart
    x = torch.randn(6, 2, 4)
    y = model(x)
    y.sum().backward()
    weight = model.rnn.W
    effective = weight * model.rnn.W_mask * model.rnn.block_hh_mask
    assert torch.count_nonzero(torch.diagonal(effective)).item() == 0
    assert torch.count_nonzero(effective[:heart, heart:]).item() == 0
    assert torch.count_nonzero(effective[heart:, :heart]).item() == 0
    assert torch.count_nonzero(weight.grad[:heart, heart:]).item() == 0
    assert torch.count_nonzero(weight.grad[heart:, :heart]).item() == 0
    assert torch.count_nonzero(torch.diagonal(weight.grad)).item() == 0
    assert model.rnn.linear_C.weight.grad.abs().sum().item() > 0

    # Zero the recurrent map. Every hidden unit still sees every input.
    model.zero_grad()
    model.rnn.W.data.zero_()
    model.rnn.A_sigmas.data.fill_(-20)
    model.rnn.linear_C.weight.data.fill_(0.25)
    model.rnn.linear_C.bias.data.zero_()
    probe = torch.ones(1, 1, 4)
    hidden, _state = model.rnn(probe)
    assert torch.allclose(hidden, torch.ones_like(hidden))


def test_weak_coupling_scales_the_recurrent_map_and_not_the_decoder():
    model = _model(**_split_on(hidden_block_weak_coupling="true"))
    heart = model.hidden_block.heart
    assert model.rnn.block_hh_mask[0, heart].item() == pytest.approx(0.01)
    assert model.rnn.block_hh_mask[0, 0].item() == pytest.approx(1.0)
    eeg = model.hidden_block.eeg[0]
    ecg = model.hidden_block.ecg[0]
    assert model.decoder_mask[eeg, 0].item() == 0.0
    assert model.decoder_mask[ecg, 0].item() == 1.0
    assert model.decoder_mask[ecg, heart].item() == 0.0


def test_decoder_cross_terms_get_no_gradient():
    model = _model(**_split_on())
    heart = model.hidden_block.heart
    y = model(torch.randn(5, 2, 4))
    y.sum().backward()
    grad = model.gen_out.weight.grad
    eeg = list(model.hidden_block.eeg)
    ecg = list(model.hidden_block.ecg)
    assert torch.count_nonzero(grad[eeg, :heart]).item() == 0
    assert torch.count_nonzero(grad[ecg, heart:]).item() == 0
    assert grad[ecg, :heart].abs().sum().item() > 0


def test_vanilla_rnn_and_lstm_mask_weight_hh_only():
    for kind in ("RNN", "LSTM"):
        model = _model(kind, **_split_on())
        heart = model.hidden_block.heart
        hidden = model.dim_rnn
        y = model(torch.randn(5, 2, 4))
        y.sum().backward()
        grad = model.rnn.weight_hh_l0.grad
        gates = 4 if kind == "LSTM" else 1
        for gate in range(gates):
            rows = slice(gate * hidden, (gate + 1) * hidden)
            block = grad[rows]
            assert torch.count_nonzero(block[:heart, heart:]).item() == 0
            assert torch.count_nonzero(block[heart:, :heart]).item() == 0
        assert model.rnn.weight_ih_l0.grad.abs().sum().item() > 0


def test_shplrnn_masks_both_factors_and_leaves_the_input_dense():
    model = _model("shPLRNN", **_split_on())
    heart = model.hidden_block.heart
    heart_sh = model.rnn.heart_sh
    y = model(torch.randn(5, 2, 4))
    y.sum().backward()
    assert torch.count_nonzero(model.rnn.W1.grad[:heart, heart_sh:]).item() == 0
    assert torch.count_nonzero(model.rnn.W1.grad[heart:, :heart_sh]).item() == 0
    assert torch.count_nonzero(model.rnn.W2.grad[:heart_sh, heart:]).item() == 0
    assert torch.count_nonzero(model.rnn.W2.grad[heart_sh:, :heart]).item() == 0
    assert model.rnn.linear_C.weight.grad.abs().sum().item() > 0
    assert model.rnn.heart_sh + model.rnn.brain_sh == 6


def test_mlp_decoder_and_stacked_rnn_are_rejected():
    with pytest.raises(ValueError, match="dense_h_x"):
        BaseRNN(
            x_dim=4,
            activation="relu",
            dense_x=[],
            dense_h_x=[8],
            dim_rnn=4,
            num_rnn=1,
            type_rnn="RNN",
            dropout_p=0,
            device="cpu",
            cfg=_cfg("RNN", **_split_on()),
        )
    with pytest.raises(ValueError, match="num_rnn"):
        BaseRNN(
            x_dim=4,
            activation="relu",
            dense_x=[],
            dense_h_x=[],
            dim_rnn=4,
            num_rnn=2,
            type_rnn="RNN",
            dropout_p=0,
            device="cpu",
            cfg=_cfg("RNN", **_split_on()),
        )


def test_per_block_alphas_tile_or_span_and_blank_keeps_the_shared_list():
    shared = MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=4,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=4,
        cfg=_cfg(**_split_on()),
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )
    got = shared.alphas_per_unit().detach()
    assert torch.allclose(got, torch.tensor([0.2, 0.8, 0.2, 0.8]))

    replaced = MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=4,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=4,
        cfg=_cfg(
            **_split_on(heart_alphas="0.1", brain_alpha_range="0.3,0.9")
        ),
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )
    got = replaced.alphas_per_unit().detach()
    assert torch.allclose(got[:2], torch.tensor([0.1, 0.1]))
    assert got[2].item() == pytest.approx(0.3)
    assert got[3].item() == pytest.approx(0.9)


def test_subband_loss_is_zero_on_a_match_and_differentiable():
    step = torch.arange(64, dtype=torch.float32)
    slow = torch.sin(2.0 * torch.pi * 2.0 * step / 64.0)
    fast = torch.sin(2.0 * torch.pi * 10.0 * step / 64.0)
    target = slow.view(-1, 1, 1).repeat(1, 2, 1)
    same = subband_power_loss(target, target, (0,), 64.0, 8.0, None)
    assert same.item() == pytest.approx(0.0, abs=1e-6)
    predicted = fast.view(-1, 1, 1).repeat(1, 2, 1).detach().requires_grad_(True)
    mismatch = subband_power_loss(predicted, target, (0,), 64.0, 8.0, None)
    assert mismatch.item() > 0.0
    mismatch.backward()
    assert predicted.grad is not None
    assert predicted.grad.abs().sum().item() > 0.0


def test_decorr_is_the_mean_squared_cross_correlation():
    step = torch.arange(32, dtype=torch.float32)
    sine = torch.sin(2.0 * torch.pi * step / 32.0)
    cosine = torch.cos(2.0 * torch.pi * step / 32.0)
    hidden = torch.stack([sine, sine], dim=1).unsqueeze(1)
    assert decorr_hidden_loss(hidden, 1, 1).item() == pytest.approx(1.0, abs=1e-5)
    crossed = torch.stack([sine, cosine], dim=1).unsqueeze(1)
    assert decorr_hidden_loss(crossed, 1, 1).item() == pytest.approx(0.0, abs=1e-5)

    output = torch.stack([sine, cosine, sine, cosine], dim=1).unsqueeze(1)
    matched = decorr_output_loss(output, (0, 1), (2, 3))
    assert matched.item() == pytest.approx(0.5, abs=1e-5)


def test_warmup_progress_ramps_only_when_asked():
    assert warmup_progress(0, 4, 6, True) == 0.0
    assert warmup_progress(5, 4, 6, True) == pytest.approx(0.5)
    assert warmup_progress(0, 4, 6, False) == 1.0
    assert warmup_progress(0, 0, 0, True) == 1.0


def test_regularizer_settings_stay_off_until_the_weight_is_set():
    cfg = configparser.ConfigParser()
    cfg["Training"] = {"subband_loss_weight": "", "decorr_loss_weight": "off"}
    settings = parse_regularizer_settings(cfg, y_dim=4, hidden_block=None)
    assert settings.subband_on is False
    assert settings.decorr_on is False
    cfg["Training"] = {
        "decorr_loss_weight": "0.1",
        "decorr_mode": "hidden",
    }
    with pytest.raises(ValueError, match="hidden_block_split"):
        parse_regularizer_settings(cfg, y_dim=4, hidden_block=None)
    cfg["Training"] = {
        "subband_loss_weight": "0.2",
        "subband_loss_warmup": "true",
        "subband_channels": "1",
        "subband_fs": "64",
        "subband_low_hz": "8",
        "decorr_loss_weight": "0.1",
        "decorr_mode": "output",
        "decorr_groups": "0,1|2,3",
    }
    settings = parse_regularizer_settings(cfg, y_dim=4, hidden_block=None)
    assert settings.subband_warmup is True
    assert settings.subband_channels == (1,)
    assert settings.decorr_groups == ((0, 1), (2, 3))
