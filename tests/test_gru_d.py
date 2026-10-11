"""GRU-D cell on synthetic tensors. No dataset files."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dvae.model.gru_d import GRUD, build_GRU_D, masked_mse
from dvae.model.rnn import build_RNN
from dvae.utils.read_config import myconf

ROOT = Path(__file__).resolve().parents[1]


def _cfg(path):
    cfg = myconf()
    assert cfg.read(path)
    return cfg


def test_example_ini_builds_gru_d_and_rnn_config_still_builds_rnn():
    grud_cfg = _cfg(ROOT / "config" / "lorenz63" / "cfg_gru_d.ini")
    assert grud_cfg.get("Network", "name") == "GRU_D"
    grud = build_GRU_D(grud_cfg, device="cpu")
    assert grud.x_dim == 3
    assert grud.type_rnn == "GRU_D"

    rnn_cfg = _cfg(ROOT / "config" / "lorenz63" / "cfg_rnn.ini")
    assert rnn_cfg.get("Network", "name") == "RNN"
    assert rnn_cfg.get("Network", "type_rnn") == "RNN"
    rnn = build_RNN(rnn_cfg, device="cpu")
    assert type(rnn).__name__ == "RNN"
    assert rnn.type_rnn == "RNN"


def test_num_rnn_must_be_one():
    cfg = _cfg(ROOT / "config" / "lorenz63" / "cfg_gru_d.ini")
    cfg.set("Network", "num_rnn", "2")
    with pytest.raises(ValueError, match="num_rnn"):
        build_GRU_D(cfg, device="cpu")


def test_zero_decay_at_delta_zero_and_decay_toward_mean():
    model = GRUD(x_dim=2, dim_rnn=4, dense_h_x=[], device="cpu", cfg=None)
    model.set_empirical_mean([3.0, -4.0])
    gamma = model.input_decay_gamma(torch.zeros(1, 2))
    assert torch.allclose(gamma, torch.ones(1, 2))

    with torch.no_grad():
        model.input_decay.weight.fill_(1.0)
        model.input_decay.bias.zero_()
    delta = torch.full((1, 2), 50.0)
    gamma = model.input_decay_gamma(delta)
    assert float(gamma.detach().max()) < 1e-6
    decayed = model.decayed_input(torch.zeros(1, 2), gamma)
    assert torch.allclose(decayed, torch.tensor([[3.0, -4.0]]), atol=1e-4)


def test_missing_steps_do_not_refresh_last_observation():
    model = GRUD(x_dim=1, dim_rnn=4, dense_h_x=[], device="cpu", cfg=None)
    series = torch.tensor([[[1.0]], [[float("nan")]], [[2.0]]])
    out = model(series)
    assert out.shape == (3, 1, 1)
    assert torch.isfinite(out).all()
    assert torch.allclose(model.x_last, torch.tensor([[2.0]]))
    assert torch.allclose(model.delta, torch.tensor([[1.0]]))


def test_all_missing_keeps_mean_and_grows_delta():
    model = GRUD(x_dim=2, dim_rnn=4, dense_h_x=[], device="cpu", cfg=None)
    model.set_empirical_mean([0.5, -0.5])
    series = torch.full((2, 1, 2), float("nan"))
    out = model(series)
    assert torch.isfinite(out).all()
    assert torch.allclose(model.x_last, torch.tensor([[0.5, -0.5]]))
    assert torch.allclose(model.delta, torch.full((1, 2), 2.0))


def test_closed_loop_does_not_need_future_observations():
    torch.manual_seed(0)
    model = GRUD(x_dim=3, dim_rnn=8, dense_h_x=[], device="cpu", cfg=None)
    series = torch.randn(12, 2, 3)
    series[1:] = float("nan")
    mode = torch.zeros(12)
    mode[1:] = 1.0
    out = model(series, mode_selector=mode)
    assert out.shape == (12, 2, 3)
    assert torch.isfinite(out).all()


def test_masked_mse_ignores_nan_targets_and_backward_runs():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [3.0, 8.0]])
    loss = masked_mse(pred, target)
    # Errors are 0, 0, 4 on the three finite entries. Mean is 16/3.
    assert torch.allclose(loss, torch.tensor(16.0 / 3.0))
    loss.backward()
    assert pred.grad is not None


def test_observed_sine_loss_drops():
    torch.manual_seed(0)
    time = torch.linspace(0, 6.28, 48)
    signal = torch.stack((torch.sin(time), torch.cos(time)), dim=-1)
    series = signal.unsqueeze(1).expand(48, 8, 2).contiguous()
    series = series.clone()
    series[10:14, :, 0] = float("nan")
    model = GRUD(x_dim=2, dim_rnn=16, dense_h_x=[], dropout_p=0.0, device="cpu", cfg=None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        pred = model(series)
        loss = masked_mse(pred, series)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5
