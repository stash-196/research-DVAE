"""Unit tests for the in-repo shPLRNN cell (Hess / Durstewitz shallow form).

Uses fake tensors only. Does not implement or exercise GTF.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import torch

from dvae.model import build_MT_RNN, build_RNN
from dvae.model.plrnn import shPLRNN
from dvae.utils.read_config import myconf

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CFG = REPO_ROOT / "config" / "lorenz63" / "cfg_sh_plrnn.ini"


def _network_cfg(
    *,
    name="RNN",
    type_rnn="shPLRNN",
    x_dim=3,
    dim_rnn=8,
    hidden_sh_size="16",
    dense_x="",
    dense_h_x="",
    alphas="0.1, 0.5",
    extra_training="",
):
    hidden_line = (
        f"hidden_sh_size = {hidden_sh_size}\n" if hidden_sh_size is not None else ""
    )
    text = f"""
[Network]
name = {name}
tag = {type_rnn}
activation = relu
x_dim = {x_dim}
dense_x = {dense_x}
dense_h_x = {dense_h_x}
dim_rnn = {dim_rnn}
{hidden_line}num_rnn = 1
type_rnn = {type_rnn}
dropout_p = 0
alphas = {alphas}

[Training]
beta = 1
{extra_training}
"""
    cfg = myconf()
    cfg.read_file(StringIO(text))
    return cfg


def test_shplrnn_cell_matches_hess_shallow_form():
    """z_t = A ⊙ z + W1 φ(W2 z + h2) + h1 + C x  (official step + linear input)."""
    M, L, K = 5, 11, 3
    cell = shPLRNN(input_size=K, hidden_size=M, hidden_sh_size=L)
    assert cell.W1.shape == (M, L)
    assert cell.W2.shape == (L, M)
    assert cell.b1.shape == (M,)
    assert cell.b2.shape == (L,)
    assert cell.A.shape == (M,)
    assert cell.linear_C.in_features == K
    assert cell.linear_C.out_features == M

    seq_len, batch = 4, 2
    x = torch.randn(seq_len, batch, K)
    hx = torch.randn(1, batch, M)
    out, new_hx = cell(x, hx)
    assert out.shape == (seq_len, batch, M)
    assert new_hx.shape == (1, batch, M)

    # Manual one-step check against the official Julia `step` plus C x.
    z = hx[0]
    x0 = x[0]
    expected = (
        cell.A * z
        + torch.matmul(torch.relu(torch.matmul(z, cell.W2.T) + cell.b2), cell.W1.T)
        + cell.b1
        + cell.linear_C(x0)
    )
    torch.testing.assert_close(out[0], expected)


def test_build_rnn_reads_hidden_sh_size_and_forward_shapes():
    M, L, x_dim = 8, 16, 3
    cfg = _network_cfg(dim_rnn=M, hidden_sh_size=str(L), x_dim=x_dim)
    model = build_RNN(cfg, device="cpu")
    assert model.type_rnn == "shPLRNN"
    assert model.hidden_sh_size == L
    assert model.rnn.hidden_sh_size == L
    assert model.rnn.W1.shape == (M, L)
    assert model.rnn.W2.shape == (L, M)

    seq_len, batch = 7, 4
    x = torch.randn(seq_len, batch, x_dim)
    y = model(x, mode_selector=torch.zeros(seq_len, batch, x_dim))
    assert y.shape == (seq_len, batch, x_dim)
    assert model.h.shape == (seq_len, batch, M)


def test_hidden_sh_size_falls_back_to_dim_rnn_when_missing():
    M = 6
    cfg = _network_cfg(dim_rnn=M, hidden_sh_size=None)
    model = build_RNN(cfg, device="cpu")
    assert model.hidden_sh_size == M
    assert model.rnn.W1.shape == (M, M)


def test_teacher_forcing_vs_autonomous_hooks():
    x_dim, seq_len, batch = 2, 6, 3
    cfg = _network_cfg(x_dim=x_dim, dim_rnn=5, hidden_sh_size="7")
    model = build_RNN(cfg, device="cpu")
    x = torch.randn(seq_len, batch, x_dim)
    tf = torch.zeros(seq_len, batch, x_dim)
    auto = torch.ones(seq_len, batch, x_dim)
    y_tf = model(x, mode_selector=tf)
    y_auto = model(x, mode_selector=auto)
    assert y_tf.shape == y_auto.shape == (seq_len, batch, x_dim)
    # After t=0 the TF mix uses ground truth vs previous prediction.
    assert not torch.allclose(y_tf[1:], y_auto[1:])


def test_one_train_step_and_checkpoint_reload(tmp_path):
    x_dim, seq_len, batch = 3, 5, 2
    cfg = _network_cfg(x_dim=x_dim, dim_rnn=8, hidden_sh_size="12")
    model = build_RNN(cfg, device="cpu")
    model.train()
    x = torch.randn(seq_len, batch, x_dim)
    target = torch.randn(seq_len, batch, x_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    y = model(x, mode_selector=torch.zeros(seq_len, batch, x_dim))
    loss = torch.nn.functional.mse_loss(y, target)
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt = tmp_path / "shplrnn_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        ckpt,
    )

    reloaded = build_RNN(cfg, device="cpu")
    payload = torch.load(ckpt, weights_only=False)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    model.eval()
    x_eval = torch.randn(seq_len, batch, x_dim)
    with torch.no_grad():
        y_orig = model(x_eval, mode_selector=torch.zeros(seq_len, batch, x_dim))
        y_reload = reloaded(x_eval, mode_selector=torch.zeros(seq_len, batch, x_dim))
    torch.testing.assert_close(y_orig, y_reload)


def test_smoke_ini_is_selectable_as_shplrnn():
    assert SMOKE_CFG.is_file()
    cfg = myconf()
    cfg.read(SMOKE_CFG)
    assert cfg.get("Network", "name") == "RNN"
    assert cfg.get("Network", "type_rnn") == "shPLRNN"
    assert cfg.getint("Network", "dim_rnn") == 8
    assert cfg.getint("Network", "hidden_sh_size") == 16
    model = build_RNN(cfg, device="cpu")
    assert model.rnn.W1.shape == (8, 16)
    seq_len, batch, x_dim = 4, 2, cfg.getint("Network", "x_dim")
    y = model(torch.randn(seq_len, batch, x_dim))
    assert y.shape == (seq_len, batch, x_dim)


def test_mt_rnn_shplrnn_forward_uses_existing_tf_path():
    cfg = _network_cfg(
        name="MT_RNN",
        type_rnn="shPLRNN",
        x_dim=2,
        dim_rnn=6,
        hidden_sh_size="10",
        alphas="0.1, 0.5",
    )
    model = build_MT_RNN(cfg, device="cpu")
    seq_len, batch, x_dim = 5, 3, 2
    y = model(
        torch.randn(seq_len, batch, x_dim),
        mode_selector=torch.zeros(seq_len, batch, x_dim),
    )
    assert y.shape == (seq_len, batch, x_dim)
    assert model.rnn.W1.shape == (6, 10)
