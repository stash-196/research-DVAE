"""Jacobian and Benettin Lyapunov diagnostics.

Analytic shallow / dendritic PLRNN Jacobians are checked against autodiff.
The spectrum is checked on linear maps with a known Lyapunov spectrum
(diagonal, and the Arnold cat map) plus the logistic map at r=4.
The eval flag defaults off and does not write lyap_* / jac_* keys.
"""

from __future__ import annotations

import math
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from dvae.eval.utils.jacobian_lyapunov import (
    attach_lyapunov_metrics,
    benettin_lyapunov_spectrum,
    build_autonomous_map,
    clipped_shplrnn_step,
    jacobian_alpha_mix,
    jacobian_autodiff,
    jacobian_plrnn,
    jacobian_shplrnn,
    resolve_lyapunov_settings,
    teacher_forced_initial_state,
    trajectory_jacobian_summaries,
)
from dvae.model.mt_rnn import build_MT_RNN
from dvae.model.plrnn import PLRNN, shPLRNN, shPLRNN_wo_A
from dvae.model.rnn import build_RNN
from dvae.utils.read_config import myconf

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SIGNAL = REPO_ROOT / "src" / "dvae" / "eval" / "eval_signal.py"


def _cfg(
    *,
    name="RNN",
    type_rnn="shPLRNN",
    x_dim=3,
    dim_rnn=4,
    hidden_sh_size="6",
    dense_x="",
    alphas="0.2, 0.8",
):
    text = f"""
[Network]
name = {name}
tag = {type_rnn}
activation = relu
x_dim = {x_dim}
dense_x = {dense_x}
dense_h_x =
dim_rnn = {dim_rnn}
hidden_sh_size = {hidden_sh_size}
num_rnn = 1
type_rnn = {type_rnn}
dropout_p = 0
alphas = {alphas}

[Training]
beta = 1
"""
    cfg = myconf()
    cfg.read_file(StringIO(text))
    return cfg


def _build(name, type_rnn, **kwargs):
    cfg = _cfg(name=name, type_rnn=type_rnn, **kwargs)
    if name == "MT_RNN":
        model = build_MT_RNN(cfg, device="cpu")
    else:
        model = build_RNN(cfg, device="cpu")
    model.eval()
    return model


def test_shplrnn_jacobian_matches_cell_autodiff():
    torch.manual_seed(0)
    M, L, K = 4, 7, 3
    cell = shPLRNN(input_size=K, hidden_size=M, hidden_sh_size=L)
    cell.eval()

    def step(v):
        x = torch.zeros(1, 1, K, dtype=v.dtype)
        _, hx = cell(x, v.view(1, 1, -1))
        return hx.reshape(-1)

    for _ in range(6):
        z = torch.randn(M)
        J = jacobian_shplrnn(z, cell.A, cell.W1, cell.W2, cell.b2)
        J_auto = torch.autograd.functional.jacobian(step, z)
        torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)

    # ReLU kink: derivative is 0 at 0, so J collapses to diag(A).
    with torch.no_grad():
        cell.b2.zero_()
    z0 = torch.zeros(M)
    J0 = jacobian_shplrnn(z0, cell.A, cell.W1, cell.W2, cell.b2)
    torch.testing.assert_close(J0, torch.diag(cell.A))
    torch.testing.assert_close(J0, torch.autograd.functional.jacobian(step, z0))


def test_shplrnn_jacobian_batch_matches_rows():
    torch.manual_seed(1)
    M, L = 3, 5
    cell = shPLRNN(input_size=2, hidden_size=M, hidden_sh_size=L)
    rows = torch.randn(4, M)
    Jb = jacobian_shplrnn(rows, cell.A, cell.W1, cell.W2, cell.b2)
    assert Jb.shape == (4, M, M)
    for i in range(4):
        Ji = jacobian_shplrnn(rows[i], cell.A, cell.W1, cell.W2, cell.b2)
        torch.testing.assert_close(Jb[i], Ji)


def test_clipped_shplrnn_gate_matches_autodiff():
    """Clipped basis is not shPLRNN.forward; the gate is still the Durstewitz one."""
    torch.manual_seed(2)
    M, L = 4, 6
    A = torch.rand(M)
    W1 = torch.randn(M, L)
    W2 = torch.randn(L, M)
    b2 = torch.randn(L)
    b1 = torch.randn(M)

    def step(v):
        return clipped_shplrnn_step(v, A, W1, W2, b2, b1)

    for _ in range(5):
        z = torch.randn(M)
        J = jacobian_shplrnn(z, A, W1, W2, b2, clipped=True)
        J_auto = torch.autograd.functional.jacobian(step, z)
        torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)
        plain = jacobian_shplrnn(z, A, W1, W2, b2, clipped=False)
        # The two gates differ on a generic state (not a proof, a tripwire).
        if not torch.allclose(plain, J):
            break
    else:
        raise AssertionError("clipped and plain gates did not differ")


def test_plrnn_jacobian_matches_cell_and_ignores_diagonal_W():
    torch.manual_seed(3)
    M, K = 5, 2
    cell = PLRNN(input_size=K, hidden_size=M)
    cell.eval()

    def step(v):
        x = torch.zeros(1, 1, K, dtype=v.dtype)
        _, hx = cell(x, v.view(1, 1, -1))
        return hx.reshape(-1)

    z = torch.randn(M)
    J = jacobian_plrnn(z, cell.A, cell.W, cell.W_mask)
    J_auto = torch.autograd.functional.jacobian(step, z)
    torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)

    W_diag = cell.W.detach().clone()
    W_diag.fill_diagonal_(50.0)
    J_ignored = jacobian_plrnn(z, cell.A, W_diag, cell.W_mask)
    torch.testing.assert_close(J, J_ignored)


def test_shplrnn_wo_A_is_analytic_jacobian_with_zero_A():
    torch.manual_seed(4)
    M, L, K = 3, 5, 2
    cell = shPLRNN_wo_A(input_size=K, hidden_size=M, hidden_sh_size=L)

    def step(v):
        x = torch.zeros(1, 1, K)
        _, hx = cell(x, v.view(1, 1, -1))
        return hx.reshape(-1)

    z = torch.randn(M)
    J = jacobian_shplrnn(z, torch.zeros(M), cell.W1, cell.W2, cell.b2)
    J_auto = torch.autograd.functional.jacobian(step, z)
    torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)


def test_alpha_mix_matches_row_scale_and_autodiff():
    torch.manual_seed(5)
    M = 4
    J_f = torch.randn(M, M)
    alpha = torch.tensor([0.2, 0.4, 0.7, 1.0])
    J = jacobian_alpha_mix(J_f, alpha)
    manual = torch.diag(1 - alpha) + torch.diag(alpha) @ J_f
    torch.testing.assert_close(J, manual)

    def step(h):
        return (1 - alpha) * h + alpha * (J_f @ h)

    z = torch.randn(M)
    J_auto = torch.autograd.functional.jacobian(step, z)
    torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "name,type_rnn",
    [
        ("RNN", "shPLRNN"),
        ("RNN", "PLRNN"),
        ("RNN", "RNN"),
        ("MT_RNN", "shPLRNN"),
        ("MT_RNN", "PLRNN"),
        ("MT_RNN", "RNN"),
    ],
)
def test_autonomous_map_jacobian_matches_autodiff(name, type_rnn):
    torch.manual_seed(6)
    model = _build(name, type_rnn, dim_rnn=4, hidden_sh_size="6", x_dim=3)
    amap = build_autonomous_map(model)
    assert amap.dim == 4
    z = torch.randn(4)
    J = amap.jacobian(z)
    J_auto = jacobian_autodiff(amap.step_fn, z)
    torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)
    assert J.shape == (4, 4)
    if name == "MT_RNN" and type_rnn == "shPLRNN":
        cell = model.rnn
        J_cell = jacobian_shplrnn(z, cell.A, cell.W1, cell.W2, cell.b2)
        assert not torch.allclose(J, J_cell, atol=1e-5, rtol=1e-4)
        assert "alpha_mix" in amap.kind


def test_nonzero_feature_extractor_is_held_at_observation_zero():
    """u = feature(0), not the zero cell-input, and it stays out of J's ∂u/∂z."""
    torch.manual_seed(7)
    model = _build(
        "RNN", "shPLRNN", dim_rnn=4, hidden_sh_size="5", x_dim=3, dense_x="7"
    )
    linear = model.feature_extractor_x[0]
    assert isinstance(linear, torch.nn.Linear)
    with torch.no_grad():
        linear.bias.fill_(0.4)
    amap = build_autonomous_map(model)
    u = model.feature_extractor_x(torch.zeros(1, 3)).reshape(-1).detach()
    assert u.shape == (7,)
    assert float(u.abs().sum()) > 0

    z = torch.randn(4)
    expected, _ = model.recurrence(u.view(1, 1, -1), z.view(1, 1, -1))
    torch.testing.assert_close(amap.step(z), expected.reshape(-1))
    # A hard-coded zero cell input is a different orbit point.
    zero_next, _ = model.recurrence(
        torch.zeros(1, 1, 7), z.view(1, 1, -1)
    )
    assert not torch.allclose(amap.step(z), zero_next.reshape(-1))
    J = amap.jacobian(z)
    J_auto = jacobian_autodiff(amap.step_fn, z)
    torch.testing.assert_close(J, J_auto, atol=1e-5, rtol=1e-4)


def test_benettin_diagonal_linear_map_is_log_abs_eig_sorted():
    scales = torch.tensor([0.25, 2.0, 0.5], dtype=torch.float64)

    def step(z):
        return scales * z

    def jac(z):
        return torch.diag(scales)

    lam = benettin_lyapunov_spectrum(
        jac, step, torch.ones(3, dtype=torch.float64), n_steps=15, n_transient=4
    )
    expected = torch.log(scales.abs()).sort(descending=True).values
    torch.testing.assert_close(lam, expected, atol=1e-8, rtol=1e-6)
    assert lam.shape == (3,)
    assert torch.isfinite(lam).all()
    assert torch.all(lam[:-1] >= lam[1:] - 1e-12)
    assert lam[0] == pytest.approx(torch.log(torch.tensor(2.0)).item())

    lam_k = benettin_lyapunov_spectrum(
        jac,
        step,
        torch.ones(3, dtype=torch.float64),
        n_steps=10,
        n_transient=2,
        k=2,
    )
    # Initial tangent frame is the first two coordinate axes, which here
    # are already eigendirections (exponents log 0.25 and log 2).
    expected_k = torch.log(scales[:2].abs()).sort(descending=True).values
    torch.testing.assert_close(lam_k, expected_k, atol=1e-8, rtol=1e-6)


def test_benettin_cat_map_matches_log_abs_eigenvalues():
    # Arnold cat map. LEs are log|λ| of [[2, 1], [1, 1]], and they sum to 0.
    A = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64)

    def step(z):
        return A @ z

    def jac(z):
        return A

    lam = benettin_lyapunov_spectrum(
        jac,
        step,
        torch.tensor([0.3, 0.7], dtype=torch.float64),
        n_steps=40,
        n_transient=20,
    )
    eig = torch.log(torch.linalg.eigvals(A).abs()).real
    expected, _ = torch.sort(eig, descending=True)
    torch.testing.assert_close(lam, expected, atol=1e-6, rtol=1e-5)
    assert float(lam.sum()) == pytest.approx(0.0, abs=1e-6)

    lam_max = benettin_lyapunov_spectrum(
        jac,
        step,
        torch.tensor([0.3, 0.7], dtype=torch.float64),
        n_steps=30,
        n_transient=30,
        k=1,
    )
    torch.testing.assert_close(lam_max, expected[:1], atol=1e-5, rtol=1e-5)


def test_benettin_logistic_map_r4():
    # Chaotic logistic map: λ = ln(2).
    def step(z):
        x = z.reshape(())
        return (4 * x * (1 - x)).reshape(1)

    def jac(z):
        x = z.reshape(())
        return (4 * (1 - 2 * x)).reshape(1, 1)

    lam = benettin_lyapunov_spectrum(
        jac,
        step,
        torch.tensor([0.1], dtype=torch.float64),
        n_steps=4000,
        n_transient=500,
    )
    assert lam.shape == (1,)
    assert torch.isfinite(lam).all()
    assert float(lam[0]) == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=0.05)


def test_trajectory_jacobian_summaries_on_linear_map():
    scales = torch.tensor([0.5, 0.25, -0.1], dtype=torch.float64)

    def step(z):
        return scales * z

    def jac(z):
        return torch.diag(scales)

    stats = trajectory_jacobian_summaries(
        jac, step, torch.ones(3, dtype=torch.float64), n_steps=6, n_transient=2
    )
    assert stats["jac_opnorm_mean"] == pytest.approx(0.5)
    assert stats["jac_opnorm_max"] == pytest.approx(0.5)
    assert stats["jac_rho_max"] == pytest.approx(0.5)
    assert stats["jac_rho_gt1_frac"] == pytest.approx(0.0)

    expanding = torch.tensor([2.0, 0.2], dtype=torch.float64)
    stats_exp = trajectory_jacobian_summaries(
        lambda z: torch.diag(expanding),
        lambda z: expanding * z,
        torch.ones(2, dtype=torch.float64),
        n_steps=4,
        n_transient=0,
    )
    assert stats_exp["jac_opnorm_max"] == pytest.approx(2.0)
    assert stats_exp["jac_rho_max"] == pytest.approx(2.0)
    assert stats_exp["jac_rho_gt1_frac"] == pytest.approx(1.0)


def test_flag_off_leaves_metrics_and_model_untouched():
    metrics = {"mse_tf": 1.0, "local_drift_avg_d_norm": 0.25}
    model = MagicMock()
    out = attach_lyapunov_metrics(metrics, model, enabled=False)
    assert out is metrics
    assert out == {"mse_tf": 1.0, "local_drift_avg_d_norm": 0.25}
    model.assert_not_called()
    model.recurrence.assert_not_called()
    model.eval.assert_not_called()
    assert "lyap_max" not in metrics
    assert "jac_opnorm_mean" not in metrics


def test_flag_on_shplrnn_writes_finite_summary_keys():
    torch.manual_seed(8)
    model = _build("RNN", "shPLRNN", dim_rnn=4, hidden_sh_size="6")
    batch = torch.randn(12, 2, 3)
    metrics = {"mse_tf": 0.5}
    attach_lyapunov_metrics(
        metrics,
        model,
        batch_data=batch,
        enabled=True,
        n_steps=12,
        n_transient=3,
        n_warmup=5,
        batch_index=1,
    )
    assert "lyapunov_skip_reason" not in metrics
    for key in (
        "lyap_max",
        "jac_opnorm_mean",
        "jac_opnorm_max",
        "jac_rho_max",
        "jac_rho_gt1_frac",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])
    assert len(metrics["lyap_spectrum"]) == 4
    assert all(math.isfinite(v) for v in metrics["lyap_spectrum"])
    assert metrics["lyap_spectrum"] == sorted(
        metrics["lyap_spectrum"], reverse=True
    )
    assert metrics["lyap_max"] == metrics["lyap_spectrum"][0]
    assert metrics["lyap_n_steps"] == 12
    assert metrics["lyap_n_transient"] == 3
    assert metrics["lyap_n_warmup"] == 5
    assert metrics["lyap_k"] == 4
    assert metrics["lyap_autonomous_input"] == "external_observation_zero"
    assert "shPLRNN" in metrics["lyap_map"]
    assert metrics["mse_tf"] == 0.5


def test_teacher_forced_warmup_matches_recurrence_and_is_not_the_origin():
    torch.manual_seed(9)
    model = _build("RNN", "shPLRNN", dim_rnn=4, hidden_sh_size="6", x_dim=3)
    data = torch.randn(8, 2, 3)
    z0, steps = teacher_forced_initial_state(
        model, data, n_warmup=4, batch_index=1
    )
    assert steps == 4
    h = torch.zeros(1, 2, 4)
    for t in range(4):
        feat = model.feature_extractor_x(data[t])
        h, _ = model.recurrence(feat.unsqueeze(0), h)
    torch.testing.assert_close(z0, h[0, 1])
    assert not torch.allclose(z0, torch.zeros_like(z0))


def test_lstm_and_unknown_models_skip_without_raising():
    lstm = _build("RNN", "LSTM", dim_rnn=4, hidden_sh_size="")
    metrics = {"mse_tf": 1.0}
    with pytest.warns(RuntimeWarning, match="LSTM"):
        attach_lyapunov_metrics(
            metrics,
            lstm,
            enabled=True,
            n_steps=4,
            n_transient=1,
            n_warmup=0,
        )
    assert "lyap_max" not in metrics
    assert "LSTM" in metrics["lyapunov_skip_reason"]
    assert metrics["mse_tf"] == 1.0

    metrics_unknown = {}
    with pytest.warns(RuntimeWarning):
        attach_lyapunov_metrics(
            metrics_unknown,
            object(),
            enabled=True,
            n_steps=2,
            n_transient=0,
            n_warmup=0,
        )
    assert "lyap_spectrum" not in metrics_unknown
    assert "lyapunov_skip_reason" in metrics_unknown


def test_resolve_lyapunov_settings_defaults_off_and_cli_wins():
    off = resolve_lyapunov_settings({})
    assert off["enabled"] is False
    assert off["n_steps"] == 1000
    assert off["n_transient"] == 100
    assert off["n_warmup"] == 50
    assert off["k"] is None

    cfg = myconf()
    cfg.read_string(
        """
[Evaluation]
compute_lyapunov = true
lyap_n_steps = 12
lyap_n_transient = 3
lyap_n_warmup = 4
lyap_k = 2
"""
    )
    from_cfg = resolve_lyapunov_settings({"compute_lyapunov": None}, cfg)
    assert from_cfg["enabled"] is True
    assert from_cfg["n_steps"] == 12
    assert from_cfg["n_transient"] == 3
    assert from_cfg["n_warmup"] == 4
    assert from_cfg["k"] == 2

    cli_off = resolve_lyapunov_settings(
        {"compute_lyapunov": False, "lyap_n_steps": 7}, cfg
    )
    assert cli_off["enabled"] is False
    assert cli_off["n_steps"] == 7


def test_eval_signal_gates_lyapunov_and_keeps_local_drift():
    src = EVAL_SIGNAL.read_text()
    assert "compute_local_drift_statistics" in src
    assert "resolve_lyapunov_settings" in src
    assert 'if lyap_settings["enabled"]' in src
    gate = src.index('if lyap_settings["enabled"]')
    call = src.index("attach_lyapunov_metrics(", gate)
    assert gate < call
    assert "external observation = 0" in src

    from dvae.eval.eval_signal import Options

    opt = Options()
    opt._initial()
    action = next(a for a in opt.parser._actions if a.dest == "compute_lyapunov")
    assert action.default is None
