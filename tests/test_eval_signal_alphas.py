"""Eval timescales: MT alphas, PLRNN/shPLRNN diagonal A, skip vanilla RNN.

Repro: Lorenz smoke (cfg_sh_plrnn.ini) trains Network.name=RNN /
type_rnn=shPLRNN, then eval_signal.py crashed with:

    AttributeError: 'RNN' object has no attribute 'alphas_per_unit'

The Training.optimize_alphas flag is False (not None) in that config, so
the old `if learning_algo.optimize_alphas is not None` gate still called
an MT-only method. shPLRNN/PLRNN now expose diagonal A via the same
``alphas_per_unit()`` name so eval records and plots those timescales.
Vanilla RNN/LSTM still skip (no invented alphas).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dvae.eval.maybe_alphas import alphas_to_metric_list, maybe_alphas_per_unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SIGNAL = REPO_ROOT / "src" / "dvae" / "eval" / "eval_signal.py"


def test_eval_signal_call_site_uses_guard_not_bare_method():
    src = EVAL_SIGNAL.read_text()
    assert "maybe_alphas_per_unit(dvae)" in src
    assert "dvae.alphas_per_unit()" not in src
    assert "from dvae.eval.maybe_alphas import alphas_to_metric_list, maybe_alphas_per_unit" in src
    assert 'metrics["alphas_per_unit"]' in src


def test_maybe_alphas_skips_model_without_method():
    dvae = SimpleNamespace()  # no alphas_per_unit
    assert maybe_alphas_per_unit(dvae) is None
    assert alphas_to_metric_list(None) is None


def test_maybe_alphas_skips_non_callable_attribute():
    dvae = SimpleNamespace(alphas_per_unit=0.5)
    assert maybe_alphas_per_unit(dvae) is None


def test_maybe_alphas_reads_callable():
    expected = object()
    dvae = SimpleNamespace(alphas_per_unit=MagicMock(return_value=expected))
    assert maybe_alphas_per_unit(dvae) is expected
    dvae.alphas_per_unit.assert_called_once_with()


def test_eval_path_does_not_attributeerror_when_optimize_alphas_is_false():
    """Exact smoke-config gate: optimize_alphas=False is still not None."""
    learning_algo = SimpleNamespace(optimize_alphas=False)
    dvae = SimpleNamespace()

    if learning_algo.optimize_alphas is not None:
        with pytest.raises(AttributeError):
            dvae.alphas_per_unit()

    assert maybe_alphas_per_unit(dvae) is None


def test_maybe_alphas_on_built_models_records_a_or_skips():
    """Vanilla RNN skips; shPLRNN/PLRNN return diagonal A; MT_RNN keeps alphas."""
    torch = pytest.importorskip("torch")
    from io import StringIO

    from dvae.model.mt_rnn import build_MT_RNN
    from dvae.model.plrnn import PLRNN, shPLRNN
    from dvae.model.rnn import build_RNN
    from dvae.utils.read_config import myconf

    def _cfg(name, type_rnn, *, alphas="0.1, 0.5", hidden_sh_size=""):
        hidden_line = (
            f"hidden_sh_size = {hidden_sh_size}\n" if hidden_sh_size else ""
        )
        text = f"""
[Network]
name = {name}
tag = {type_rnn}
activation = relu
x_dim = 3
dense_x =
dense_h_x =
dim_rnn = 4
{hidden_line}num_rnn = 1
type_rnn = {type_rnn}
dropout_p = 0
alphas = {alphas}

[Training]
beta = 1
"""
        cfg = myconf()
        cfg.read_file(StringIO(text))
        return cfg

    plain = build_RNN(_cfg("RNN", "RNN"), device="cpu")
    lstm = build_RNN(_cfg("RNN", "LSTM"), device="cpu")
    sh = build_RNN(_cfg("RNN", "shPLRNN", hidden_sh_size="6"), device="cpu")
    pl = build_RNN(_cfg("RNN", "PLRNN"), device="cpu")
    mt = build_MT_RNN(_cfg("MT_RNN", "RNN"), device="cpu")

    assert maybe_alphas_per_unit(plain) is None
    assert maybe_alphas_per_unit(lstm) is None

    sh_alphas = maybe_alphas_per_unit(sh)
    assert sh_alphas is not None
    torch.testing.assert_close(sh_alphas, sh.rnn.A)
    assert tuple(sh_alphas.shape) == (4,)
    assert alphas_to_metric_list(sh_alphas) == [
        float(a) for a in sh.rnn.A.detach()
    ]

    pl_alphas = maybe_alphas_per_unit(pl)
    assert pl_alphas is not None
    torch.testing.assert_close(pl_alphas, pl.rnn.A)
    assert tuple(pl_alphas.shape) == (4,)

    mt_alphas = maybe_alphas_per_unit(mt)
    assert mt_alphas is not None
    assert tuple(mt_alphas.shape) == (4,)
    torch.testing.assert_close(mt_alphas, mt.alphas_per_unit())
    assert torch.isfinite(mt_alphas).all()

    cell = shPLRNN(input_size=3, hidden_size=5, hidden_sh_size=7)
    torch.testing.assert_close(cell.alphas_per_unit(), cell.A)
    pl_cell = PLRNN(input_size=3, hidden_size=5)
    torch.testing.assert_close(pl_cell.alphas_per_unit(), pl_cell.A)
