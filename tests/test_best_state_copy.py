"""Best-epoch weights are copies, not live views of the parameters."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.checkpoint_state import (
    clone_module_state_dict,
    clone_optimizer_state_dict,
)
from tests.test_train_smoke import _write_lorenz


def _clone_cpu(state):
    cloned = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            cloned[key] = _clone_cpu(value)
        else:
            cloned[key] = value
    return cloned


def _states_equal(left, right) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                return False
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            if not torch.equal(a.cpu(), b.cpu()):
                return False
        elif isinstance(a, dict) or isinstance(b, dict):
            if not (isinstance(a, dict) and isinstance(b, dict)):
                return False
            if not _states_equal(a, b):
                return False
    return True


def test_cloned_weights_and_optimizer_ignore_later_steps():
    torch.manual_seed(0)
    layer = torch.nn.Linear(2, 2, bias=False)
    opt = torch.optim.Adam(layer.parameters(), lr=0.1)
    data = torch.randn(4, 2)

    def step():
        opt.zero_grad()
        layer(data).pow(2).mean().backward()
        opt.step()

    step()
    best_weights = clone_module_state_dict(layer.state_dict())
    best_optim = clone_optimizer_state_dict(opt.state_dict())
    weight_at_best = best_weights["weight"].clone()
    exp_at_best = best_optim["state"][0]["exp_avg"].clone()

    step()

    assert not torch.equal(layer.state_dict()["weight"], weight_at_best)
    assert torch.equal(best_weights["weight"], weight_at_best)
    live_exp = opt.state_dict()["state"][0]["exp_avg"]
    assert not torch.equal(live_exp, exp_at_best)
    assert torch.equal(best_optim["state"][0]["exp_avg"], exp_at_best)


def _config(epochs, lr, patience) -> str:
    # Written directly so warmup_patience_start and save_spectrum_comparison
    # are not dropped by the smoke helper's section whitelist.
    return f"""[User]
experiment_name = beststate
logger_type = 1
print_model = False

[Network]
name = RNN
tag = RNN
x_dim = 1
activation = relu
dense_x =
dense_h_x =
dim_rnn = 4
num_rnn = 1
type_rnn = RNN
dropout_p = 0
hidden_sh_size =

[Training]
use_cuda = False
optimization = adam
beta = 1
lr = {lr}
epochs = {epochs}
early_stop_patience = {patience}
warmup_patience_start = {patience}
save_frequency = 1
save_spectrum_comparison = false
gradient_clip = 0
optimize_alphas = False
sampling_method = ptf
sampling_ratio = 0
val_criteria = recon
val_spectrum_n_windows = 1
val_spectrum_window_len = 24
val_spectrum_tf_steps = 8
val_drift_horizon = 5
val_rosenstein = false

[DataFrame]
dataset_name = Lorenz63
dataset_label = synthetic
mask_label = None
shuffle = False
batch_size = 2
num_workers = 0
sequence_len = 8
initial_sequence_len = 8
sample_rate = 1
skip_rate = 1
val_indices = 0.25
observation_process = only_x
overlap = False
with_nan = False
"""


def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_saved_best_epoch_is_not_the_stopping_weights(tmp_path, monkeypatch):
    """The final file matches the best epoch and not the last epoch.

    ``save_frequency`` is 1, so the checkpoint is written twice per epoch
    (before the figures and after the plot timer). The first write of the
    best epoch is the snapshot taken when that epoch was selected.
    """
    monkeypatch.setattr(
        "dvae.learning_algo.run_post_training_eval",
        lambda *args, **kwargs: {"eval_failed": False},
    )
    checkpoint_snaps = []
    real_save = torch.save

    def spy(obj, path, *args, **kwargs):
        name = os.path.basename(str(path))
        if name.endswith("_checkpoint.pt") and isinstance(obj, dict):
            checkpoint_snaps.append(_clone_cpu(obj["model_state_dict"]))
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", spy)

    n_epochs = 4
    data_root = tmp_path / "data"
    saved_root = tmp_path / "saved"
    _write_lorenz(data_root)
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(_config(n_epochs, lr="1.0", patience=100))
    _seed()

    from dvae.learning_algo import LearningAlgorithm

    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "best",
            "reload": False,
            "device_config": {
                "saved_root": str(saved_root),
                "data_dir": str(data_root),
                "device_name": "cpu",
            },
        }
    )
    algo.train()

    run_dir = next(Path(saved_root).rglob("loss_model.pckl")).parent
    final_path = next(run_dir.glob("*_final_epoch*.pt"))
    latest_path = next(run_dir.glob("*_latest.pt"))
    best_epoch = int(final_path.stem.split("final_epoch")[-1])
    assert best_epoch < n_epochs - 1, (
        f"best epoch {best_epoch} is the last epoch, so this run cannot "
        "show that the saved weights stayed at the best epoch"
    )
    assert len(checkpoint_snaps) == 2 * n_epochs

    final = _clone_cpu(torch.load(final_path, map_location="cpu", weights_only=False))
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    true_best = checkpoint_snaps[best_epoch * 2]
    assert _states_equal(final, true_best)
    assert _states_equal(final, _clone_cpu(latest["best_state_dict"]))
    assert not _states_equal(final, _clone_cpu(latest["model_state_dict"]))
