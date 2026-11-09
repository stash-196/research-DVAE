"""Resume from the latest checkpoint continues the same training run."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.checkpoint_state import curriculum_state_at_best_epoch
from tests.test_train_smoke import _write_lorenz


def _config(epochs: int) -> str:
    # Nonzero learning rate. Patience start 0 so the first epoch that
    # does not improve the validation loss bumps the window (8 -> 16).
    # This rate stops improving within a few epochs on the sine series,
    # which is what makes that bump fire. Spectrum windows stay short.
    return f"""[User]
experiment_name = resume
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
lr = 100
epochs = {epochs}
early_stop_patience = 100
warmup_patience_start = 0
warmup_order = window_first
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
sequence_len = 16
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


def _train(tmp: Path, epochs: int, *, reload: bool, model_dir, job: str):
    from dvae.learning_algo import LearningAlgorithm

    cfg_path = tmp / f"cfg_{job}.ini"
    cfg_path.write_text(_config(epochs))
    params = {
        "cfg": str(cfg_path),
        "job_id": job,
        "reload": reload,
        "device_config": {
            "saved_root": str(tmp / "saved"),
            "data_dir": str(tmp / "data"),
            "device_name": "cpu",
        },
    }
    if model_dir is not None:
        params["model_dir"] = str(model_dir)
    algo = LearningAlgorithm(params)
    algo.train()
    runs = list((tmp / "saved").rglob("loss_model.pckl"))
    assert len(runs) == 1
    return runs[0].parent


def _history(run_dir: Path) -> dict:
    with open(run_dir / "loss_model.pckl", "rb") as handle:
        return pickle.load(handle)


def test_curriculum_rebuild_uses_the_bump_not_the_pre_bump_series():
    loss_log = {
        "auto_warm_values": [0.0, 0.0, 0.0],
        "kl_warm_values": [0.0, 0.0, 0.1],
        "noise_warm_values": [1.0, 1.0, 1.0],
        "sequence_len_values": [8, 8, 10],
        "cpt_patience_epochs": [0, 1, 0],
        "bump_events": [
            {"epoch": 1, "type": "window", "new": 10},
            {"epoch": 2, "type": "auto", "new": 0.2},
            {"epoch": 2, "type": "KL", "new": 0.1},
        ],
    }
    defaults = {
        "auto": 0.0,
        "kl": 0.0,
        "noise": 1.0,
        "sequence_len": 8,
        "cpt_patience": 0,
    }
    at_window = curriculum_state_at_best_epoch(loss_log, 1, defaults)
    assert at_window["current_sequence_len"] == 10
    assert at_window["warm_up_happened"] is True
    assert at_window["auto_kl_done"] == 0
    assert at_window["cpt_patience"] == 1
    assert at_window["current_auto_warm"] == 0.0

    at_auto = curriculum_state_at_best_epoch(loss_log, 2, defaults)
    assert at_auto["current_auto_warm"] == pytest.approx(0.2)
    assert at_auto["current_kl_warm"] == pytest.approx(0.1)
    assert at_auto["auto_kl_done"] == 2
    assert at_auto["current_sequence_len"] == 10


def test_resume_matches_uninterrupted_losses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dvae.learning_algo.run_post_training_eval",
        lambda *args, **kwargs: {"eval_failed": False},
    )
    for name in ("full", "partial"):
        root = tmp_path / name
        root.mkdir()
        _write_lorenz(root / "data")

    _seed()
    uninterrupted = _train(tmp_path / "full", 6, reload=False, model_dir=None, job="full")
    _seed()
    partial = _train(tmp_path / "partial", 3, reload=False, model_dir=None, job="part")
    latest = next(partial.glob("*_latest.pt"))
    loaded = torch.load(latest, map_location="cpu", weights_only=False)
    assert int(loaded["format_version"]) == 1
    assert int(loaded["epoch"]) == 2
    assert "rng_state" in loaded
    assert "best_state_dict" in loaded

    resumed = _train(
        tmp_path / "partial",
        6,
        reload=True,
        model_dir=partial,
        job="resume",
    )
    assert resumed == partial

    left = _history(uninterrupted)
    right = _history(resumed)
    keys = (
        "train_loss",
        "val_loss",
        "auto_warm_values",
        "sequence_len_values",
        "cpt_patience_epochs",
    )
    diffs = {}
    for key in keys:
        a = np.asarray(left[key], dtype=np.float64).reshape(-1)
        b = np.asarray(right[key], dtype=np.float64).reshape(-1)
        assert a.shape == b.shape
        diffs[key] = float(np.max(np.abs(a - b)))
        assert np.array_equal(a, b), (key, diffs[key])
    sequence = np.asarray(left["sequence_len_values"], dtype=np.float64)
    assert sequence.max() > sequence.min(), sequence
    print("resume max abs diff", diffs)
