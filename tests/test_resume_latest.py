"""Resume from the latest checkpoint continues the same training run."""

from __future__ import annotations

import os
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.checkpoint_state import (
    auto_kl_done_through_epoch,
    curriculum_state_at_best_epoch,
)
from dvae.learning_algo import overlay_nan_autonomous
from dvae.utils import create_autonomous_mode_selector
from tests.test_train_smoke import _base_config, _write_lorenz


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


def _load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _numpy_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _quiet_config(epochs: int, **extra) -> str:
    # No window curriculum and no early stop. Recon only, so a short run
    # does not spend the epoch on spectrum validation.
    lines = {
        "sampling_method": "ptf",
        "sampling_ratio": "0",
        "shuffle": "False",
        "sequence_len": "8",
        "initial_sequence_len": "8",
        "lr": "0",
        "early_stop_patience": "100",
        "warmup_patience_start": "100",
        "val_criteria": "recon",
        "save_spectrum_comparison": "false",
    }
    lines.update(extra)
    return _ini(epochs, lines)


def _ini(epochs: int, lines: dict) -> str:
    base = {
        "experiment_name": "resume",
        "logger_type": "1",
        "print_model": "False",
        "name": "RNN",
        "tag": "RNN",
        "x_dim": "1",
        "activation": "relu",
        "dense_x": "",
        "dense_h_x": "",
        "dim_rnn": "4",
        "num_rnn": "1",
        "type_rnn": "RNN",
        "dropout_p": "0",
        "hidden_sh_size": "",
        "use_cuda": "False",
        "optimization": "adam",
        "beta": "1",
        "lr": "0",
        "epochs": str(epochs),
        "early_stop_patience": "100",
        "warmup_patience_start": "100",
        "warmup_order": "window_first",
        "save_frequency": "1",
        "gradient_clip": "0",
        "optimize_alphas": "False",
        "sampling_method": "ptf",
        "sampling_ratio": "0",
        "auto_warm_start": "0",
        "val_criteria": "recon",
        "val_spectrum_n_windows": "1",
        "val_spectrum_window_len": "24",
        "val_spectrum_tf_steps": "8",
        "val_drift_horizon": "5",
        "val_rosenstein": "false",
        "dataset_name": "Lorenz63",
        "dataset_label": "synthetic",
        "mask_label": "None",
        "shuffle": "False",
        "batch_size": "2",
        "num_workers": "0",
        "sequence_len": "16",
        "initial_sequence_len": "8",
        "sample_rate": "1",
        "skip_rate": "1",
        "val_indices": "0.25",
        "observation_process": "only_x",
        "overlap": "False",
        "with_nan": "False",
    }
    base.update(lines)
    base["epochs"] = str(epochs)
    user = ["experiment_name", "logger_type", "print_model"]
    network = [
        "name",
        "tag",
        "x_dim",
        "activation",
        "dense_x",
        "dense_h_x",
        "dim_rnn",
        "num_rnn",
        "type_rnn",
        "dropout_p",
        "hidden_sh_size",
    ]
    training = [
        "use_cuda",
        "optimization",
        "beta",
        "lr",
        "epochs",
        "early_stop_patience",
        "warmup_patience_start",
        "warmup_order",
        "save_frequency",
        "save_spectrum_comparison",
        "gradient_clip",
        "optimize_alphas",
        "sampling_method",
        "sampling_ratio",
        "auto_warm_start",
        "val_spectrum_n_windows",
        "val_spectrum_window_len",
        "val_spectrum_tf_steps",
        "val_drift_horizon",
        "val_rosenstein",
        "val_criteria",
    ]
    frame = [
        "dataset_name",
        "dataset_label",
        "mask_label",
        "shuffle",
        "batch_size",
        "num_workers",
        "sequence_len",
        "initial_sequence_len",
        "sample_rate",
        "skip_rate",
        "val_indices",
        "observation_process",
        "overlap",
        "with_nan",
    ]
    sections = {"User": user, "Network": network, "Training": training, "DataFrame": frame}
    chunks = []
    for section, keys in sections.items():
        body = [f"{key} = {base[key]}" for key in keys if key in base and base[key] is not None]
        chunks.append("[" + section + "]\n" + "\n".join(body))
    return "\n\n".join(chunks) + "\n"


def _run(tmp: Path, epochs: int, *, reload: bool, model_dir, job: str, config_text: str):
    from dvae.learning_algo import LearningAlgorithm

    cfg_path = tmp / f"cfg_{job}.ini"
    cfg_path.write_text(config_text)
    params = {
        "cfg": str(cfg_path),
        "job_id": job,
        "reload": reload,
        "device_config": {
            "saved_root": str(tmp / "saved" / job),
            "data_dir": str(tmp / "data"),
            "device_name": "cpu",
        },
    }
    if model_dir is not None:
        params["model_dir"] = str(model_dir)
    algo = LearningAlgorithm(params)
    algo.train()
    if model_dir is not None:
        return Path(model_dir)
    runs = list((tmp / "saved" / job).rglob("loss_model.pckl"))
    assert len(runs) == 1
    return runs[0].parent


def _skip_eval(monkeypatch):
    monkeypatch.setattr(
        "dvae.learning_algo.run_post_training_eval",
        lambda *args, **kwargs: {"eval_failed": False},
    )


def test_dataloader_rebuild_copies_pin_memory_and_workers():
    from dvae.learning_algo import _dataloader_replacing

    dataset = torch.utils.data.TensorDataset(torch.zeros(4, 2))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    rebuilt = _dataloader_replacing(loader, batch_size=1, shuffle=True)
    assert rebuilt.pin_memory is False
    assert rebuilt.num_workers == 0
    assert rebuilt.drop_last is True
    assert rebuilt.batch_size == 1


def test_rng_restore_failure_keeps_the_latest_weights(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    _write_lorenz(tmp_path / "data")
    partial = _run(
        tmp_path,
        2,
        reload=False,
        model_dir=None,
        job="part",
        config_text=_quiet_config(2),
    )
    latest = _load_pt(next(partial.glob("*_latest.pt")))
    assert int(latest["epoch"]) == 1

    def boom(state):
        raise RuntimeError("CUDA device count mismatch")

    monkeypatch.setattr("dvae.learning_algo.restore_rng_state", boom)
    resumed = _run(
        tmp_path,
        3,
        reload=True,
        model_dir=partial,
        job="resume",
        config_text=_quiet_config(3),
    )
    log = (resumed / "log.txt").read_text(encoding="utf-8")
    assert "Continuing with the weights already loaded" in log
    assert "falling back to the best-epoch checkpoint" not in log.lower()
    assert "continuing at epoch 2" in log
    history = _history(resumed)
    assert len(np.asarray(history["train_loss"]).reshape(-1)) == 3


def test_legacy_resume_drops_bump_rows_after_the_best_epoch(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    _write_lorenz(tmp_path / "data")
    trained = _run(
        tmp_path,
        3,
        reload=False,
        model_dir=None,
        job="base",
        config_text=_quiet_config(3),
    )
    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    shutil.copytree(trained, clean)
    shutil.copytree(trained, dirty)
    for run in (clean, dirty):
        next(run.glob("*_latest.pt")).unlink()
    checkpoint_path = next(dirty.glob("*_checkpoint.pt"))
    checkpoint = _load_pt(checkpoint_path)
    assert int(checkpoint["epoch"]) == 0
    checkpoint["loss_log"].setdefault("bump_events", []).append(
        {
            "epoch": 1,
            "type": "auto",
            "old": 0.0,
            "new": 0.777,
            "patience": 1.0,
            "patience_after": 1.0,
            "cpt_patience": 1,
        }
    )
    torch.save(checkpoint, checkpoint_path)

    clean_dir = _run(
        tmp_path,
        3,
        reload=True,
        model_dir=clean,
        job="clean",
        config_text=_quiet_config(3),
    )
    dirty_dir = _run(
        tmp_path,
        3,
        reload=True,
        model_dir=dirty,
        job="dirty",
        config_text=_quiet_config(3),
    )
    dirty_history = _history(dirty_dir)
    clean_history = _history(clean_dir)
    stale = [
        row
        for row in dirty_history["bump_events"]
        if abs(float(row["new"]) - 0.777) < 1e-6
    ]
    assert stale == []
    last = int(np.asarray(dirty_history["train_loss"]).reshape(-1).size) - 1
    assert auto_kl_done_through_epoch(
        dirty_history["bump_events"], last
    ) == auto_kl_done_through_epoch(clean_history["bump_events"], last)
    dirty_latest = _load_pt(next(dirty_dir.glob("*_latest.pt")))
    clean_latest = _load_pt(next(clean_dir.glob("*_latest.pt")))
    assert int(dirty_latest["auto_kl_done"]) == int(clean_latest["auto_kl_done"])


def test_legacy_resume_rebuilds_curriculum_at_the_best_epoch(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    _write_lorenz(tmp_path / "data")
    # The window bump config: patience start 0, initial length 8, target 16.
    trained = _run(
        tmp_path,
        4,
        reload=False,
        model_dir=None,
        job="bump",
        config_text=_config(4),
    )
    checkpoint = _load_pt(next(trained.glob("*_checkpoint.pt")))
    best = int(checkpoint["epoch"])
    rebuilt = curriculum_state_at_best_epoch(
        checkpoint["loss_log"],
        best,
        {
            "auto": 0.0,
            "kl": 1.0,
            "noise": 1.0,
            "sequence_len": 8,
            "cpt_patience": 0,
        },
    )
    next(trained.glob("*_latest.pt")).unlink()
    resumed = _run(
        tmp_path,
        6,
        reload=True,
        model_dir=trained,
        job="legacy",
        config_text=_config(6),
    )
    history = _history(resumed)
    assert len(np.asarray(history["train_loss"]).reshape(-1)) == 6
    assert int(history["sequence_len_values"][best + 1]) == int(
        rebuilt["current_sequence_len"]
    )
    assert float(history["auto_warm_values"][best + 1]) == pytest.approx(
        rebuilt["current_auto_warm"]
    )
    log = (resumed / "log.txt").read_text(encoding="utf-8")
    assert "Rebuilt warm-up curriculum" in log


def test_spectrum_flag_does_not_move_rng_after_one_epoch(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    # The 256-step Lorenz series has one full-length window, so the long
    # figure is skipped and the spectrum forward never runs. Cap the
    # figure at 32 so this epoch actually draws it.
    monkeypatch.setattr(
        "dvae.learning_algo.cap_eval_seq_len",
        lambda dataset, requested: 32,
    )
    states = {}
    for flag in ("true", "false"):
        root = tmp_path / flag
        root.mkdir()
        _write_lorenz(root / "data")
        _seed()
        run = _run(
            root,
            1,
            reload=False,
            model_dir=None,
            job=flag,
            config_text=_quiet_config(
                1,
                save_spectrum_comparison=flag,
                sampling_method="ss",
                sampling_ratio="0.5",
                auto_warm_start="0.5",
                shuffle="True",
                sequence_len="16",
                initial_sequence_len="16",
            ),
        )
        spectrum = list(run.rglob("spectrum_comparison/**/*.npz"))
        if flag == "true":
            assert spectrum, "spectrum comparison did not run"
        else:
            assert spectrum == []
        numpy_state = np.random.get_state()
        states[flag] = (
            torch.get_rng_state().clone(),
            numpy_state,
            random.getstate(),
        )
    assert torch.equal(states["true"][0], states["false"][0])
    _numpy_equal(states["true"][1], states["false"][1])
    assert states["true"][2] == states["false"][2]


def test_stochastic_resume_matches_with_spectrum_comparison_on(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    monkeypatch.setattr(
        "dvae.learning_algo.cap_eval_seq_len",
        lambda dataset, requested: 32,
    )

    def config(epochs):
        # save_spectrum_comparison left unset: the training default is on.
        return _ini(
            epochs,
            {
                "sampling_method": "ss",
                "sampling_ratio": "0.5",
                "auto_warm_start": "0.5",
                "shuffle": "True",
                "sequence_len": "16",
                "initial_sequence_len": "16",
                "lr": "0",
                "early_stop_patience": "100",
                "warmup_patience_start": "100",
                "val_criteria": "recon",
                "save_spectrum_comparison": None,
            },
        )

    for name in ("full", "partial"):
        root = tmp_path / name
        root.mkdir()
        _write_lorenz(root / "data")
    _seed()
    uninterrupted = _run(
        tmp_path / "full",
        3,
        reload=False,
        model_dir=None,
        job="full",
        config_text=config(3),
    )
    assert list(uninterrupted.rglob("spectrum_comparison/**/*.npz"))
    _seed()
    partial = _run(
        tmp_path / "partial",
        1,
        reload=False,
        model_dir=None,
        job="part",
        config_text=config(1),
    )
    resumed = _run(
        tmp_path / "partial",
        3,
        reload=True,
        model_dir=partial,
        job="resume",
        config_text=config(3),
    )
    left = _history(uninterrupted)
    right = _history(resumed)
    for key in ("train_loss", "val_loss"):
        a = np.asarray(left[key], dtype=np.float64).reshape(-1)
        b = np.asarray(right[key], dtype=np.float64).reshape(-1)
        assert a.shape == b.shape
        assert np.array_equal(a, b), (key, float(np.max(np.abs(a - b))))


def test_training_figures_draw_the_nan_overlaid_selector(tmp_path, monkeypatch):
    _skip_eval(monkeypatch)
    monkeypatch.setattr(
        "dvae.learning_algo.cap_eval_seq_len",
        lambda dataset, requested: 32,
    )
    calls = []

    def fake(batch, dvae, auto_mode_selector, save_path, **kwargs):
        calls.append(
            {
                "path": str(save_path),
                "batch": batch.detach().cpu().clone(),
                "selector": auto_mode_selector.detach().cpu().clone(),
            }
        )

    monkeypatch.setattr(
        "dvae.learning_algo.visualize_teacherforcing_2_autonomous",
        fake,
    )
    from dvae.dataset.lorenz63_dataset import Lorenz63

    original_get = Lorenz63.__getitem__

    def with_nan(self, index):
        item = original_get(self, index)
        if int(self.seq_len) == 32:
            item = np.array(item, dtype=np.float32, copy=True)
            item[1, 0] = np.float32(np.nan)
        return item

    monkeypatch.setattr(Lorenz63, "__getitem__", with_nan)
    _write_lorenz(tmp_path / "data")
    _run(
        tmp_path,
        1,
        reload=False,
        model_dir=None,
        job="nanviz",
        config_text=_base_config(
            epochs="1",
            early_stop_patience="100",
            lr="0",
            val_criteria="recon",
            sequence_len="8",
            initial_sequence_len="8",
            save_frequency="1",
        ),
    )
    found = {"even_bursts": False, "half_half": False}
    for call in calls:
        path = call["path"]
        if "even_bursts" in path:
            mode = "even_bursts"
            ratio = 0.1
        elif path.rstrip("/").endswith("half_half") or "/half_half" in path:
            mode = "half_half"
            ratio = 0.0
        else:
            continue
        batch = call["batch"]
        selector = call["selector"]
        assert torch.isnan(batch[:, 0, 0]).any()
        expected = overlay_nan_autonomous(
            batch,
            create_autonomous_mode_selector(
                batch.size(0),
                mode=mode,
                autonomous_ratio=ratio,
                batch_size=batch.size(1),
                x_dim=batch.size(2),
                device=batch.device,
            ),
        )
        assert torch.equal(selector, expected)
        assert float(selector[1, 0, 0]) == 1.0
        assert float(selector[0, 0, 0]) == 0.0
        found[mode] = True
    assert found["even_bursts"] and found["half_half"]

