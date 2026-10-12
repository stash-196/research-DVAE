"""Train-path smoke tests.

``DatasetConfig(**windowing_from_config(cfg))`` is the first thing
``LearningAlgorithm.train`` does. ``legacy_defaults`` is an alias of
``legacy_windowing`` and must be a field on that dataclass, or every
training run dies at dataset setup. The end-to-end cases call ``train``
on tiny synthetic series so a new unpacked key fails here.
"""

from __future__ import annotations

import configparser
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import fields

from dvae.dataset.dataset_builder import DatasetConfig
from dvae.dataset.xhro_proper_dataset import (
    XhroProper,
    coarse_dataframe_kwargs,
    windowing_from_config,
)


def _base_config(**overrides) -> str:
    lines = {
        "experiment_name": "smoke",
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
        "epochs": "4",
        "early_stop_patience": "1",
        "save_frequency": "1",
        "gradient_clip": "0",
        "optimize_alphas": "False",
        "sampling_method": "ptf",
        "sampling_ratio": "0",
        "dataset_name": "Lorenz63",
        "dataset_label": "synthetic",
        "mask_label": "None",
        "shuffle": "False",
        "batch_size": "2",
        "num_workers": "0",
        "sequence_len": "10",
        "initial_sequence_len": "8",
        "sample_rate": "1",
        "skip_rate": "1",
        "val_indices": "0.25",
        "observation_process": "only_x",
        "overlap": "False",
        "with_nan": "False",
        "val_spectrum_n_windows": "1",
        "val_spectrum_window_len": "24",
        "val_spectrum_tf_steps": "8",
        "val_drift_horizon": "5",
        "val_rosenstein": "false",
        "val_criteria": "recon,spectrum",
        "warmup_order": "window_first",
    }
    lines.update(overrides)
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
        "save_frequency",
        "gradient_clip",
        "optimize_alphas",
        "sampling_method",
        "sampling_ratio",
        "val_spectrum_n_windows",
        "val_spectrum_window_len",
        "val_spectrum_tf_steps",
        "val_drift_horizon",
        "val_rosenstein",
        "val_criteria",
        "warmup_order",
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
        "legacy_defaults",
    ]
    sections = {"User": user, "Network": network, "Training": training, "DataFrame": frame}
    chunks = []
    for section, keys in sections.items():
        body = [f"{key} = {lines[key]}" for key in keys if key in lines]
        if body:
            chunks.append("[" + section + "]\n" + "\n".join(body))
    return "\n\n".join(chunks) + "\n"


def _write_lorenz(root: Path) -> None:
    folder = root / "lorenz63" / "data" / "synthetic"
    folder.mkdir(parents=True)
    t = np.linspace(0, 8 * np.pi, 256)
    series = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1).astype(np.float64)
    for name in ("complete_dataset_train.pkl", "complete_dataset_test.pkl"):
        with open(folder / name, "wb") as handle:
            pickle.dump(series, handle)


def _write_damped(root: Path) -> None:
    folder = root / "damped_sho" / "data" / "synthetic"
    folder.mkdir(parents=True)
    t = np.linspace(0, 8 * np.pi, 256)
    series = np.sin(t).astype(np.float64).reshape(-1, 1)
    for name in ("complete_dataset_train.pkl", "complete_dataset_test.pkl"):
        with open(folder / name, "wb") as handle:
            pickle.dump(series, handle)


def _dataset_config_from_cfg(cfg) -> DatasetConfig:
    """Same unpack ``train`` and ``eval_signal`` use for the window keys."""
    return DatasetConfig(
        data_dir=".",
        x_dim=1,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="only_x",
        overlap=False,
        with_nan=False,
        seq_len=8,
        device="cpu",
        **windowing_from_config(cfg),
        **coarse_dataframe_kwargs(cfg),
    )


def test_unpacked_window_and_coarse_keys_are_dataset_config_fields():
    names = {item.name for item in fields(DatasetConfig)}
    blank = configparser.ConfigParser()
    blank.read_string("[DataFrame]\n")
    aliased = configparser.ConfigParser()
    aliased.read_string("[DataFrame]\nlegacy_defaults = true\n")
    for cfg in (blank, aliased):
        extra = (set(windowing_from_config(cfg)) | set(coarse_dataframe_kwargs(cfg))) - names
        assert extra == set()
    off = _dataset_config_from_cfg(blank)
    assert off.legacy_windowing is False
    assert off.legacy_defaults is False
    on = _dataset_config_from_cfg(aliased)
    assert on.legacy_windowing is True
    assert on.legacy_defaults is True
    only_alias = DatasetConfig(
        data_dir=".",
        x_dim=1,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="only_x",
        overlap=False,
        with_nan=False,
        seq_len=8,
        device="cpu",
        legacy_defaults=True,
    )
    assert only_alias.legacy_windowing is True
    assert only_alias.legacy_defaults is True


def test_xhro_proper_maps_legacy_defaults_onto_windowing(tmp_path):
    from tests.test_xhro_proper import _fake_session

    _fake_session(tmp_path)
    dataset = XhroProper(
        data_dir=str(tmp_path),
        dataset_label="REC1",
        mask_label="realtime",
        split="train",
        seq_len=16,
        x_dim=1,
        sample_rate=1,
        skip_rate=1,
        val_indices=0.0,
        observation_process="raw_ch1",
        device="cpu",
        overlap=False,
        shuffle=False,
        legacy_defaults=True,
    )
    assert dataset.legacy_defaults is True
    assert dataset.legacy_windowing is True
    assert dataset.start_on_observed is False
    assert dataset.strict_non_overlap is False
    assert dataset.max_gap_window_frac is None


def _run_train(tmp_path: Path, kind: str) -> Path:
    from dvae.learning_algo import LearningAlgorithm

    data_root = tmp_path / "data"
    saved_root = tmp_path / "saved"
    if kind == "lorenz":
        _write_lorenz(data_root)
        text = _base_config()
    elif kind == "damped":
        _write_damped(data_root)
        text = _base_config(
            dataset_name="DampedSHO",
            observation_process="mixed_1d",
        )
    elif kind == "xhro":
        from tests.test_xhro_proper import _fake_session

        _fake_session(data_root)
        text = _base_config(
            dataset_name="XhroProper",
            dataset_label="REC1",
            mask_label="realtime",
            observation_process="raw_ch1",
            legacy_defaults="true",
            # Explicit, so the legacy switch still restores old window
            # placement without also dropping spectrum and drift.
            val_criteria="recon,spectrum",
            warmup_order="window_first",
        )
    else:
        raise AssertionError(kind)
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(text)
    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "smoke",
            "reload": False,
            "device_config": {
                "saved_root": str(saved_root),
                "data_dir": str(data_root),
                "device_name": "cpu",
            },
        }
    )
    algo.build_model()
    algo.train()
    runs = list(saved_root.rglob("loss_model.pckl"))
    assert len(runs) == 1
    return runs[0].parent


@pytest.mark.parametrize("kind", ["lorenz", "damped", "xhro"])
def test_train_runs_epochs_with_metrics_figures_and_checkpoint(tmp_path, kind):
    run_dir = _run_train(tmp_path, kind)
    checkpoints = list(run_dir.glob("*_checkpoint.pt"))
    finals = list(run_dir.glob("*_final_epoch*.pt"))
    assert checkpoints
    assert finals
    figures = list((run_dir / "vis_during_training").rglob("*.png"))
    assert figures
    with open(run_dir / "loss_model.pckl", "rb") as handle:
        history = pickle.load(handle)
    for key in ("val_spectrum", "val_base_error", "val_auto_drift", "val_ftle", "time_plot"):
        values = np.asarray(history[key], dtype=float)
        assert values.size >= 1
        assert np.isfinite(values).all()
    events = history["bump_events"]
    assert any(row["type"] == "window" for row in events)
    assert "legacy_defaults" in (run_dir / "config.ini").read_text() or kind != "xhro"
