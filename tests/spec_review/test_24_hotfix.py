"""Pins for hotfix PR #24. Synthetic series only.

DatasetConfig must accept ``legacy_defaults`` and ``legacy_windowing`` as
one switch. ``train()`` on tiny Lorenz, DampedSHO, and a fake Xhro session
must reach a window bump, spectrum and drift series, figures, and a
checkpoint. Every key unpacked into DatasetConfig on the train and eval
paths must be a field. The end-of-training eval child must be this
interpreter.
"""

from __future__ import annotations

import ast
import configparser
import os
import pickle
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.dataset.dataset_builder import DATASET_REGISTRY, DatasetConfig
from dvae.dataset.xhro_proper_dataset import (
    coarse_dataframe_kwargs,
    windowing_from_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _config_text(**overrides) -> str:
    lines = {
        "experiment_name": "spec24",
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
    sections = {
        "User": ["experiment_name", "logger_type", "print_model"],
        "Network": [
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
        ],
        "Training": [
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
        ],
        "DataFrame": [
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
            "legacy_windowing",
        ],
    }
    chunks = []
    for section, keys in sections.items():
        body = [f"{key} = {lines[key]}" for key in keys if key in lines]
        chunks.append("[" + section + "]\n" + "\n".join(body))
    return "\n\n".join(chunks) + "\n"


def _parser(text: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read_string(text)
    return cfg


def _dataset_config_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name == "DatasetConfig":
            found.append(node)
    return found


def _minimal_dataset_config(**extra) -> DatasetConfig:
    params = dict(
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
    )
    params.update(extra)
    return DatasetConfig(**params)


def test_legacy_names_are_accepted_and_equivalent():
    """Either name alone is the legacy bundle. Both names survive unpacking."""
    only_defaults = _minimal_dataset_config(legacy_defaults=True)
    only_windowing = _minimal_dataset_config(legacy_windowing=True)
    neither = _minimal_dataset_config()
    both = _minimal_dataset_config(legacy_defaults=True, legacy_windowing=False)
    for turned_on in (only_defaults, only_windowing, both):
        assert turned_on.legacy_defaults is True
        assert turned_on.legacy_windowing is True
    assert neither.legacy_defaults is False
    assert neither.legacy_windowing is False

    blank = _parser("[DataFrame]\n")
    on = _parser("[DataFrame]\nlegacy_defaults = true\n")
    other = _parser("[DataFrame]\nlegacy_windowing = true\n")
    for cfg in (blank, on, other):
        built = _minimal_dataset_config(
            **windowing_from_config(cfg),
            **coarse_dataframe_kwargs(cfg),
        )
        assert isinstance(built, DatasetConfig)
    assert _minimal_dataset_config(**windowing_from_config(on)).legacy_windowing is True
    assert (
        _minimal_dataset_config(**windowing_from_config(other)).legacy_defaults
        is True
    )
    assert (
        _minimal_dataset_config(**windowing_from_config(blank)).legacy_defaults
        is False
    )


def test_train_and_eval_unpack_only_dataset_config_fields():
    """No starred or explicit DatasetConfig key on those paths is unknown."""
    names = {item.name for item in fields(DatasetConfig)}
    paths = [
        ROOT / "src/dvae/learning_algo.py",
        ROOT / "src/dvae/eval/eval_signal.py",
    ]
    starred_funcs = {
        "windowing_from_config": windowing_from_config,
        "coarse_dataframe_kwargs": coarse_dataframe_kwargs,
    }
    cfg = _parser(
        "\n".join(
            [
                "[DataFrame]",
                "legacy_defaults = true",
                "legacy_windowing = false",
                "coarse_include_hr = false",
                "coarse_test_split = usable_tail",
                "start_on_observed = true",
                "strict_non_overlap = true",
                "max_window_nan_excess = 0.2",
                "max_gap_window_frac = 0.25",
                "gap_split_quantile = 0.99",
            ]
        )
    )
    for path in paths:
        calls = _dataset_config_calls(path)
        assert calls, path.name
        for call in calls:
            explicit = set()
            for keyword in call.keywords:
                if keyword.arg is None:
                    func = keyword.value.func
                    fname = func.id if isinstance(func, ast.Name) else func.attr
                    assert fname in starred_funcs, fname
                    extra = set(starred_funcs[fname](cfg)) - names
                    assert extra == set(), (path.name, fname, extra)
                else:
                    explicit.add(keyword.arg)
            unknown = explicit - names
            assert unknown == set(), (path.name, unknown)
            _minimal_dataset_config(
                **windowing_from_config(cfg),
                **coarse_dataframe_kwargs(cfg),
            )


def test_registry_constructors_accept_dataset_config_keys():
    """build_dataloader forwards every DatasetConfig field. None may TypeError."""
    import inspect

    sample = _minimal_dataset_config(
        **windowing_from_config(_parser("[DataFrame]\n")),
        **coarse_dataframe_kwargs(_parser("[DataFrame]\n")),
    )
    payload = {key: getattr(sample, key) for key in sample.__dataclass_fields__}
    payload.update(
        split="train",
        data_dir=".",
        dataset_label="synthetic",
        mask_label="None",
    )
    rejected = {}
    for name, cls in DATASET_REGISTRY.items():
        signature = inspect.signature(cls.__init__)
        has_var_keyword = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if has_var_keyword:
            continue
        missing = [
            key
            for key in payload
            if key not in signature.parameters and key != "self"
        ]
        if missing:
            rejected[name] = missing
    assert rejected == {}


def test_eval_child_is_sys_executable():
    source = (ROOT / "src/dvae/learning_algo.py").read_text()
    tree = ast.parse(source)
    launches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "run":
            continue
        owner = func.value
        if isinstance(owner, ast.Name) and owner.id == "subprocess":
            launches.append(node)
    assert len(launches) == 1
    argv = launches[0].args[0]
    assert isinstance(argv, ast.List)
    first = argv.elts[0]
    assert isinstance(first, ast.Attribute)
    assert first.attr == "executable"
    assert isinstance(first.value, ast.Name) and first.value.id == "sys"


def _write_series(root: Path, kind: str) -> None:
    if kind == "lorenz":
        folder = root / "lorenz63" / "data" / "synthetic"
        folder.mkdir(parents=True)
        t = np.linspace(0, 8 * np.pi, 256)
        series = np.stack(
            [np.sin(t), np.cos(t), np.sin(2 * t)], axis=1
        ).astype(np.float64)
    elif kind == "damped":
        folder = root / "damped_sho" / "data" / "synthetic"
        folder.mkdir(parents=True)
        t = np.linspace(0, 8 * np.pi, 256)
        # One column. squeeze() must keep this as one instance.
        series = np.sin(t).astype(np.float64).reshape(-1, 1)
    else:
        raise AssertionError(kind)
    for name in ("complete_dataset_train.pkl", "complete_dataset_test.pkl"):
        with open(folder / name, "wb") as handle:
            pickle.dump(series, handle)


def _run_train(tmp_path: Path, kind: str, monkeypatch) -> tuple[Path, list]:
    from dvae.learning_algo import LearningAlgorithm

    launches = []

    def _record(argv, *args, **kwargs):
        launches.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("dvae.learning_algo.subprocess.run", _record)
    data_root = tmp_path / "data"
    saved_root = tmp_path / "saved"
    if kind == "xhro":
        from tests.test_xhro_proper import _fake_session

        _fake_session(data_root)
        text = _config_text(
            dataset_name="XhroProper",
            dataset_label="REC1",
            mask_label="realtime",
            observation_process="raw_ch1",
            legacy_defaults="true",
            val_criteria="recon,spectrum",
            warmup_order="window_first",
        )
    elif kind == "damped":
        _write_series(data_root, "damped")
        text = _config_text(
            dataset_name="DampedSHO",
            observation_process="mixed_1d",
        )
    else:
        _write_series(data_root, "lorenz")
        text = _config_text()
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(text)
    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "spec24",
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
    return runs[0].parent, launches


@pytest.mark.parametrize("kind", ["lorenz", "damped", "xhro"])
def test_train_reaches_window_bump_metrics_figures_and_checkpoint(
    tmp_path, kind, monkeypatch
):
    run_dir, launches = _run_train(tmp_path, kind, monkeypatch)
    assert launches, kind
    argv = launches[0]
    assert argv[0] == sys.executable
    assert argv[1].endswith("eval_signal.py")
    assert "--saved_dict" in argv
    assert list(run_dir.glob("*_checkpoint.pt"))
    assert list(run_dir.glob("*_final_epoch*.pt"))
    assert list((run_dir / "vis_during_training").rglob("*.png"))
    with open(run_dir / "loss_model.pckl", "rb") as handle:
        history = pickle.load(handle)
    for key in (
        "val_spectrum",
        "val_base_error",
        "val_auto_drift",
        "val_ftle",
        "time_plot",
    ):
        values = np.asarray(history[key], dtype=float)
        assert values.size >= 1
        assert np.isfinite(values).all(), key
    assert any(row["type"] == "window" for row in history["bump_events"])
