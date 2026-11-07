"""The baselines stay out of the live training and eval pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from dvae.utils.read_config import myconf

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    path = ROOT / "bin" / "generate_gru_d_configs.py"
    spec = importlib.util.spec_from_file_location("generate_gru_d_configs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

UNTOUCHED = (
    "src/dvae/learning_algo.py",
    "src/dvae/eval/eval_signal.py",
    "src/dvae/eval/aggregate_evaluation_results.py",
    "src/dvae/eval/aggregate_plot_style.py",
    "src/dvae/dataset/physionet2012_dataset.py",
    "src/dvae/dataset/xhro_proper_dataset.py",
    "bin/generate_config_file_for_multiple.py",
    "config/lorenz63/cfg_rnn.ini",
    "config/lorenz63/cfg_mt_rnn.ini",
    "config/general_signal/cfg_base_template.ini",
)


def test_live_pipeline_files_do_not_mention_the_new_baselines():
    needles = ("GRU_D", "build_GRU_D", "pysindy", "fit_sindy")
    for relative in UNTOUCHED:
        text = (ROOT / relative).read_text()
        for needle in needles:
            assert needle not in text, f"{needle} found in {relative}"


def test_committed_rnn_config_is_still_an_rnn():
    cfg = myconf()
    assert cfg.read(ROOT / "config" / "lorenz63" / "cfg_rnn.ini")
    assert cfg.get("Network", "name") == "RNN"
    assert cfg.get("Network", "type_rnn") == "RNN"
    mt = myconf()
    assert mt.read(ROOT / "config" / "lorenz63" / "cfg_mt_rnn.ini")
    assert mt.get("Network", "name") == "MT_RNN"


def test_generator_writes_only_gru_d_inis(tmp_path):
    before = {
        relative: (ROOT / relative).read_bytes()
        for relative in UNTOUCHED
    }
    written = _generator().generate_gru_d_configs(
        tmp_path,
        "unit",
        {
            "observation_process": ["all_xyz", "only_x"],
            "mask_label": ["None"],
            "dim_rnn": [16],
            "dropout_p": [0.0],
            "lr": [0.001],
            "epochs": [2],
            "early_stop_patience": [1],
            "save_frequency": [1],
            "gradient_clip": [1.0],
            "seed": [0],
            "dataset_name": ["Lorenz63"],
            "dataset_label": ["sigma10_rho28_beta8d3_N108k_dt0.01"],
            "shuffle": ["True"],
            "batch_size": [8],
            "num_workers": [0],
            "sequence_len": [200],
            "val_indices": [0.2],
        },
    )
    assert len(written) == 2
    widths = {}
    for path in written:
        assert path.parent == tmp_path
        assert path.name.startswith("cfg_gru_d_")
        cfg = myconf()
        assert cfg.read(path)
        assert cfg.get("Network", "name") == "GRU_D"
        assert cfg.get("Network", "num_rnn") == "1"
        widths[cfg.get("DataFrame", "observation_process")] = cfg.getint("Network", "x_dim")
    assert widths == {"all_xyz": 3, "only_x": 1}
    for relative, payload in before.items():
        assert (ROOT / relative).read_bytes() == payload


def test_generator_refuses_handwritten_lorenz_config_dir(tmp_path):
    protected = tmp_path / "lorenz63"
    protected.mkdir()
    with pytest.raises(RuntimeError, match="refusing"):
        _generator().assert_output_dir_allowed(protected)
