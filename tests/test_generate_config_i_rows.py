"""Focused checks for Planned trainings rows (i) and (i.2)."""

from __future__ import annotations

import configparser
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "bin" / "generate_config_file_for_multiple.py"

EXPECTED_COUNTS = {
    "i": 36,
    "i.2": 36,
}
I_OBS = ("coarse_eeg", "coarse_eeg_acc", "coarse_eeg_acc_tod")
I_XDIM = {
    "coarse_eeg": "4",
    "coarse_eeg_acc": "5",
    "coarse_eeg_acc_tod": "7",
}
I_SUBJECT = "xhro_01_XH032"
I2_SUBJECTS = (
    "xhro_01_XH032",
    "xhro_04_XH058",
    "xhro_01_XH048",
    "xhro_01_XH006",
    "xhro_02_XH039",
)
I_DATA_ROOT = "/bucket/DoyaU/stash/suntory/xhro_dataset_v2/grok_output"
SS_RATIOS = {"0.0", "0.3", "0.5", "0.7"}
SEQ_LENS = {"128", "256", "512"}


def _generate(tmp_path: Path, rows: str) -> list[Path]:
    out = tmp_path / rows.replace(".", "_")
    out.mkdir()
    result = subprocess.run(
        [sys.executable, str(GEN), "--rows", rows, "--output-dir", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "shPLRNN" not in result.stdout
    assert "Number of configurations for MT_RNN: 36" in result.stdout
    return sorted(out.glob("cfg_*.ini"))


def _read(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def _n_alphas(cfg: configparser.ConfigParser) -> int:
    raw = cfg["Network"]["alphas"]
    return len([part for part in raw.split(",") if part.strip()])


def _assert_shared(cfg: configparser.ConfigParser, prefix: str) -> None:
    net = cfg["Network"]
    train = cfg["Training"]
    frame = cfg["DataFrame"]
    assert net["name"] == "MT_RNN"
    assert net["type_rnn"] == "RNN"
    assert "shPLRNN" not in net["type_rnn"]
    assert net["dim_rnn"] == "200"
    assert _n_alphas(cfg) == 9
    assert train["val_criteria"] == "recon"
    assert train["optimize_alphas"].lower() == "true"
    assert train["sampling_method"] == "ss"
    assert train["epochs"] == "20000"
    assert float(train["gradient_clip"]) == 10.0
    assert train["early_stop_patience"] == "500"
    assert train["warmup_order"] == "window_first"
    assert train["auto_warm_start"] == "0.0"
    assert frame["dataset_name"] == "XhroProper"
    assert frame["corpus"] == "multi"
    assert frame["data_root"] == I_DATA_ROOT
    assert frame["mask_label"] == "None"
    assert frame["coarse_test_split"] == "usable_tail"
    assert frame.get("contiguous_val_split", "").lower() not in {
        "true",
        "1",
        "yes",
    }
    assert frame.get("segments", "") == ""
    assert frame.get("crop_start_sec", "") == ""
    assert frame.get("crop_end_sec", "") == ""
    assert frame.get("block_mask_file", "") == ""
    obs = frame["observation_process"]
    assert obs in I_OBS
    assert net["x_dim"] == I_XDIM[obs]
    assert net["dense_x"] == I_XDIM[obs]
    assert frame["sequence_len"] in SEQ_LENS
    assert train["sampling_ratio"] in SS_RATIOS
    assert net["tag"].startswith(f"MT_RNN_{prefix}_")
    assert obs in net["tag"]


def _assert_grid(files: list[Path], cfgs: list[configparser.ConfigParser]) -> None:
    obs = Counter(cfg["DataFrame"]["observation_process"] for cfg in cfgs)
    assert set(obs) == set(I_OBS)
    assert all(count == 12 for count in obs.values())
    ratios = Counter(cfg["Training"]["sampling_ratio"] for cfg in cfgs)
    assert set(ratios) == SS_RATIOS
    assert all(count == 9 for count in ratios.values())
    lengths = Counter(cfg["DataFrame"]["sequence_len"] for cfg in cfgs)
    assert set(lengths) == SEQ_LENS
    assert all(count == 12 for count in lengths.values())
    for path, cfg in zip(files, cfgs):
        seq = cfg["DataFrame"]["sequence_len"]
        assert f"sequence_len-{seq}" in path.name
        assert "L128-256-512" in path.name


@pytest.mark.parametrize("rows, count", list(EXPECTED_COUNTS.items()))
def test_row_counts(tmp_path, rows, count):
    files = _generate(tmp_path, rows)
    assert len(files) == count


def test_i_xh032_coarse_sweep(tmp_path):
    files = _generate(tmp_path, "i")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 36
    _assert_grid(files, cfgs)
    for cfg in cfgs:
        _assert_shared(cfg, "i")
        assert cfg["DataFrame"]["dataset_label"] == I_SUBJECT
        assert cfg["Network"]["tag"].startswith("MT_RNN_i_coarse_")


def test_i2_five_subjects_pooled(tmp_path):
    files = _generate(tmp_path, "i.2")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 36
    _assert_grid(files, cfgs)
    for cfg in cfgs:
        _assert_shared(cfg, "i2")
        label = cfg["DataFrame"]["dataset_label"]
        for subject in I2_SUBJECTS:
            assert subject in label
        assert [part.strip() for part in label.split(",")] == list(I2_SUBJECTS)
        assert cfg["Network"]["tag"].startswith("MT_RNN_i2_coarse_")


def test_bcd_still_emits_twenty_four_without_shplrnn(tmp_path):
    out = tmp_path / "bcd"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, str(GEN), "--rows", "bcd", "--output-dir", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(out.glob("cfg_*.ini"))
    assert len(files) == 24
    assert "shPLRNN" not in result.stdout
    cfg = _read(files[0])
    assert cfg["Network"]["type_rnn"] == "RNN"
    assert cfg["DataFrame"]["dataset_name"] == "XhroProper"
    assert cfg["Training"]["val_criteria"] == "recon"
    assert cfg["DataFrame"]["contiguous_val_split"] == "true"
