"""Focused checks for Planned trainings rows m, n, and o after weight calibration."""

from __future__ import annotations

import configparser
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "bin" / "generate_config_file_for_multiple.py"

EXPECTED_COUNTS = {"m": 8, "n": 8, "o": 16}
SS_RATIOS = {"0.0", "0.3", "0.5", "0.7"}
SUBBAND_WEIGHT = 170.0
DECORR_WEIGHT_HIDDEN = 0.9
DECORR_WEIGHT_OUTPUT = 1.2


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
    return sorted(out.glob("cfg_*.ini"))


def _read(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def _n_alphas(cfg: configparser.ConfigParser) -> int:
    raw = cfg["Network"]["alphas"]
    return len([part for part in raw.split(",") if part.strip()])


def _unset(raw: str) -> bool:
    return raw.strip() in {"", "off", "none", "false"}


def _assert_shared(cfg: configparser.ConfigParser) -> None:
    net = cfg["Network"]
    train = cfg["Training"]
    frame = cfg["DataFrame"]
    assert net["name"] == "MT_RNN"
    assert net["type_rnn"] == "RNN"
    assert "shPLRNN" not in net["type_rnn"]
    assert net["dim_rnn"] == "200"
    assert train["val_criteria"] == "recon"
    assert train["optimize_alphas"].lower() == "true"
    assert train["sampling_method"] == "ss"
    assert train["epochs"] == "20000"
    assert float(train["gradient_clip"]) == 10.0
    assert train["early_stop_patience"] == "500"
    assert train["warmup_order"] == "window_first"
    assert frame["dataset_name"] == "Xhro"
    assert frame["dataset_label"] == "XHRO_02_XH070"
    assert frame["observation_process"] == "raw_all"
    assert frame["sequence_len"] == "1000"
    assert net["x_dim"] == "4"
    assert net["dense_x"] == "4"
    assert frame.get("contiguous_val_split", "").lower() not in {"true", "1", "yes"}


def _assert_grid(cfgs: list[configparser.ConfigParser], n_per_ratio: int) -> None:
    ratios = Counter(cfg["Training"]["sampling_ratio"] for cfg in cfgs)
    assert set(ratios) == SS_RATIOS
    assert all(count == n_per_ratio for count in ratios.values())
    alpha_counts = Counter(_n_alphas(cfg) for cfg in cfgs)
    assert alpha_counts[9] == n_per_ratio * 2
    assert alpha_counts[200] == n_per_ratio * 2


@pytest.mark.parametrize("rows, count", list(EXPECTED_COUNTS.items()))
def test_row_counts(tmp_path, rows, count):
    files = _generate(tmp_path, rows)
    assert len(files) == count


def test_m_block_split_without_regularizer_weights(tmp_path):
    files = _generate(tmp_path, "m")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 8
    _assert_grid(cfgs, n_per_ratio=2)
    for cfg in cfgs:
        _assert_shared(cfg)
        assert cfg["Network"]["hidden_block_split"] == "half"
        assert cfg["Network"]["ecg_channels"] == "0,1"
        assert cfg["Network"]["eeg_channels"] == "2,3"
        assert _unset(cfg["Training"].get("subband_loss_weight", ""))
        assert _unset(cfg["Training"].get("decorr_loss_weight", ""))
        assert cfg["Network"]["tag"].startswith("MT_RNN_m_")


def test_n_subband_weight_is_calibrated(tmp_path):
    files = _generate(tmp_path, "n")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 8
    _assert_grid(cfgs, n_per_ratio=2)
    for cfg in cfgs:
        _assert_shared(cfg)
        assert float(cfg["Training"]["subband_loss_weight"]) == SUBBAND_WEIGHT
        assert cfg["Training"]["subband_loss_warmup"].lower() == "true"
        assert cfg["Training"]["subband_channels"] == "2,3"
        assert cfg["Training"]["subband_fs"] == "250"
        assert float(cfg["Training"]["subband_low_hz"]) == 4.0
        assert _unset(cfg["Training"].get("decorr_loss_weight", ""))
        assert cfg["Network"]["tag"].startswith("MT_RNN_n_")


def test_o_decorr_weights_match_mode(tmp_path):
    files = _generate(tmp_path, "o")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 16
    _assert_grid(cfgs, n_per_ratio=4)
    hidden = [cfg for cfg in cfgs if cfg["Training"]["decorr_mode"] == "hidden"]
    output = [cfg for cfg in cfgs if cfg["Training"]["decorr_mode"] == "output"]
    assert len(hidden) == 8
    assert len(output) == 8
    for cfg in hidden:
        _assert_shared(cfg)
        assert float(cfg["Training"]["decorr_loss_weight"]) == DECORR_WEIGHT_HIDDEN
        assert cfg["Training"]["decorr_loss_warmup"].lower() == "true"
        assert cfg["Network"]["hidden_block_split"] == "half"
        assert cfg["Network"]["ecg_channels"] == "0,1"
        assert cfg["Network"]["eeg_channels"] == "2,3"
        assert _unset(cfg["Training"].get("subband_loss_weight", ""))
        assert cfg["Network"]["tag"].startswith("MT_RNN_o_")
        assert cfg["Network"]["tag"].endswith("_hidden") or "_hidden" in cfg["Network"]["tag"]
    for cfg in output:
        _assert_shared(cfg)
        assert float(cfg["Training"]["decorr_loss_weight"]) == DECORR_WEIGHT_OUTPUT
        assert cfg["Training"]["decorr_loss_warmup"].lower() == "true"
        assert cfg["Training"]["decorr_groups"] == "0,1|2,3"
        assert _unset(cfg["Training"].get("subband_loss_weight", ""))
        assert "_output" in cfg["Network"]["tag"]



def test_other_rows_do_not_pick_up_mno_calib_weights(tmp_path):
    """bcd and g must not receive the calibrated n/o regularizer weights."""
    for rows, count in (("bcd", 24), ("g", 4)):
        files = _generate(tmp_path, rows)
        assert len(files) == count
        for cfg in (_read(path) for path in files):
            train = cfg["Training"]
            sub = train.get("subband_loss_weight", "").strip()
            dec = train.get("decorr_loss_weight", "").strip()
            if sub:
                assert float(sub) != SUBBAND_WEIGHT
            if dec:
                assert float(dec) not in {DECORR_WEIGHT_HIDDEN, DECORR_WEIGHT_OUTPUT}
            assert cfg["Training"]["val_criteria"] == "recon"
