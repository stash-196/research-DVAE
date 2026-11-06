"""Focused checks for Planned trainings rows k.1, k.2a, k.2b, g, g.2, g.3."""

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
    "k.1": 24,
    "k.2a": 24,
    "k.2b": 24,
    "g": 4,
    "g.2": 4,
    "g.3": 8,
}
K_OBS = ("bipolar_eeg", "ecg_removed_eeg_scale", "ecg_removed_eeg")
PKTLOSS_RETRANS = "XHRO3506_20260622T142411000+0900"
PKTLOSS_BLOCK_MASK = (
    "/flash/DoyaU/stash/grok_scratch/xhro_pl_windows/pl_representative_flags_v2.npz"
)
MULTI_DATA_ROOT = "/bucket/DoyaU/stash/suntory/xhro_dataset_v2/grok_output"
SS_RATIOS = {"0.0", "0.3", "0.5", "0.7"}
G2_RANGES = (
    "46080-46200",
    "65460-65580",
    "123450-123570",
    "180000-180120",
    "213360-213480",
)
G3_PARTS = (
    "xhro_01_XH032:180180-180300",
    "xhro_04_XH058:57810-57930",
    "xhro_01_XH048:204150-204270",
    "xhro_02_XH039:112560-112680",
)
G3_SUBJECTS = (
    "xhro_01_XH032",
    "xhro_04_XH058",
    "xhro_01_XH048",
    "xhro_02_XH039",
)


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
    assert frame["contiguous_val_split"] == "true"
    assert frame["sequence_len"] == "1000"
    assert frame["dataset_name"] == "XhroProper"
    assert train["val_spectrum_window_len"] == "1000"
    assert train["val_spectrum_tf_steps"] == "500"
    assert train["val_window_stride"] == "500"
    assert train["val_window_max_gap_steps"] == "10"


def _assert_k_grid(cfgs: list[configparser.ConfigParser]) -> None:
    obs = Counter(cfg["DataFrame"]["observation_process"] for cfg in cfgs)
    assert set(obs) == set(K_OBS)
    assert all(count == 8 for count in obs.values())
    alpha_counts = Counter(_n_alphas(cfg) for cfg in cfgs)
    assert alpha_counts == {9: 12, 200: 12}
    ratios = Counter(cfg["Training"]["sampling_ratio"] for cfg in cfgs)
    assert set(ratios) == SS_RATIOS
    assert all(count == 6 for count in ratios.values())
    for cfg in cfgs:
        assert cfg["Network"]["x_dim"] == "1"
        assert cfg["Network"]["dense_x"] == "1"


def _assert_g_shared(cfgs: list[configparser.ConfigParser]) -> None:
    for cfg in cfgs:
        frame = cfg["DataFrame"]
        assert frame["observation_process"] == "raw_all"
        assert cfg["Network"]["x_dim"] == "4"
        assert cfg["Network"]["dense_x"] == "4"
        assert frame["corpus"] == "multi"
        assert frame["data_root"] == MULTI_DATA_ROOT
        assert frame["mask_label"] == "None"
        assert frame["block_mask_file"] == ""
    ratios = Counter(cfg["Training"]["sampling_ratio"] for cfg in cfgs)
    assert set(ratios) == SS_RATIOS


@pytest.mark.parametrize("rows, count", list(EXPECTED_COUNTS.items()))
def test_row_counts(tmp_path, rows, count):
    files = _generate(tmp_path, rows)
    assert len(files) == count


def test_k1_packet_loss_retrans_seg_b(tmp_path):
    files = _generate(tmp_path, "k.1")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 24
    _assert_k_grid(cfgs)
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        assert frame["dataset_label"] == PKTLOSS_RETRANS
        assert frame["mask_label"] == "retrans"
        assert float(frame["crop_start_sec"]) == 5070.0
        assert float(frame["crop_end_sec"]) == 5670.0
        assert frame["block_mask_file"] == PKTLOSS_BLOCK_MASK
        assert frame["block_mask_mode"] == "not_good"
        assert frame.get("corpus", "None") in {"", "None"}
        assert frame.get("segments", "") == ""
        assert cfg["Network"]["tag"].startswith("MT_RNN_k1_")


def test_k2a_xh070_thesis_slice(tmp_path):
    files = _generate(tmp_path, "k.2a")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 24
    _assert_k_grid(cfgs)
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        assert frame["dataset_label"] == "xhro_02_XH070"
        assert frame["corpus"] == "multi"
        assert frame["data_root"] == MULTI_DATA_ROOT
        assert frame["mask_label"] == "None"
        assert float(frame["crop_start_sec"]) == 30909.713
        assert float(frame["crop_end_sec"]) == 31509.713
        assert frame["block_mask_file"] == ""
        assert frame.get("segments", "") == ""
        assert cfg["Network"]["tag"].startswith("MT_RNN_k2a_")


def test_k2b_xh070_clean_slice(tmp_path):
    files = _generate(tmp_path, "k.2b")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 24
    _assert_k_grid(cfgs)
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        assert frame["dataset_label"] == "xhro_02_XH070"
        assert frame["corpus"] == "multi"
        assert frame["data_root"] == MULTI_DATA_ROOT
        assert float(frame["crop_start_sec"]) == 33660.0
        assert float(frame["crop_end_sec"]) == 34260.0
        assert frame["block_mask_file"] == ""
        assert frame.get("segments", "") == ""
        assert cfg["Network"]["tag"].startswith("MT_RNN_k2b_")


def test_g_xh032_single_crop(tmp_path):
    files = _generate(tmp_path, "g")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 4
    _assert_g_shared(cfgs)
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        assert frame["dataset_label"] == "xhro_01_XH032"
        assert float(frame["crop_start_sec"]) == 53458.0
        assert float(frame["crop_end_sec"]) == 54058.0
        assert frame.get("segments", "") == ""
        assert _n_alphas(cfg) == 9
        assert cfg["Network"]["tag"].startswith("MT_RNN_g_")
    ratios = Counter(cfg["Training"]["sampling_ratio"] for cfg in cfgs)
    assert ratios == {ratio: 1 for ratio in SS_RATIOS}


def test_g2_xh032_five_segments(tmp_path):
    files = _generate(tmp_path, "g.2")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 4
    _assert_g_shared(cfgs)
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        assert frame["dataset_label"] == "xhro_01_XH032"
        assert frame["crop_start_sec"] == ""
        assert frame["crop_end_sec"] == ""
        segments = frame["segments"]
        assert segments.startswith("xhro_01_XH032:")
        for piece in G2_RANGES:
            assert piece in segments
        assert _n_alphas(cfg) == 9
        assert cfg["Network"]["tag"].startswith("MT_RNN_g2_")


def test_g3_four_subjects_pooled(tmp_path):
    files = _generate(tmp_path, "g.3")
    cfgs = [_read(path) for path in files]
    assert len(cfgs) == 8
    _assert_g_shared(cfgs)
    alpha_counts = Counter(_n_alphas(cfg) for cfg in cfgs)
    assert alpha_counts == {9: 4, 200: 4}
    for cfg in cfgs:
        _assert_shared(cfg)
        frame = cfg["DataFrame"]
        label = frame["dataset_label"]
        for subject in G3_SUBJECTS:
            assert subject in label
        assert frame["crop_start_sec"] == ""
        assert frame["crop_end_sec"] == ""
        segments = frame["segments"]
        for part in G3_PARTS:
            assert part in segments
        assert cfg["Network"]["tag"].startswith("MT_RNN_g3_")


def test_bcd_still_emits_twenty_four_without_shplrnn(tmp_path):
    files = _generate(tmp_path, "bcd")
    assert len(files) == 24
    cfg = _read(files[0])
    _assert_shared(cfg)
    assert cfg["Network"]["x_dim"] == "4"
    assert "shPLRNN" not in cfg["Network"]["type_rnn"]
