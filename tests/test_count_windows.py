"""bin/count_windows.py prints counts, never sample values."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from dvae.dataset.dataset_builder import build_dataloader
from dvae.dataset.xhro_proper_dataset import training_initial_sequence_len
from dvae.learning_algo import training_dataset_config
from dvae.utils import myconf

_PY = os.path.expanduser("~/grok_bot/wenv/bin/python")
if not os.path.isfile(_PY):
    _PY = "python"
_ROOT = Path(__file__).resolve().parents[1]


def _write_session(root: Path, n: int = 500, fs: float = 250.0) -> None:
    t = np.arange(n, dtype=np.float64) / fs
    tone = np.sin(2.0 * np.pi * 10.0 * t)
    base = pd.Timestamp("2026-01-01", tz="UTC").value
    path = (
        root
        / "xhro_packet_loss"
        / "grok_output"
        / "REC1_realtime"
        / "intermediates"
        / "stage05_ecg_filtered.npz"
    )
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        t=t,
        fs=np.asarray(fs),
        datetime_ns=base + np.rint(t * 1e9).astype(np.int64),
        tz=np.asarray("UTC"),
        ch1=tone,
        ch2=tone,
        ecg=tone,
        present=np.ones(n, dtype=bool),
    )


def _ini(path: Path) -> None:
    path.write_text(
        """[User]
experiment_name = count
logger_type = 1
print_model = False

[Network]
name = RNN
tag = RNN
x_dim = 1

[Training]
use_cuda = False
optimization = adam
beta = 1
lr = 0.001
epochs = 1
early_stop_patience = 5
sampling_method = ptf
sampling_ratio = 0
val_criteria = recon

[DataFrame]
dataset_name = XhroProper
dataset_label = REC1
mask_label = realtime
shuffle = False
batch_size = 2
num_workers = 0
sequence_len = 40
initial_sequence_len = 40
sample_rate = 1
skip_rate = 1
val_indices = 0.25
observation_process = raw_ch1
overlap = False
with_nan = False
legacy_defaults = true
"""
    )


def _run(ini: Path, data_dir: Path, seq_len: str | None = None) -> str:
    cmd = [_PY, str(_ROOT / "bin" / "count_windows.py"), str(ini), "--data-dir", str(data_dir)]
    if seq_len is not None:
        cmd.extend(["--seq-len", seq_len])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        cmd, cwd=str(_ROOT), env=env, check=True, capture_output=True, text=True
    )
    return result.stdout


def _parse(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 7
    out = {}
    for line in lines:
        assert "[" not in line
        key, value = line.split("=", 1)
        out[key] = value
    return out


def _direct_counts(ini: Path, data_dir: Path, seq_len: int | None):
    cfg = myconf()
    cfg.read(ini)
    config = training_dataset_config(
        cfg,
        data_dir=str(data_dir),
        device="cpu",
        batch_size=cfg.getint("DataFrame", "batch_size"),
        shuffle=cfg.getboolean("DataFrame", "shuffle"),
        num_workers=0,
        dataset_label="REC1",
        mask_label="realtime",
        seq_len=training_initial_sequence_len(cfg),
    )
    train_loader, val_loader, _, _ = build_dataloader("XhroProper", config, "train")
    if seq_len is not None:
        train_loader.dataset.update_sequence_length(seq_len)
        val_loader.dataset.update_sequence_length(seq_len)
    return len(train_loader.dataset), len(val_loader.dataset)


def test_count_windows_matches_the_training_builder(tmp_path):
    _write_session(tmp_path)
    ini = tmp_path / "cfg.ini"
    _ini(ini)
    parsed = _parse(_run(ini, tmp_path))
    n_train, n_val = _direct_counts(ini, tmp_path, None)
    assert parsed["n_train_windows"] == str(n_train)
    assert parsed["n_val_windows"] == str(n_val)
    assert parsed["n_val_spectrum_windows"] == "na"
    assert parsed["train_finite_sec"] != "na"
    assert parsed["test_finite_sec"] != "na"
    assert float(parsed["missing_fraction"]) == 0.0
    assert "e+" not in "".join(parsed.values())
    longer = _parse(_run(ini, tmp_path, seq_len="20"))
    n_train_20, n_val_20 = _direct_counts(ini, tmp_path, 20)
    assert longer["n_train_windows"] == str(n_train_20)
    assert longer["n_val_windows"] == str(n_val_20)
    assert int(longer["n_train_windows"]) > n_train
