#!/usr/bin/env python3
"""Train Professor Forcing, or the hidden-match ablation, on Lorenz63.

This script does not call LearningAlgorithm. Generated sweep configs are
written by bin/generate_professor_forcing_configs.py.
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import date
from pathlib import Path

from dvae.baselines.professor_forcing.config_io import read_ini
from dvae.baselines.professor_forcing.data import load_lorenz_loaders, resolved_x_dim
from dvae.baselines.professor_forcing.trainer import train
from dvae.utils.path_manager import find_project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Path to a Professor Forcing ini.")
    parser.add_argument("--job_id", default=None)
    parser.add_argument("--saved_root", default=None, help="Override the output root.")
    parser.add_argument("--data_dir", default=None, help="Override the dataset root.")
    args = parser.parse_args()

    cfg = read_ini(args.cfg)
    root = find_project_root(__file__)
    data_dir, saved_root = _resolve_paths(root, args.data_dir, args.saved_root)
    cfg["x_dim"] = resolved_x_dim(cfg["observation_process"], cfg["x_dim"])
    job_id = args.job_id or os.environ.get("SLURM_JOBID") or uuid.uuid4().hex[:8]
    experiment = cfg["experiment_name"]
    tag = (
        f"{cfg['variant']}_{cfg['observation_process']}_{cfg['mask_label'] or 'nomask'}"
        f"_{cfg['behavior']}_{cfg['generator_objective']}_{job_id}"
    )
    save_dir = Path(saved_root) / date.today().isoformat() / "professor_forcing" / experiment / tag
    print(f"[professor-forcing] data_dir={data_dir}")
    print(f"[professor-forcing] save_dir={save_dir}")
    train_loader, val_loader, x_dim = load_lorenz_loaders(
        data_dir,
        dataset_label=cfg["dataset_label"],
        mask_label=cfg["mask_label"],
        observation_process=cfg["observation_process"],
        x_dim=cfg["x_dim"],
        sequence_len=cfg["sequence_len"],
        batch_size=cfg["batch_size"],
        val_indices=cfg["val_indices"],
        shuffle=cfg["shuffle"],
        num_workers=cfg["num_workers"],
        device="cpu",
    )
    cfg["x_dim"] = x_dim
    train(cfg, train_loader, val_loader, save_dir)


def _resolve_paths(root: Path, data_override, saved_override):
    data_dir = data_override
    saved_root = saved_override
    if data_dir and saved_root:
        return data_dir, saved_root
    try:
        from dvae.utils.config_handler import load_device_paths

        paths = load_device_paths(str(root / "config" / "device_paths.yaml"))
        data_dir = data_dir or paths["data_dir"]
        saved_root = saved_root or paths["saved_root"]
    except Exception as exc:
        if not data_dir or not saved_root:
            raise SystemExit(
                "Could not resolve data_dir and saved_root from config/device_paths.yaml "
                f"({exc}). Pass --data_dir and --saved_root."
            ) from exc
    return data_dir, saved_root


if __name__ == "__main__":
    main()
