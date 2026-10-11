#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a saved GRU-D checkpoint on reconstruction, imputation, and rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dvae.baselines.gru_d_runtime import (
    evaluate_from_cfg,
    load_gru_d,
    read_cfg,
    resolve_io_paths,
)
from dvae.utils.path_manager import find_project_root
from dvae.utils.read_config import myconf


def _cfg_from_checkpoint(checkpoint, cfg_path):
    if cfg_path:
        return read_cfg(cfg_path)
    text = checkpoint.get("cfg_text") if isinstance(checkpoint, dict) else None
    if not text:
        raise SystemExit("pass --cfg or a checkpoint that stores cfg_text")
    cfg = myconf()
    cfg.read_string(text)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GRU-D checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg", default=None, help="ini; default is the text stored in the checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    root = Path(find_project_root(__file__))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(checkpoint, args.cfg)
    device = "cuda" if torch.cuda.is_available() and cfg.getboolean(
        "Training", "use_cuda", fallback=False
    ) else "cpu"
    data_dir, _, _ = resolve_io_paths(cfg, root / "config" / "device_paths.yaml")
    model = load_gru_d(checkpoint, cfg, device)
    summary = evaluate_from_cfg(
        cfg,
        model,
        data_dir=data_dir,
        device=device,
        output_dir=args.output_dir,
        dt=args.dt,
    )
    print(
        "[GRU-D eval] mse_tf={mse_tf} imputation_mse={imputation_mse} "
        "mse_auto={mse_auto} kld_auto={kld_auto} "
        "spectrum_error_auto={spectrum_error_auto}".format(**summary)
    )


if __name__ == "__main__":
    main()
