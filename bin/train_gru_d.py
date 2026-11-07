#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train GRU-D from an ini file.

This does not call ``LearningAlgorithm`` or ``bin/train_model.py``.
On Deigo, ``scripts/slurm/train/run_gru_d_multiple.sh`` submits one job per
ini produced by ``bin/generate_gru_d_configs.py``.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dvae.baselines.gru_d_runtime import (
    evaluate_from_cfg,
    read_cfg,
    train_gru_d,
)
from dvae.utils.path_manager import find_project_root


def main():
    parser = argparse.ArgumentParser(description="Train a GRU-D baseline")
    parser.add_argument("--cfg", required=True, help="path to a GRU-D ini")
    parser.add_argument("--job_id", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="train only; do not score the test split",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="sample period used for spectrum and KL (Lorenz default 0.01)",
    )
    args = parser.parse_args()
    root = Path(find_project_root(__file__))
    cfg_path = Path(args.cfg)
    if not cfg_path.is_file():
        cfg_path = root / args.cfg
    cfg = read_cfg(cfg_path)
    if cfg.get("Network", "name") != "GRU_D":
        raise SystemExit(
            f"{cfg_path} has Network.name={cfg.get('Network', 'name')!r}; "
            "this entry point only trains GRU_D."
        )
    job_id = args.job_id or str(uuid.uuid4())[:8]
    print(f"[GRU-D] config={cfg_path} job_id={job_id}")
    result = train_gru_d(
        cfg,
        cfg_path=str(cfg_path),
        device_paths_file=str(root / "config" / "device_paths.yaml"),
        job_id=job_id,
        save_dir=args.save_dir,
    )
    print(f"[GRU-D] saved {result['save_dir']} best_epoch={result['best_epoch']}")
    if args.skip_eval:
        return
    try:
        summary = evaluate_from_cfg(
            cfg,
            result["model"],
            data_dir=result["data_dir"],
            device=result["device"],
            output_dir=result["save_dir"],
            dt=args.dt,
        )
    except FileNotFoundError as exc:
        print(f"[GRU-D] skipping eval, test data not found: {exc}")
        return
    print(
        "[GRU-D] mse_tf={mse_tf} mse_auto={mse_auto} "
        "kld_auto={kld_auto} spectrum_error_auto={spectrum_error_auto} "
        "imputation_mse={imputation_mse}".format(**summary)
    )


if __name__ == "__main__":
    main()
