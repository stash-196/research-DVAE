#!/usr/bin/env python3
"""Fit and evaluate filtering imputation baselines.

Does not call ``train_model.py`` or ``eval_signal.py``. Generated configs are
read as JSON. Example (Deigo, inside the project container, data bind ``/data``)::

    python3 bin/fit_eval_filtering_baselines.py \\
        --dataset Lorenz63 \\
        --data-dir /data \\
        --dataset-label sigma10_rho28_beta8d3_N108k_dt0.01 \\
        --mask-label Bernoulli_0.3 \\
        --observation-process only_x \\
        --method kalman \\
        --state-dim 3 \\
        --out-dir /saved_model/filtering_baselines/kalman_only_x

``--dataset demo`` runs a short masked Lorenz trajectory with no data files.
``--dataset PhysioNet2012`` is documented in the README; it needs the processed
parquet from the PhysioNet loader and does not edit that loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="kalman",
        choices=("kalman", "tresp", "oracle_ekf", "oracle_ukf", "all"),
    )
    parser.add_argument(
        "--dataset",
        default="demo",
        choices=("demo", "Lorenz63", "PhysioNet2012", "npz"),
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--mask-label", default=None)
    parser.add_argument("--observation-process", default="only_x")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--out-dir", default="outputs/filtering_baselines/demo")
    parser.add_argument("--config", default=None, help="JSON file; CLI overrides it.")
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--em-iters", type=int, default=25)
    parser.add_argument("--metrics-seq-len", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--flip-point", type=int, default=None)
    parser.add_argument("--time-delay", type=int, default=None)
    parser.add_argument("--delay-dims", type=int, default=None)
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--tresp-train",
        default="observed_targets",
        choices=("observed_targets", "multiple_imputation"),
    )
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--demo-steps", type=int, default=300)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = {}
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    cli = {
        "method": args.method,
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "dataset_label": args.dataset_label,
        "mask_label": args.mask_label,
        "observation_process": args.observation_process,
        "npz": args.npz,
        "state_dim": args.state_dim,
        "em_iters": args.em_iters,
        "metrics_seq_len": args.metrics_seq_len,
        "max_windows": args.max_windows,
        "flip_point": args.flip_point,
        "time_delay": args.time_delay,
        "delay_dims": args.delay_dims,
        "compute_geometry": not args.no_geometry,
        "seed": args.seed,
        "delay": args.delay,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "tresp_train": args.tresp_train,
        "sigma": args.sigma,
        "rho": args.rho,
        "beta": args.beta,
        "dt": args.dt,
        "demo_steps": args.demo_steps,
    }
    # Config file wins only for keys the CLI left at parser defaults when a
    # config is present. Explicit CLI values always override.
    if args.config:
        for key, value in cli.items():
            if key not in cfg or _cli_explicit(argv, key):
                cfg[key] = value
    else:
        cfg.update(cli)
    cfg = {key: value for key, value in cfg.items() if value is not None}

    from dvae.baselines.filtering.runner import load_bundle_from_cfg, run_method

    bundle = load_bundle_from_cfg(cfg)
    methods = (
        ("kalman", "tresp", "oracle_ekf", "oracle_ukf")
        if cfg.get("method") == "all"
        else (cfg.get("method", "kalman"),)
    )
    out_root = Path(args.out_dir)
    for method in methods:
        summary = run_method(method, bundle, cfg, out_root / method)
        mse_tf = summary.get("mse_tf")
        mse_auto = summary.get("mse_auto")
        role = summary.get("role")
        print(
            f"[filtering] {method} role={role} "
            f"mse_tf={mse_tf} mse_auto={mse_auto} "
            f"-> {out_root / method / 'metrics.json'}"
        )
    return 0


def _cli_explicit(argv, key: str) -> bool:
    if argv is None:
        argv = sys.argv[1:]
    flag = "--" + key.replace("_", "-")
    return flag in argv


if __name__ == "__main__":
    raise SystemExit(main())
