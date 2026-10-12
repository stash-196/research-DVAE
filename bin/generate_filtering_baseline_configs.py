#!/usr/bin/env python3
"""Write JSON configs for filtering baselines.

This is separate from ``bin/generate_config_file_for_multiple.py`` (that file
is the OTF grid generator and is left untouched). Configs land under
``config/filtering_baselines/generated/<experiment>/``, which is gitignored by
``config/**/generated*``.

The printed command is what to run on Deigo after ``git pull``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default="lorenz_filtering_baselines",
    )
    parser.add_argument(
        "--out-root",
        default="config/filtering_baselines/generated",
    )
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument(
        "--dataset-label",
        default="sigma10_rho28_beta8d3_N108k_dt0.01",
    )
    parser.add_argument(
        "--observation-processes",
        nargs="+",
        default=["only_x", "only_xy", "all_xyz"],
    )
    parser.add_argument(
        "--mask-labels",
        nargs="+",
        default=["Bernoulli_0.3", "Bernoulli_0.5"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["kalman", "tresp", "oracle_ekf", "oracle_ukf"],
    )
    parser.add_argument("--state-dim", type=int, default=3)
    parser.add_argument("--em-iters", type=int, default=30)
    parser.add_argument("--metrics-seq-len", type=int, default=2000)
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=32)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    experiment_dir = Path(args.out_root) / args.experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for method in args.methods:
        for process in args.observation_processes:
            for mask in args.mask_labels:
                # The oracle needs the true field; it does not use the mask
                # label as a learned parameter, but it is scored on the same
                # masked series as the fair baselines.
                payload = {
                    "method": method,
                    "dataset": "Lorenz63",
                    "data_dir": args.data_dir,
                    "dataset_label": args.dataset_label,
                    "mask_label": mask,
                    "observation_process": process,
                    "state_dim": args.state_dim,
                    "em_iters": args.em_iters,
                    "metrics_seq_len": args.metrics_seq_len,
                    "max_windows": args.max_windows,
                    "epochs": args.epochs,
                    "hidden": args.hidden,
                    "delay": 1 if process != "only_x" else 3,
                    "tresp_train": "observed_targets",
                    "compute_geometry": True,
                }
                name = f"{method}__{process}__{mask}.json"
                (experiment_dir / name).write_text(json.dumps(payload, indent=2))
                count += 1
    print(f"[filtering] wrote {count} configs in {experiment_dir}")
    print(
        "On Deigo, after git pull:\n"
        "  ml singularity\n"
        "  singularity exec /bucket/DoyaU/stash/containers/generic_ml_container.sif \\\n"
        f"    python3 bin/generate_filtering_baseline_configs.py --experiment {args.experiment}\n"
        f"  bash scripts/slurm/baselines/run_filtering_baselines.sh {args.experiment}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
