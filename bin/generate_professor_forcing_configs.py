#!/usr/bin/env python3
"""Write Lorenz63 Professor Forcing ini files.

Does not edit bin/generate_config_file_for_multiple.py. Outputs land under
config/lorenz63/generated_professor_forcing/, which the repo gitignores
(config/**/generated*).

The active lists below are the comparison grid. Commented entries are the
wider Lorenz observation and mask settings used by the OTF sweeps.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from dvae.baselines.professor_forcing.config_io import write_ini
from dvae.baselines.professor_forcing.data import resolved_x_dim
from dvae.utils.path_manager import find_project_root

EXPERIMENT_NAME = "professor_forcing_lorenz63"

# Active comparison grid. Uncomment neighbors to widen it.
OBSERVATION_PROCESSES = [
    "only_x",
    # "only_xy",
    # "only_xz",
    "all_xyz",
    "only_x_indicate",
]
MASK_LABELS = [
    "None",
    # "Markov_AvgLen15_0.1",
    # "Markov_AvgLen15_0.3",
    "Markov_AvgLen15_0.5",
    # "Markov_AvgLen15_0.7",
]
# Paper section 4.1 behavior sequences. hidden / hidden_and_output /
# hidden_and_input are accepted by the trainer but are not in this grid.
BEHAVIORS = [
    "pre_tanh",
    "pre_tanh_and_output",
]
OBJECTIVES = [
    "nll_cf",
    "nll_cf_ct",
]

SHARED = {
    "experiment_name": EXPERIMENT_NAME,
    "free_run_feedback": "mean",
    "observation_std": 1.0,
    "hidden_size": 64,
    "disc_hidden_size": 64,
    "logit_clip": 10.0,
    "hidden_match_weight": 1.0,
    "lr": 0.0001,
    "disc_lr": 0.0001,
    "epochs": 20000,
    "early_stop_patience": 20,
    "gradient_clip": 5.0,
    "acc_low": 0.75,
    "acc_high": 0.99,
    "eval_every": 10,
    "max_eval_batches": 4,
    "max_train_batches": 0,
    "seed": 0,
    "use_cuda": True,
    "dataset_name": "Lorenz63",
    "dataset_label": "sigma10_rho28_beta8d3_N108k_dt0.01",
    "sequence_len": 1000,
    "batch_size": 128,
    "val_indices": 0.2,
    "shuffle": True,
    "num_workers": 0,
}


def iter_configs():
    for observation, mask in product(OBSERVATION_PROCESSES, MASK_LABELS):
        base = dict(SHARED)
        base["observation_process"] = observation
        base["mask_label"] = None if mask in ("None", "none", "") else mask
        base["x_dim"] = resolved_x_dim(observation, 1)
        ablation = dict(base)
        ablation["variant"] = "hidden_match"
        ablation["behavior"] = "pre_tanh"
        ablation["generator_objective"] = "nll_cf"
        yield ablation
        for behavior, objective in product(BEHAVIORS, OBJECTIVES):
            row = dict(base)
            row["variant"] = "professor_forcing"
            row["behavior"] = behavior
            row["generator_objective"] = objective
            yield row


def config_filename(cfg: dict) -> str:
    mask = cfg["mask_label"] or "nomask"
    return (
        f"cfg_{cfg['variant']}_{cfg['observation_process']}_{mask}"
        f"_{cfg['behavior']}_{cfg['generator_objective']}.ini"
    )


def generate(output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg in iter_configs():
        path = output_dir / config_filename(cfg)
        write_ini(path, cfg)
        written.append(path)
    return written


def main():
    root = find_project_root(__file__)
    default_out = root / "config" / "lorenz63" / "generated_professor_forcing" / EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(default_out))
    args = parser.parse_args()
    written = generate(args.output_dir)
    print(f"[professor-forcing] wrote {len(written)} configs under {args.output_dir}")
    print("Submit with:")
    print(f"  bash scripts/slurm/baselines/run_professor_forcing.sh {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
