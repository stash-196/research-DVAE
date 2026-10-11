#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write PhysioNet 2012 OTF / interpolate / indicate training configs.

This script does not import or edit ``bin/generate_config_file_for_multiple.py``.
Generated ``.ini`` files land under ``config/general_signal/generated/`` and
are gitignored. Submit them with the existing Deigo helper::

    bash scripts/slurm/train/run_training_multiple.sh physionet2012_otf_benchmark

``dataset_name`` stays ``PhysioNet2012``, which the training loop already
accepts. The new observation names are opt-in. ``sampling_ratio`` is 0 so
scheduled autonomy is off: OTF autonomy comes only from NaN entries.

Default grid is the 37-variable parquet (``hourly_benchmark_v1``).
``--subset vitals`` targets the existing 5-vital ``hourly_v1`` file.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


class _DefaultDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


def write_config(template: Path, output_dir: Path, experiment_name: str, test_keys: list, config: dict) -> Path:
    payload = dict(config)
    payload["test_keys"] = ", ".join(test_keys)
    for key, value in list(payload.items()):
        if isinstance(value, list):
            payload[key] = ", ".join(map(str, value))
    content = template.read_text()
    content = content.format_map(_DefaultDict(lambda: "", **payload))
    label_keys = [f"{key}-{payload[key]}" for key in test_keys]
    output_file_name = "_".join([experiment_name, payload["tag"], *label_keys]).replace(" ", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"cfg_{output_file_name}.ini"
    output_file.write_text(content)
    return output_file


def benchmark_grid(subset: str) -> tuple[str, str, list[dict]]:
    if subset == "vitals":
        dataset_label = "hourly_v1"
        specs = [
            ("benchmark_vitals_otf", 5),
            ("benchmark_vitals_interpolate", 5),
            ("benchmark_vitals_indicate", 10),
        ]
        experiment = "physionet2012_otf_benchmark_vitals"
    elif subset == "full":
        dataset_label = "hourly_benchmark_v1"
        specs = [
            ("benchmark_otf", 37),
            ("benchmark_interpolate", 37),
            ("benchmark_indicate", 74),
        ]
        experiment = "physionet2012_otf_benchmark"
    else:
        raise ValueError(subset)

    configs = []
    for observation, x_dim in specs:
        configs.append(
            {
                "experiment_name": experiment,
                "name": "RNN",
                "tag": "RNN",
                "x_dim": x_dim,
                "z_dim": 16,
                "activation": "relu",
                "dense_x": x_dim,
                "dense_z": "",
                "dim_rnn": 64,
                "hidden_sh_size": 64,
                "type_rnn": "RNN",
                "dropout_p": 0.0,
                "alphas": "",
                "lr": 0.001,
                "alpha_lr": 0.01,
                "epochs": 400,
                "early_stop_patience": 40,
                "save_frequency": 40,
                "gradient_clip": 10.0,
                "optimize_alphas": False,
                "sampling_method": "ptf",
                "sampling_ratio": 0.0,
                "mask_autonomous_filled": False,
                "auto_warm_start": 0.0,
                "loss_mask_mode": "none",
                "noise_init_ratio": 0.0,
                "noise_mix_ratio": 0.0,
                "noise_std_factor": 0.1,
                "noise_target": "both",
                "noise_sampling_method": "none",
                "tie_noise_to_auto": False,
                "noise_warm_reset_on_window": True,
                "noise_window_size": 100,
                "dataset_name": "PhysioNet2012",
                "dataset_label": dataset_label,
                "mask_label": "None",
                "corpus": "None",
                "data_root": "None",
                "s_dim": 1,
                "shuffle": True,
                "batch_size": 64,
                "num_workers": 0,
                "sequence_len": 48,
                "val_indices": 0.2,
                "observation_process": observation,
                "coarse_test_split": "usable_tail",
                "coarse_eeg_artifact_mask": "none",
                "coarse_eeg_artifact_k": "None",
                "coarse_eeg_max_rel_delta": "None",
            }
        )
    return experiment, dataset_label, configs


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("full", "vitals"),
        default="full",
        help="full: 37 variables (hourly_benchmark_v1). vitals: 5 vitals (hourly_v1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the generated-config directory (tests).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "config" / "general_signal" / "cfg_base_template.ini",
    )
    args = parser.parse_args(argv)

    experiment, dataset_label, configs = benchmark_grid(args.subset)
    output_dir = args.output_dir or (
        root / "config" / "general_signal" / "generated" / experiment
    )
    test_keys = ["observation_process", "x_dim"]
    written = []
    for config in configs:
        written.append(write_config(args.template, output_dir, experiment, test_keys, config))
    manifest = [
        {
            "model": "RNN",
            "dataset_name": "PhysioNet2012",
            "dataset_label": dataset_label,
            "observation_process": config["observation_process"],
            "x_dim": config["x_dim"],
            "sampling_method": config["sampling_method"],
            "sampling_ratio": config["sampling_ratio"],
        }
        for config in configs
    ]
    (output_dir / "params_being_compared.json").write_text(json.dumps(manifest, indent=2))
    print(f"Experiment name: {experiment}")
    print(f"Wrote {len(written)} configs to {output_dir}")
    for path in written:
        print(f"  {path.name}")
    print(
        "Submit with:\n"
        f"  bash scripts/slurm/train/run_training_multiple.sh {experiment}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
