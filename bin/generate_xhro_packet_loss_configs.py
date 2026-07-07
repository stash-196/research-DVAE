#!/usr/bin/env python3
"""Generate training configs for xhro_packet_loss (OTF / PTF experiments)."""

from __future__ import annotations

import itertools
import json
import os
import sys

# Reuse generator helpers from the multi-dataset script
sys.path.insert(0, os.path.dirname(__file__))
import generate_config_file_for_multiple as gen_multi  # noqa: E402

generate_config_file = gen_multi.generate_config_file
get_configurations_for_model = gen_multi.get_configurations_for_model

RECORDING_ID = "XHRO3506_20260622T142410000+0900"
BASE_TEMPLATE = "config/general_signal/cfg_base_template.ini"


def build_experiment_params(
    *,
    experiment_name: str,
    mode: str = "pilot",
) -> dict:
    """Return model_params dict keyed by model name (RNN / MT_RNN / ...)."""
    if mode == "pilot":
        epochs = [300]
        early_stop_patience = [80]
        sampling_ratio = [0.5]
        mask_label = ["realtime"]
        observation_process = ["raw_all"]
    elif mode == "ptf_sweep":
        epochs = [20_000]
        early_stop_patience = [500]
        sampling_ratio = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        mask_label = ["realtime", "recovered"]
        observation_process = ["raw_all"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    shared = {
        "experiment_name": [experiment_name],
        "name": ["MT_RNN"],
        "tag": ["MT_RNN"],
        "type_rnn": ["RNN"],
        "x_dim": [4],
        "dense_x": [4],
        "z_dim": [9],
        "dense_z": [[16, 32]],
        "dim_rnn": [200],
        "activation": ["relu"],
        "dropout_p": [0.0],
        "alphas": [[0.1] * 9],
        "lr": [0.001],
        "alpha_lr": [0.01],
        "epochs": epochs,
        "early_stop_patience": early_stop_patience,
        "save_frequency": [50],
        "gradient_clip": [10.0],
        "optimize_alphas": [True],
        "sampling_method": ["ptf"],
        "sampling_ratio": sampling_ratio,
        "mask_autonomous_filled": [False],
        "auto_warm_start": [0.0],
        "loss_mask_mode": ["none"],
        "noise_init_ratio": [0.0],
        "noise_mix_ratio": [0.0],
        "noise_std_factor": [1.0],
        "noise_target": ["none"],
        "noise_sampling_method": ["none"],
        "tie_noise_to_auto": [False],
        "noise_warm_reset_on_window": [False],
        "noise_window_size": [100],
        "dataset_name": ["XhroPacketLoss"],
        "dataset_label": [RECORDING_ID],
        "mask_label": mask_label,
        "s_dim": [1],
        "shuffle": [True],
        "batch_size": [128],
        "num_workers": [4],
        "sequence_len": [1000],
        "val_indices": [0.2],
        "observation_process": observation_process,
    }

    model_params = {
        "RNN": {k: v for k, v in shared.items() if k not in ("alphas", "alpha_lr", "optimize_alphas")},
        "MT_RNN": shared,
    }
    for key, value in model_params["RNN"].items():
        if key not in model_params["MT_RNN"]:
            model_params["MT_RNN"][key] = value
    return model_params


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate xhro_packet_loss training configs")
    parser.add_argument(
        "--mode",
        choices=["pilot", "ptf_sweep"],
        default="pilot",
        help="pilot = single quick run; ptf_sweep = full PTF × variant grid",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    experiment_name = args.experiment_name or (
        "20260702-XHRO_packet_loss_pilot"
        if args.mode == "pilot"
        else "20260702-XHRO_packet_loss_ptf_sweep"
    )
    output_dir = os.path.join("config/xhro_packet_loss/generated", experiment_name)
    model_params = build_experiment_params(experiment_name=experiment_name, mode=args.mode)

    keys_being_compared = [
        key for key, value in model_params["MT_RNN"].items() if len(value) > 1
    ]
    configs = get_configurations_for_model(model_params["MT_RNN"])
    print(f"Generating {len(configs)} configs -> {output_dir}")
    gen_multi.keys_being_compared = keys_being_compared
    for config in configs:
        generate_config_file(
            BASE_TEMPLATE,
            output_dir,
            experiment_name,
            keys_being_compared,
            config,
        )

    json_params = [
        {key: config.get(key) for key in keys_being_compared} for config in configs
    ]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "params_being_compared.json"), "w") as f:
        json.dump(json_params, f, indent=2)

    print("Done. Example run:")
    print(
        f"  python bin/train_model.py --cfg {output_dir}/cfg_<name>.ini"
    )


if __name__ == "__main__":
    main()