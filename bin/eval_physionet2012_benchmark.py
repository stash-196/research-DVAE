#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Imputation MAE/MSE on the PhysioNet 2012 held-out mask, plus optional AUROC.

Loads the same ``PhysioNet2012`` benchmark split the training config names.
With ``--saved_dict``, runs the checkpoint (NaN overlay + indicate mask-channel
teacher forcing, matching ``learning_algo``) and scores signal channels on
held-out entries. Mean-fill and per-stay linear interpolation are always
scored on that same mask. ``--mortality`` fits a logistic probe on the last
hidden state when ``in_hospital_death`` is in the parquet.

The checkpoint may be a raw ``state_dict`` (``*_final_epoch*.pt``) or a
training checkpoint dict with ``model_state_dict``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import configparser
import numpy as np
import torch

from dvae.dataset.physionet2012_benchmark import (
    mortality_auroc,
    plot_imputation_comparison,
    predict_dataset,
    score_predictions,
)
from dvae.dataset.physionet2012_dataset import PhysioNet2012
from dvae.model import build_MT_RNN, build_MT_VRNN, build_RNN, build_VRNN
from dvae.utils import find_project_root, load_device_paths

_BUILDERS = {
    "RNN": build_RNN,
    "VRNN": build_VRNN,
    "MT_RNN": build_MT_RNN,
    "MT_VRNN": build_MT_VRNN,
}


def _cfg(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(path)
    return parser


def _dataset_from_cfg(cfg, data_dir: str, split: str, device: str):
    return PhysioNet2012(
        data_dir=data_dir,
        dataset_label=cfg.get("DataFrame", "dataset_label", fallback="None"),
        mask_label=cfg.get("DataFrame", "mask_label", fallback="None"),
        split=split,
        seq_len=cfg.getint("DataFrame", "sequence_len"),
        x_dim=cfg.getint("Network", "x_dim"),
        sample_rate=cfg.getint("DataFrame", "sample_rate", fallback=1),
        skip_rate=cfg.getint("DataFrame", "skip_rate", fallback=1),
        val_indices=cfg.getfloat("DataFrame", "val_indices"),
        observation_process=cfg.get("DataFrame", "observation_process"),
        device=device,
        overlap=cfg.getboolean("DataFrame", "overlap", fallback=False),
        shuffle=False,
    )


def _load_state(path: Path, device: str):
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    root = find_project_root(__file__)
    default_data = None
    try:
        device_paths = load_device_paths(str(Path(root) / "config" / "device_paths.yaml"))
        default_data = device_paths.get("data_dir") if isinstance(device_paths, dict) else None
    except (FileNotFoundError, ValueError):
        device_paths = None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--saved_dict", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path(default_data) if default_data else None)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mortality",
        dest="mortality",
        action="store_true",
        default=True,
        help="Fit the optional in-hospital mortality probe (default: on).",
    )
    parser.add_argument(
        "--no-mortality",
        dest="mortality",
        action="store_false",
        help="Skip the mortality probe.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.data_dir is None:
        print("Error: --data-dir is required.", file=sys.stderr)
        return 1
    cfg = _cfg(args.cfg)
    observation = cfg.get("DataFrame", "observation_process")
    if not str(observation).startswith("benchmark_"):
        print(
            f"Error: {observation!r} is not a benchmark observation. "
            "Point --cfg at a physionet2012_otf_benchmark ini.",
            file=sys.stderr,
        )
        return 1

    device = args.device
    dataset = _dataset_from_cfg(cfg, str(args.data_dir), args.split, device)
    sampling_ratio = cfg.getfloat("Training", "sampling_ratio", fallback=0.0)
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = (
            args.saved_dict.parent
            if args.saved_dict is not None
            else Path("outputs") / "physionet2012_benchmark"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "observation_process": observation,
        "split": args.split,
        "sampling_ratio": sampling_ratio,
        "holdout_fraction": dataset.benchmark_holdout_fraction,
        "holdout_seed": dataset.benchmark_holdout_seed,
        "n_windows": len(dataset),
        "columns": list(dataset.benchmark_columns),
    }

    if args.saved_dict is None:
        from dvae.dataset.physionet2012_benchmark import (
            reference_linear_windows,
            reference_mean_windows,
            imputation_errors,
            stack_windows,
            window_slices,
        )

        slices = window_slices(dataset)
        target = stack_windows(dataset.benchmark_target, slices)
        holdout = stack_windows(dataset.benchmark_holdout, slices)
        report["model"] = None
        report["reference_linear"] = imputation_errors(
            reference_linear_windows(dataset), target, holdout
        )
        report["reference_mean"] = imputation_errors(
            reference_mean_windows(dataset), target, holdout
        )
        report["auroc"] = None
    else:
        name = cfg.get("Network", "name")
        if name not in _BUILDERS:
            print(f"Error: unsupported model {name!r}", file=sys.stderr)
            return 1
        model = _BUILDERS[name](cfg, device=device)
        model.load_state_dict(_load_state(args.saved_dict, device))
        model.to(device)
        predictions, hidden = predict_dataset(
            model,
            dataset,
            sampling_ratio=sampling_ratio,
            batch_size=args.batch_size,
            device=device,
        )
        report.update(score_predictions(dataset, predictions))
        if args.mortality and args.split == "test":
            train_ds = _dataset_from_cfg(cfg, str(args.data_dir), "train", device)
            _, train_hidden = predict_dataset(
                model,
                train_ds,
                sampling_ratio=sampling_ratio,
                batch_size=args.batch_size,
                device=device,
            )
            report["mortality"] = mortality_auroc(train_ds, train_hidden, dataset, hidden)
        elif args.mortality:
            report["mortality"] = {
                "auroc": None,
                "reason": "mortality probe is scored on the test split",
            }
        else:
            report["mortality"] = None

    out_json = out_dir / f"imputation_{args.split}_{observation}.json"
    out_json.write_text(json.dumps(_jsonable(report), indent=2))
    print(f"Wrote {out_json}")

    rows = []
    if report.get("model"):
        rows.append(
            {
                "label": observation,
                "mae": report["model"]["mae"],
                "mse": report["model"]["mse"],
            }
        )
    if report.get("reference_linear"):
        rows.append(
            {
                "label": "linear",
                "mae": report["reference_linear"]["mae"],
                "mse": report["reference_linear"]["mse"],
            }
        )
    if report.get("reference_mean"):
        rows.append(
            {
                "label": "mean",
                "mae": report["reference_mean"]["mae"],
                "mse": report["reference_mean"]["mse"],
            }
        )
    finite_rows = [
        row
        for row in rows
        if row["mae"] is not None
        and row["mse"] is not None
        and np.isfinite(row["mae"])
        and np.isfinite(row["mse"])
    ]
    if finite_rows:
        fig_path = out_dir / f"imputation_{args.split}_{observation}.png"
        plot_imputation_comparison(finite_rows, fig_path)
        print(f"Wrote {fig_path}")
    model_metrics = report.get("model") or {}
    if model_metrics:
        print(
            f"MAE={model_metrics.get('mae')} MSE={model_metrics.get('mse')} "
            f"n={model_metrics.get('n_holdout')}"
        )
    if report.get("mortality"):
        print(f"mortality={report['mortality']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
