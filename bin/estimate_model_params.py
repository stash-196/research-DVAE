#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate model parameter counts from a config file without training."""

import argparse
import json
import os

import torch

from dvae.model import build_MT_RNN, build_MT_VRNN, build_RNN, build_VRNN
from dvae.utils import find_project_root, myconf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a model from cfg and print parameter counts."
    )
    parser.add_argument("--cfg", required=True, help="Path to config.ini")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device used only for model instantiation (default: cpu)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON",
    )
    parser.add_argument(
        "--module-breakdown",
        action="store_true",
        help="Also print per-top-module parameter counts",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Params] CUDA not available, falling back to cpu")
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[Params] MPS not available, falling back to cpu")
        return "cpu"
    return requested


def build_model_from_cfg(cfg, device):
    model_name = cfg.get("Network", "name")

    builders = {
        "VRNN": build_VRNN,
        "RNN": build_RNN,
        "MT_RNN": build_MT_RNN,
        "MT_VRNN": build_MT_VRNN,
    }

    if model_name not in builders:
        supported = ", ".join(sorted(builders.keys()))
        raise ValueError(
            f"Unsupported model '{model_name}' for this estimator. Supported: {supported}"
        )

    return model_name, builders[model_name](cfg=cfg, device=device)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def module_breakdown(model):
    breakdown = []
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        breakdown.append(
            {
                "module": name,
                "total_params": int(total),
                "trainable_params": int(trainable),
            }
        )
    return breakdown


def main():
    args = parse_args()

    root_dir = find_project_root(__file__)
    cfg_path = args.cfg
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(root_dir, cfg_path)

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = myconf()
    cfg.read(cfg_path)

    device = resolve_device(args.device)
    model_name, model = build_model_from_cfg(cfg, device)

    total, trainable = count_params(model)

    result = {
        "config": cfg_path,
        "model_name": model_name,
        "device": device,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_m": total / 1000000.0,
        "trainable_params_m": trainable / 1000000.0,
    }

    if args.module_breakdown:
        result["module_breakdown"] = module_breakdown(model)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Config: {result['config']}")
    print(f"Model: {result['model_name']} ({result['device']})")
    print(f"Total params: {result['total_params']:,}")
    print(f"Trainable params: {result['trainable_params']:,}")
    print(f"Total params (M): {result['total_params_m']:.6f}")
    print(f"Trainable params (M): {result['trainable_params_m']:.6f}")

    if args.module_breakdown:
        print("\nPer-module breakdown:")
        for row in result["module_breakdown"]:
            print(
                f"- {row['module']}: {row['total_params']:,} "
                f"(trainable {row['trainable_params']:,})"
            )


if __name__ == "__main__":
    main()
