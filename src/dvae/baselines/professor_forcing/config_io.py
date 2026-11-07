"""Read and write the Professor Forcing ini files."""

from __future__ import annotations

import configparser
from pathlib import Path


def read_ini(path: str | Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    cfg = {
        "experiment_name": parser.get("User", "experiment_name", fallback="professor_forcing"),
        "variant": parser.get("Model", "variant"),
        "behavior": parser.get("Model", "behavior", fallback="pre_tanh"),
        "generator_objective": parser.get("Model", "generator_objective", fallback="nll_cf"),
        "free_run_feedback": parser.get("Model", "free_run_feedback", fallback="mean"),
        "observation_std": parser.getfloat("Model", "observation_std", fallback=1.0),
        "hidden_size": parser.getint("Model", "hidden_size"),
        "disc_hidden_size": parser.getint("Model", "disc_hidden_size", fallback=64),
        "logit_clip": parser.getfloat("Model", "logit_clip", fallback=10.0),
        "hidden_match_weight": parser.getfloat("Model", "hidden_match_weight", fallback=1.0),
        "x_dim": parser.getint("Model", "x_dim"),
        "lr": parser.getfloat("Training", "lr"),
        "disc_lr": parser.getfloat("Training", "disc_lr", fallback=parser.getfloat("Training", "lr")),
        "epochs": parser.getint("Training", "epochs"),
        "early_stop_patience": parser.getint("Training", "early_stop_patience", fallback=10),
        "gradient_clip": parser.getfloat("Training", "gradient_clip", fallback=5.0),
        "acc_low": parser.getfloat("Training", "acc_low", fallback=0.75),
        "acc_high": parser.getfloat("Training", "acc_high", fallback=0.99),
        "eval_every": parser.getint("Training", "eval_every", fallback=5),
        "max_eval_batches": parser.getint("Training", "max_eval_batches", fallback=4),
        "max_train_batches": parser.getint("Training", "max_train_batches", fallback=0),
        "seed": parser.getint("Training", "seed", fallback=0),
        "use_cuda": parser.getboolean("Training", "use_cuda", fallback=True),
        "dataset_name": parser.get("DataFrame", "dataset_name", fallback="Lorenz63"),
        "dataset_label": parser.get("DataFrame", "dataset_label"),
        "mask_label": _none(parser.get("DataFrame", "mask_label", fallback="None")),
        "observation_process": parser.get("DataFrame", "observation_process"),
        "sequence_len": parser.getint("DataFrame", "sequence_len"),
        "batch_size": parser.getint("DataFrame", "batch_size"),
        "val_indices": parser.getfloat("DataFrame", "val_indices"),
        "shuffle": parser.getboolean("DataFrame", "shuffle", fallback=True),
        "num_workers": parser.getint("DataFrame", "num_workers", fallback=0),
    }
    return cfg


def write_ini(path: str | Path, cfg: dict) -> None:
    parser = configparser.ConfigParser()
    parser["User"] = {"experiment_name": str(cfg["experiment_name"])}
    parser["Model"] = {
        "variant": str(cfg["variant"]),
        "behavior": str(cfg["behavior"]),
        "generator_objective": str(cfg["generator_objective"]),
        "free_run_feedback": str(cfg["free_run_feedback"]),
        "observation_std": str(cfg["observation_std"]),
        "hidden_size": str(cfg["hidden_size"]),
        "disc_hidden_size": str(cfg["disc_hidden_size"]),
        "logit_clip": str(cfg["logit_clip"]),
        "hidden_match_weight": str(cfg["hidden_match_weight"]),
        "x_dim": str(cfg["x_dim"]),
    }
    parser["Training"] = {
        "lr": str(cfg["lr"]),
        "disc_lr": str(cfg["disc_lr"]),
        "epochs": str(cfg["epochs"]),
        "early_stop_patience": str(cfg["early_stop_patience"]),
        "gradient_clip": str(cfg["gradient_clip"]),
        "acc_low": str(cfg["acc_low"]),
        "acc_high": str(cfg["acc_high"]),
        "eval_every": str(cfg["eval_every"]),
        "max_eval_batches": str(cfg["max_eval_batches"]),
        "max_train_batches": str(cfg.get("max_train_batches", 0)),
        "seed": str(cfg["seed"]),
        "use_cuda": str(cfg["use_cuda"]),
    }
    parser["DataFrame"] = {
        "dataset_name": str(cfg["dataset_name"]),
        "dataset_label": str(cfg["dataset_label"]),
        "mask_label": "None" if cfg["mask_label"] in (None, "", "None") else str(cfg["mask_label"]),
        "observation_process": str(cfg["observation_process"]),
        "sequence_len": str(cfg["sequence_len"]),
        "batch_size": str(cfg["batch_size"]),
        "val_indices": str(cfg["val_indices"]),
        "shuffle": str(cfg["shuffle"]),
        "num_workers": str(cfg["num_workers"]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(
            "# Professor Forcing / hidden-match ablation. Not consumed by bin/train_model.py.\n"
        )
        parser.write(handle)


def _none(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "none"):
        return None
    return text
