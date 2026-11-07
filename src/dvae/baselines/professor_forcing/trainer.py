"""Training loop for Professor Forcing and the hidden-match ablation.

This is a separate entry point. It does not call ``LearningAlgorithm``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dvae.baselines.professor_forcing.data import batch_to_time_major
from dvae.baselines.professor_forcing.discriminator import BehaviorDiscriminator
from dvae.baselines.professor_forcing.generator import BEHAVIORS, ProfessorForcingGenerator
from dvae.baselines.professor_forcing.losses import indicate_process
from dvae.baselines.professor_forcing.metrics import rollout_pair, score_batch
from dvae.baselines.professor_forcing.plotting import plot_metric_history
from dvae.baselines.professor_forcing.step import apply_step, hidden_match_loss, professor_forcing_loss


def build_modules(cfg: dict, device: torch.device):
    indicate = indicate_process(cfg["observation_process"])
    generator = ProfessorForcingGenerator(
        x_dim=int(cfg["x_dim"]),
        hidden_size=int(cfg["hidden_size"]),
        indicate=indicate,
    ).to(device)
    discriminator = None
    if cfg["variant"] == "professor_forcing":
        discriminator = BehaviorDiscriminator(
            input_size=generator.feature_dim(cfg["behavior"]),
            hidden_size=int(cfg["disc_hidden_size"]),
            logit_clip=float(cfg["logit_clip"]),
        ).to(device)
    elif cfg["variant"] != "hidden_match":
        raise ValueError(
            "variant must be 'professor_forcing' or 'hidden_match' (ablation), "
            f"got {cfg['variant']!r}."
        )
    return generator, discriminator


def make_optimizers(generator, discriminator, cfg):
    opt_g = torch.optim.Adam(generator.parameters(), lr=float(cfg["lr"]))
    opt_d = None
    if discriminator is not None:
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["disc_lr"]))
    return opt_g, opt_d


def train(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str | Path,
    device: torch.device | None = None,
) -> dict:
    cfg = _with_defaults(cfg)
    if device is None:
        device = _select_device(cfg)
    _seed(int(cfg["seed"]))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    generator, discriminator = build_modules(cfg, device)
    opt_g, opt_d = make_optimizers(generator, discriminator, cfg)

    history = []
    best_val = float("inf")
    bad_epochs = 0
    best_path = save_dir / "best.pt"
    epoch = -1

    for epoch in range(int(cfg["epochs"])):
        train_log = _run_epoch(
            generator, discriminator, opt_g, opt_d, train_loader, cfg, device, train_mode=True
        )
        record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_log.items()}}
        record["nll"] = train_log.get("nll")
        record["disc_accuracy"] = train_log.get("disc_accuracy")
        record["hidden_match"] = train_log.get("hidden_match")

        do_eval = (epoch % int(cfg["eval_every"]) == 0) or (epoch == int(cfg["epochs"]) - 1)
        if do_eval:
            val_log = _run_epoch(
                generator,
                discriminator,
                opt_g,
                opt_d,
                val_loader,
                cfg,
                device,
                train_mode=False,
            )
            metrics = _evaluate_metrics(generator, val_loader, cfg, device)
            record.update(val_log)
            record.update(metrics)
            record["nll"] = val_log.get("nll", record["nll"])
            val_nll = val_log.get("nll", float("inf"))
            if np.isfinite(val_nll) and val_nll < best_val:
                best_val = float(val_nll)
                bad_epochs = 0
                _save_checkpoint(best_path, generator, discriminator, epoch, cfg)
            else:
                bad_epochs += 1
        history.append(record)
        _write_history(save_dir / "metrics.json", history)
        print(
            f"[professor-forcing] epoch {epoch} nll={record.get('nll')} "
            f"mse_auto={record.get('mse_auto')} acc={record.get('disc_accuracy')}"
        )
        if bad_epochs >= int(cfg["early_stop_patience"]):
            print(f"[professor-forcing] early stop at epoch {epoch}")
            break

    plot_metric_history(history, save_dir / "training_metrics.png")
    _save_checkpoint(save_dir / "last.pt", generator, discriminator, epoch, cfg)
    return {"history": history, "save_dir": str(save_dir), "best_val_nll": best_val}


def _run_epoch(generator, discriminator, opt_g, opt_d, loader, cfg, device, train_mode: bool) -> dict:
    if train_mode:
        generator.train()
        if discriminator is not None:
            discriminator.train()
    else:
        generator.eval()
        if discriminator is not None:
            discriminator.eval()

    totals = {}
    counts = {}
    batches = 0
    for batch in loader:
        x = batch_to_time_major(batch).to(device)
        if train_mode:
            result = _loss(generator, discriminator, x, cfg)
            apply_step(result, generator, discriminator, opt_g, opt_d, float(cfg["gradient_clip"]))
        else:
            with torch.no_grad():
                result = _loss(generator, discriminator, x, cfg)
        for key, value in result.as_log().items():
            if isinstance(value, (int, float)) and value is not None and np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        batches += 1
        if train_mode and int(cfg.get("max_train_batches") or 0) and batches >= int(cfg["max_train_batches"]):
            break
        if (not train_mode) and batches >= int(cfg["max_eval_batches"]):
            break
    return {key: totals[key] / counts[key] for key in totals}


def _loss(generator, discriminator, x, cfg):
    if cfg["variant"] == "hidden_match":
        return hidden_match_loss(
            generator,
            x,
            observation_process=cfg["observation_process"],
            feedback=cfg["free_run_feedback"],
            observation_std=float(cfg["observation_std"]),
            hidden_match_weight=float(cfg["hidden_match_weight"]),
        )
    return professor_forcing_loss(
        generator,
        discriminator,
        x,
        behavior=cfg["behavior"],
        generator_objective=cfg["generator_objective"],
        observation_process=cfg["observation_process"],
        feedback=cfg["free_run_feedback"],
        observation_std=float(cfg["observation_std"]),
        acc_low=float(cfg["acc_low"]),
        acc_high=float(cfg["acc_high"]),
    )


@torch.no_grad()
def _evaluate_metrics(generator, loader, cfg, device) -> dict:
    generator.eval()
    rows = []
    for index, batch in enumerate(loader):
        if index >= int(cfg["max_eval_batches"]):
            break
        x = batch_to_time_major(batch).to(device)
        pred_tf, pred_half = rollout_pair(generator, x)
        rows.append(
            score_batch(
                x.detach().cpu().numpy(),
                pred_tf.detach().cpu().numpy(),
                pred_half.detach().cpu().numpy(),
                cfg["observation_process"],
                cfg.get("dataset_name", "Lorenz63"),
            )
        )
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "per_channel"]
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows if np.isfinite(row[key])]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def _save_checkpoint(path: Path, generator, discriminator, epoch: int, cfg: dict) -> None:
    payload = {
        "epoch": epoch,
        "variant": cfg["variant"],
        "generator": generator.state_dict(),
        "discriminator": None if discriminator is None else discriminator.state_dict(),
        "cfg": cfg,
    }
    torch.save(payload, path)


def _write_history(path: Path, history: list) -> None:
    path.write_text(json.dumps(_jsonable(history), indent=2))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_device(cfg: dict) -> torch.device:
    use_cuda = bool(cfg.get("use_cuda", True))
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _with_defaults(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update({key: value for key, value in cfg.items() if value is not None})
    if merged["behavior"] not in BEHAVIORS:
        raise ValueError(f"behavior must be one of {BEHAVIORS}, got {merged['behavior']!r}.")
    if merged["variant"] not in ("professor_forcing", "hidden_match"):
        raise ValueError(f"Unknown variant {merged['variant']!r}.")
    if merged["generator_objective"] not in ("nll_cf", "nll_cf_ct"):
        raise ValueError(f"Unknown generator_objective {merged['generator_objective']!r}.")
    if merged["free_run_feedback"] not in ("mean", "gaussian"):
        raise ValueError(f"Unknown free_run_feedback {merged['free_run_feedback']!r}.")
    return merged


DEFAULTS = {
    "variant": "professor_forcing",
    "behavior": "pre_tanh",
    "generator_objective": "nll_cf",
    "free_run_feedback": "mean",
    "observation_std": 1.0,
    "hidden_size": 64,
    "disc_hidden_size": 64,
    "logit_clip": 10.0,
    "hidden_match_weight": 1.0,
    "x_dim": 1,
    "lr": 1e-4,
    "disc_lr": 1e-4,
    "epochs": 10,
    "early_stop_patience": 10,
    "gradient_clip": 5.0,
    "acc_low": 0.75,
    "acc_high": 0.99,
    "eval_every": 1,
    "max_eval_batches": 2,
    "max_train_batches": 0,
    "seed": 0,
    "use_cuda": True,
    "observation_process": "only_x",
    "dataset_name": "Lorenz63",
}
