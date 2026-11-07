#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate GRU-D without entering ``LearningAlgorithm``.

Dataloaders come from the existing ``build_dataloader``. Checkpoints and
``evaluation_summary.yaml`` are written under a GRU-D directory so they do
not collide with the OTF runs.
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import torch
import yaml

from dvae.baselines.plotting import plot_loss_curves, plot_metric_bars, plot_trajectories
from dvae.baselines.rollout_metrics import score_trajectory_pair
from dvae.dataset.dataset_builder import DatasetConfig, build_dataloader
from dvae.model.gru_d import build_GRU_D, masked_mse
from dvae.utils.model_mode_selector import create_autonomous_mode_selector
from dvae.utils.read_config import myconf


def read_cfg(path):
    cfg = myconf()
    read = cfg.read(path)
    if not read:
        raise FileNotFoundError(path)
    return cfg


def resolve_io_paths(cfg, device_paths_file):
    """Prefer ``device_paths.yaml`` when this host is listed.

    Deigo's hostname contains ``deigo``, so a pulled checkout uses the same
    flash/bucket roots as ``train_model.py``. Any other host keeps the ini
    ``[User]`` paths. A missing match is not an error: this entry point is
    also used off-cluster.
    """
    data_dir = cfg.get("User", "data_dir", fallback="./data")
    saved_root = cfg.get("User", "saved_root", fallback="./saved_model")
    device_name = "ini"
    path = Path(device_paths_file)
    if path.is_file():
        try:
            from dvae.utils.config_handler import load_device_paths

            paths = load_device_paths(str(path))
        except (ValueError, FileNotFoundError, OSError):
            paths = None
        if paths:
            data_dir = paths["data_dir"]
            saved_root = paths["saved_root"]
            device_name = paths["device_name"]
    return data_dir, saved_root, device_name


def _optional_str(cfg, section, option):
    if not cfg.has_option(section, option):
        return None
    raw = cfg.get(section, option).strip()
    if raw == "" or raw.lower() == "none":
        return None
    return raw


def dataset_config_from_cfg(cfg, data_dir, device, seq_len):
    return DatasetConfig(
        data_dir=data_dir,
        x_dim=cfg.getint("Network", "x_dim"),
        batch_size=cfg.getint("DataFrame", "batch_size"),
        shuffle=cfg.getboolean("DataFrame", "shuffle"),
        num_workers=cfg.getint("DataFrame", "num_workers"),
        sample_rate=cfg.getint("DataFrame", "sample_rate"),
        skip_rate=cfg.getint("DataFrame", "skip_rate"),
        val_indices=cfg.getfloat("DataFrame", "val_indices"),
        observation_process=cfg.get("DataFrame", "observation_process"),
        overlap=cfg.getboolean("DataFrame", "overlap"),
        with_nan=cfg.getboolean("DataFrame", "with_nan", fallback=False),
        seq_len=int(seq_len),
        device=device,
        dataset_label=_optional_str(cfg, "DataFrame", "dataset_label"),
        mask_label=_optional_str(cfg, "DataFrame", "mask_label"),
        data_root=_optional_str(cfg, "DataFrame", "data_root"),
        corpus=_optional_str(cfg, "DataFrame", "corpus"),
    )


def select_device(cfg):
    use_cuda = cfg.getboolean("Training", "use_cuda", fallback=False)
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def batch_to_model(batch, device):
    """DataLoader batches are ``(batch, time, dim)``; the model wants time first."""
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(np.asarray(batch), dtype=torch.float32)
    batch = batch.to(device).float()
    if batch.dim() == 2:
        batch = batch.transpose(0, 1).unsqueeze(-1)
    elif batch.dim() == 3:
        batch = batch.permute(1, 0, 2)
    else:
        raise ValueError(f"unexpected batch rank {batch.dim()}")
    return batch


def empirical_mean_from_loader(loader, x_dim, device, max_batches=None):
    total = np.zeros(x_dim, dtype=np.float64)
    count = np.zeros(x_dim, dtype=np.float64)
    for i, batch in enumerate(loader):
        series = batch_to_model(batch, "cpu").numpy()
        flat = series.reshape(-1, series.shape[-1])
        if flat.shape[1] != x_dim:
            raise ValueError(
                f"batch width {flat.shape[1]} != model x_dim {x_dim}. "
                "Set Network.x_dim to the dataset width "
                "(1 for only_x, 3 for all_xyz, 5 for PhysioNet raw_vitals)."
            )
        finite = np.isfinite(flat)
        total += np.where(finite, flat, 0.0).sum(axis=0)
        count += finite.sum(axis=0)
        if max_batches is not None and i + 1 >= max_batches:
            break
    mean = np.zeros(x_dim, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return torch.tensor(mean, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    seen = 0
    for batch in loader:
        series = batch_to_model(batch, device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            pred = model(series)
            loss = masked_mse(pred, series)
            if train and loss.requires_grad:
                loss.backward()
                clip = getattr(model, "_gradient_clip", 0.0)
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
        batch_size = series.shape[1]
        total += float(loss.detach()) * batch_size
        seen += batch_size
    if seen == 0:
        return float("nan")
    return total / seen


def _build_optimizer(cfg, model):
    lr = cfg.getfloat("Training", "lr")
    name = cfg.get("Training", "optimization", fallback="adam").lower()
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    return torch.optim.Adam(params, lr=lr, weight_decay=1e-5)


def align_model_width(cfg, train_loader):
    """Match ``Network.x_dim`` to the width the dataset actually yields."""
    resolved = int(getattr(train_loader.dataset, "x_dim", cfg.getint("Network", "x_dim")))
    if resolved != cfg.getint("Network", "x_dim"):
        if not cfg.has_section("Network"):
            cfg.add_section("Network")
        cfg.set("Network", "x_dim", str(resolved))
    return resolved


def train_gru_d(cfg, cfg_path, device_paths_file, job_id, save_dir=None):
    device = select_device(cfg)
    data_dir, saved_root, device_name = resolve_io_paths(cfg, device_paths_file)
    seq_len = cfg.getint("DataFrame", "sequence_len")
    dataset_name = cfg.get("DataFrame", "dataset_name")
    dataset_config = dataset_config_from_cfg(cfg, data_dir, device, seq_len)
    train_loader, val_loader, train_num, val_num = build_dataloader(
        dataset_name, dataset_config, "train"
    )
    align_model_width(cfg, train_loader)
    model = build_GRU_D(cfg, device=device)
    model._gradient_clip = cfg.getfloat("Training", "gradient_clip", fallback=0.0)
    mean = empirical_mean_from_loader(train_loader, model.x_dim, device)
    model.set_empirical_mean(mean)

    if save_dir is None:
        experiment = cfg.get("User", "experiment_name", fallback="gru_d")
        stamp = cfg.get("User", "run_stamp", fallback="")
        folder = f"{experiment}_{job_id}" if not stamp else f"{stamp}_{experiment}_{job_id}"
        save_dir = Path(saved_root) / "gru_d" / folder
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    aligned_cfg = save_dir / "cfg.ini"
    _write_cfg(cfg, aligned_cfg)
    cfg_path = str(aligned_cfg)
    optimizer = _build_optimizer(cfg, model)
    epochs = cfg.getint("Training", "epochs")
    patience = cfg.getint("Training", "early_stop_patience", fallback=epochs)
    save_every = cfg.getint("Training", "save_frequency", fallback=epochs)
    seed = cfg.getint("Training", "seed", fallback=0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_hist = []
    val_hist = []
    best = float("inf")
    best_epoch = -1
    stale = 0
    log_lines = [
        f"host={socket.gethostname()} device_paths={device_name} train_device={device}",
        f"samples train={train_num} val={val_num} x_dim={model.x_dim}",
        f"x_mean={mean.detach().cpu().tolist()}",
    ]
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        log_lines.append(f"epoch={epoch} train={train_loss:.6g} val={val_loss:.6g}")
        print(log_lines[-1])
        improved = np.isfinite(val_loss) and val_loss < best
        if improved:
            best = val_loss
            best_epoch = epoch
            stale = 0
            _save_checkpoint(save_dir / "GRU_D_best.pt", model, cfg_path, epoch, best)
        elif np.isfinite(val_loss):
            stale += 1
        if save_every and (epoch + 1) % save_every == 0:
            _save_checkpoint(
                save_dir / f"GRU_D_epoch{epoch}.pt", model, cfg_path, epoch, val_loss
            )
        if patience and stale >= patience:
            log_lines.append(f"early_stop epoch={epoch} best_epoch={best_epoch}")
            print(log_lines[-1])
            break
    _save_checkpoint(save_dir / "GRU_D_last.pt", model, cfg_path, best_epoch, best)
    plot_loss_curves(train_hist, val_hist, str(save_dir / "loss_curves.png"))
    (save_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
    return {
        "save_dir": str(save_dir),
        "best_epoch": best_epoch,
        "best_val": best,
        "model": model,
        "device": device,
        "data_dir": data_dir,
    }


def _write_cfg(cfg, path):
    with open(path, "w") as handle:
        cfg.write(handle)


def _save_checkpoint(path, model, cfg_path, epoch, val_loss):
    torch.save(
        {
            "model_name": "GRU_D",
            "model_state": model.state_dict(),
            "epoch": int(epoch),
            "val_loss": None if val_loss is None else float(val_loss),
            "cfg_text": Path(cfg_path).read_text() if cfg_path else "",
            "x_dim": int(model.x_dim),
            "dim_rnn": int(model.dim_rnn),
        },
        path,
    )


def load_gru_d(checkpoint, cfg, device):
    model = build_GRU_D(cfg, device=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def _channel_names(dataset, x_dim):
    process = getattr(dataset, "observation_process", "") or ""
    if x_dim == 3 and process in ("all_xyz", "xyz_to_xyz", "xyz_to_xyz_w_noise"):
        return ["x", "y", "z"]
    if x_dim == 1:
        return ["x"]
    return [f"ch{i}" for i in range(x_dim)]


def _lorenz_ground_truth(dataset):
    """Unmasked series sliced like the observation, or None if it cannot be aligned."""
    full = getattr(dataset, "full_sequence", None)
    if full is None:
        return None
    full = np.asarray(full)
    if full.ndim != 2 or full.shape[1] < 1:
        return None
    process = getattr(dataset, "observation_process", "")
    try:
        from dvae.dataset.lorenz63_dataset import LORENZ63_SLICE_CHANNELS
    except Exception:
        return None
    channels = LORENZ63_SLICE_CHANNELS.get(process)
    if channels:
        index = {"x": 0, "y": 1, "z": 2}
        cols = [index[name] for name in channels]
        return full[:, cols]
    if process in ("xyz_to_xyz", "xyz_to_xyz_w_noise"):
        return full[:, :3]
    return None


def _windows_from_dataset(dataset):
    seq = dataset.seq
    if torch.is_tensor(seq):
        seq = seq.detach().cpu().numpy()
    seq = np.asarray(seq)
    if seq.ndim == 1:
        seq = seq[:, None]
    gt_series = _lorenz_ground_truth(dataset)
    windows = []
    for start in list(dataset.data_idx):
        stop = min(start + dataset.seq_len, len(seq))
        observed = seq[start:stop]
        if len(observed) < dataset.seq_len:
            continue
        gt = None
        if gt_series is not None and stop <= len(gt_series):
            gt = gt_series[start:stop]
            if gt.shape != observed.shape:
                gt = None
        windows.append((observed, gt))
    return windows


def evaluate_gru_d(model, dataset, device, output_dir, dataset_name, dt, max_windows=4):
    """Teacher-forced reconstruction, imputation where GT exists, and a half-series free run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    windows = _windows_from_dataset(dataset)
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    names = _channel_names(dataset, model.x_dim)
    tf_rows = []
    auto_rows = []
    imp_values = []
    example = None
    for observed, gt in windows:
        series = torch.tensor(observed, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            pred_tf = model(series).squeeze(1).detach().cpu().numpy()
        # Reconstruction scores ignore missing inputs. The complete Lorenz
        # trajectory, when we have it, is reserved for imputation and for
        # the free-run comparison.
        tf_rows.append(
            score_trajectory_pair(
                observed,
                pred_tf,
                dataset_name=dataset_name,
                dt=dt,
                prefix="tf",
                channel_names=names,
            )
        )
        if gt is not None:
            missing = ~np.isfinite(observed)
            if missing.any() and np.isfinite(gt[missing]).any():
                imp_values.append(
                    float(np.mean((pred_tf[missing] - gt[missing]) ** 2))
                )
        seq_len = series.shape[0]
        selector = create_autonomous_mode_selector(
            seq_len,
            mode="half_half",
            batch_size=1,
            x_dim=model.x_dim,
            device=device,
        )
        with torch.no_grad():
            pred_auto_full = model(series, mode_selector=selector).squeeze(1).detach().cpu().numpy()
        flip = seq_len // 2
        target_auto = (gt if gt is not None else observed)[flip:]
        plot_gt = gt if gt is not None else observed
        auto_rows.append(
            score_trajectory_pair(
                target_auto,
                pred_auto_full[flip:],
                dataset_name=dataset_name,
                dt=dt,
                prefix="auto",
                channel_names=names,
            )
        )
        if example is None:
            example = {
                "gt": plot_gt,
                "pred_tf": pred_tf,
                "pred_auto": pred_auto_full,
                "flip": flip,
                "observed": observed,
            }
    summary = {
        "model_name": "GRU_D",
        "dataset_name": dataset_name,
        "observation_process": getattr(dataset, "observation_process", None),
        "mask_label": getattr(dataset, "mask_label", None),
        "n_windows": len(windows),
        "dt": float(dt),
        "mse_tf": _mean_metric(tf_rows, "mse_tf"),
        "mse_tf_mean": _mean_metric(tf_rows, "mse_tf_mean"),
        "kld_tf": _mean_metric(tf_rows, "kld_tf"),
        "kld_tf_mean": _mean_metric(tf_rows, "kld_tf_mean"),
        "spectrum_error_tf": _mean_metric(tf_rows, "spectrum_error_tf"),
        "spectrum_error_tf_mean": _mean_metric(tf_rows, "spectrum_error_tf_mean"),
        "mse_auto": _mean_metric(auto_rows, "mse_auto"),
        "mse_auto_mean": _mean_metric(auto_rows, "mse_auto_mean"),
        "mse_auto_short": _mean_metric(auto_rows, "mse_auto_short"),
        "kld_auto": _mean_metric(auto_rows, "kld_auto"),
        "kld_auto_mean": _mean_metric(auto_rows, "kld_auto_mean"),
        "spectrum_error_auto": _mean_metric(auto_rows, "spectrum_error_auto"),
        "spectrum_error_auto_mean": _mean_metric(auto_rows, "spectrum_error_auto_mean"),
        "imputation_mse": float(np.mean(imp_values)) if imp_values else None,
        "notes": (
            "GRU-D is trained teacher-forced on observed samples only "
            "(Che et al.). mse_tf is the one-step error on those samples. "
            "imputation_mse is the error at NaN inputs when an unmasked "
            "Lorenz trajectory is available; PhysioNet has no such labels, "
            "so imputation_mse stays empty there. mse_auto / kld_auto / "
            "spectrum_error_auto are a half_half closed-loop rollout, the "
            "same schedule and metric names as the OTF eval. GRU-D is not "
            "trained with scheduled autonomous steps."
        ),
    }
    _dump_yaml(output_dir / "evaluation_summary.yaml", summary)
    plot_metric_bars(
        summary,
        ["mse_tf", "mse_auto", "mse_auto_short", "kld_auto", "spectrum_error_auto"],
        str(output_dir / "metrics.png"),
    )
    if example is not None:
        plot_trajectories(
            example["gt"],
            example["pred_tf"],
            str(output_dir / "teacher_forced_window0.png"),
            channel_names=names,
            pred_label="GRU-D TF",
        )
        plot_trajectories(
            example["gt"],
            example["pred_auto"],
            str(output_dir / "rollout_window0.png"),
            flip_index=example["flip"],
            channel_names=names,
            pred_label="GRU-D half-series",
        )
    return summary


def _mean_metric(rows, key):
    values = []
    for row in rows:
        value = row.get(key, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _dump_yaml(path, payload):
    def _convert(value):
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, (np.floating, float)):
            value = float(value)
            if not np.isfinite(value):
                return None
            return value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, np.ndarray):
            return _convert(value.tolist())
        return value

    path.write_text(yaml.safe_dump(_convert(payload), sort_keys=False))


def evaluate_from_cfg(cfg, model, data_dir, device, output_dir, dt):
    dataset_name = cfg.get("DataFrame", "dataset_name")
    seq_len = cfg.getint("DataFrame", "sequence_len")
    dataset_config = dataset_config_from_cfg(cfg, data_dir, device, seq_len)
    # Test split, no shuffle effect on indices beyond the dataset's own rule.
    dataset_config.shuffle = False
    test_loader = build_dataloader(dataset_name, dataset_config, "test")
    return evaluate_gru_d(
        model,
        test_loader.dataset,
        device,
        output_dir,
        dataset_name=dataset_name,
        dt=dt,
    )
