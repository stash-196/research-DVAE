"""Lorenz63 loaders for the Professor Forcing entry point.

The dataset class is the repo's ``Lorenz63``. This module does not register
anything in ``dataset_builder``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dvae.dataset.lorenz63_dataset import Lorenz63, lorenz63_resolved_x_dim
from dvae.baselines.professor_forcing.losses import indicate_process


def resolved_x_dim(observation_process: str, x_dim: int) -> int:
    if indicate_process(observation_process):
        if observation_process == "only_x_indicate":
            return 2
        base = observation_process[: -len("_indicate")]
        return 2 * int(lorenz63_resolved_x_dim(base, x_dim))
    return int(lorenz63_resolved_x_dim(observation_process, x_dim))


def load_lorenz_loaders(
    data_dir: str,
    *,
    dataset_label: str,
    mask_label: str | None,
    observation_process: str,
    x_dim: int,
    sequence_len: int,
    batch_size: int,
    val_indices: float,
    shuffle: bool,
    num_workers: int,
    device: str,
):
    x_dim = resolved_x_dim(observation_process, x_dim)
    common = dict(
        data_dir=data_dir,
        dataset_label=dataset_label,
        mask_label=mask_label,
        seq_len=sequence_len,
        x_dim=x_dim,
        sample_rate=1,
        skip_rate=1,
        val_indices=val_indices,
        observation_process=observation_process,
        device=device,
        overlap=False,
        with_nan=True,
        shuffle=shuffle,
        eval_mode=False,
    )
    train_ds = Lorenz63(split="train", **common)
    val_ds = Lorenz63(split="valid", **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_ds.x_dim


def batch_to_time_major(batch) -> torch.Tensor:
    """DataLoader stacks items as ``(B, T, D)``. The generator wants ``(T, B, D)``."""
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    batch = batch.float()
    if batch.dim() == 2:
        batch = batch.unsqueeze(-1)
    if batch.dim() != 3:
        raise ValueError(f"Expected a batch of sequences, got shape {tuple(batch.shape)}.")
    return batch.permute(1, 0, 2).contiguous()
