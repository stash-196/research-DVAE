"""Evaluation forward passes using create_autonomous_mode_selector schedules."""

from typing import Optional, Tuple

import torch

from dvae.utils import create_autonomous_mode_selector


def is_indicate_observation(observation_process: Optional[str]) -> bool:
    if observation_process is None:
        return False
    if observation_process == "only_x_indicate":
        return True
    return isinstance(observation_process, str) and observation_process.endswith(
        "_indicate"
    )


def apply_indicate_mask_tf_forcing(
    mode_selector: torch.Tensor, observation_process: Optional[str]
) -> torch.Tensor:
    """Force pure teacher-forcing on the mask/indicator channel during evaluation."""
    if is_indicate_observation(observation_process) and mode_selector.size(2) >= 2:
        mode_selector = mode_selector.clone()
        mode_selector[:, :, 1] = 0.0
    return mode_selector


def build_mode_selector(
    seq_len: int,
    batch_size: int,
    x_dim: int,
    device: str,
    mode: str,
    observation_process: Optional[str] = None,
    autonomous_ratio: float = 0.0,
    flip_point: Optional[int] = None,
) -> torch.Tensor:
    mode_selector = create_autonomous_mode_selector(
        seq_len,
        mode=mode,
        autonomous_ratio=autonomous_ratio,
        batch_size=batch_size,
        x_dim=x_dim,
        device=device,
        flip_point=flip_point,
    )
    return apply_indicate_mask_tf_forcing(mode_selector, observation_process)


def run_forward_with_mode(
    dvae,
    batch_data: torch.Tensor,
    mode: str,
    observation_process: Optional[str] = None,
    autonomous_ratio: float = 0.0,
    flip_point: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one forward pass with a schedule from create_autonomous_mode_selector.

    Args:
        dvae: Trained model in eval mode.
        batch_data: (seq_len, batch_size, x_dim) on any device.
        mode: Schedule name (e.g. all_0, half_half, flip_at_index).
        observation_process: Dataset observation process for indicate-mask handling.
        autonomous_ratio: Used by ratio-based modes (e.g. flip_at_middle).
        flip_point: Used by flip_at_index.

    Returns:
        recon: (seq_len, batch_size, x_dim) tensor on dvae.device.
        mode_selector: (seq_len, batch_size, x_dim) tensor used for the pass.
    """
    seq_len, batch_size, x_dim = batch_data.shape
    device = dvae.device

    mode_selector = build_mode_selector(
        seq_len=seq_len,
        batch_size=batch_size,
        x_dim=x_dim,
        device=device,
        mode=mode,
        observation_process=observation_process,
        autonomous_ratio=autonomous_ratio,
        flip_point=flip_point,
    )

    with torch.no_grad():
        recon = dvae(
            batch_data.to(device),
            mode_selector=mode_selector,
            inference_mode=True,
        )

    return recon, mode_selector


def get_flip_point_for_mode(
    seq_len: int, mode: str, flip_point: Optional[int] = None
) -> int:
    """Return the timestep where autonomous mode begins for contiguous scoring."""
    if mode == "half_half":
        return seq_len // 2
    if mode == "flip_at_index":
        if flip_point is None:
            raise ValueError("flip_at_index requires flip_point.")
        return int(flip_point)
    if mode == "all_0":
        return seq_len
    if mode == "all_1":
        return 0
    raise ValueError(
        f"Cannot infer flip_point for mode={mode}. Use half_half or flip_at_index."
    )