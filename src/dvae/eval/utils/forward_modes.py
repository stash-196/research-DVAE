"""Evaluation forward passes using create_autonomous_mode_selector schedules."""

from typing import List, Optional, Tuple

import numpy as np
import torch

from dvae.utils import create_autonomous_mode_selector
from dvae.utils.model_mode_selector import create_autonomous_mode_selector_1d


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
    block_len: Optional[int] = None,
) -> torch.Tensor:
    mode_selector = create_autonomous_mode_selector(
        seq_len,
        mode=mode,
        autonomous_ratio=autonomous_ratio,
        batch_size=batch_size,
        x_dim=x_dim,
        device=device,
        flip_point=flip_point,
        block_len=block_len,
    )
    return apply_indicate_mask_tf_forcing(mode_selector, observation_process)


def run_forward_with_mode(
    dvae,
    batch_data: torch.Tensor,
    mode: str,
    observation_process: Optional[str] = None,
    autonomous_ratio: float = 0.0,
    flip_point: Optional[int] = None,
    block_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one forward pass with a schedule from create_autonomous_mode_selector.

    Args:
        dvae: Trained model in eval mode.
        batch_data: (seq_len, batch_size, x_dim) on any device.
        mode: Schedule name (e.g. all_0, half_half, alternating_blocks).
        observation_process: Dataset observation process for indicate-mask handling.
        autonomous_ratio: Used by ratio-based modes (e.g. flip_at_middle, even_bursts).
        flip_point: Used by flip_at_index.
        block_len: Used by alternating_blocks (paper-style fixed TF/Auto blocks).

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
        block_len=block_len,
    )

    with torch.no_grad():
        recon = dvae(
            batch_data.to(device),
            mode_selector=mode_selector,
            inference_mode=True,
        )

    return recon, mode_selector


def get_flip_point_for_mode(
    seq_len: int,
    mode: str,
    flip_point: Optional[int] = None,
    block_len: Optional[int] = None,
) -> int:
    """
    Return a representative timestep where autonomous mode begins.

    For single-transition modes this is the free-run start. For alternating
    multi-block modes this is the start of the *first* Auto block (used only
    as a summary; scoring should use the full mode mask).
    """
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
    if mode == "alternating_blocks":
        bl = int(block_len) if block_len is not None else 1000
        return min(bl, seq_len)
    if mode == "even_bursts":
        # Legacy even_bursts uses ratio * seq_len as flip interval; without ratio
        # treat first half-boundary as a coarse summary only.
        return seq_len // 2
    raise ValueError(
        f"Cannot infer flip_point for mode={mode}. "
        "Use half_half, flip_at_index, alternating_blocks, or even_bursts."
    )


def mode_selector_to_1d(mode_selector: torch.Tensor) -> np.ndarray:
    """Reduce (seq, batch, x_dim) or lower-rank selector to shape (seq_len,)."""
    if not torch.is_tensor(mode_selector):
        arr = np.asarray(mode_selector, dtype=np.float32)
    else:
        arr = mode_selector.detach().cpu().numpy().astype(np.float32)

    if arr.ndim == 3:
        # Signal channel 0, batch 0 (mask channel may be forced TF)
        arr = arr[:, 0, 0]
    elif arr.ndim == 2:
        arr = arr[:, 0]
    return arr.reshape(-1)


def get_auto_mask_1d(
    seq_len: int,
    mode: str,
    autonomous_ratio: float = 0.0,
    flip_point: Optional[int] = None,
    block_len: Optional[int] = None,
) -> np.ndarray:
    """Boolean mask of autonomous timesteps for the given schedule (True = auto)."""
    selector = create_autonomous_mode_selector_1d(
        seq_len,
        mode=mode,
        autonomous_ratio=autonomous_ratio,
        flip_point=flip_point,
        block_len=block_len,
    )
    return selector > 0.5


def count_auto_blocks(auto_mask: np.ndarray) -> int:
    """Count contiguous True runs in a 1D auto mask."""
    if auto_mask.size == 0:
        return 0
    m = np.asarray(auto_mask, dtype=bool).reshape(-1)
    # Rising edges
    padded = np.concatenate([[False], m])
    return int(np.sum((~padded[:-1]) & padded[1:]))


def list_auto_block_ranges(auto_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open [start, end) ranges for each contiguous auto block."""
    m = np.asarray(auto_mask, dtype=bool).reshape(-1)
    ranges: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(m):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(m)))
    return ranges
