"""Wall-clock spans for the training loop.

``time.perf_counter`` around the measured block. On CUDA the device is
synchronised first so the span includes the kernels still in flight.
CPU devices are not synchronised.
"""

from __future__ import annotations

import time

import torch


def sync_cuda(device) -> None:
    """Wait for ``device`` when it is a CUDA device and CUDA is available."""
    if device is None:
        return
    try:
        dev = torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return
    if dev.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(dev)


class WallClock:
    """Context manager. ``elapsed`` is seconds after the block."""

    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0
        self._t0 = 0.0

    def __enter__(self):
        sync_cuda(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_cuda(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False
