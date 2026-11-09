"""The spectrum-comparison forward must not move the training RNG."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.visualizers.visualizers import visualize_teacherforcing_2_autonomous


class _Stochastic(torch.nn.Module):
    """Draws torch, NumPy, and Python random on every forward."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, mode_selector=None, noise_selector=None, inference_mode=False):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        _ = np.random.random()
        _ = random.random()
        return x + noise * 0 + self.bias


def _snapshot():
    numpy_state = np.random.get_state()
    return (
        torch.get_rng_state().clone(),
        (numpy_state[0], np.array(numpy_state[1], copy=True), *numpy_state[2:]),
        random.getstate(),
    )


def _same(left, right):
    assert torch.equal(left[0], right[0])
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert left[2] == right[2]


def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _plot(tmp, compare):
    batch = torch.randn(8, 2, 1)
    selector = torch.zeros(8, 2, 1)
    selector[4:, :, :] = 1
    visualize_teacherforcing_2_autonomous(
        batch,
        _Stochastic(),
        auto_mode_selector=selector,
        save_path=str(tmp / ("on" if compare else "off")),
        explain="epoch0",
        inference_mode=True,
        spectrum_compare=compare,
        sample_rate=32.0,
        epoch=0,
        spectrum_mode_name="half_half",
    )


def test_spectrum_compare_does_not_change_rng_state(tmp_path):
    _seed()
    before = _snapshot()
    _plot(tmp_path, False)
    after_off = _snapshot()
    # The waveform forward still draws. The flag must not add more.
    assert not torch.equal(before[0], after_off[0])

    _seed()
    _plot(tmp_path, True)
    after_on = _snapshot()
    _same(after_off, after_on)

    _seed()
    _plot(tmp_path, False)
    off_draw = torch.randn(4)
    off_numpy = np.random.random()
    off_python = random.random()
    _seed()
    _plot(tmp_path, True)
    on_draw = torch.randn(4)
    on_numpy = np.random.random()
    on_python = random.random()
    assert torch.equal(off_draw, on_draw)
    assert off_numpy == on_numpy
    assert off_python == on_python
