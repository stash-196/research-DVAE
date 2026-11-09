"""Short-sequence evaluation plots keep the mask on the same axes."""

from __future__ import annotations

import torch

from dvae.eval.eval_signal import aligned_short_sequence_viz_inputs
from dvae.learning_algo import overlay_nan_autonomous


def test_short_sequence_inputs_are_not_sliced():
    batch = torch.zeros(6, 3, 2)
    selector = torch.zeros(6, 3, 2)
    mask = torch.zeros(6, 3, 1)
    mask[0, 0, 0] = 1.0
    out_batch, out_sel, out_mask = aligned_short_sequence_viz_inputs(
        batch, selector, mask
    )
    assert out_batch.shape == batch.shape
    assert out_sel.shape == selector.shape
    assert out_mask.shape == mask.shape
    assert out_mask[0, 0, 0] == 1.0
    assert aligned_short_sequence_viz_inputs(batch, selector, None)[2] is None


def test_nan_overlay_uses_the_selector_it_is_given():
    batch = torch.zeros(4, 1, 1)
    batch[1, 0, 0] = float("nan")
    plotted = torch.zeros(4, 1, 1)
    plotted[2, 0, 0] = 1.0
    other = torch.ones(4, 1, 1)
    overlaid = overlay_nan_autonomous(batch, plotted)
    assert float(overlaid[1, 0, 0]) == 1.0
    assert float(overlaid[2, 0, 0]) == 1.0
    assert float(overlaid[0, 0, 0]) == 0.0
    # The other selector is not what gets the NaN fill.
    assert float(other[0, 0, 0]) == 1.0
    assert float(plotted[1, 0, 0]) == 0.0
