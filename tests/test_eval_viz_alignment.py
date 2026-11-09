"""Short-sequence evaluation plots keep the mask on the same axes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dvae.eval.eval_signal import (
    aligned_short_sequence_viz_inputs,
    load_eval_weights,
    plot_eval_short_sequence,
)
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


def test_plot_eval_short_sequence_passes_unsliced_inputs(monkeypatch):
    captured = {}

    def fake(batch, dvae, auto_mode_selector, save_path, **kwargs):
        captured["batch"] = batch
        captured["selector"] = auto_mode_selector
        captured["mask"] = kwargs.get("missing_mask")
        captured["explain"] = kwargs.get("explain")

    monkeypatch.setattr(
        "dvae.eval.eval_signal.visualize_teacherforcing_2_autonomous",
        fake,
    )
    batch = torch.zeros(6, 3, 2)
    batch[1, 2, 0] = float("nan")
    selector = torch.zeros(6, 3, 2)
    selector[4, 0, 1] = 1.0
    mask = torch.zeros(6, 3, 1)
    mask[1, 2, 0] = 1.0
    plot_eval_short_sequence(
        batch,
        object(),
        selector,
        mask,
        "unused",
        explain="final_short_inference_mode_half_half_short",
        is_segmented_1d=False,
        hide_mask_output=False,
    )
    assert captured["batch"].shape == batch.shape
    assert captured["selector"].shape == selector.shape
    assert captured["mask"].shape == mask.shape
    assert torch.isnan(captured["batch"][1, 2, 0])
    assert float(captured["selector"][4, 0, 1]) == 1.0
    assert float(captured["mask"][1, 2, 0]) == 1.0
    assert captured["explain"] == "final_short_inference_mode_half_half_short"


def test_short_inference_call_site_passes_the_unsliced_batch():
    source = Path("src/dvae/eval/eval_signal.py").read_text(encoding="utf-8")
    needle = (
        "plot_eval_short_sequence(\n"
        "                    batch_data_long,\n"
        "                    dvae,\n"
        "                    autonomous_mode_selector_long,\n"
        "                    missing_mask_long,"
    )
    assert needle in source
    assert "batch_data_long[:, 1:" not in source


def _load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key])


def test_load_eval_weights_uses_content_not_the_filename(tmp_path):
    model = torch.nn.Linear(2, 2)
    raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
    final_path = tmp_path / "RNN_final_epoch0.pt"
    checkpoint_path = tmp_path / "RNN_checkpoint.pt"
    latest_path = tmp_path / "RNN_latest.pt"
    torch.save(raw, final_path)
    torch.save({"epoch": 0, "model_state_dict": raw}, checkpoint_path)
    torch.save({"epoch": 1, "model_state_dict": raw, "cur_best_epoch": 0}, latest_path)

    for path in (final_path, checkpoint_path, latest_path):
        fresh = torch.nn.Linear(2, 2)
        with torch.no_grad():
            fresh.weight.fill_(0)
            fresh.bias.fill_(0)
        load_eval_weights(fresh, path)
        _state_equal(fresh.state_dict(), raw)

    unread = tmp_path / "not_a_checkpoint.pt"
    unread.write_text("not torch\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_eval_weights(torch.nn.Linear(2, 2), unread)

    tensor_path = tmp_path / "tensor.pt"
    torch.save(torch.zeros(2), tensor_path)
    with pytest.raises(SystemExit):
        load_eval_weights(torch.nn.Linear(2, 2), tensor_path)

    mismatch = tmp_path / "mismatch.pt"
    torch.save({"weight": torch.zeros(3, 3)}, mismatch)
    with pytest.raises(SystemExit):
        load_eval_weights(torch.nn.Linear(2, 2), mismatch)
    assert _load_pt(final_path).keys() == raw.keys()
