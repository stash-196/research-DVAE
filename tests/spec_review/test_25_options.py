"""Pins for options PR #25. Synthetic tensors and series only.

New switches are off unless the config turns them on, and a blank config
does not change a reconstruction step. The split, subband loss,
decorrelation penalty, blank warm-up patience floor, and coarse HR gate
are checked against the spec, including where the stored weights are not
exactly zero.
"""

from __future__ import annotations

import configparser
import os
import pickle
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from dvae.dataset.xhro_proper_dataset import COARSE_OBS_COLUMNS, coarse_feature_columns
from dvae.model.base_rnn import BaseRNN
from dvae.model.hidden_block import parse_hidden_block_split
from dvae.model.mt_rnn import MT_RNN
from dvae.regularizers import (
    decorr_hidden_loss,
    decorr_output_loss,
    effective_weight,
    parse_regularizer_settings,
    regularizer_losses,
    subband_power_loss,
)
from dvae.warmup_curriculum import (
    patience_start_from_fraction,
    resolve_warmup_patience_start,
    warmup_settings_from_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _cfg(type_rnn="PLRNN", dim="4", x_dim="4", **network):
    parser = configparser.ConfigParser()
    base = {
        "x_dim": x_dim,
        "dim_rnn": dim,
        "hidden_sh_size": "6",
        "type_rnn": type_rnn,
        "dense_x": "",
        "dense_h_x": "",
        "alphas": "0.2,0.8",
        "hidden_block_split": "",
    }
    base.update({key: str(value) for key, value in network.items()})
    parser["Network"] = base
    parser["Training"] = {"noise_std_factor": "0"}
    return parser


def _split_on(**network):
    network.setdefault("hidden_block_split", "half")
    network.setdefault("ecg_channels", "0,1")
    network.setdefault("eeg_channels", "2,3")
    return network


def _model(type_rnn="PLRNN", dim=4, x_dim=4, **network):
    cfg = _cfg(type_rnn, str(dim), str(x_dim), **network)
    return BaseRNN(
        x_dim=x_dim,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=dim,
        num_rnn=1,
        type_rnn=type_rnn,
        dropout_p=0,
        device="cpu",
        cfg=cfg,
    )


def _off_block(weight, heart, gates=1):
    """Off-block slices of a recurrent map. LSTM repeats the block per gate."""
    hidden = weight.shape[1]
    pieces = []
    for gate in range(gates):
        rows = weight[gate * hidden : (gate + 1) * hidden]
        pieces.append(rows[:heart, heart:])
        pieces.append(rows[heart:, :heart])
    return pieces


def test_blank_options_leave_the_step_unchanged():
    """Off is the default, and the added terms do not move the weights."""
    from collections import defaultdict

    class _Blank(defaultdict):
        def __missing__(self, key):
            return self.default_factory()

    template = (ROOT / "config/general_signal/cfg_base_template.ini").read_text()
    rendered = template.format_map(_Blank(lambda: ""))
    cfg = configparser.ConfigParser()
    cfg.read_string(rendered)
    assert parse_hidden_block_split(cfg, 4, 4) is None
    settings = parse_regularizer_settings(cfg, 4, None)
    assert settings.subband_weight == 0.0
    assert settings.decorr_weight == 0.0
    assert settings.subband_on is False
    assert settings.decorr_on is False

    torch.manual_seed(0)
    left = _model()
    right = _model()
    right.load_state_dict(left.state_dict())
    assert left.hidden_block is None
    assert getattr(left, "decoder_mask", None) is None
    batch = torch.randn(5, 2, 4)
    torch.manual_seed(1)
    left(batch)
    torch.manual_seed(1)
    right(batch)
    assert torch.equal(left.y, right.y)
    sub, dec = regularizer_losses(left, batch, settings)
    total = (left.y - batch).pow(2).mean()
    total = total + effective_weight(settings.subband_weight, False, 0, 4, 6) * sub
    total = total + effective_weight(settings.decorr_weight, False, 0, 4, 6) * dec
    recon = (right.y - batch).pow(2).mean()
    opt_a = torch.optim.Adam(left.parameters(), lr=0.05)
    opt_b = torch.optim.Adam(right.parameters(), lr=0.05)
    opt_a.zero_grad()
    total.backward()
    opt_a.step()
    opt_b.zero_grad()
    recon.backward()
    opt_b.step()
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.allclose(a, b), a.shape


def test_half_and_explicit_sizes_and_block_alphas():
    even = parse_hidden_block_split(_cfg(**_split_on(hidden_block_split="half")), 4, 4)
    assert (even.heart, even.brain) == (2, 2)
    odd = parse_hidden_block_split(
        _cfg(dim="5", **_split_on(hidden_block_split="true")), 5, 4
    )
    assert (odd.heart, odd.brain) == (2, 3)
    sized = parse_hidden_block_split(
        _cfg(**_split_on(hidden_block_split="1,3")), 4, 4
    )
    assert (sized.heart, sized.brain) == (1, 3)

    shared = MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=4,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=4,
        cfg=_cfg(**_split_on()),
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )
    assert torch.allclose(
        shared.alphas_per_unit().detach(), torch.tensor([0.2, 0.8, 0.2, 0.8])
    )
    ranged = MT_RNN(
        alphas=[0.2, 0.8],
        x_dim=4,
        activation="relu",
        dense_x=[],
        dense_h_x=[],
        dim_rnn=6,
        cfg=_cfg(
            dim="6",
            **_split_on(hidden_block_split="2,4", heart_alpha_range="0.2,0.4", brain_alphas="0.7"),
        ),
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        device="cpu",
    )
    got = ranged.alphas_per_unit().detach()
    assert torch.allclose(got[:2], torch.linspace(0.2, 0.4, 2))
    assert torch.allclose(got[2:], torch.full((4,), 0.7))


def test_mask_is_applied_every_step_and_the_input_map_stays_dense():
    model = _model(**_split_on())
    heart = model.hidden_block.heart
    model.rnn.A_sigmas.data.fill_(-30)
    model.rnn.W.data.fill_(1.0)
    model.rnn.linear_C.weight.data.zero_()
    model.rnn.linear_C.bias.data.zero_()
    hx = torch.zeros(1, 1, 4)
    hx[0, 0, :heart] = 1.0
    outputs, _ = model.rnn(torch.zeros(4, 1, 4), hx)
    assert outputs[:, :, heart:].abs().max().item() == 0.0
    assert outputs[:, :, :heart].abs().max().item() > 0.0
    effective = model.rnn.W * model.rnn.W_mask * model.rnn.block_hh_mask
    assert torch.count_nonzero(torch.diagonal(effective)).item() == 0

    probe = _model(**_split_on())
    probe.zero_grad()
    probe(torch.randn(4, 2, 4)).sum().backward()
    assert probe.rnn.linear_C.weight.grad[:heart].abs().sum().item() > 0
    assert probe.rnn.linear_C.weight.grad[heart:].abs().sum().item() > 0

    shallow = _model("shPLRNN", **_split_on())
    heart_sh = shallow.rnn.heart_sh
    w1 = shallow.rnn.W1 * shallow.rnn.block_w1_mask
    w2 = shallow.rnn.W2 * shallow.rnn.block_w2_mask
    assert w1[:heart, heart_sh:].abs().max().item() == 0.0
    assert w1[heart:, :heart_sh].abs().max().item() == 0.0
    assert w2[:heart_sh, heart:].abs().max().item() == 0.0
    assert w2[heart_sh:, :heart].abs().max().item() == 0.0
    assert shallow.rnn.linear_C.weight.abs().sum().item() > 0

    for kind in ("RNN", "LSTM"):
        cell = _model(kind, **_split_on())
        hidden = cell.dim_rnn
        cell.rnn.weight_ih_l0.data.zero_()
        if cell.rnn.bias_ih_l0 is not None:
            cell.rnn.bias_ih_l0.data.zero_()
        if cell.rnn.bias_hh_l0 is not None:
            cell.rnn.bias_hh_l0.data.zero_()
        cell.rnn.weight_hh_l0.data.fill_(1.0)
        state = torch.zeros(1, 1, hidden)
        state[0, 0, :heart] = 1.0
        if kind == "LSTM":
            cell.h_t = state
            # Drive recurrence, which is the path train() uses.
            out_h, _ = cell.recurrence(
                torch.zeros(1, 1, 4), state, torch.zeros(1, 1, hidden)
            )
        else:
            out_h, _ = cell.recurrence(torch.zeros(1, 1, 4), state)
        assert out_h[0, 0, heart:].abs().max().item() == 0.0


def test_a_lone_channel_zero_is_a_valid_index():
    """0-based channel 0 is an index, not the off switch."""
    parsed = parse_hidden_block_split(
        _cfg(**_split_on(ecg_channels="0", eeg_channels="1")), 4, 2
    )
    assert parsed.ecg == (0,)
    assert parsed.eeg == (1,)


def test_decoder_writes_heart_to_ecg_and_brain_to_eeg_only():
    model = _model(**_split_on(ecg_channels="0,1", eeg_channels="3"))
    heart = model.hidden_block.heart
    model.gen_out.bias.data.zero_()
    model.gen_out.weight.data.fill_(1.0)
    hidden = torch.zeros(1, 4)
    hidden[0, :heart] = 1.0
    y = model.generation_x(hidden)
    ecg = list(model.hidden_block.ecg)
    eeg = list(model.hidden_block.eeg)
    assert torch.allclose(y[0, ecg], torch.full((len(ecg),), float(heart)))
    assert y[0, eeg].abs().max().item() == 0.0
    other = [index for index in range(model.y_dim) if index not in ecg and index not in eeg]
    if other:
        assert y[0, other].abs().max().item() == 0.0
    hidden = torch.zeros(1, 4)
    hidden[0, heart:] = 1.0
    y = model.generation_x(hidden)
    assert y[0, ecg].abs().max().item() == 0.0
    assert torch.allclose(y[0, eeg], torch.full((len(eeg),), float(model.hidden_block.brain)))


def test_masked_parameters_stay_exactly_zero_after_adam():
    """The spec: masked recurrent entries are exactly 0 after optimizer steps."""
    problems = []
    for kind in ("PLRNN", "shPLRNN", "RNN", "LSTM"):
        model = _model(kind, **_split_on())
        heart = model.hidden_block.heart
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        model(torch.randn(6, 2, 4)).square().mean().backward()
        opt.step()
        if kind == "PLRNN":
            pieces = _off_block(model.rnn.W.data, heart)
            pieces.append(torch.diagonal(model.rnn.W.data))
        elif kind == "shPLRNN":
            heart_sh = model.rnn.heart_sh
            pieces = [
                model.rnn.W1.data[:heart, heart_sh:],
                model.rnn.W1.data[heart:, :heart_sh],
                model.rnn.W2.data[:heart_sh, heart:],
                model.rnn.W2.data[heart_sh:, :heart],
            ]
        else:
            gates = 4 if kind == "LSTM" else 1
            pieces = _off_block(model.rnn.weight_hh_l0.data, heart, gates)
        worst = max(float(piece.abs().max()) for piece in pieces)
        if worst != 0.0:
            problems.append(f"{kind} masked parameter max={worst}")
        cross = model.gen_out.weight.data
        eeg = list(model.hidden_block.eeg)
        ecg = list(model.hidden_block.ecg)
        decoder_worst = max(
            float(cross[eeg, :heart].abs().max()),
            float(cross[ecg, heart:].abs().max()),
        )
        if decoder_worst != 0.0:
            problems.append(f"{kind} decoder cross max={decoder_worst}")
    assert not problems, "; ".join(problems)


def test_zeroed_masked_entries_are_not_refilled_by_adam():
    model = _model(**_split_on())
    heart = model.hidden_block.heart
    with torch.no_grad():
        model.rnn.W.data[:heart, heart:] = 0
        model.rnn.W.data[heart:, :heart] = 0
        model.rnn.W.data.fill_diagonal_(0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.1)
    model(torch.randn(6, 2, 4)).square().mean().backward()
    opt.step()
    assert model.rnn.W.data[:heart, heart:].abs().max().item() == 0.0
    assert model.rnn.W.data[heart:, :heart].abs().max().item() == 0.0
    assert torch.diagonal(model.rnn.W.data).abs().max().item() == 0.0


def _reference_subband(predicted, target, channels, fs, low, high):
    pred = np.nan_to_num(predicted[:, :, list(channels)], nan=0.0)
    gt = np.nan_to_num(target[:, :, list(channels)], nan=0.0)
    freqs = np.fft.rfftfreq(pred.shape[0], d=1.0 / fs)
    nyquist = fs / 2.0 if high is None else high
    band = (freqs >= low) & (freqs <= nyquist + 1e-8)
    pred_power = np.abs(np.fft.rfft(pred, axis=0)) ** 2
    gt_power = np.abs(np.fft.rfft(gt, axis=0)) ** 2
    pred_band = pred_power[band]
    gt_band = gt_power[band]
    pred_norm = pred_band / np.maximum(pred_band.sum(axis=0, keepdims=True), 1e-8)
    gt_norm = gt_band / np.maximum(gt_band.sum(axis=0, keepdims=True), 1e-8)
    return float(np.mean((pred_norm - gt_norm) ** 2))


def test_subband_band_may_start_at_zero_hertz():
    """0 Hz is the bottom of a band, not the off switch."""
    cfg = configparser.ConfigParser()
    cfg["Training"] = {
        "subband_loss_weight": "1",
        "subband_channels": "0",
        "subband_fs": "64",
        "subband_low_hz": "0",
    }
    settings = parse_regularizer_settings(cfg, 2, None)
    assert settings.subband_on is True
    assert settings.subband_low_hz == 0.0
    assert settings.subband_channels == (0,)


def test_subband_is_band_normalised_rfft_mse_on_the_chosen_channels():
    step = np.arange(64, dtype=np.float64)
    slow = np.sin(2 * np.pi * 2 * step / 64)
    mid = np.sin(2 * np.pi * 3 * step / 64)
    fast = np.sin(2 * np.pi * 16 * step / 64)
    other = np.sin(2 * np.pi * 1 * step / 64)
    target = np.stack([slow, other], axis=1)[:, None, :]
    predicted = np.stack([mid, other * 5], axis=1)[:, None, :]
    # Channel 1 is outside the chosen set. 2 Hz and 3 Hz are both inside 0..4 Hz.
    got = subband_power_loss(
        torch.tensor(predicted, dtype=torch.float64),
        torch.tensor(target, dtype=torch.float64),
        (0,),
        64.0,
        0.0,
        4.0,
    )
    ref = _reference_subband(predicted, target, (0,), 64.0, 0.0, 4.0)
    assert ref > 0.0
    assert got.item() == pytest.approx(ref, rel=1e-5, abs=1e-8)
    outside = np.stack([slow + fast, other * 5], axis=1)[:, None, :]
    matched = subband_power_loss(
        torch.tensor(outside, dtype=torch.float64),
        torch.tensor(target, dtype=torch.float64),
        (0,),
        64.0,
        0.0,
        4.0,
    )
    assert matched.item() == pytest.approx(0.0, abs=1e-6)


def test_decorr_is_mean_squared_cross_correlation_only():
    step = torch.arange(32, dtype=torch.float32)
    sine = torch.sin(2 * torch.pi * step / 32)
    cosine = torch.cos(2 * torch.pi * step / 32)
    hidden = torch.stack([sine, sine, cosine, cosine], dim=1).unsqueeze(1)
    # Within a block the series match. Only the cross block is scored.
    assert decorr_hidden_loss(hidden, 2, 2).item() == pytest.approx(0.0, abs=1e-5)
    copied = torch.stack([sine, cosine, sine, cosine], dim=1).unsqueeze(1)
    assert decorr_hidden_loss(copied, 2, 2).item() == pytest.approx(0.5, abs=1e-5)
    output = copied
    assert decorr_output_loss(output, (0, 1), (2, 3)).item() == pytest.approx(
        0.5, abs=1e-5
    )
    # Within-group correlation is 0 for (sine, cosine), and that must not
    # enter the output penalty. The cross block is zero here.
    crossed = torch.stack([sine, cosine, cosine, sine], dim=1).unsqueeze(1)
    # (0,2)=0, (0,3)=1, (1,2)=1, (1,3)=0 → mean of squares 0.5
    assert decorr_output_loss(crossed, (0, 1), (2, 3)).item() == pytest.approx(
        0.5, abs=1e-5
    )


def test_warmup_scale_follows_auto_and_kl_bumps_only():
    assert effective_weight(2.0, False, 0, 4, 6) == pytest.approx(2.0)
    assert effective_weight(2.0, True, 0, 0, 0) == pytest.approx(2.0)
    assert effective_weight(2.0, True, 0, 4, 6) == pytest.approx(0.0)
    assert effective_weight(2.0, True, 5, 4, 6) == pytest.approx(1.0)
    assert effective_weight(2.0, True, 10, 4, 6) == pytest.approx(2.0)


def test_blank_patience_start_is_at_least_one_for_patience_1_and_2():
    assert patience_start_from_fraction(0.2, 1) == 1
    assert patience_start_from_fraction(0.2, 2) == 1
    assert patience_start_from_fraction(0.2, 200) == 40
    assert patience_start_from_fraction(0.2, 500) == 100
    for patience in (1, 2):
        assert resolve_warmup_patience_start(
            None, legacy=False, frac=0.2, early_stop_patience=patience
        ) == 1.0
    cfg = configparser.ConfigParser()
    cfg["Training"] = {"early_stop_patience": "2"}
    _order, start = warmup_settings_from_config(cfg)
    assert start == 1.0
    explicit = resolve_warmup_patience_start(
        0, legacy=False, frac=0.2, early_stop_patience=2
    )
    assert explicit == 0.0


def test_hr_coarse_modes_raise_unless_opted_in_and_the_example_is_ecg_free():
    hr_modes = [
        name
        for name, columns in COARSE_OBS_COLUMNS.items()
        if "hr_mean_bpm" in columns
    ]
    assert hr_modes
    for name in hr_modes:
        with pytest.raises(ValueError, match="coarse_include_hr"):
            coarse_feature_columns(name, include_hr=False)
        with pytest.warns(UserWarning, match="hr_mean_bpm"):
            columns = coarse_feature_columns(name, include_hr=True)
        assert "hr_mean_bpm" in columns
    for name in ("coarse_eeg", "coarse_eeg_acc", "coarse_eeg_acc_tod"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            columns = coarse_feature_columns(name, include_hr=False)
        assert "hr_mean_bpm" not in columns

    from bin.generate_config_file_for_multiple import xhro_proper_coarse_example

    example = xhro_proper_coarse_example()
    assert example["observation_process"] == ["coarse_eeg_acc_tod"]
    width = len(COARSE_OBS_COLUMNS["coarse_eeg_acc_tod"])
    assert width == 7
    assert example["x_dim"] == [width]


def _train_text() -> str:
    return """
[User]
experiment_name = spec25
logger_type = 1
print_model = False

[Network]
name = RNN
tag = RNN
x_dim = 3
activation = relu
dense_x =
dense_h_x =
dim_rnn = 4
num_rnn = 1
type_rnn = RNN
dropout_p = 0
hidden_sh_size =
hidden_block_split = half
ecg_channels = 1
eeg_channels = 2

[Training]
use_cuda = False
optimization = adam
beta = 1
lr = 0
epochs = 6
early_stop_patience = 1
warmup_patience_start = 0
warmup_order = window_first
save_frequency = 1
gradient_clip = 0
optimize_alphas = False
sampling_method = ss
sampling_ratio = 1
auto_warm_start = 0.8
val_criteria = recon
val_rosenstein = false
legacy_defaults = true
subband_loss_weight = 2
subband_loss_warmup = true
subband_channels = 1
subband_fs = 1
subband_low_hz = 0.1
decorr_loss_weight = 3
decorr_loss_warmup = true
decorr_mode = hidden

[DataFrame]
dataset_name = Lorenz63
dataset_label = synthetic
mask_label = None
shuffle = False
batch_size = 2
num_workers = 0
sequence_len = 10
initial_sequence_len = 8
sample_rate = 1
skip_rate = 1
val_indices = 0.25
observation_process = all_xyz
overlap = False
with_nan = False
legacy_defaults = true
""".strip() + "\n"


def test_subband_and_decorr_are_logged_stored_and_ramped_on_auto_bumps(
    tmp_path, monkeypatch
):
    from dvae.learning_algo import LearningAlgorithm

    def _record(argv, *args, **kwargs):
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("dvae.learning_algo.subprocess.run", _record)

    data_root = tmp_path / "data"
    folder = data_root / "lorenz63" / "data" / "synthetic"
    folder.mkdir(parents=True)
    t = np.linspace(0, 6 * np.pi, 128)
    series = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    for name in ("complete_dataset_train.pkl", "complete_dataset_test.pkl"):
        with open(folder / name, "wb") as handle:
            pickle.dump(series.astype(np.float64), handle)
    cfg_path = tmp_path / "cfg.ini"
    cfg_path.write_text(_train_text())
    algo = LearningAlgorithm(
        {
            "cfg": str(cfg_path),
            "job_id": "spec25",
            "reload": False,
            "device_config": {
                "saved_root": str(tmp_path / "saved"),
                "data_dir": str(data_root),
                "device_name": "cpu",
            },
        }
    )
    algo.build_model()
    assert algo.model.hidden_block is not None
    algo.train()
    run = next((tmp_path / "saved").rglob("loss_model.pckl"))
    log = (run.parent / "log.txt").read_text()
    assert "Train => subband" in log
    assert "Train => decorr" in log
    assert "Train => tot:" in log
    with open(run, "rb") as handle:
        history = pickle.load(handle)
    sub_w = np.asarray(history["subband_loss_weight_values"], dtype=float)
    dec_w = np.asarray(history["decorr_loss_weight_values"], dtype=float)
    sub_scale = np.asarray(history["subband_warm_values"], dtype=float)
    dec_scale = np.asarray(history["decorr_warm_values"], dtype=float)
    assert history["subband_enabled"] is True
    assert history["decorr_enabled"] is True
    np.testing.assert_allclose(sub_scale * 2.0, sub_w)
    np.testing.assert_allclose(dec_scale * 3.0, dec_w)
    sub = np.asarray(history["train_subband"], dtype=float)
    dec = np.asarray(history["train_decorr"], dtype=float)
    assert np.isfinite(sub_w).all()
    assert np.isfinite(sub).all() and np.isfinite(dec).all()
    events = history["bump_events"]
    assert any(row["type"] == "window" for row in events)
    assert any(row["type"] == "auto" for row in events)
    sub_bumps = [row["epoch"] for row in events if row["type"] == "subband"]
    dec_bumps = [row["epoch"] for row in events if row["type"] == "decorr"]
    auto_bumps = [row["epoch"] for row in events if row["type"] == "auto"]
    assert sub_bumps and dec_bumps == sub_bumps
    assert set(sub_bumps) <= set(auto_bumps)
    assert "type=subband" in log
    assert "type=decorr" in log
    done = 0
    auto_total = sum(1 for row in events if row["type"] == "auto")
    # KL is off for an RNN, so the ramp denominator is the auto-bump count.
    assert auto_total >= 1
    for epoch, value in enumerate(sub_w):
        if epoch == 0 and not any(row["epoch"] == 0 for row in events):
            # Epoch 0 is the improving epoch and still uses a zero ramp.
            pass
        expected = 2.0 * (0.0 if auto_total == 0 else min(done / auto_total, 1.0))
        assert value == pytest.approx(expected), (epoch, value, expected, done)
        assert dec_w[epoch] == pytest.approx(expected * 3.0 / 2.0)
        done += sum(
            1
            for row in events
            if row["epoch"] == epoch and row["type"] in ("auto", "KL")
        )
    window_epochs = [row["epoch"] for row in events if row["type"] == "window"]
    for epoch in window_epochs:
        later_auto = any(
            row["type"] == "auto" and row["epoch"] < epoch for row in events
        )
        if not later_auto:
            assert sub_w[epoch] == pytest.approx(0.0)
