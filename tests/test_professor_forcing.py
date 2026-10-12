"""Synthetic tests for the Professor Forcing baseline. No Lorenz files required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dvae.baselines.professor_forcing.config_io import read_ini, write_ini
from dvae.baselines.professor_forcing.discriminator import BehaviorDiscriminator
from dvae.baselines.professor_forcing.generator import ProfessorForcingGenerator
from dvae.baselines.professor_forcing.gru import ProfessorGRUCell
from dvae.baselines.professor_forcing.losses import (
    adversarial_gates,
    hidden_match_penalty,
    loss_mask_for,
    repo_masked_mse,
)
from dvae.baselines.professor_forcing.metrics import (
    half_half_auto_mask,
    rollout_pair,
    score_sequence,
)
from dvae.baselines.professor_forcing.plotting import plot_metric_history
from dvae.baselines.professor_forcing.step import apply_step, professor_forcing_loss
from dvae.baselines.professor_forcing.trainer import train
from dvae.eval.utils.run_mse_analysis import calc_masked_mse
from dvae.utils.loss import loss_MSE


def test_package_does_not_reference_the_otf_trainer():
    root = Path(__file__).resolve().parents[1] / "src" / "dvae" / "baselines" / "professor_forcing"
    for path in root.glob("*.py"):
        assert "learning_algo" not in path.read_text()


def test_gru_pre_tanh_matches_pytorch_cell():
    reference = nn.GRUCell(3, 5)
    cell = ProfessorGRUCell(3, 5)
    cell.load_state_dict(reference.state_dict())
    x = torch.randn(4, 3)
    h = torch.randn(4, 5)
    h_new, pre_tanh = cell(x, h)
    assert torch.allclose(h_new, reference(x, h), atol=1e-6)

    hidden = cell.hidden_size
    gi = nn.functional.linear(x, cell.weight_ih, cell.bias_ih)
    gh = nn.functional.linear(h, cell.weight_hh, cell.bias_hh)
    x_reset, _, x_new = gi.chunk(3, dim=-1)
    h_reset, _, h_new_gate = gh.chunk(3, dim=-1)
    reset = torch.sigmoid(x_reset + h_reset)
    manual = x_new + reset * h_new_gate
    assert manual.shape[-1] == hidden
    assert torch.allclose(pre_tanh, manual, atol=1e-6)


def test_teacher_forcing_uses_the_observation_and_lags_the_input():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(3, 4)
    x = torch.randn(6, 2, 3)
    teacher = generator.rollout(x, mode="teacher")
    assert torch.allclose(teacher.inputs_used, x)

    shifted = x.clone()
    shifted[1] = shifted[1] + 1
    other = generator.rollout(shifted, mode="teacher")
    assert torch.allclose(teacher.emission[1], other.emission[1])
    assert not torch.allclose(teacher.emission[2], other.emission[2])


def test_free_running_mean_feedback_ignores_the_observation():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(3, 4)
    x = torch.randn(5, 2, 3)
    free = generator.rollout(x, mode="free", feedback="mean")
    assert torch.allclose(free.inputs_used, free.emission)
    perturbed = x.clone()
    perturbed += 4
    other = generator.rollout(perturbed, mode="free", feedback="mean")
    assert torch.allclose(free.emission, other.emission)


def test_nan_teacher_force_fills_with_the_previous_emission():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(2, 4)
    x = torch.randn(4, 1, 2)
    x[2, 0, 0] = float("nan")
    teacher = generator.rollout(x, mode="teacher")
    assert torch.allclose(teacher.inputs_used[2, 0, 0], teacher.emission[1, 0, 0])
    assert torch.allclose(teacher.inputs_used[2, 0, 1], x[2, 0, 1])
    assert torch.allclose(teacher.inputs_used[0], torch.nan_to_num(x[0], nan=0.0))


def test_indicate_channels_stay_clamped_in_free_run():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(2, 4, indicate=True)
    x = torch.rand(6, 2, 2)
    x[:, :, 1] = (x[:, :, 1] > 0.4).float()
    free = generator.rollout(x, mode="free", feedback="mean")
    assert torch.allclose(free.inputs_used[:, :, 1], x[:, :, 1])
    assert torch.allclose(free.inputs_used[:, :, 0], free.emission[:, :, 0])


def test_half_half_matches_teacher_forcing_before_the_flip():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(1, 4)
    x = torch.randn(10, 2, 1)
    teacher, mixed = rollout_pair(generator, x)
    flip = x.shape[0] // 2
    assert torch.allclose(teacher[:flip], mixed[:flip])
    mask = half_half_auto_mask(x.shape[0])
    assert mask[:flip].sum() == 0
    assert mask[flip:].all()


def test_repo_nll_matches_loss_mse_reduction():
    torch.manual_seed(0)
    target = torch.randn(5, 2, 3)
    target[1, 0, 1] = float("nan")
    pred = torch.randn_like(target)
    mask = loss_mask_for(target, None)
    seq_len, batch, _ = target.shape
    manual = loss_MSE(torch.nan_to_num(target * mask, nan=0.0), pred * mask) / (seq_len * batch)
    assert torch.allclose(repo_masked_mse(pred, target, mask), manual)

    indicate = target.clone()
    indicate_mask = loss_mask_for(indicate, "only_x_indicate")
    assert torch.count_nonzero(indicate_mask[:, :, 1::2]) == 0
    assert torch.count_nonzero(indicate_mask[:, :, 0::2]) > 0


def test_accuracy_gate_matches_the_paper_thresholds():
    assert adversarial_gates(0.75) == (False, True)
    assert adversarial_gates(0.75 + 1e-6) == (True, True)
    assert adversarial_gates(0.99) == (True, True)
    assert adversarial_gates(0.99 + 1e-6) == (True, False)
    assert adversarial_gates(0.5) == (False, True)


def _pair():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(1, 4)
    discriminator = BehaviorDiscriminator(generator.feature_dim("pre_tanh"), 4, logit_clip=0)
    x = torch.randn(6, 2, 1)
    return generator, discriminator, x


def test_low_accuracy_generator_grad_is_nll_only():
    generator, discriminator, x = _pair()
    teacher = generator.rollout(x, mode="teacher")
    mask = loss_mask_for(x, "only_x")
    nll = repo_masked_mse(teacher.emission, x, mask)
    nll.backward()
    nll_grads = {name: param.grad.detach().clone() for name, param in generator.named_parameters()}
    generator.zero_grad()

    result = professor_forcing_loss(
        generator,
        discriminator,
        x,
        behavior="pre_tanh",
        generator_objective="nll_cf",
        observation_process="only_x",
        accuracy_override=0.5,
    )
    assert result.update_generator_adversarial is False
    assert result.update_discriminator is True
    result.generator_loss.backward()
    for name, param in generator.named_parameters():
        assert torch.allclose(param.grad, nll_grads[name])


def test_in_band_adversarial_term_changes_generator_grad():
    generator, discriminator, x = _pair()
    nll_only = professor_forcing_loss(
        generator,
        discriminator,
        x,
        behavior="pre_tanh",
        generator_objective="nll_cf",
        observation_process="only_x",
        accuracy_override=0.5,
    )
    nll_only.generator_loss.backward()
    before = {name: param.grad.detach().clone() for name, param in generator.named_parameters()}
    generator.zero_grad()
    discriminator.zero_grad()

    both = professor_forcing_loss(
        generator,
        discriminator,
        x,
        behavior="pre_tanh",
        generator_objective="nll_cf_ct",
        observation_process="only_x",
        accuracy_override=0.8,
    )
    assert both.update_generator_adversarial is True
    assert both.update_discriminator is True
    both.generator_loss.backward()
    changed = False
    for name, param in generator.named_parameters():
        if not torch.allclose(param.grad, before[name], atol=1e-8):
            changed = True
    assert changed


def test_high_accuracy_does_not_step_the_discriminator():
    generator, discriminator, x = _pair()
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-2)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-2)
    before = {name: param.detach().clone() for name, param in discriminator.named_parameters()}
    result = professor_forcing_loss(
        generator,
        discriminator,
        x,
        behavior="pre_tanh",
        generator_objective="nll_cf",
        observation_process="only_x",
        accuracy_override=0.995,
    )
    assert result.update_discriminator is False
    assert result.discriminator_loss is None
    apply_step(result, generator, discriminator, opt_g, opt_d, gradient_clip=5.0)
    for name, param in discriminator.named_parameters():
        assert torch.equal(param, before[name])


def test_hidden_match_stopgrad_blocks_teacher_hidden_grad():
    torch.manual_seed(0)
    generator = ProfessorForcingGenerator(1, 4)
    x = torch.randn(5, 2, 1)
    teacher = generator.rollout(x, mode="teacher")
    free = generator.rollout(x, mode="free")
    teacher.h.retain_grad()
    free.h.retain_grad()
    hidden_match_penalty(free.h, teacher.h).backward()
    assert teacher.h.grad is None
    assert free.h.grad is not None
    assert float(free.h.grad.abs().sum()) > 0

    teacher2 = generator.rollout(x, mode="teacher")
    free2 = generator.rollout(x, mode="free")
    generator.zero_grad()
    torch.mean((free2.h - teacher2.h) ** 2).backward()
    with_tf = {name: param.grad.detach().clone() for name, param in generator.named_parameters()}
    generator.zero_grad()
    teacher3 = generator.rollout(x, mode="teacher")
    free3 = generator.rollout(x, mode="free")
    hidden_match_penalty(free3.h, teacher3.h).backward()
    changed = False
    for name, param in generator.named_parameters():
        if not torch.allclose(param.grad, with_tf[name], atol=1e-8):
            changed = True
    assert changed


def test_score_sequence_mse_matches_calc_masked_mse():
    rng = np.random.default_rng(0)
    length = 40
    gt = rng.normal(size=(length, 1))
    pred_tf = gt + 0.1
    pred_half = gt + 0.1
    pred_half[length // 2 :] += 0.5
    gt[3, 0] = np.nan
    scored = score_sequence(gt, pred_tf, pred_half, "only_x", "Lorenz63")
    assert scored["mse_tf"] == pytest.approx(
        calc_masked_mse(gt[:, 0], pred_tf[:, 0], None)
    )
    auto = half_half_auto_mask(length)
    assert scored["mse_auto"] == pytest.approx(
        calc_masked_mse(gt[auto, 0], pred_half[auto, 0], None)
    )
    assert "spectrum_error_auto" in scored
    assert "kld_auto" in scored
    assert np.isfinite(scored["mse_auto"])


def test_config_roundtrip_and_grid(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "bin" / "generate_professor_forcing_configs.py"
    spec = importlib.util.spec_from_file_location("generate_professor_forcing_configs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    written = module.generate(tmp_path)
    assert len(written) == 3 * 2 * (1 + 2 * 2)
    names = [path.name for path in written]
    assert any(name.startswith("cfg_hidden_match_") for name in names)
    assert any("professor_forcing_only_x_indicate" in name for name in names)
    assert any("pre_tanh_and_output_nll_cf_ct" in name for name in names)

    sample = read_ini(written[0])
    again = tmp_path / "roundtrip.ini"
    write_ini(again, sample)
    assert read_ini(again)["variant"] == sample["variant"]
    assert read_ini(again)["acc_low"] == pytest.approx(0.75)
    assert read_ini(again)["acc_high"] == pytest.approx(0.99)

    widths = {read_ini(path)["observation_process"]: read_ini(path)["x_dim"] for path in written}
    assert widths["only_x"] == 1
    assert widths["all_xyz"] == 3
    assert widths["only_x_indicate"] == 2


def test_synthetic_train_writes_metrics_and_plot(tmp_path: Path):
    torch.manual_seed(0)
    series = torch.sin(torch.linspace(0, 6, 40)).view(1, 40, 1).repeat(4, 1, 1)
    series = series + 0.01 * torch.randn_like(series)
    loader = DataLoader(TensorDataset(series), batch_size=2)
    cfg = {
        "variant": "professor_forcing",
        "behavior": "pre_tanh",
        "generator_objective": "nll_cf",
        "free_run_feedback": "mean",
        "observation_process": "only_x",
        "dataset_name": "Lorenz63",
        "x_dim": 1,
        "hidden_size": 4,
        "disc_hidden_size": 4,
        "epochs": 1,
        "eval_every": 1,
        "max_train_batches": 1,
        "max_eval_batches": 1,
        "early_stop_patience": 5,
        "lr": 1e-3,
        "disc_lr": 1e-3,
        "use_cuda": False,
        "seed": 0,
    }
    result = train(cfg, loader, loader, tmp_path, device=torch.device("cpu"))
    history = result["history"]
    assert history and np.isfinite(history[-1]["mse_tf"])
    assert np.isfinite(history[-1]["mse_auto"])
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "training_metrics.png").is_file()
    assert plot_metric_history(history, tmp_path / "again.png") is not None

    cfg["variant"] = "hidden_match"
    ablation = train(cfg, loader, loader, tmp_path / "ablation", device=torch.device("cpu"))
    assert np.isfinite(ablation["history"][-1]["hidden_match"])
    assert ablation["history"][-1]["disc_accuracy"] is None
