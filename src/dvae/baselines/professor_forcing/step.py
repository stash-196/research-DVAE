"""One optimization step for Professor Forcing or the hidden-match ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dvae.baselines.professor_forcing.discriminator import BehaviorDiscriminator
from dvae.baselines.professor_forcing.generator import ProfessorForcingGenerator
from dvae.baselines.professor_forcing.losses import (
    adversarial_gates,
    classification_accuracy,
    discriminator_loss,
    generator_adversarial_loss,
    hidden_match_penalty,
    loss_mask_for,
    repo_masked_mse,
)


@dataclass
class StepResult:
    generator_loss: torch.Tensor
    nll: torch.Tensor
    update_generator_adversarial: bool
    update_discriminator: bool
    discriminator_loss: torch.Tensor | None = None
    adversarial_loss: torch.Tensor | None = None
    hidden_match: torch.Tensor | None = None
    accuracy: float | None = None

    def as_log(self) -> dict:
        def _value(tensor):
            if tensor is None:
                return None
            return float(tensor.detach().item())

        return {
            "loss": _value(self.generator_loss),
            "nll": _value(self.nll),
            "c_d": _value(self.discriminator_loss),
            "c_adv": _value(self.adversarial_loss),
            "hidden_match": _value(self.hidden_match),
            "disc_accuracy": self.accuracy,
            "updated_g_adv": self.update_generator_adversarial,
            "updated_d": self.update_discriminator,
        }


def professor_forcing_loss(
    generator: ProfessorForcingGenerator,
    discriminator: BehaviorDiscriminator,
    x: torch.Tensor,
    *,
    behavior: str,
    generator_objective: str,
    observation_process: str | None,
    feedback: str = "mean",
    observation_std: float = 1.0,
    acc_low: float = 0.75,
    acc_high: float = 0.99,
    accuracy_override: float | None = None,
) -> StepResult:
    """Build the paper's generator and discriminator losses for one minibatch.

    The accuracy gate uses a stop-gradient classification of this minibatch,
    unless ``accuracy_override`` is set (tests). Discriminator parameters are
    not stepped here.
    """
    if generator_objective not in ("nll_cf", "nll_cf_ct"):
        raise ValueError(
            "generator_objective must be 'nll_cf' or 'nll_cf_ct', "
            f"got {generator_objective!r}."
        )
    teacher = generator.rollout(x, mode="teacher", feedback=feedback, observation_std=observation_std)
    free = generator.rollout(x, mode="free", feedback=feedback, observation_std=observation_std)
    mask = loss_mask_for(x, observation_process)
    nll = repo_masked_mse(teacher.emission, x, mask)

    b_tf = generator.behavior_features(teacher, behavior)
    b_free = generator.behavior_features(free, behavior)
    with torch.no_grad():
        judged_tf = discriminator(b_tf.detach())
        judged_free = discriminator(b_free.detach())
        accuracy = classification_accuracy(judged_tf, judged_free)
    if accuracy_override is not None:
        accuracy = float(accuracy_override)
    update_g_adv, update_d = adversarial_gates(accuracy, acc_low, acc_high)

    adversarial = None
    generator_loss = nll
    if update_g_adv:
        free_out = discriminator(b_free)
        teacher_out = discriminator(b_tf) if generator_objective == "nll_cf_ct" else None
        adversarial = generator_adversarial_loss(
            teacher_out, free_out, include_teacher_term=(generator_objective == "nll_cf_ct")
        )
        generator_loss = nll + adversarial

    disc_loss = None
    if update_d:
        disc_loss = discriminator_loss(discriminator(b_tf.detach()), discriminator(b_free.detach()))

    return StepResult(
        generator_loss=generator_loss,
        nll=nll,
        update_generator_adversarial=update_g_adv,
        update_discriminator=update_d,
        discriminator_loss=disc_loss,
        adversarial_loss=adversarial,
        accuracy=accuracy,
    )


def hidden_match_loss(
    generator: ProfessorForcingGenerator,
    x: torch.Tensor,
    *,
    observation_process: str | None,
    feedback: str = "mean",
    observation_std: float = 1.0,
    hidden_match_weight: float = 1.0,
) -> StepResult:
    """Ablation: NLL plus ``weight * ||h_free - stopgrad(h_tf)||^2`` (mean).

    There is no discriminator and no accuracy gate.
    """
    teacher = generator.rollout(x, mode="teacher", feedback=feedback, observation_std=observation_std)
    free = generator.rollout(x, mode="free", feedback=feedback, observation_std=observation_std)
    mask = loss_mask_for(x, observation_process)
    nll = repo_masked_mse(teacher.emission, x, mask)
    match = hidden_match_penalty(free.h, teacher.h)
    return StepResult(
        generator_loss=nll + float(hidden_match_weight) * match,
        nll=nll,
        update_generator_adversarial=False,
        update_discriminator=False,
        hidden_match=match,
        accuracy=None,
    )


def apply_step(
    result: StepResult,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> None:
    """Step the generator, then the discriminator when the gate allows it."""
    opt_g.zero_grad(set_to_none=True)
    if opt_d is not None:
        opt_d.zero_grad(set_to_none=True)
    result.generator_loss.backward()
    _clip(generator.parameters(), gradient_clip)
    opt_g.step()
    if opt_d is not None:
        opt_d.zero_grad(set_to_none=True)
    if result.update_discriminator and result.discriminator_loss is not None and opt_d is not None:
        result.discriminator_loss.backward()
        _clip(discriminator.parameters(), gradient_clip)
        opt_d.step()


def _clip(parameters, gradient_clip: float) -> None:
    if gradient_clip and float(gradient_clip) > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
