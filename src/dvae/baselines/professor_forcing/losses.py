"""Loss terms.

Teacher-forced NLL is the repo's masked sum of squares divided by
``seq_len * batch_size`` (``loss_MSE`` then ``/ (seq_len * batch)`` in
``LearningAlgorithm``). The adversarial terms are the paper's ``C_d``,
``C_f``, and optional ``C_t``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dvae.baselines.professor_forcing.discriminator import DiscriminatorOutput


def indicate_process(observation_process: str | None) -> bool:
    """Same predicate as ``dvae.eval.utils.benchmark_signals.is_indicate_observation``."""
    if not observation_process:
        return False
    if observation_process == "only_x_indicate":
        return True
    return observation_process.endswith("_indicate")


def loss_mask_for(x: torch.Tensor, observation_process: str | None) -> torch.Tensor:
    """1 where the repo would include the coordinate in the reconstruction sum."""
    finite = torch.isfinite(x).to(dtype=x.dtype)
    if indicate_process(observation_process):
        mask = torch.zeros_like(finite)
        mask[..., 0::2] = finite[..., 0::2]
        return mask
    return finite


def repo_masked_mse(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Match ``loss_MSE(target * mask, pred * mask) / (T * B)``."""
    seq_len, batch_size, _ = pred.shape
    target_masked = torch.nan_to_num(target * loss_mask, nan=0.0)
    pred_masked = pred * loss_mask
    return torch.sum((target_masked - pred_masked) ** 2) / (seq_len * batch_size)


def classification_accuracy(teacher: DiscriminatorOutput, free: DiscriminatorOutput) -> float:
    """Fraction of the 2N behavior sequences classified correctly.

    Teacher-forced behavior is the positive class (``D > 0.5``). Free-running
    behavior is the negative class.
    """
    teacher_correct = (teacher.probs > 0.5).to(dtype=teacher.probs.dtype)
    free_correct = (free.probs <= 0.5).to(dtype=free.probs.dtype)
    correct = teacher_correct.sum() + free_correct.sum()
    total = teacher_correct.numel() + free_correct.numel()
    return float((correct / total).item())


def adversarial_gates(accuracy: float, acc_low: float = 0.75, acc_high: float = 0.99):
    """Paper section 4.1.

    Adversarial gradients enter the generator only when accuracy is greater
    than 75%. The discriminator is not updated when accuracy is greater than
    99%. The two tests are separate: above 99% the generator is still updated
    adversarially.
    """
    update_generator_adversarial = accuracy > acc_low
    update_discriminator = accuracy <= acc_high
    return update_generator_adversarial, update_discriminator


def bce_logits(logits: torch.Tensor, target: float) -> torch.Tensor:
    labels = torch.full_like(logits, float(target))
    return F.binary_cross_entropy_with_logits(logits, labels)


def discriminator_loss(teacher: DiscriminatorOutput, free: DiscriminatorOutput) -> torch.Tensor:
    """``C_d``: ``-log D(b_tf) - log(1 - D(b_free))``, averaged per class then summed."""
    return bce_logits(teacher.logits, 1.0) + bce_logits(free.logits, 0.0)


def generator_adversarial_loss(
    teacher: DiscriminatorOutput | None,
    free: DiscriminatorOutput,
    include_teacher_term: bool,
) -> torch.Tensor:
    """``C_f``, plus ``C_t`` when ``include_teacher_term`` is set."""
    c_f = bce_logits(free.logits, 1.0)
    if not include_teacher_term:
        return c_f
    if teacher is None:
        raise ValueError("C_t requires the teacher-forced discriminator output.")
    c_t = bce_logits(teacher.logits, 0.0)
    return c_f + c_t


def hidden_match_penalty(h_free: torch.Tensor, h_tf: torch.Tensor) -> torch.Tensor:
    """Mean squared deviation ``||h_free - stopgrad(h_tf)||^2 / numel``.

    The stop-gradient is on the teacher-forced hidden state, as specified for
    the ablation. The reduction is the mean over time, batch, and units.
    """
    return torch.mean((h_free - h_tf.detach()) ** 2)
