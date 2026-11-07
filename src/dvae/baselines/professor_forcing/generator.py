"""Generator RNN rolled out in teacher-forced and free-running modes.

Time index follows the usual one-step predictor: the emission at time t is a
function of the hidden state that has already consumed inputs ``0 .. t-1``.
The GRU then consumes the chosen input at time t and produces the state used
for the next emission. Under teacher forcing that input is the observation
(with the repo's NaN fill). Under free running it is the emission just
produced. See FORMULATION.md for the continuous-Lorenz choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dvae.baselines.professor_forcing.gru import ProfessorGRUCell

BEHAVIORS = (
    "pre_tanh",
    "pre_tanh_and_output",
    "hidden",
    "hidden_and_output",
    "hidden_and_input",
)


@dataclass
class Rollout:
    """Tensors from one closed- or open-loop pass. Time is axis 0."""

    emission: torch.Tensor
    pre_tanh: torch.Tensor
    h: torch.Tensor
    behavior_output: torch.Tensor
    inputs_used: torch.Tensor


class ProfessorForcingGenerator(nn.Module):
    def __init__(self, x_dim: int, hidden_size: int, indicate: bool = False):
        super().__init__()
        self.x_dim = int(x_dim)
        self.hidden_size = int(hidden_size)
        self.indicate = bool(indicate)
        self.cell = ProfessorGRUCell(self.x_dim, self.hidden_size)
        self.readout = nn.Linear(self.hidden_size, self.x_dim)

    def feature_dim(self, behavior: str) -> int:
        if behavior not in BEHAVIORS:
            raise ValueError(
                f"behavior must be one of {BEHAVIORS}, got {behavior!r}."
            )
        if behavior == "pre_tanh":
            return self.hidden_size
        if behavior == "hidden":
            return self.hidden_size
        return self.hidden_size + self.x_dim

    def behavior_features(self, rollout: Rollout, behavior: str) -> torch.Tensor:
        if behavior == "pre_tanh":
            return rollout.pre_tanh
        if behavior == "pre_tanh_and_output":
            return torch.cat([rollout.pre_tanh, rollout.behavior_output], dim=-1)
        if behavior == "hidden":
            return rollout.h
        if behavior == "hidden_and_output":
            return torch.cat([rollout.h, rollout.behavior_output], dim=-1)
        if behavior == "hidden_and_input":
            return torch.cat([rollout.h, rollout.inputs_used], dim=-1)
        raise ValueError(f"Unknown behavior {behavior!r}.")

    def rollout(
        self,
        x: torch.Tensor,
        mode: str = "teacher",
        mode_selector: torch.Tensor | None = None,
        feedback: str = "mean",
        observation_std: float = 1.0,
    ) -> Rollout:
        """Roll out over ``x`` shaped ``(T, B, D)``.

        ``mode`` is ``teacher`` or ``free``. ``mode_selector`` overrides it:
        0 is teacher forcing and 1 is free running, broadcast over features.
        ``feedback`` is ``mean`` or ``gaussian`` and applies only on free-running
        coordinates.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (T, B, D), got shape {tuple(x.shape)}.")
        seq_len, batch_size, x_dim = x.shape
        if x_dim != self.x_dim:
            raise ValueError(f"x_dim {x_dim} does not match the generator ({self.x_dim}).")
        if feedback not in ("mean", "gaussian"):
            raise ValueError(f"feedback must be 'mean' or 'gaussian', got {feedback!r}.")

        free = _free_mask(x, mode, mode_selector)
        if self.indicate:
            free = free.clone()
            free[:, :, 1::2] = 0.0

        h = x.new_zeros(batch_size, self.hidden_size)
        emission = x.new_empty(seq_len, batch_size, x_dim)
        pre_tanh = x.new_empty(seq_len, batch_size, self.hidden_size)
        h_seq = x.new_empty(seq_len, batch_size, self.hidden_size)
        inputs_used = x.new_empty(seq_len, batch_size, x_dim)

        for t in range(seq_len):
            y_hat = self.readout(h)
            emission[t] = y_hat
            imputed = _impute_observation(x[t], emission[t - 1] if t else None)
            if feedback == "mean":
                sampled = y_hat
            else:
                sampled = y_hat + float(observation_std) * torch.randn_like(y_hat)
            used = free[t] * sampled + (1.0 - free[t]) * imputed
            inputs_used[t] = used
            h, pre = self.cell(used, h)
            pre_tanh[t] = pre
            h_seq[t] = h

        next_from_final = self.readout(h).unsqueeze(0)
        behavior_output = torch.cat([emission[1:], next_from_final], dim=0)
        return Rollout(
            emission=emission,
            pre_tanh=pre_tanh,
            h=h_seq,
            behavior_output=behavior_output,
            inputs_used=inputs_used,
        )


def _free_mask(x: torch.Tensor, mode: str, mode_selector: torch.Tensor | None) -> torch.Tensor:
    seq_len, batch_size, x_dim = x.shape
    if mode_selector is None:
        if mode == "teacher":
            return x.new_zeros(seq_len, batch_size, x_dim)
        if mode == "free":
            return x.new_ones(seq_len, batch_size, x_dim)
        raise ValueError(f"mode must be 'teacher' or 'free', got {mode!r}.")

    selector = mode_selector.to(device=x.device, dtype=x.dtype)
    if selector.shape == (seq_len,):
        selector = selector.view(seq_len, 1, 1).expand(seq_len, batch_size, x_dim)
    elif selector.shape == (seq_len, batch_size):
        selector = selector.unsqueeze(-1).expand(seq_len, batch_size, x_dim)
    elif selector.shape == (seq_len, batch_size, 1):
        selector = selector.expand(seq_len, batch_size, x_dim)
    elif selector.shape != (seq_len, batch_size, x_dim):
        raise ValueError(
            f"mode_selector shape {tuple(selector.shape)} is not compatible with "
            f"x shape {(seq_len, batch_size, x_dim)}."
        )
    return selector


def _impute_observation(obs: torch.Tensor, previous_emission: torch.Tensor | None) -> torch.Tensor:
    """Fill non-finite coordinates the way ``BaseModel`` fills NaNs.

    Time 0 uses zeros. Later times use the previous emission. Finite
    coordinates keep the observation.
    """
    finite = torch.isfinite(obs)
    if previous_emission is None:
        fill = torch.zeros_like(obs)
    else:
        fill = previous_emission
    return torch.where(finite, obs, fill)
