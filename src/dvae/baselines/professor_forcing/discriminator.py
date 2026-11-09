"""Bidirectional-GRU discriminator over a behavior sequence.

Section 4.1: a forward GRU and a backward GRU read the behavior sequence.
Their hidden states are concatenated at each time and passed through an MLP
that is shared across time. The MLP has three affine+ReLU layers. A final
affine layer produces a score. That score is reduced over time, clipped, and
passed through a sigmoid so ``D(b)`` is one probability per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiscriminatorOutput:
    logits: torch.Tensor
    probs: torch.Tensor


class BehaviorDiscriminator(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, logit_clip: float = 10.0):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.logit_clip = float(logit_clip)
        self.forward_gru = nn.GRU(self.input_size, self.hidden_size)
        self.backward_gru = nn.GRU(self.input_size, self.hidden_size)
        width = self.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.output = nn.Linear(width, 1)

    def forward(self, behavior: torch.Tensor) -> DiscriminatorOutput:
        """``behavior`` is ``(T, B, F)``. Returns one logit and probability per batch row."""
        if behavior.dim() != 3:
            raise ValueError(
                f"behavior must be (T, B, F), got shape {tuple(behavior.shape)}."
            )
        forward_h, _ = self.forward_gru(behavior)
        backward_h, _ = self.backward_gru(behavior.flip(0))
        backward_h = backward_h.flip(0)
        combined = torch.cat([forward_h, backward_h], dim=-1)
        time_steps, batch_size, _ = combined.shape
        features = self.mlp(combined.reshape(time_steps * batch_size, -1))
        step_logits = self.output(features).reshape(time_steps, batch_size)
        logits = step_logits.mean(dim=0)
        if self.logit_clip > 0:
            logits = logits.clamp(-self.logit_clip, self.logit_clip)
        return DiscriminatorOutput(logits=logits, probs=torch.sigmoid(logits))
