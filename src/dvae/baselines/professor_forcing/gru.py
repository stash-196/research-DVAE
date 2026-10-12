"""Single-layer GRU that exposes the candidate pre-tanh activation.

The recurrence matches ``torch.nn.GRUCell`` (reset, update, new gates):

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n_pre = W_in x + b_in + r * (W_hn h + b_hn)
    n = tanh(n_pre)
    h' = (1 - z) * n + z * h

Section 4.1 of Lamb et al. (2016) uses the pre-tanh candidate activation
``n_pre`` as the behavior sequence.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ProfessorGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -stdv, stdv)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        gates_x = nn.functional.linear(x, self.weight_ih, self.bias_ih)
        gates_h = nn.functional.linear(h, self.weight_hh, self.bias_hh)
        x_reset, x_update, x_new = gates_x.chunk(3, dim=-1)
        h_reset, h_update, h_new = gates_h.chunk(3, dim=-1)
        reset = torch.sigmoid(x_reset + h_reset)
        update = torch.sigmoid(x_update + h_update)
        pre_tanh = x_new + reset * h_new
        candidate = torch.tanh(pre_tanh)
        h_new_state = (1.0 - update) * candidate + update * h
        return h_new_state, pre_tanh
