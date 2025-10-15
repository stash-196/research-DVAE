import torch
import torch.nn as nn
import torch.nn.functional as F


class PLRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise ValueError(
                "PLRNN currently supports only num_layers=1 for simplicity."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        # Diagonal A: learnable vector for element-wise multiplication
        self.A = nn.Parameter(torch.empty(hidden_size).uniform_(0.5, 0.9))
        # Off-diagonal W: full matrix parameter
        r = 1.0 / (self.hidden_size**0.5)
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-r, r))

        # C x_t + b: standard linear
        self.linear_C = nn.Linear(input_size, hidden_size, bias=True)
        nn.init.uniform_(self.linear_C.weight, -r, r)  # Same scaling as W
        nn.init.uniform_(self.linear_C.bias, -r, r)

        # Mask for off-diagonal W (zeros on diagonal, ones elsewhere)
        self.register_buffer("W_mask", torch.ones(hidden_size, hidden_size))
        torch.diagonal(self.W_mask).zero_()

    def forward(self, input, hx=None):
        # input: (seq_len, batch_size, input_size) -- expects seq_len=1 in BaseRNN
        # hx: (1, batch_size, hidden_size)
        # Returns: output (seq_len, batch_size, hidden_size), new_hx (1, batch_size, hidden_size)

        if hx is None:
            hx = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)

        seq_len, batch_size = input.shape[:2]
        outputs = []
        h = hx[0]  # (batch_size, hidden_size)

        for t in range(seq_len):
            x = input[t]  # (batch_size, input_size)

            # Apply mask to W to zero diagonal, preserving gradient flow
            W_off = self.W * self.W_mask

            # h_t = A * h_{t-1} + W_off @ ReLU(h_{t-1}) + C x_t + b
            relu_h = F.relu(h)
            h = self.A * h + (W_off @ relu_h.T).T + self.linear_C(x)

            outputs.append(h)

        outputs = torch.stack(outputs)  # (seq_len, batch_size, hidden_size)
        new_hx = h.unsqueeze(0)  # (1, batch_size, hidden_size)
        return outputs, new_hx


class shPLRNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_sh_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise ValueError(
                "PLRNN currently supports only num_layers=1 for simplicity."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size  # dz in repo (latent dim)
        self.hidden_sh_size = hidden_sh_size  # dh in repo (shallow hidden dim)
        self.num_layers = 1

        # Follow repo init ranges for stability
        r1 = 1.0 / (self.hidden_sh_size**0.5)
        r2 = 1.0 / (self.hidden_size**0.5)

        # Diagonal A: learnable vector, init for decay (|A| < 1)
        self.A = nn.Parameter(torch.empty(self.hidden_size).uniform_(0.5, 0.9))

        # W1 (dz, dh): no off-diagonal constraint
        self.W1 = nn.Parameter(
            torch.empty(self.hidden_size, self.hidden_sh_size).uniform_(-r1, r1)
        )

        # W2 (dh, dz): no off-diagonal constraint
        self.W2 = nn.Parameter(
            torch.empty(self.hidden_sh_size, self.hidden_size).uniform_(-r2, r2)
        )

        # b1 (dz): bias, init zeros
        self.b1 = nn.Parameter(torch.zeros(self.hidden_size))

        # b2 (dh): bias in ReLU, init uniform
        self.b2 = nn.Parameter(torch.empty(self.hidden_sh_size).uniform_(-r1, r1))

        # C x_t + b: standard linear (your term, not in repo)
        self.linear_C = nn.Linear(input_size, self.hidden_size, bias=True)

    def forward(self, input, hx=None):
        # input: (seq_len, batch_size, input_size) -- works with seq_len=1
        # hx: (1, batch_size, hidden_size)
        # Returns: output (seq_len, batch_size, hidden_size), new_hx (1, batch_size, hidden_size)

        if hx is None:
            hx = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)

        seq_len, batch_size = input.shape[:2]
        outputs = []
        h = hx[0]  # (batch_size, hidden_size)

        for t in range(seq_len):
            x = input[t]  # (batch_size, input_size)

            # shPLRNN step: A * h + W1 @ ReLU(W2 @ h + b2) + b1 + linear_C(x)
            relu_term = F.relu(
                torch.matmul(h, self.W2.T) + self.b2
            )  # (batch_size, hidden_sh_size)
            h = (
                self.A * h
                + torch.matmul(relu_term, self.W1.T)
                + self.b1
                + self.linear_C(x)
            )

            outputs.append(h)

        outputs = torch.stack(outputs)  # (seq_len, batch_size, hidden_size)
        new_hx = h.unsqueeze(0)  # (1, batch_size, hidden_size)
        return outputs, new_hx
