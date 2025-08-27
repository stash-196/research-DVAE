#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements a Multi-timescale Recurrent Neural Network (RNN).

Attributes:
- alphas (torch.Tensor): Timescales for each hidden unit.
- x_dim (int): Dimensionality of the input feature vector.
- ... [Other attributes]

Methods:
- assign_alpha_per_unit: Assigns alphas to each hidden unit.
- build: Constructs the neural network layers.
- generation_x: Generates output based on the hidden state.
- recurrence: Computes the next hidden state.
- forward: Defines the forward pass of the RNN.
- get_info: Returns information about the RNN's architecture.

The code in this file is based on:
- “A Recurrent Latent Variable Model for Sequential Data” ICLR, 2015, Junyoung Chung et al.
"""

from torch import nn
import torch
from collections import OrderedDict
import math
from dvae.model.base_rnn import BaseRNN


def build_MT_RNN(cfg, device="cpu"):

    # Load parameters for VRNN
    # General
    # slit the alphas string if it contains , if not, just use the value
    alphas = [float(i) for i in cfg.get("Network", "alphas").split(",") if i != ""]

    x_dim = cfg.getint("Network", "x_dim")
    activation = cfg.get("Network", "activation")
    dropout_p = cfg.getfloat("Network", "dropout_p")
    # Feature extractor
    dense_x = (
        []
        if cfg.get("Network", "dense_x") == ""
        else [int(i) for i in cfg.get("Network", "dense_x").split(",")]
    )
    # Dense layers
    dense_h_x = (
        []
        if cfg.get("Network", "dense_h_x") == ""
        else [int(i) for i in cfg.get("Network", "dense_h_x").split(",")]
    )
    # RNN
    dim_rnn = cfg.getint("Network", "dim_rnn")
    num_rnn = cfg.getint("Network", "num_rnn")
    type_rnn = cfg.get("Network", "type_rnn")

    # Beta-vae
    beta = cfg.getfloat("Training", "beta")

    # Build model
    model = MT_RNN(
        alphas=alphas,
        x_dim=x_dim,
        activation=activation,
        dense_x=dense_x,
        dense_h_x=dense_h_x,
        dim_rnn=dim_rnn,
        num_rnn=num_rnn,
        type_rnn=type_rnn,
        dropout_p=dropout_p,
        beta=beta,
        device=device,
    ).to(device)

    return model


class MT_RNN(BaseRNN):
    def __init__(
        self,
        alphas,
        x_dim,
        activation,
        dense_x,
        dense_h_x,
        dim_rnn,
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        beta=1,
        device="cpu",
    ):
        super().__init__(
            x_dim,
            activation,
            dense_x,
            dense_h_x,
            dim_rnn,
            num_rnn,
            type_rnn,
            dropout_p,
            device,
        )

        self.alphas = alphas
        self.sigmas = nn.Parameter(
            torch.tensor(
                [self.inverse_sigmoid_10(alpha) for alpha in alphas],
                dtype=torch.float32,
            ),
            requires_grad=True,
        )

        self.beta = beta

    @staticmethod
    def inverse_sigmoid_10(y):
        y_tensor = torch.as_tensor(y)
        return -torch.log10((1 / y_tensor) - 1)

    @staticmethod
    def sigmoid_10(x):
        return 1 / (1 + 10 ** (-x))

    def base_parameters(self):
        return (p for name, p in self.named_parameters() if "sigmas" not in name)

    def alphas_per_unit(self):
        # Convert sigma to alpha using sigmoid
        alphas = self.sigmoid_10(self.sigmas)
        # If the number of hidden units is greater than the number of alphas,
        # distribute the alphas evenly among the hidden units.
        if self.dim_rnn > len(alphas):
            repeats = self.dim_rnn // len(alphas)
            remainder = self.dim_rnn % len(alphas)
            return torch.cat([alphas] * repeats + [alphas[:remainder]])
        else:
            return alphas[: self.dim_rnn]

    def recurrence(self, feature_xt, h_t, c_t=None):
        if self.type_rnn == "LSTM":
            _, (h_tp1, c_tp1) = self.rnn(feature_xt, (h_t, c_t))
        elif self.type_rnn == "RNN":
            _, h_tp1 = self.rnn(feature_xt, h_t)
            c_tp1 = None
        h_tp1 = (1 - self.alphas_per_unit()) * h_t + self.alphas_per_unit() * h_tp1
        return h_tp1, c_tp1


# feature_xt.shape
# torch.Size([1, 128, 100])
# h_t.shape
# torch.Size([1, 128, 64])
