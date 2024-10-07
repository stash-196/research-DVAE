#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “A Recurrent Latent Variable Model for Sequential Data” ICLR, 2015, Junyoung Chung et al.
WITH PROPER PRIORS (VRNN_pp)
"""

from torch import nn
import torch
from collections import OrderedDict
import math
from dvae.model.base_vrnn import BaseVRNN


def build_MT_VRNN(cfg, device="cpu"):

    # Load parameters for VRNN
    # General
    alphas = [float(i) for i in cfg.get("Network", "alphas").split(",") if i != ""]
    x_dim = cfg.getint("Network", "x_dim")
    z_dim = cfg.getint("Network", "z_dim")
    activation = cfg.get("Network", "activation")
    dropout_p = cfg.getfloat("Network", "dropout_p")
    # Feature extractor
    dense_x = (
        []
        if cfg.get("Network", "dense_x") == ""
        else [int(i) for i in cfg.get("Network", "dense_x").split(",")]
    )
    dense_z = (
        []
        if cfg.get("Network", "dense_z") == ""
        else [int(i) for i in cfg.get("Network", "dense_z").split(",")]
    )
    # Dense layers
    dense_hx_z = (
        []
        if cfg.get("Network", "dense_hx_z") == ""
        else [int(i) for i in cfg.get("Network", "dense_hx_z").split(",")]
    )
    dense_hz_x = (
        []
        if cfg.get("Network", "dense_hz_x") == ""
        else [int(i) for i in cfg.get("Network", "dense_hz_x").split(",")]
    )
    dense_h_z = (
        []
        if cfg.get("Network", "dense_h_z") == ""
        else [int(i) for i in cfg.get("Network", "dense_h_z").split(",")]
    )
    # RNN
    dim_rnn = cfg.getint("Network", "dim_rnn")
    num_rnn = cfg.getint("Network", "num_rnn")
    type_rnn = cfg.get("Network", "type_rnn")

    # Beta-vae
    beta = cfg.getfloat("Training", "beta")

    # Build model
    model = MT_VRNN(
        alphas=alphas,
        x_dim=x_dim,
        z_dim=z_dim,
        activation=activation,
        dense_x=dense_x,
        dense_z=dense_z,
        dense_hx_z=dense_hx_z,
        dense_hz_x=dense_hz_x,
        dense_h_z=dense_h_z,
        dim_rnn=dim_rnn,
        num_rnn=num_rnn,
        type_rnn=type_rnn,
        dropout_p=dropout_p,
        beta=beta,
        device=device,
    ).to(device)

    return model


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inverse_sigmoid(y):
    return -math.log((1 / y) - 1)


class MT_VRNN(BaseVRNN):
    def __init__(
        self,
        alphas,
        x_dim,
        z_dim,
        activation,
        dense_x,
        dense_z,
        dense_hx_z,
        dense_hz_x,
        dense_h_z,
        dim_rnn,
        num_rnn=1,
        type_rnn="RNN",
        dropout_p=0,
        beta=1,
        device="cpu",
    ):
        super().__init__(
            x_dim,
            z_dim,
            activation,
            dense_x,
            dense_z,
            dense_hx_z,
            dense_hz_x,
            dense_h_z,
            dim_rnn,
            num_rnn,
            type_rnn,
            dropout_p,
            device,
        )
        # Beta-loss
        self.beta = beta

        self.alphas = alphas
        self.sigmas = nn.Parameter(
            torch.tensor(
                [inverse_sigmoid(alpha) for alpha in alphas], dtype=torch.float32
            ),
            requires_grad=True,
        )

    @staticmethod
    def inverse_sigmoid(y):
        return -math.log((1 / y) - 1)

    def base_parameters(self):
        return (p for name, p in self.named_parameters() if "sigmas" not in name)

    def alphas_per_unit(self):
        # Convert sigma to alpha using sigmoid
        alphas = torch.sigmoid(self.sigmas)
        # If the number of hidden units is greater than the number of alphas,
        # distribute the alphas evenly among the hidden units.
        if self.dim_rnn > len(alphas):
            repeats = self.dim_rnn // len(alphas)
            remainder = self.dim_rnn % len(alphas)
            return torch.cat([alphas] * repeats + [alphas[:remainder]])
        else:
            return alphas[: self.dim_rnn]

    def recurrence(self, feature_xt, feature_zt, h_t, c_t=None):
        # rnn_input Shape: (1, batch_size, input_size)
        rnn_input = torch.cat((feature_xt, feature_zt), -1)
        if self.type_rnn == "LSTM":
            _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
        elif self.type_rnn == "RNN":
            _, h_tp1 = self.rnn(rnn_input, h_t)
            c_tp1 = None
        h_tp1 = (1 - self.alphas_per_unit()) * h_t + self.alphas_per_unit() * h_tp1
        return h_tp1, c_tp1
