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
"""

from torch import nn
import torch
from collections import OrderedDict
from dvae.model.base_rnn import BaseRNN


def build_RNN(cfg, device="cpu"):

    # Load parameters for VRNN
    # General
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
    model = RNN(
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


class RNN(BaseRNN):
    def __init__(
        self,
        x_dim,
        activation,
        dense_x,
        dense_h_x,
        dim_rnn,
        num_rnn,
        type_rnn,
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
        self.beta = beta

    def base_parameters(self):
        return (p for name, p in self.named_parameters() if "sigmas" not in name)
