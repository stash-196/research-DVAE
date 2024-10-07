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
from dvae.utils.logger import print_or_log
from dvae.utils.model_mode_selector import prepare_mode_selector
from dvae.model.base_vrnn import BaseVRNN


def build_VRNN(cfg, device="cpu"):

    # Load parameters for VRNN
    # General
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
    model = VRNN(
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


class VRNN(BaseVRNN):
    def __init__(
        self,
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
