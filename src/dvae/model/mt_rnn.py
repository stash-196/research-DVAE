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
from dvae.model.hidden_block import (
    apply_block_alphas,
    apply_trainable_block_alphas,
    block_alpha_init,
)


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
        cfg=cfg,
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
        cfg,
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
            cfg=cfg,
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
        self._install_block_alpha_parameters()

    def _optimize_block_alphas(self) -> bool:
        cfg = getattr(self, "cfg", None)
        if cfg is None or not cfg.has_option("Training", "optimize_alphas"):
            return False
        try:
            return bool(cfg.getboolean("Training", "optimize_alphas"))
        except ValueError:
            return False

    def _install_block_alpha_parameters(self) -> None:
        """Per-block logits, same base-10 map as ``sigmas``.

        Created only when ``optimize_alphas`` is true and that block has a
        list or a range. Otherwise the forward keeps the constant replacement.
        """
        self.heart_sigmas = None
        self.brain_sigmas = None
        block = getattr(self, "hidden_block", None)
        if block is None or not self._optimize_block_alphas():
            return
        heart = block_alpha_init(
            block.heart_alphas, block.heart_alpha_range, block.heart
        )
        brain = block_alpha_init(
            block.brain_alphas, block.brain_alpha_range, block.brain
        )
        if heart is not None:
            self.heart_sigmas = nn.Parameter(
                self.inverse_sigmoid_10(heart.float()).detach().float()
            )
        if brain is not None:
            self.brain_sigmas = nn.Parameter(
                self.inverse_sigmoid_10(brain.float()).detach().float()
            )

    def alpha_parameters(self):
        """Timescale logits. These take ``alpha_lr``, not the base learning rate."""
        yield self.sigmas
        for name in ("heart_sigmas", "brain_sigmas"):
            value = getattr(self, name, None)
            if isinstance(value, nn.Parameter):
                yield value

    def logged_sigmas(self) -> torch.Tensor:
        """Shared logits, then heart, then brain. One row per plotted alpha."""
        parts = [self.sigmas.reshape(-1)]
        for name in ("heart_sigmas", "brain_sigmas"):
            value = getattr(self, name, None)
            if isinstance(value, nn.Parameter):
                parts.append(value.reshape(-1))
        return torch.cat(parts)

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
            per_unit = torch.cat([alphas] * repeats + [alphas[:remainder]])
        else:
            per_unit = alphas[: self.dim_rnn]
        block = getattr(self, "hidden_block", None)
        if block is None:
            return per_unit
        heart = getattr(self, "heart_sigmas", None)
        brain = getattr(self, "brain_sigmas", None)
        heart_on = isinstance(heart, nn.Parameter)
        brain_on = isinstance(brain, nn.Parameter)
        if heart_on or brain_on:
            return apply_trainable_block_alphas(
                per_unit,
                block,
                self.sigmoid_10(heart) if heart_on else None,
                self.sigmoid_10(brain) if brain_on else None,
            )
        return apply_block_alphas(per_unit, block)

    def recurrence(self, feature_xt, h_t, c_t=None):
        # ``h_t`` is the hidden state entering this step. The block mask,
        # when it is on, multiplies the recurrent map that reads that
        # entering state. It does not project the updated state onto the
        # blocks afterwards.
        # The mix keeps that same entering state: (1 - alpha) * h_entering
        # + alpha * h_cell. Alpha is not applied inside the cell.
        h_tp1, c_tp1 = super().recurrence(feature_xt, h_t, c_t)
        alphas = self.alphas_per_unit()
        h_tp1 = (1 - alphas) * h_t + alphas * h_tp1
        return h_tp1, c_tp1


# feature_xt.shape
# torch.Size([1, 128, 100])
# h_t.shape
# torch.Size([1, 128, 64])
