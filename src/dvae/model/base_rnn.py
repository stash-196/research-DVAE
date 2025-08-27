#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for RNNs.
"""

from torch import nn
import torch
from collections import OrderedDict
from dvae.utils.model_mode_selector import prepare_mode_selector
from dvae.model.base_model import BaseModel


class BaseRNN(BaseModel):
    def __init__(
        self,
        x_dim,
        activation,
        dense_x,
        dense_h_x,
        dim_rnn,
        num_rnn,
        type_rnn,
        dropout_p,
        device,
    ):
        super().__init__(x_dim, activation, dropout_p, device)
        self.dense_x = dense_x
        self.dense_h_x = dense_h_x
        self.dim_rnn = dim_rnn
        self.num_rnn = num_rnn
        self.type_rnn = type_rnn
        self.build()

    def build(self):
        self.feature_extractor_x, _ = self._build_sequential_layers(
            self.dense_x, self.x_dim
        )
        self.mlp_h_x, last_dim = self._build_sequential_layers(
            self.dense_h_x, self.dim_rnn
        )
        self.gen_out = nn.Linear(last_dim, self.y_dim)
        self.rnn = self._build_recurrence()

    def _build_recurrence(self):
        input_dim = self.dense_x[-1] if self.dense_x else self.x_dim
        if self.type_rnn == "LSTM":
            return nn.LSTM(input_dim, self.dim_rnn, self.num_rnn)
        elif self.type_rnn == "RNN":
            return nn.RNN(input_dim, self.dim_rnn, self.num_rnn)
        elif self.type_rnn == "PLRNN":
            return nn.RNN(input_dim, self.dim_rnn)
        else:
            raise ValueError("Unsupported RNN type!")

    def generation_x(self, h_t):
        dec_output = self.mlp_h_x(h_t)
        y_t = self.gen_out(dec_output)
        return y_t

    def recurrence(self, feature_xt, h_t, c_t=None):
        # feature_xt: (1, batch_size, input_size)
        if self.type_rnn == "LSTM":
            _, (h_tp1, c_tp1) = self.rnn(feature_xt, (h_t, c_t))
            return h_tp1, c_tp1
        elif self.type_rnn == "RNN":
            _, h_tp1 = self.rnn(feature_xt, h_t)
            return h_tp1, None
        else:
            raise ValueError("Unsupported RNN type for recurrence!")

    def forward(
        self,
        x,
        initialize_states=True,
        mode_selector=None,
        inference_mode=False,
        logger=None,
        from_instance=None,
    ):
        seq_len, batch_size, _ = x.shape

        # Initialize or retrieve states
        if initialize_states:
            y = torch.zeros((seq_len, batch_size, self.y_dim), device=self.device)
            h = torch.zeros((seq_len, batch_size, self.dim_rnn), device=self.device)
            h_t = torch.zeros(
                self.num_rnn, batch_size, self.dim_rnn, device=self.device
            )
            if self.type_rnn == "LSTM":
                c_t = torch.zeros(
                    self.num_rnn, batch_size, self.dim_rnn, device=self.device
                )
            feature_x = torch.zeros(
                (seq_len, batch_size, self.dense_x[-1] if self.dense_x else self.x_dim),
                device=self.device,
            )
        else:
            y = self.y
            h = self.h
            h_t = self.h_t
            if self.type_rnn == "LSTM":
                c_t = self.c_t
            feature_x = self.feature_x

        # Prepare mode selector
        mode_selector = self.prepare_mode_selector(
            mode_selector, seq_len, batch_size, self.x_dim, self.device
        )

        for t in range(seq_len):
            input_t = x[t, :, :]  # Shape: (batch_size, x_dim)
            mode_selector_t = mode_selector[t, :, :]  # Shape: (batch_size, x_dim)

            # Impute NaNs with previous output
            imputed_input_t = self.impute_inputs_nans_with_output(
                input_t, y_t if t > 0 else None, t
            )

            # Mix inputs based on mode_selector and extract features
            mixed_input_t = self.mix_inputs_with_outputs(
                imputed_input_t, y_t if t > 0 else None, mode_selector_t, t
            )

            feature_xt = self.feature_extractor_x(mixed_input_t)

            # Proceed with model-specific methods
            h_t_last = h_t[-1]  # Shape: (batch_size, dim_rnn)
            y_t = self.generation_x(h_t_last)  # Expects h_t_last: (batch_size, dim_rnn)
            y[t, :, :] = y_t
            h[t, :, :] = h_t_last
            feature_x[t, :, :] = feature_xt

            # Recurrence
            if self.type_rnn == "LSTM":
                h_t, c_t = self.recurrence(feature_xt.unsqueeze(0), h_t, c_t)
            elif self.type_rnn == "RNN":
                h_t, _ = self.recurrence(feature_xt.unsqueeze(0), h_t)

        # Save states
        self.y = y
        self.h = h
        self.feature_x = feature_x
        self.h_t = h_t
        if self.type_rnn == "LSTM":
            self.c_t = c_t

        return y

    def get_info(self):
        info = []
        info.append("----- Feature extractor -----")
        for layer in self.feature_extractor_x:
            info.append(str(layer))
        info.append("----- Generation x -----")
        for layer in self.mlp_h_x:
            info.append(str(layer))
        info.append(str(self.gen_out))
        info.append("----- Recurrence -----")
        info.append(str(self.rnn))
        return info
