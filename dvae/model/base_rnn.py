#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for RNNs.
"""

from torch import nn
import torch
from collections import OrderedDict


class BaseRNN(nn.Module):
    def __init__(self, x_dim, activation, dense_x, dense_h_x, dim_rnn, num_rnn, type_rnn, dropout_p, device):
        super().__init__()
        self.x_dim = x_dim
        self.activation = self._get_activation(activation)
        self.dense_x = dense_x
        self.dense_h_x = dense_h_x
        self.dim_rnn = dim_rnn
        self.num_rnn = num_rnn
        self.type_rnn = type_rnn
        self.dropout_p = dropout_p
        self.device = device
        self.build()

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')

    def build(self):
        self.feature_extractor_x = self._build_feature_extractor()
        self.mlp_h_x, self.gen_out = self._build_dense_layers()
        self.rnn = self._build_recurrence()

    def _build_feature_extractor(self):
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dic_layers['Identity'] = nn.Identity()
        else:
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[n-1], self.dense_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        return nn.Sequential(dic_layers)

    def _build_dense_layers(self):
        dic_layers = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = self.dim_rnn
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for n in range(len(self.dense_h_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_rnn, self.dense_h_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_x[n-1], self.dense_h_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        mlp_h_x = nn.Sequential(dic_layers)
        gen_out = nn.Linear(dim_h_x, self.x_dim)
        return mlp_h_x, gen_out

    def _build_recurrence(self):
        if self.type_rnn == 'LSTM':
            return nn.LSTM(self.dense_x[-1] if len(self.dense_x) > 0 else self.x_dim, self.dim_rnn, self.num_rnn)
        elif self.type_rnn == 'RNN':
            return nn.RNN(self.dense_x[-1] if len(self.dense_x) > 0 else self.x_dim, self.dim_rnn, self.num_rnn)
        else:
            raise SystemExit('Wrong RNN type!')

    def forward(self, x, initialize_states=True, update_states=True, mode_selector=None, inference_mode=False):
        raise NotImplementedError("Subclasses should implement this method")

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
