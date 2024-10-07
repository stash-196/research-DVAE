#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import torch
import math

class MultiTimescaleMixin:
    def __init__(self, alphas, dim_rnn):
        super().__init__()
        self.alphas = alphas
        self.sigmas = nn.Parameter(torch.tensor([self.inverse_sigmoid(alpha) for alpha in alphas], dtype=torch.float32), requires_grad=True)
        self.dim_rnn = dim_rnn
    
    def alphas_per_unit(self):
        alphas = torch.sigmoid(self.sigmas)
        if self.dim_rnn > len(alphas):
            num_repeats = self.dim_rnn // len(alphas)
            remainder = self.dim_rnn % len(alphas)
            return torch.cat([alphas] * num_repeats + [alphas[:remainder]])
        else:
            return alphas[:self.dim_rnn]

    def recurrence(self, feature_xt, h_t, c_t=None):
        h_tp1, c_tp1 = super().recurrence(feature_xt, h_t, c_t)
        h_tp1 = (1 - self.alphas_per_unit()) * h_t + self.alphas_per_unit() * h_tp1
        return h_tp1, c_tp1

    @staticmethod
    def inverse_sigmoid(y):
        return -math.log((1 / y) - 1)
