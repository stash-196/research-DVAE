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


def build_RNN(cfg, device='cpu'):

    # Load parameters for VRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Feature extractor
    dense_x = [] if cfg.get('Network', 'dense_x') == '' else [
        int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    # Dense layers
    dense_h_x = [] if cfg.get('Network', 'dense_h_x') == '' else [
        int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]
    # RNN
    dim_rnn = cfg.getint('Network', 'dim_rnn')
    num_rnn = cfg.getint('Network', 'num_rnn')
    type_rnn = cfg.get('Network', 'type_rnn')

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = RNN(x_dim=x_dim, activation=activation,
                dense_x=dense_x,
                dense_h_x=dense_h_x,
                dim_rnn=dim_rnn, num_rnn=num_rnn, type_rnn=type_rnn,
                dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


class RNN(nn.Module):

    def __init__(self, x_dim, activation,
                 dense_x,
                 dense_h_x,
                 dim_rnn, num_rnn, type_rnn,
                 dropout_p=0, beta=1, device='cpu'):

        super().__init__()
        # General parameters
        self.x_dim = x_dim
        self.dropout_p = dropout_p
        self.y_dim = self.x_dim
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        # Feature extractor
        self.dense_x = dense_x
        # Dense layers
        self.dense_h_x = dense_h_x
        # RNN
        self.dim_rnn = dim_rnn
        self.num_rnn = num_rnn
        self.type_rnn = type_rnn
        # Beta-loss
        self.beta = beta

        self.build()

    def build(self):

        ###########################
        #### Feature extractor ####
        ###########################
        # x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_feature_x = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear' +
                               str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear' +
                               str(n)] = nn.Linear(self.dense_x[n-1], self.dense_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_x = nn.Sequential(dic_layers)

        ######################
        #### Dense layers ####
        ######################
        # h_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = self.dim_rnn
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for n in range(len(self.dense_h_x)):
                if n == 0:
                    dic_layers['linear' +
                               str(n)] = nn.Linear(self.dim_rnn, self.dense_h_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(
                        self.dense_h_x[n-1], self.dense_h_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_h_x, self.y_dim)

        ####################
        #### Recurrence ####
        ####################
        if self.type_rnn == 'LSTM':
            self.rnn = nn.LSTM(dim_feature_x, self.dim_rnn, self.num_rnn)
        elif self.type_rnn == 'RNN':
            self.rnn = nn.RNN(dim_feature_x, self.dim_rnn, self.num_rnn)
        else:
            raise SystemExit('Wrong RNN type!')

    def generation_x(self, h_t):
        dec_input = h_t
        dec_output = self.mlp_h_x(dec_input)
        y_t = self.gen_out(dec_output)
        return y_t

    def recurrence(self, feature_xt, h_t, c_t=None):

        rnn_input = feature_xt

        if self.type_rnn == 'LSTM':
            _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
        elif self.type_rnn == 'RNN':
            _, h_tp1 = self.rnn(rnn_input, h_t)
            c_tp1 = None

        return h_tp1, c_tp1

    def forward(self, x, initialize_states=True, update_states=True, mode_selector=None, inference_mode=False, logger=None, from_instance=None):

        # need input:  (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape

        if initialize_states:
            # create variable holder and send to GPU if needed
            y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
            h = torch.zeros((seq_len, batch_size, self.dim_rnn)
                            ).to(self.device)
            h_t = torch.zeros(self.num_rnn, batch_size,
                              self.dim_rnn).to(self.device)
            if self.type_rnn == 'LSTM':
                c_t = torch.zeros(self.num_rnn, batch_size,
                                  self.dim_rnn).to(self.device)
        else:
            y = self.y
            h = self.h
            h_t = self.h_t
            if self.type_rnn == 'LSTM':
                c_t = self.c_t

        # main part
        feature_x = self.feature_extractor_x(x)

        for t in range(seq_len):
            if mode_selector is not None:
                # Calculate the mix of autonomous and teacher-forced inputs
                mix_ratio = mode_selector[t]
            else:
                mix_ratio = 0.0  # Default to full teacher forcing if mode_selector is not provided

            # Generate features for both teacher-forced and autonomous mode
            feature_tf = feature_x[t, :, :].unsqueeze(
                0)  # Teacher-forced feature
            feature_auto = self.feature_extractor_x(
                y_t) if t > 0 else feature_x[0, :, :].unsqueeze(0)  # Autonomous feature

            # Mix the features based on the ratio
            feature_xt = mix_ratio * feature_auto + \
                (1 - mix_ratio) * feature_tf

            h_t_last = h_t.view(self.num_rnn, 1, batch_size,
                                self.dim_rnn)[-1, :, :, :]
            y_t = self.generation_x(h_t_last)
            y[t, :, :] = torch.squeeze(y_t, 0)
            h[t, :, :] = torch.squeeze(h_t_last, 0)

            if self.type_rnn == 'LSTM':
                h_t, c_t = self.recurrence(
                    feature_xt, h_t, c_t)  # recurrence for t+1
            elif self.type_rnn == 'RNN':
                h_t, _ = self.recurrence(feature_xt, h_t)  # recurrence for t+1

        self.y = y
        self.h = h
        self.feature_x = feature_x
        self.h_t = h_t
        if self.type_rnn == 'LSTM':
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
