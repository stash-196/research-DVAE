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

    ### Load parameters for VRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Dense layers
    dense_h_x = [] if cfg.get('Network', 'dense_h_x') == '' else [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]
    # RNN
    dim_RNN = cfg.getint('Network', 'dim_RNN')
    num_RNN = cfg.getint('Network', 'num_RNN')

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = RNN(x_dim=x_dim, activation=activation,
                 dense_h_x=dense_h_x, 
                 dim_RNN=dim_RNN, num_RNN=num_RNN,
                 dropout_p= dropout_p, beta=beta, device=device).to(device)

    return model


    
class RNN(nn.Module):

    def __init__(self, x_dim, activation = 'tanh',
                 dense_x=[128],
                 dense_h_x=[128],
                 dim_RNN=128, num_RNN=1, type_RNN='RNN',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters
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
        ### Feature extractor
        self.dense_x = dense_x
        ### Dense layers
        self.dense_h_x = dense_h_x
        ### RNN
        self.dim_RNN = dim_RNN
        self.num_RNN = num_RNN
        self.type_RNN = type_RNN
        ### Beta-loss
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
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[n-1], self.dense_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_x = nn.Sequential(dic_layers)

        ######################
        #### Dense layers ####
        ######################
        # h_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = self.dim_RNN
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for n in range(len(self.dense_h_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN, self.dense_h_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_x[n-1], self.dense_h_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_h_x, self.y_dim)
        
        ####################
        #### Recurrence ####
        ####################
        if self.type_RNN == 'LSTM':
            self.rnn = nn.LSTM(dim_feature_x, self.dim_RNN, self.num_RNN)
        elif self.type_RNN == 'RNN':
            self.rnn = nn.RNN(dim_feature_x, self.dim_RNN, self.num_RNN)
        else:
            raise SystemExit('Wrong RNN type!')


    def generation_x(self, h_t):
        dec_input = h_t
        dec_output = self.mlp_h_x(dec_input)
        y_t = self.gen_out(dec_output)
        return y_t
        


    def recurrence(self, feature_xt, h_t, c_t=None):

        rnn_input = feature_xt

        if self.type_RNN == 'LSTM':
            _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
        elif self.type_RNN == 'RNN':
            _, h_tp1 = self.rnn(rnn_input, h_t)
            c_tp1 = None

        return h_tp1, c_tp1


    def forward(self, x):

        # need input:  (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape

        # create variable holder and send to GPU if needed
        self.y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
        self.h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.device)
        h_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)
        if self.type_RNN == 'LSTM':
            c_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)

        # main part
        feature_x = self.feature_extractor_x(x)
        for t in range(seq_len):
            feature_xt = feature_x[t,:,:].unsqueeze(0)
            h_t_last = h_t.view(self.num_RNN, 1, batch_size, self.dim_RNN)[-1,:,:,:]
            y_t = self.generation_x(h_t_last)
            self.y[t,:,:] = torch.squeeze(y_t)
            self.h[t,:,:] = torch.squeeze(h_t_last)

            if self.type_RNN == 'LSTM':
                h_t, c_t = self.recurrence(feature_xt, h_t, c_t) # recurrence for t+1 
            elif self.type_RNN == 'RNN':
                h_t, _ = self.recurrence(feature_xt, h_t) # recurrence for t+1

        return self.y

        
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


