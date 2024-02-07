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

def build_MT_RNN(cfg, device='cpu'):

    ### Load parameters for VRNN
    # General
    # slit the alphas string if it contains , if not, just use the value
    alphas = [float(i) for i in cfg.get('Network', 'alphas').split(',') if i != '']

    x_dim = cfg.getint('Network', 'x_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Feature extractor
    dense_x = [] if cfg.get('Network', 'dense_x') == '' else [int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    # Dense layers
    dense_h_x = [] if cfg.get('Network', 'dense_h_x') == '' else [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]
    # RNN
    dim_RNN = cfg.getint('Network', 'dim_RNN')
    num_RNN = cfg.getint('Network', 'num_RNN')
    type_RNN = cfg.get('Network', 'type_RNN')

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = MT_RNN(alphas=alphas, x_dim=x_dim, activation=activation,
                   dense_x=dense_x, 
                   dense_h_x=dense_h_x, 
                   dim_RNN=dim_RNN, num_RNN=num_RNN, type_RNN=type_RNN,
                   dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model



def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(y):
    return -math.log((1 / y) - 1)


class MT_RNN(nn.Module):

    def __init__(self, alphas, x_dim, activation,
                 dense_x,
                 dense_h_x,
                 dim_RNN, num_RNN=1, type_RNN='RNN',
                 dropout_p=0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters
        self.alphas = alphas
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
        # # Create an array of alphas for each hidden unit
        # Convert alphas to sigma using inverse sigmoid
        self.sigmas = nn.Parameter(torch.tensor([inverse_sigmoid(alpha) for alpha in alphas], dtype=torch.float32), requires_grad=True)

        # self.register_buffer('alphas_per_unit', self.assign_alpha_per_unit())
        # self.alphas_per_unit = nn.Parameter(self.assign_alpha_per_unit())

        # Build the model
        self.build()


    def base_parameters(self):
        return (p for name, p in self.named_parameters() if 'sigmas' not in name)

    
    def alphas_per_unit(self):
        # Convert sigma to alpha using sigmoid
        alphas = torch.sigmoid(self.sigmas)
        
        # If the number of hidden units is greater than the number of alphas,
        # distribute the alphas evenly among the hidden units.
        if self.dim_RNN > len(alphas):
            num_repeats = self.dim_RNN // len(alphas)
            remainder = self.dim_RNN % len(alphas)
            return torch.cat([alphas] * num_repeats + [alphas[:remainder]])
        else:
            return alphas[:self.dim_RNN]



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


        # # Add the weight initialization here
        # for name, param in self.rnn.named_parameters():
        #     if 'weight_ih' in name:
        #         torch.nn.init.xavier_uniform_(param.data)
        #     elif 'weight_hh' in name:
        #         torch.nn.init.orthogonal_(param.data)
        #     elif 'bias' in name:
        #         param.data.fill_(0)



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

        h_tp1 = (1 - self.alphas_per_unit()) * h_t + self.alphas_per_unit() * h_tp1

        return h_tp1, c_tp1


    def forward(self, x, initialize_states=True, update_states=True, mode_selector=None, inference_mode=False):

        # need input:  (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape

        if initialize_states:
            # create variable holder and send to GPU if needed
            y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
            h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.device)
            h_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)
            if self.type_RNN == 'LSTM':
                c_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)
        else:
            y = self.y
            h = self.h
            h_t = self.h_t
            if self.type_RNN == 'LSTM':
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
            feature_tf = feature_x[t,:,:].unsqueeze(0)  # Teacher-forced feature
            feature_auto = self.feature_extractor_x(y_t) if t > 0 else feature_x[0,:,:].unsqueeze(0)  # Autonomous feature

            # Mix the features based on the ratio
            feature_xt = mix_ratio * feature_auto + (1 - mix_ratio) * feature_tf

            h_t_last = h_t.view(self.num_RNN, 1, batch_size, self.dim_RNN)[-1,:,:,:]
            y_t = self.generation_x(h_t_last)
            y[t,:,:] = torch.squeeze(y_t, 0)
            h[t,:,:] = torch.squeeze(h_t_last, 0)

            if self.type_RNN == 'LSTM':
                h_t, c_t = self.recurrence(feature_xt, h_t, c_t) # recurrence for t+1 
            elif self.type_RNN == 'RNN':
                h_t, _ = self.recurrence(feature_xt, h_t)
        
        # save states
        self.y = y
        self.h = h
        self.feature_x = feature_x
        self.h_t = h_t
        if self.type_RNN == 'LSTM':
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


