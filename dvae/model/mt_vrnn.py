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

def build_MT_VRNN_pp(cfg, device='cpu'):

    ### Load parameters for VRNN
    # General
    alphas = [float(i) for i in cfg.get('Network', 'alphas').split(',') if i != '']
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Feature extractor
    dense_x = [] if cfg.get('Network', 'dense_x') == '' else [int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    dense_z = [] if cfg.get('Network', 'dense_z') == '' else [int(i) for i in cfg.get('Network', 'dense_z').split(',')]
    # Dense layers
    dense_hx_z = [] if cfg.get('Network', 'dense_hx_z') == '' else [int(i) for i in cfg.get('Network', 'dense_hx_z').split(',')]
    dense_hz_x = [] if cfg.get('Network', 'dense_hz_x') == '' else [int(i) for i in cfg.get('Network', 'dense_hz_x').split(',')]
    dense_h_z = [] if cfg.get('Network', 'dense_h_z') == '' else [int(i) for i in cfg.get('Network', 'dense_h_z').split(',')]
    # RNN
    dim_RNN = cfg.getint('Network', 'dim_RNN')
    num_RNN = cfg.getint('Network', 'num_RNN')
    type_RNN = cfg.get('Network', 'type_RNN')

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = MT_VRNN(alphas=alphas, x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x=dense_x, dense_z=dense_z,
                 dense_hx_z=dense_hx_z, dense_hz_x=dense_hz_x, 
                 dense_h_z=dense_h_z,
                 dim_RNN=dim_RNN, num_RNN=num_RNN, type_RNN=type_RNN,
                 dropout_p= dropout_p, beta=beta, device=device).to(device)

    return model


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(y):
    return -math.log((1 / y) - 1)



class MT_VRNN(nn.Module):

    def __init__(self, alphas, x_dim, z_dim, activation,
                 dense_x, dense_z,
                 dense_hx_z, dense_hz_x, dense_h_z,
                 dim_RNN, num_RNN=1, type_RNN='RNN',
                 dropout_p=0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters
        self.alphas = alphas
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        self.y_dim = self.x_dim
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device

        ### Feature extractors
        self.dense_x = dense_x
        self.dense_z = dense_z
        ### Dense layers
        self.dense_hx_z = dense_hx_z
        self.dense_hz_x = dense_hz_x
        self.dense_h_z = dense_h_z
        ### RNN
        self.dim_RNN = dim_RNN
        self.num_RNN = num_RNN
        self.type_RNN = type_RNN
        ### Beta-loss
        self.beta = beta

        self.sigmas = nn.Parameter(torch.tensor([inverse_sigmoid(alpha) for alpha in alphas], dtype=torch.float32), requires_grad=True)

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
        # z
        dic_layers = OrderedDict()
        if len(self.dense_z) == 0:
            dim_feature_z = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_z = self.dense_z[-1]
            for n in range(len(self.dense_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z[n-1], self.dense_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_z = nn.Sequential(dic_layers)
        
        ######################
        #### Dense layers ####
        ######################
        # 1. h_t, x_t to z_t (Inference)
        dic_layers = OrderedDict()
        if len(self.dense_hx_z) == 0:
            dim_hx_z = self.dim_RNN + dim_feature_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_z = self.dense_hx_z[-1]
            for n in range(len(self.dense_hx_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[-1] + self.dim_RNN, self.dense_hx_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hx_z[n-1], self.dense_hx_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_hx_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_hx_z, self.z_dim)
        
        # 2. h_t to z_t (Generation z)
        dic_layers = OrderedDict()
        if len(self.dense_h_z) == 0:
            dim_h_z = self.dim_RNN
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_z = self.dense_h_z[-1]
            for n in range(len(self.dense_h_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN, self.dense_h_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_z[n-1], self.dense_h_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_z = nn.Sequential(dic_layers)
        self.prior_mean = nn.Linear(dim_h_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_h_z, self.z_dim)

        # 3. h_t, z_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN + dim_feature_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN + dim_feature_z, self.dense_hz_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hz_x[n-1], self.dense_hz_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_hz_x, self.y_dim)
        
        ####################
        #### Recurrence ####
        ####################
        if self.type_RNN == 'LSTM':
            self.rnn = nn.LSTM(dim_feature_x+dim_feature_z, self.dim_RNN, self.num_RNN)
        elif self.type_RNN == 'RNN':
            self.rnn = nn.RNN(dim_feature_x+dim_feature_z, self.dim_RNN, self.num_RNN)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)


    def generation_x(self, feature_zt, h_t):

        dec_input = torch.cat((feature_zt, h_t), 2)
        dec_output = self.mlp_hz_x(dec_input)
        y_t = self.gen_out(dec_output)

        return y_t
        

    def generation_z(self, h):

        prior_output = self.mlp_h_z(h)
        mean_prior = self.prior_mean(prior_output)
        logvar_prior = self.prior_logvar(prior_output)

        return mean_prior, logvar_prior


    def inference(self, feature_xt, h_t):

        enc_input = torch.cat((feature_xt, h_t), 2)
        enc_output = self.mlp_hx_z(enc_input)
        mean_zt = self.inf_mean(enc_output)
        logvar_zt = self.inf_logvar(enc_output)

        return mean_zt, logvar_zt


    def recurrence(self, feature_xt, feature_zt, h_t, c_t):

        rnn_input = torch.cat((feature_xt, feature_zt), -1)

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

        # create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
        x_features = torch.zeros((seq_len, batch_size, self.dense_x[-1])).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)
        h_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)

        if self.type_RNN == 'LSTM':
            c_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)

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
            mean_zt, logvar_zt = self.inference(feature_xt, h_t_last)

            if inference_mode:
                z_t = mean_zt
            else:
                z_t = self.reparameterization(mean_zt, logvar_zt)

            feature_zt = self.feature_extractor_z(z_t)
            y_t = self.generation_x(feature_zt, h_t_last)
            z_mean[t,:,:] = mean_zt
            z_logvar[t,:,:] = logvar_zt
            z[t,:,:] = torch.squeeze(z_t, 0)
            y[t,:,:] = torch.squeeze(y_t, 0)
            h[t,:,:] = torch.squeeze(h_t_last, 0)


            if self.type_RNN == 'LSTM':
                h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t) # recurrence for t+1 
            elif self.type_RNN == 'RNN':
                h_t, _ = self.recurrence(feature_xt, feature_zt, h_t, None)

        z_mean_p, z_logvar_p  = self.generation_z(h)

        # save all attributes
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.y = y
        self.z = z
        self.h = h
        self.feature_x = feature_x
        self.z_t = z_t
        self.h_t = h_t
        if self.type_RNN == 'LSTM':
            self.c_t = c_t
        self.z_mean_p = z_mean_p
        self.z_logvar_p = z_logvar_p
        
        return self.y

        
    def get_info(self):

        info = []
        info.append("----- Feature extractor -----")
        for layer in self.feature_extractor_x:
            info.append(str(layer))
        for layer in self.feature_extractor_z:
            info.append(str(layer))
        info.append("----- Inference -----")
        for layer in self.mlp_hx_z:
            info.append(str(layer))
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))
        info.append("----- Generation x -----")
        for layer in self.mlp_hz_x:
            info.append(str(layer))
        info.append(str(self.gen_out))
        info.append("----- Recurrence -----")
        info.append(str(self.rnn))
        info.append("----- Generation z -----")
        for layer in self.mlp_h_z:
            info.append(str(layer))
        info.append(str(self.prior_mean))
        info.append(str(self.prior_logvar))

        return info

