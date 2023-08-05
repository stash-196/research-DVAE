#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""
import torch

def loss_ISD(x, y):
    # Itakura-Saito divergence
    # loss can be used when you want to encourage similarity or dissimilarity between two probability distributions.
    y = y + 1e-10
    ret = torch.sum( x/y - torch.log(x/y) - 1)
    return ret

def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    # This loss is used when you want to encourage the encoded distribution to match a prior distribution.

    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret

def loss_JointNorm(x, y, nfeats=3):
    #  This loss is useful when you want to penalize differences between corresponding elements in two sequences, often used in sequence-to-sequence tasks or alignment problems.
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = torch.sum(torch.norm(x-y, dim=-1))
    return ret

def loss_MPJPE(x, y, nfeats=3):
    # commonly used in tasks such as human pose estimation to assess the accuracy of predicted joint positions compared to ground truth.
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = (x-y).norm(dim=-1).mean(dim=-1).sum()
    return ret


""" 
The reconstruction loss (negative log likelihood of the data under the model's 
distribution) is computed. This is often done with a mean squared error (MSE) 
or binary cross-entropy loss for real-valued or binary data respectively.
"""

def loss_MSE(x, y):
    # commonly used for regression tasks.
    ret = torch.sum((x-y).pow(2))
    return ret

# def loss_ISD(x, y):
#     seq_len, bs, _ = x.shape
#     ret = torch.sum( x/y - torch.log(x/y) - 1)
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
#     if len(z_mean.shape) == 3:
#         seq_len, bs, _ = z_mean.shape
#     elif len(z_mean.shape) == 2:
#         seq_len = 1
#         bs, _ = z_mean.shape
#     ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
#                 - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_JointNorm(x, y, nfeats=3):
#     seq_len, bs, _ = x.shape
#     x = x.reshape(seq_len, bs, -1, nfeats)
#     y = y.reshape(seq_len, bs, -1, nfeats)
#     return torch.mean(torch.norm(x-y, dim=-1))



