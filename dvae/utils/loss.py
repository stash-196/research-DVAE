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

def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0, clamp_min=1e-6, clamp_max=1e6, logger=None, from_instance=None):
    # Ensure z_logvar_p is within a reasonable range
    z_logvar_p_exp = z_logvar_p.exp().clamp(min=clamp_min, max=clamp_max)
    z_logvar_exp = z_logvar.exp().clamp(min=clamp_min, max=clamp_max)

    # Logging values for debugging
    print_or_log(f"[KLD] z_logvar: min={z_logvar.min()}, max={z_logvar.max()}", logger=logger, from_instance=from_instance)
    print_or_log(f"[KLD] z_logvar_p: min={z_logvar_p.min()}, max={z_logvar_p.max()}", logger=logger, from_instance=from_instance)
    print_or_log(f"[KLD] z_mean: min={z_mean.min()}, max={z_mean.max()}", logger=logger, from_instance=from_instance)
    print_or_log(f"[KLD] z_mean_p: min={z_mean_p.min()}, max={z_mean_p.max()}", logger=logger, from_instance=from_instance)

    # Print warning if any values are NaN
    if torch.isnan(z_logvar).any().item() or torch.isnan(z_logvar_p).any().item() or torch.isnan(z_mean).any().item() or torch.isnan(z_mean_p).any().item():
        print_or_log(f"[KLD] Warning: NaN values...> z_logvar: {torch.sum(torch.isnan(z_logvar))}, z_logvar_p: {torch.sum(torch.isnan(z_logvar_p))}, z_mean: {torch.sum(torch.isnan(z_mean))}, z_mean_p: {torch.sum(torch.isnan(z_mean_p))}", logger=logger, from_instance=from_instance, log_type="warning")

    # Check if any values need to be clamped
    clamped_min = (z_logvar_p_exp < clamp_min).any().item() or (z_logvar_exp < clamp_min).any().item()
    clamped_max = (z_logvar_p_exp > clamp_max).any().item() or (z_logvar_exp > clamp_max).any().item()
    
    # Print warning if clamping occurred
    if clamped_min or clamped_max:
        print_or_log(f"[KLD] Warning: Clamping applied to avoid numerical instability.", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_p_exp Min Values: {torch.sum(z_logvar_p_exp < clamp_min)}, z_logvar_p_exp Max Values: {torch.sum(z_logvar_p_exp > clamp_max)}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_exp Min Values: {torch.sum(z_logvar_exp < clamp_min)}, z_logvar_exp Max Values: {torch.sum(z_logvar_exp > clamp_max)}", logger=logger, from_instance=from_instance, log_type="warning")

    # Compute the KL divergence loss
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p - torch.div(z_logvar_exp + (z_mean - z_mean_p).pow(2), z_logvar_p_exp))
    
    # Print ret to see if it's NaN
    if torch.isnan(ret).any().item():
        print_or_log(f"[KLD] Warning: KL divergence loss is NaN. NaN values...> z_logvar: {torch.sum(torch.isnan(z_logvar))}, z_logvar_p: {torch.sum(torch.isnan(z_logvar_p))}, z_mean: {torch.sum(torch.isnan(z_mean))}, z_mean_p: {torch.sum(torch.isnan(z_mean_p))}", logger, from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_p_exp NaN Values: {torch.sum(torch.isnan(z_logvar_p_exp))}, z_logvar_exp NaN Values: {torch.sum(torch.isnan(z_logvar_exp))}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_p_exp <{clamp_min} Values: {torch.sum(z_logvar_p_exp < clamp_min)}, z_logvar_p_exp >{clamp_max} Values: {torch.sum(z_logvar_p_exp > clamp_max)}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_p_exp Min Values: {torch.min(z_logvar_p_exp)}, z_logvar_p_exp Max Values: {torch.max(z_logvar_p_exp)}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_exp NaN Values: {torch.sum(torch.isnan(z_logvar_exp))}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_exp <{clamp_min} Values: {torch.sum(z_logvar_exp < clamp_min)}, z_logvar_exp >{clamp_max} Values: {torch.sum(z_logvar_exp > clamp_max)}", logger=logger, from_instance=from_instance, log_type="warning")
        print_or_log(f"[KLD] |-> z_logvar_exp Min Values: {torch.min(z_logvar_exp)}, z_logvar_exp Max Values: {torch.max(z_logvar_exp)}", logger=logger, from_instance=from_instance, log_type="warning")
    
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



def print_or_log(str, logger=None, from_instance=None, log_type="info"):
    if from_instance:
        prefix = f"{from_instance} "
    else:
        prefix = ""

    if logger:
        if log_type == "info":
            logger.info(prefix + str)
        elif log_type == "warning":
            logger.warning(prefix + str)
    else:
        print(prefix + str)
