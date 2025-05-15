#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import numpy as np
# from pypesq import pesq
# from pystoi import stoi


def compute_median(data):
    median = np.median(data, axis=0)
    q75, q25 = np.quantile(data, [.75 ,.25], axis=0)    
    iqr = q75 - q25
    CI = 1.57*iqr/np.sqrt(data.shape[0])
    if np.any(np.isnan(data)):
        raise NameError('nan in data')
    return median, CI


def compute_rmse(x_est, x_ref):

    # scaling, to get minimum nomrlized-rmse
    alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
    # x_est_ = np.expand_dims(x_est, axis=1)
    # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
    x_est_scaled = alpha * x_est

    return np.sqrt(np.square(x_est_scaled - x_ref).mean())


"""
as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
and one estimate.
see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
"""
# def compute_sisdr(x_est, x_ref):

#     eps = np.finfo(x_est.dtype).eps
#     reference = x_ref.reshape(x_ref.size, 1)
#     estimate = x_est.reshape(x_est.size, 1)
#     Rss = np.dot(reference.T, reference)
#     # get the scaling factor for clean sources
#     a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

#     e_true = a * reference
#     e_res = estimate - e_true

#     Sss = (e_true**2).sum()
#     Snn = (e_res**2).sum()

#     return 10 * np.log10((eps+ Sss)/(eps + Snn))


# class EvalMetrics():

#     def __init__(self, metric='all'):

#         self.metric = metric

#     def eval(self, audio_est, audio_ref):

#         x_est, fs_est = sf.read(audio_est)
#         x_ref, fs_ref = sf.read(audio_ref)
#         # mono channel
#         if len(x_est.shape) > 1:
#             x_est = x_est[:,0]
#         if len(x_ref.shape) > 1:
#             x_ref = x_ref[:,0]
#         # align
#         len_x = np.min([len(x_est), len(x_ref)])
#         x_est = x_est[:len_x]
#         x_ref = x_ref[:len_x]

#         # x_ref = x_ref / np.max(np.abs(x_ref))

#         if fs_est != fs_ref:
#             raise ValueError('Sampling rate is different amon estimated audio and reference audio')

#         if self.metric  == 'rmse':
#             return compute_rmse(x_est, x_ref)
#         elif self.metric == 'sisdr':
#             return compute_sisdr(x_est, x_ref)
#         elif self.metric == 'pesq':
#             return pesq(x_ref, x_est, fs_est)
#         elif self.metric == 'stoi':
#             return stoi(x_ref, x_est, fs_est, extended=False)
#         elif self.metric == 'estoi':
#             return stoi(x_ref, x_est, fs_est, extended=True)
#         elif self.metric == 'all':
#             score_rmse = compute_rmse(x_est, x_ref)
#             score_sisdr = compute_sisdr(x_est, x_ref)
#             score_pesq = pesq(x_ref, x_est, fs_est)
#             score_estoi = stoi(x_ref, x_est, fs_est, extended=True)
#             return score_rmse, score_sisdr, score_pesq, score_estoi
#         else:
#             raise ValueError('Evaluation only support: rmse, pesq, (e)stoi, all')

import torch
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter1d
from torch.fft import rfft
from sklearn.mixture import GaussianMixture
from torch.nn.functional import mse_loss

def laplace_smoothing(hist, alpha=1e-5):
    """Apply Laplace smoothing to the histogram."""
    return (hist + alpha) / (hist.sum() + alpha * np.prod(hist.shape))

def state_space_divergence(X, X_hat, n_bins=10, binning='default', sigma_squared=None, max_T=10000, mc_samples=1000):
    """Compute state space divergence between two time series."""
    if binning == 'default':
        lo = X.min(axis=1, keepdims=True) - 0.1 * X.std(axis=1, keepdims=True)
        hi = X.max(axis=1, keepdims=True) + 0.1 * X.std(axis=1, keepdims=True)
    elif binning == 'legacy':
        sigma = np.sqrt(sigma_squared) if sigma_squared is not None else X.std(axis=1, keepdims=True)
        lo, hi = -2 * sigma, 2 * sigma
    else:
        raise ValueError("Unsupported binning scheme.")

    # Bin edges for histogram
    bin_edges = np.linspace(lo, hi, n_bins + 1, axis=1)

    # Compute histograms
    hist_true, _ = np.histogramdd(X.T, bins=bin_edges)
    hist_gen, _ = np.histogramdd(X_hat.T, bins=bin_edges)

    # Apply Laplace smoothing
    p = laplace_smoothing(hist_true.flatten())
    q = laplace_smoothing(hist_gen.flatten())

    # Compute KL divergence
    return entropy(p, q)

def power_spectrum_error(PS, PS_hat, filter_std=0, return_unreduced=False):
    """Compute power spectrum error from precomputed power spectra using Hellinger distance."""
    def normalize_and_smooth_power_spectrum(PS, sigma):
        if sigma > 0:
            PS = gaussian_filter1d(PS, sigma, axis=0)
        PS /= PS.sum(axis=0, keepdims=True)
        return PS
    
    def hellinger_distance(P, Q):
        return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2, axis=0)) / np.sqrt(2)
    
    PS_normalized = normalize_and_smooth_power_spectrum(PS, filter_std)
    PS_hat_normalized = normalize_and_smooth_power_spectrum(PS_hat, filter_std)
    hd = hellinger_distance(PS_normalized, PS_hat_normalized)

    return (np.mean(hd), hd) if return_unreduced else np.mean(hd)


def prediction_error(dvae_model, X, n, S=None):
    """Compute mean squared prediction error for a DVAE model."""
    T = X.shape[0]
    T_tilde = T - n

    X_input = X[:T_tilde]  # Use the appropriate part of X based on your model's input format
    X_pred = dvae_model(X_input)

    mse = mse_loss(X_pred, X[n:])
    return mse.item()

def calculate_expected_accuracy(true_data, reconstructed_data, accuracy_measure):
    """
    Calculate the expected accuracy measure over batch data.

    Args:
    - true_data: The true data in batch format (seq_len, batch_size, x_dim).
    - reconstructed_data: The reconstructed data in the same format as true_data.
    - accuracy_measure: Function to compute the accuracy measure (e.g., RMSE).

    Returns:
    - A tensor of accuracy values averaged over batches.
    """
    seq_len, batch_size, num_iterations, x_dim = true_data.shape

    reshaped_true = true_data.reshape(seq_len * x_dim, batch_size, num_iterations)
    reshaped_recon = reconstructed_data.reshape(
        seq_len * x_dim, batch_size, num_iterations
    )

    accuracy_values = torch.zeros(seq_len * x_dim, device=true_data.device)
    variance_values = torch.zeros(seq_len * x_dim, device=true_data.device)

    for t in range(seq_len * x_dim):
        accuracy_t = accuracy_measure(reshaped_true[t], reshaped_recon[t])
        accuracy_values[t] = accuracy_t.mean()
        variance_values[t] = accuracy_t.var()

    return accuracy_values, variance_values

def rmse(true_data, pred_data):
    # RMSE per batch item
    return torch.sqrt(torch.mean((true_data - pred_data) ** 2, dim=-1))

def r_squared(true_data, pred_data, epsilon=1e-8):
    """
    Coefficient of Determination (R^2) calculation with stability check.

    Args:
    - true_data: True data tensor.
    - pred_data: Predicted data tensor.
    - epsilon: A small value to ensure numerical stability.

    Returns:
    - R^2 value.
    """
    ss_total = torch.sum(
        (true_data - true_data.mean(dim=-1, keepdim=True)) ** 2, dim=-1
    )
    ss_res = torch.sum((true_data - pred_data) ** 2, dim=-1)

    # Avoid division by very small or zero values
    stable_ss_total = torch.where(
        ss_total > epsilon, ss_total, torch.ones_like(ss_total) * epsilon
    )
    r2 = 1 - ss_res / stable_ss_total
    return r2  # R^2 per batch item

def expand_autonomous_mode_selector(selector, x_dim):
    return np.repeat(selector, x_dim)

def calculate_power_spectrum_error(
    power_spectrum_lst, true_signal_index, filter_std
):
    """
    Calculate the power spectrum error for each signal in the list.

    Args:
    - power_spectrum_lst: A list of power spectra.
    - true_signal_index: The index of the true signal in the list.

    Returns:
    - A list of power spectrum errors.
    """
    true_power_spectrum = power_spectrum_lst[true_signal_index]
    power_spectrum_errors = []
    for power_spectrum in power_spectrum_lst:
        power_spectrum_errors.append(
            power_spectrum_error(
                power_spectrum, true_power_spectrum, filter_std=filter_std
            )
        )

    return power_spectrum_errors



# GMM-based state space divergence not included due to complexity; it requires significant additional implementation for fitting and comparing GMMs.
