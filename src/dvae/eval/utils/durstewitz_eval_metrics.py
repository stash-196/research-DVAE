import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
import torch
from torch.nn.functional import mse_loss

from dvae.eval.utils import (
    run_spectrum_analysis,
    compute_delay_embedding,
)


def power_spectrum_error(PS_hat, PS, filter_std=3, return_unreduced=False):
    PS_normalized = normalize_and_smooth_power_spectrum(PS, filter_std)
    PS_hat_normalized = normalize_and_smooth_power_spectrum(PS_hat, filter_std)
    hd = hellinger_distance(PS_normalized, PS_hat_normalized)
    return (np.mean(hd), hd) if return_unreduced else np.mean(hd)


def normalize_and_smooth_power_spectrum(PS, filter_std):
    PS_smoothed = gaussian_filter1d(PS, sigma=filter_std, axis=-1)  # Per freq
    return PS_smoothed / np.sum(PS_smoothed, axis=-1, keepdims=True) + 1e-10


def hellinger_distance(p, q):
    return np.sqrt(1 - np.sum(np.sqrt(p * q), axis=1))


def n_step_prediction_error(model, X, n, mode_selector=None):
    T = X.shape[0]
    if mode_selector is None:
        mode_selector = torch.zeros_like(X)  # Full TF
    recon = model(X, mode_selector=mode_selector)  # Initial recon
    # Autonomous forecast for last n steps
    auto_selector = torch.ones(n, *X.shape[1:], device=X.device)
    forecast = model(recon[-n:], mode_selector=auto_selector, initialize_states=False)
    pe = mse_loss(forecast, X[-n:])  # Or MASE: divide by naive diff
    return pe.item()


def state_space_kl(
    true_traj,
    gen_traj,
    bins_per_dim=30,
    use_gmm=False,
    n_components=10,
    embed_if_1d=True,
    tau=1,
    embed_dim=3,
    handle_nan="remove",
):
    """
    Updated to receive optionally pre-embedded trajectories (via compute_delay_embedding).
    If 1D and embed_if_1d=True, embeds internally with NaN handling.
    """
    # Handle NaNs in inputs if not pre-embedded
    if np.any(np.isnan(true_traj)) or np.any(np.isnan(gen_traj)):
        true_traj = (
            np.nan_to_num(true_traj, nan=0.0) if handle_nan == "zero" else true_traj
        )
        gen_traj = (
            np.nan_to_num(gen_traj, nan=0.0) if handle_nan == "zero" else gen_traj
        )

    # Embed if 1D
    if embed_if_1d and true_traj.ndim == 1:
        print("[Eval] Embedding 1D trajectories for KL computation.")
        true_traj = compute_delay_embedding(
            true_traj, delay=tau, dimensions=embed_dim, handle_nan=handle_nan
        )
        gen_traj = compute_delay_embedding(
            gen_traj, delay=tau, dimensions=embed_dim, handle_nan=handle_nan
        )

    if use_gmm:  # GMM approx (from klx.py MC)
        gmm_true = GaussianMixture(n_components).fit(true_traj)
        gmm_gen = GaussianMixture(n_components).fit(gen_traj)
        samples = np.random.normal(size=(10000, true_traj.shape[1]))  # Monte Carlo
        log_p_true = gmm_true.score_samples(samples)
        log_p_gen = gmm_gen.score_samples(samples)
        kl = np.mean(log_p_true - log_p_gen)
    else:  # Binned KL (from klx.py histogramdd + Laplace)
        hist_true, edges = np.histogramdd(true_traj, bins=bins_per_dim)
        hist_gen, _ = np.histogramdd(gen_traj, bins=edges)
        p_true = laplace_smoothing(
            hist_true / hist_true.sum() + 1e-10
        )  # Align with klx alpha=1e-5
        p_gen = laplace_smoothing(hist_gen / hist_gen.sum() + 1e-10)
        kl = entropy(
            p_true.flatten(), p_gen.flatten()
        )  # SciPy entropy = KL(p_true || p_gen)

    return kl


def laplace_smoothing(hist, alpha=1e-5):
    return (hist + alpha) / (hist.sum() + alpha * np.prod(hist.shape))
