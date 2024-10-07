#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def create_autonomous_mode_selector(
    seq_len, mode="all_1", autonomous_ratio=0.0, batch_size=None
):
    """
    Creates a mode selector array for RNN training.

    :param seq_len: Length of the sequence.
    :param mode: The mode of operation - 'all_1', 'all_0', 'half_half', 'flip_at_middle',
                 'even_sampling', 'even_bursts', 'mix_sampling', or 'bernoulli_sampling'.
    :param autonomous_ratio: Ratio of autonomous steps for applicable modes.
    :param batch_size: Number of sequences in a batch. If None, returns a 1D array.
    :return: A NumPy array of shape (seq_len, batch_size) where 0 represents teacher-forcing
             mode and 1 represents autonomous mode.
    """
    if mode == "all_1":
        mode_selector = (
            np.ones((seq_len, batch_size)) if batch_size else np.ones(seq_len)
        )

    elif mode == "all_0":
        mode_selector = (
            np.zeros((seq_len, batch_size)) if batch_size else np.zeros(seq_len)
        )

    elif mode == "half_half":
        half_len = seq_len // 2
        half_array = [0] * half_len + [1] * (seq_len - half_len)
        mode_selector = (
            np.tile(half_array, (batch_size, 1)).T
            if batch_size
            else np.array(half_array)
        )

    elif mode == "flip_at_middle":
        autonomous_len = int(seq_len * autonomous_ratio)
        half_array = [0] * (seq_len - autonomous_len) + [1] * autonomous_len
        mode_selector = (
            np.tile(half_array, (batch_size, 1)).T
            if batch_size
            else np.array(half_array)
        )

    elif mode == "even_sampling":
        autonomous_step = int(1 / autonomous_ratio) if autonomous_ratio > 0 else seq_len
        mode_selector = np.array([i % autonomous_step == 0 for i in range(seq_len)])
        mode_selector = (
            np.tile(mode_selector, (batch_size, 1)).T if batch_size else mode_selector
        )

    elif mode == "even_bursts":
        mode_selector = []
        flip_autonomous = int(seq_len * autonomous_ratio)
        mode = 1
        for i in range(seq_len):
            if i % flip_autonomous == 0:
                mode = 1 - mode
            mode_selector.append(mode)
        mode_selector = (
            np.tile(mode_selector, (batch_size, 1)).T
            if batch_size
            else np.array(mode_selector)
        )

    elif mode == "mix_sampling":
        mode_selector = (
            np.full((seq_len, batch_size), autonomous_ratio)
            if batch_size
            else np.full(seq_len, autonomous_ratio)
        )

    elif mode == "bernoulli_sampling":
        mode_selector = (
            np.random.binomial(1, autonomous_ratio, (seq_len, batch_size))
            if batch_size
            else np.random.binomial(1, autonomous_ratio, seq_len)
        )

    else:
        raise ValueError("Invalid mode selected.")

    return mode_selector


import torch


def prepare_mode_selector(mode_selector, seq_len, batch_size, x_dim, device):
    if mode_selector is not None:
        if not torch.is_tensor(mode_selector):
            mode_selector = torch.tensor(mode_selector, device=device)
        else:
            mode_selector = mode_selector.to(device)
    else:
        mode_selector = torch.zeros(seq_len, device=device)

    if mode_selector.dim() == 0:
        # Scalar mode_selector, expand to (seq_len, batch_size, x_dim)
        mode_selector = (
            mode_selector.view(1, 1, 1).expand(seq_len, batch_size, x_dim).float()
        )
    elif mode_selector.dim() == 1:
        # mode_selector of shape (seq_len,)
        mode_selector = (
            mode_selector.view(seq_len, 1, 1).expand(seq_len, batch_size, x_dim).float()
        )
    elif mode_selector.dim() == 2:
        # mode_selector of shape (seq_len, batch_size)
        mode_selector = (
            mode_selector.view(seq_len, batch_size, 1)
            .expand(seq_len, batch_size, x_dim)
            .float()
        )
    else:
        raise ValueError(f"Unsupported mode_selector shape: {mode_selector.shape}")

    return mode_selector
