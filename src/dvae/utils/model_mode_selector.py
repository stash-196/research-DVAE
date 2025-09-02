#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


import numpy as np
import torch


def create_autonomous_mode_selector_1d(seq_len, mode="all_1", autonomous_ratio=0.0):
    """
    Creates a 1D mode selector array for RNN training.

    :param seq_len: Length of the sequence.
    :param mode: The mode of operation.
    :param autonomous_ratio: Ratio of autonomous steps for applicable modes.
    :return: A NumPy array of shape (seq_len,) where 0 represents teacher-forcing
             mode and 1 represents autonomous mode.
    """
    if mode == "all_1":
        mode_selector = np.ones(seq_len, dtype=np.float32)
    elif mode == "all_0":
        mode_selector = np.zeros(seq_len, dtype=np.float32)
    elif mode == "half_half":
        half_len = seq_len // 2
        half_array = [0] * half_len + [1] * (seq_len - half_len)
        mode_selector = np.array(half_array, dtype=np.float32)
    elif mode == "flip_at_middle":
        autonomous_len = int(seq_len * autonomous_ratio)
        half_array = [0] * (seq_len - autonomous_len) + [1] * autonomous_len
        mode_selector = np.array(half_array, dtype=np.float32)
    elif mode == "even_sampling":
        autonomous_step = int(1 / autonomous_ratio) if autonomous_ratio > 0 else seq_len
        # Ensure boolean array is converted to float (0.0/1.0)
        mode_selector = np.array(
            [i % autonomous_step == 0 for i in range(seq_len)], dtype=np.float32
        )
    elif mode == "even_bursts":
        mode_selector_list = []  # Use a list to build efficiently
        flip_interval = (
            int(1 / autonomous_ratio) if autonomous_ratio > 0 else seq_len + 1
        )  # Avoid division by zero, make it never flip if ratio is 0
        current_mode = 0  # Start with teacher forcing
        for i in range(seq_len):
            if i > 0 and (i % flip_interval == 0):  # Flip at interval, not at start
                current_mode = 1 - current_mode
            mode_selector_list.append(current_mode)
        mode_selector = np.array(mode_selector_list, dtype=np.float32)
    elif mode == "mix_sampling":
        mode_selector = np.full(seq_len, autonomous_ratio, dtype=np.float32)
    elif mode == "bernoulli_sampling":
        mode_selector = np.random.binomial(1, autonomous_ratio, seq_len).astype(
            np.float32
        )
    else:
        raise ValueError("Invalid mode selected.")

    return mode_selector


def create_autonomous_mode_selector(
    seq_len,
    mode="all_1",
    autonomous_ratio=0.0,
    batch_size=None,
    x_dim=None,
    device="cpu",
):
    """
    Creates a mode selector array for RNN training,
    handling batch_size and x_dim dimensions efficiently.

    :param seq_len: Length of the sequence.
    :param mode: The mode of operation - 'all_1', 'all_0', 'half_half', 'flip_at_middle',
                'even_sampling', 'even_bursts', 'mix_sampling', or 'bernoulli_sampling'.
    :param autonomous_ratio: Ratio of autonomous steps for applicable modes.
    :param batch_size: Number of sequences in a batch. If None, batch dimension is not added.
    :param x_dim: Dimensionality of the input features. If None, x_dim dimension is not added.
    :return: A NumPy array of shape (seq_len, [batch_size], [x_dim]) where 0 represents teacher-forcing
             mode and 1 represents autonomous mode. Dimensions are added only if batch_size/x_dim are provided.
    """
    # 1. Create the base 1D mode selector using the helper function
    mode_selector_1d = create_autonomous_mode_selector_1d(
        seq_len, mode, autonomous_ratio
    )

    # Convert to PyTorch tensor immediately for expand/broadcast operations
    mode_selector_tensor = torch.from_numpy(mode_selector_1d).float()

    # 2. Reshape and expand based on batch_size and x_dim
    # Start with (seq_len, 1, 1) to prepare for easy expansion
    mode_selector_tensor = mode_selector_tensor.view(seq_len, 1, 1)

    # Prepare target dimensions for expansion
    target_dims = [seq_len, 1, 1]

    if batch_size is not None:
        target_dims[1] = batch_size

    if x_dim is not None:
        target_dims[2] = x_dim

    # Use .expand() for efficient broadcasting.
    # Note: .expand() creates a new view of the tensor, it does not allocate new memory
    # for the expanded dimensions. This is highly efficient.
    # If a dimension is 1, it will be expanded to the target size.
    # If a dimension is already the target size, it remains as is.
    # If a dimension is not 1 and not the target size, it will raise an error (which is good).
    mode_selector = mode_selector_tensor.expand(*target_dims)

    # If both batch_size and x_dim were None, we want to return a 1D tensor
    if batch_size is None and x_dim is None:
        return mode_selector.squeeze()  # Remove the added 1-dimensions
    elif x_dim is None and batch_size is not None:
        return mode_selector.squeeze(
            -1
        )  # Remove the last x_dim dimension if only batch_size was specified
    elif x_dim is not None and batch_size is None:
        return mode_selector.squeeze(
            1
        )  # Remove the batch_size dimension if only x_dim was specified

    return mode_selector.to(device)


# def create_autonomous_mode_selector(
#     seq_len, mode="all_1", autonomous_ratio=0.0, batch_size=None, x_dim=None
# ):
#     """
#     Creates a mode selector array for RNN training.

#     :param seq_len: Length of the sequence.
#     :param mode: The mode of operation - 'all_1', 'all_0', 'half_half', 'flip_at_middle',
#                  'even_sampling', 'even_bursts', 'mix_sampling', or 'bernoulli_sampling'.
#     :param autonomous_ratio: Ratio of autonomous steps for applicable modes.
#     :param batch_size: Number of sequences in a batch. If None, returns a 1D array.
#     :return: A NumPy array of shape (seq_len, batch_size) where 0 represents teacher-forcing
#              mode and 1 represents autonomous mode.
#     """
#     if mode == "all_1":
#         mode_selector = (
#             np.ones((seq_len, batch_size)) if batch_size else np.ones(seq_len)
#         )

#     elif mode == "all_0":
#         mode_selector = (
#             np.zeros((seq_len, batch_size)) if batch_size else np.zeros(seq_len)
#         )

#     elif mode == "half_half":
#         half_len = seq_len // 2
#         half_array = [0] * half_len + [1] * (seq_len - half_len)
#         mode_selector = (
#             np.tile(half_array, (batch_size, 1)).T
#             if batch_size
#             else np.array(half_array)
#         )

#     elif mode == "flip_at_middle":
#         autonomous_len = int(seq_len * autonomous_ratio)
#         half_array = [0] * (seq_len - autonomous_len) + [1] * autonomous_len
#         mode_selector = (
#             np.tile(half_array, (batch_size, 1)).T
#             if batch_size
#             else np.array(half_array)
#         )

#     elif mode == "even_sampling":
#         autonomous_step = int(1 / autonomous_ratio) if autonomous_ratio > 0 else seq_len
#         mode_selector = np.array([i % autonomous_step == 0 for i in range(seq_len)])
#         mode_selector = (
#             np.tile(mode_selector, (batch_size, 1)).T if batch_size else mode_selector
#         )

#     elif mode == "even_bursts":
#         mode_selector = []
#         flip_autonomous = int(seq_len * autonomous_ratio)
#         mode = 1
#         for i in range(seq_len):
#             if i % flip_autonomous == 0:
#                 mode = 1 - mode
#             mode_selector.append(mode)
#         mode_selector = (
#             np.tile(mode_selector, (batch_size, 1)).T
#             if batch_size
#             else np.array(mode_selector)
#         )

#     elif mode == "mix_sampling":
#         mode_selector = (
#             np.full((seq_len, batch_size), autonomous_ratio)
#             if batch_size
#             else np.full(seq_len, autonomous_ratio)
#         )

#     elif mode == "bernoulli_sampling":
#         mode_selector = (
#             np.random.binomial(1, autonomous_ratio, (seq_len, batch_size))
#             if batch_size
#             else np.random.binomial(1, autonomous_ratio, seq_len)
#         )

#     else:
#         raise ValueError("Invalid mode selected.")

#     return mode_selector


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
    elif mode_selector.dim() == 3:
        # mode_selector of shape (seq_len, batch_size, x_dim)
        mode_selector = mode_selector.float()
    else:
        raise ValueError(f"Unsupported mode_selector shape: {mode_selector.shape}")

    return mode_selector
