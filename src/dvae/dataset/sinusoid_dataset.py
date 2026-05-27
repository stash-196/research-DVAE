#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sinusoid_dataset.py: A script to handle sinusoid datasets for RNN-based system identification.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils
import pickle
import os


def build_dataloader(cfg, device):
    # Load dataset params for Sinusoid
    data_dir = cfg.get("User", "data_dir")
    x_dim = cfg.getint("Network", "x_dim")
    shuffle = cfg.getboolean("DataFrame", "shuffle")
    batch_size = cfg.getint("DataFrame", "batch_size")
    num_workers = cfg.getint("DataFrame", "num_workers")
    sequence_len = cfg.getint("DataFrame", "sequence_len")
    sample_rate = cfg.getint("DataFrame", "sample_rate")
    skip_rate = cfg.getint("DataFrame", "skip_rate")
    val_indices = cfg.getfloat("DataFrame", "val_indices")
    observation_process = cfg.get("DataFrame", "observation_process")
    overlap = cfg.getboolean("DataFrame", "overlap")

    if cfg.has_option("DataFrame", "long"):
        long = cfg.getboolean("DataFrame", "long")
    else:
        long = False

    if cfg.has_option("DataFrame", "s_dim"):
        s_dim = cfg.getint("DataFrame", "s_dim")
    else:
        s_dim = 1

    # Load dataset
    train_dataset = Sinusoid(
        path_to_data=data_dir,
        split="train",
        seq_len=sequence_len,
        x_dim=x_dim,
        sample_rate=sample_rate,
        skip_rate=skip_rate,
        val_indices=val_indices,
        observation_process=observation_process,
        device=device,
        overlap=overlap,
        s_dim=s_dim,
    )
    val_dataset = Sinusoid(
        path_to_data=data_dir,
        split="valid",
        seq_len=sequence_len,
        x_dim=x_dim,
        sample_rate=sample_rate,
        skip_rate=skip_rate,
        val_indices=val_indices,
        observation_process=observation_process,
        device=device,
        overlap=overlap,
        s_dim=s_dim,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    return train_dataloader, val_dataloader, train_num, val_num


class Sinusoid(Dataset):
    def __init__(
        self,
        path_to_data,
        split,
        seq_len,
        x_dim,
        sample_rate,
        skip_rate,
        val_indices,
        observation_process,
        device,
        overlap,
        s_dim,
        shuffle=True,
    ):
        self.path_to_data = path_to_data
        self.x_dim = x_dim
        self.seq_len = seq_len
        self.split = split
        self.sample_rate = sample_rate
        self.skip_rate = skip_rate
        self.val_indices = val_indices
        self.observation_process = observation_process
        self.overlap = overlap
        self.shuffle = shuffle
        self.device = device
        self.s_dim = s_dim

        if split == "test":
            filename = os.path.join(
                self.path_to_data, "sinusoid", f"dataset_{s_dim}d_test.pkl"
            )
        else:
            filename = os.path.join(
                self.path_to_data, "sinusoid", f"dataset_{s_dim}d_train.pkl"
            )

        with open(filename, "rb") as f:
            the_sequence = np.array(pickle.load(f))

        self.full_sequence = the_sequence

        # Extract and store the missing mask before applying observation process
        self.missing_mask = self._extract_missing_mask(the_sequence)

        # Apply observation process if necessary
        the_sequence = self.apply_observation_process(the_sequence)

        # Generate sequences with or without overlap
        if self.overlap:
            the_sequence = self.create_moving_window_sequences(the_sequence, self.x_dim)
        else:
            the_sequence = np.array(
                [
                    the_sequence[i : i + x_dim]
                    for i in range(0, len(the_sequence), x_dim)
                    if i + x_dim <= len(the_sequence)
                ]
            )

        self.seq = torch.from_numpy(the_sequence).float().to(device)

        if seq_len is None:
            self.seq_len = len(self.seq)
            self.data_idx = [0]
        else:
            num_frames = self.seq.shape[0]
            all_indices = data_utils.find_indices(
                num_frames, self.seq_len, num_frames // self.seq_len
            )
            train_indices, validation_indices = self.split_dataset(
                all_indices, self.val_indices
            )
            if self.split == "train":
                valid_frames = train_indices
            else:
                valid_frames = validation_indices

            self.data_idx = list(valid_frames)

    def _extract_missing_mask(self, sequence):
        """
        Extracts a boolean mask indicating which values are missing (NaN) in the input sequence.
        This is called before observation_process so the mask is tied to the original data.

        Returns:
            np.ndarray: Boolean mask where True indicates a missing value (NaN).
        """
        return np.isnan(sequence)

    def apply_observation_process(self, sequence):
        """
        Applies an observation process to the sequence data.
        """
        if self.observation_process == "3dto3d":
            pass  # For a 3D to 3D observation process
        elif self.observation_process == "3dto3d_w_noise":
            pass  # For a 3D to 3D observation process with noise
        elif self.observation_process == "mix1d":
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v  # Vector product to convert 3D to 1D
        elif self.observation_process == "mix1dnoise":
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v + np.random.normal(
                0, 0.3, sequence.shape[0]
            )  # Add Gaussian noise
        elif self.observation_process == "only_x_interpolate":
            # take only x (first dimension) and linearly interpolate NaNs
            x = (
                sequence[:, 0].astype(np.float64)
                if sequence.ndim > 1
                else sequence.astype(np.float64)
            )
            nan_mask = np.isnan(x)
            n_nan_before = int(np.sum(nan_mask))
            if n_nan_before > 0:
                idx = np.arange(x.shape[0])
                valid = ~nan_mask
                if valid.any():
                    x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
                else:
                    x[:] = 0.0
            n_nan_after = int(np.isnan(x).sum())
            print(
                f"[Sinusoid][only_x_interpolate] NaNs before: {n_nan_before}, after: {n_nan_after}"
            )
            sequence = x
        elif self.observation_process == "only_x_indicate":
            # produce two-dimensional observation: [x_imputed, is_observed]
            x = (
                sequence[:, 0].astype(np.float32)
                if sequence.ndim > 1
                else sequence.astype(np.float32)
            )
            missing_mask = np.isnan(x)
            missing_count = int(np.sum(missing_mask))
            x_imputed = np.nan_to_num(x, nan=0.0).astype(np.float32)
            is_observed = (~missing_mask).astype(np.float32)
            result = np.stack([x_imputed, is_observed], axis=1)
            print(
                f"[Sinusoid][only_x_indicate] missing_count: {missing_count}, result_shape: {result.shape}"
            )
            sequence = result
        else:
            raise ValueError("Observation process not recognized.")
        return sequence

    @staticmethod
    def create_moving_window_sequences(sequence, window_size):
        return np.lib.stride_tricks.sliding_window_view(
            sequence, window_shape=window_size
        )

    def split_dataset(self, indices, val_indices):
        if self.shuffle:
            np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_indices))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.seq))
        return self.seq[start_frame:end_frame]

    def get_full_xyz(self, index):
        # Method might not be applicable for sinusoid data, adjust as needed
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.full_sequence))
        return self.full_sequence[
            start_frame:end_frame
        ]  # Adjust this part if needed for specific sinusoid data handling

    def get_missing_mask(self, index):
        """
        Retrieves the missing value mask for the given index.
        Indicates which values were originally NaN before imputation/interpolation.

        Args:
            index (int): Index of the desired data sequence.

        Returns:
            np.ndarray: Boolean mask where True indicates a missing value.
        """
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.missing_mask))
        return self.missing_mask[start_frame:end_frame]
