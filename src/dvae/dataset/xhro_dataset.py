#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd


class Xhro(Dataset):
    def __init__(
        self,
        data_dir,
        dataset_label,
        mask_label,
        split,
        seq_len,
        x_dim,
        sample_rate,
        skip_rate,
        val_indices,
        observation_process,
        device,
        overlap,
        shuffle=True,
        **kwargs,
    ):
        """
        :param path_to_data: path to the data folder
        :param dataset_label: label for the dataset
        :param split: 'train' or 'valid'
        :param seq_len: length of the sequences
        :param x_dim: dimension of the data
        :param sample_rate: downsampling rate
        :param skip_rate: skip rate for data sampling
        :param val_indices: proportion of data used for validation
        :param observation_process: process to apply to observations
        :param device: computation device
        :param overlap: whether to use overlapping sequences
        :param shuffle: whether to shuffle the data indices
        """

        self.path_to_data = data_dir
        self.dataset_label = dataset_label
        self.mask_label = mask_label
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

        # Read data from file
        # filename = (
        #     f"{self.path_to_data}/xhro/{self.dataset_label}/preprocessed_data.npy"
        # )
        filename = os.path.join(
            f"{self.path_to_data}",
            "xhro",
            "processed",
            f"{self.dataset_label}",
            "coarse_features.parquet",
        )
        the_sequence = pd.read_parquet(filename)
        # see if directory of filename exists

        if self.split == "test":  # use the Latter 20% of the data for testing
            the_sequence = the_sequence[-the_sequence.shape[0] // 5 :]
        else:
            the_sequence = the_sequence[: -the_sequence.shape[0] // 5]

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # Generate sequences with or without overlap
        if self.overlap:
            the_sequence = self.create_moving_window_sequences(
                the_sequence, self.seq_len
            )
        else:
            total_frames = the_sequence.shape[0]
            num_sequences = total_frames // self.seq_len  # Number of sequences
            the_sequence = the_sequence[
                : num_sequences * self.seq_len
            ]  # Trim data to fit sequences
            the_sequence = the_sequence.reshape(num_sequences, self.seq_len, -1)

        self.seq = the_sequence

        # Split dataset into training and validation indices
        num_sequences = self.seq.shape[0]
        all_indices = np.arange(num_sequences)
        train_indices, validation_indices = self.split_dataset(
            all_indices, self.val_indices
        )

        # Select appropriate indices based on the split
        if self.split == "train":
            self.data_idx = train_indices
        else:
            self.data_idx = validation_indices

    def apply_observation_process(self, sequence):
        """
        Applies an observation process to the sequence data.
        """
        if self.observation_process in select_columns.keys():
            data = sequence[select_columns[self.observation_process]].to_numpy(
                dtype=np.float32
            )
            normalized_data = self.normalize(data)
            return torch.from_numpy(normalized_data)
        else:
            raise ValueError(f"Invalid observation process: {self.observation_process}")

    def normalize(self, data):
        """
        Normalize the data.
        """
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        data = (data - mean) / std
        return data

    @staticmethod
    def create_moving_window_sequences(sequence, window_size):
        """
        Converts a time series into a 3D array of overlapping sequences.
        """
        num_samples = sequence.shape[0]
        stride = sequence.strides[0]
        num_sequences = num_samples - window_size + 1
        shape = (num_sequences, window_size, sequence.shape[1])
        strides = (stride, stride, sequence.strides[1])
        return np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)

    def split_dataset(self, indices, val_ratio):
        """
        Splits the dataset indices into training and validation sets.
        """
        if self.shuffle:
            np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_ratio))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        idx = self.data_idx[index]
        sequence = self.seq[idx]
        return sequence

    def update_sequence_length(self, new_seq_len):
        self.seq_len = new_seq_len
        # Regenerate sequences with the new sequence length
        if self.overlap:
            the_sequence = self.create_moving_window_sequences(
                self.full_sequence, self.seq_len
            )
        else:
            total_frames = self.full_sequence.shape[0]
            num_sequences = total_frames // self.seq_len
            the_sequence = self.full_sequence[: num_sequences * self.seq_len]
            the_sequence = the_sequence.reshape(num_sequences, self.seq_len, -1)

        self.seq = torch.from_numpy(the_sequence).float()

        # Update data indices
        num_sequences = self.seq.shape[0]
        all_indices = np.arange(num_sequences)
        train_indices, validation_indices = self.split_dataset(
            all_indices, self.val_indices
        )

        # Select appropriate indices based on the split
        if self.split == "train":
            self.data_idx = train_indices
        else:
            self.data_idx = validation_indices


select_columns = {
    "ch4_relative_powers": [
        ("ch4", "relative_alpha_power"),
        ("ch4", "relative_beta_power"),
        ("ch4", "relative_delta_power"),
        ("ch4", "relative_theta_power"),
        ("ch4", "relative_gamma_low_power"),
        ("ch4", "relative_gamma_high_power"),
        ("ch4", "total_power"),
    ],
    "ch4_3_vars": [
        ("ch4", "relative_alpha_power"),
        ("ch4", "relative_beta_power"),
        ("ch4", "total_power"),
    ],
}
