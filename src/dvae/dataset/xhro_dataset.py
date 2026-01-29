#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from .utils import data_utils
from .utils.visualize_nans import plot_nan_heatmap, plot_segment_length_hist


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
        self.sampling_freq = None

        # Read data from file
        if (
            self.observation_process
            in select_columns_for_obs_conditions["original"].keys()
        ):
            filename = f"{self.path_to_data}/xhro/processed/{self.dataset_label}/filtered_data.parquet"
            the_sequence = pd.read_parquet(filename)
            self.sampling_freq = 250
            print(f"Downloaded data from {filename}")
        elif (
            self.observation_process
            in select_columns_for_obs_conditions["coarsed"].keys()
        ):
            filename = os.path.join(
                f"{self.path_to_data}",
                "xhro",
                "processed",
                f"{self.dataset_label}",
                "coarse_features.parquet",
            )
            the_sequence = pd.read_parquet(filename)
            self.sampling_freq = 1 / (60 * 5)
            print(f"Downloaded data from {filename}")
        else:
            raise ValueError(
                f"Invalid observation process: {self.observation_process}. "
                "Must be one of the keys in select_columns_for_obs_conditions."
            )
        # see if directory of filename exists
        the_sequence = select_observation_window_slice(self.dataset_label, the_sequence)

        if self.split == "test":  # use the Latter 20% of the data for testing
            the_sequence = the_sequence[-the_sequence.shape[0] // 5 :]
        else:
            the_sequence = the_sequence[: -the_sequence.shape[0] // 5]

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # the_sequence should be squeezed before this takes place
        the_sequence = the_sequence.squeeze()

        if self.x_dim is None:
            if the_sequence.ndim == 1:
                self.x_dim = 1
            elif the_sequence.ndim == 2:
                self.x_dim = the_sequence.shape[1]
            else:
                raise ValueError(
                    f"Expected x is {the_sequence.ndim} dimensions, got x_dim {self.x_dim} instead."
                )

        # Generate sequences with or without overlap
        self.is_segmented_1d = False
        if the_sequence.ndim == 1:
            if self.x_dim > 1:
                self.is_segmented_1d = True
            if self.overlap:
                the_sequence = self.create_moving_window_sequences(
                    the_sequence, self.x_dim
                )
            else:  # Remove the last sequence if it is not the correct length
                the_sequence = np.array(
                    [
                        the_sequence[i : i + x_dim]
                        for i in range(0, len(the_sequence), x_dim)
                        if i + x_dim <= len(the_sequence)
                    ]
                )
        elif the_sequence.shape[1] != self.x_dim:
            raise ValueError(
                f"Expected x is {the_sequence.ndim} dimensions, got x_dim {self.x_dim} instead."
            )
        # Now the_sequence is a 2D tensor
        self.seq = the_sequence
        # torch.Size([80000, 1])

        self.update_sequence_length(self.seq_len)

    def apply_observation_process(self, sequence) -> torch.Tensor:
        """
        Applies an observation process to the sequence data.
        """
        if (
            self.observation_process
            in select_columns_for_obs_conditions["coarsed"].keys()
        ):
            data = sequence[
                select_columns_for_obs_conditions["coarsed"][self.observation_process]
            ].to_numpy(dtype=np.float32)
            normalized_data = self.normalize(data)
            return torch.from_numpy(normalized_data)
        elif (
            self.observation_process
            in select_columns_for_obs_conditions["original"].keys()
        ):
            # Find the first non-NaN index
            first_valid_index = sequence.first_valid_index()
            data = (
                sequence[
                    select_columns_for_obs_conditions["original"][
                        self.observation_process
                    ]
                ]
                .loc[first_valid_index:]
                .to_numpy(dtype=np.float32)
            )
            normalized_data = self.normalize(data)
            return torch.tensor(normalized_data, dtype=torch.float32)
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

    def get_full_xyz(self, index):
        """
        Retrieves the full xyz variables for the given index.

        Args:
            index (int): Index of the desired data sequence.

        Returns:
            torch.Tensor: The full xyz sequence data for the given index.
        """
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.full_sequence))
        # Return the full (x, y, z) data
        return self.full_sequence[start_frame:end_frame]

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
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.seq))
        return self.seq[start_frame:end_frame]

    def update_sequence_length(self, new_seq_len=None, minimum_nan_ratio=None):
        # Only one index representing the start of each sequence
        if new_seq_len is not None:
            self.seq_len = new_seq_len
            # Recalculate data_idx based on the new sequence length
            num_frames = self.seq.shape[0]
            all_indices = data_utils.find_indices(
                num_frames, self.seq_len, num_frames // self.seq_len
            )

            if minimum_nan_ratio is not None:
                total_frames_before = len(all_indices)  # Total frames before filtering
                print(f"[Dataset] Total frames before filtering: {total_frames_before}")
                valid_frames = []
                for idx in all_indices:
                    seq_slice = self.seq[idx : idx + self.seq_len]
                    nan_ratio = np.float16(np.isnan(seq_slice)).mean()
                    if nan_ratio <= minimum_nan_ratio:
                        valid_frames.append(idx)
                valid_frames = np.array(valid_frames)
                total_frames_after = len(valid_frames)  # Total frames after filtering
                disregarded_frames = (
                    total_frames_before - total_frames_after
                )  # Disregarded frames
                print(f"[Dataset] Total frames after filtering: {total_frames_after}")
                print(
                    f"[Dataset] Disregarded frames (due to NaN ratio > {minimum_nan_ratio}): {disregarded_frames}"
                )
            else:
                valid_frames = all_indices
                total_frames_after = len(valid_frames)
                disregarded_frames = 0
                print(
                    f"[Dataset] No filtering applied (minimum_nan_ratio is None). Total frames: {total_frames_after}"
                )

            train_indices, validation_indices = self.split_dataset(
                valid_frames, self.val_indices
            )
            if self.split == "train":
                valid_frames = train_indices
            else:
                valid_frames = validation_indices
            self.data_idx = list(valid_frames)
        else:
            # Use entire sequence if seq_len is None
            self.data_idx = [0]
            print("[Dataset] Using entire sequence (seq_len is None). Total frames: 1")

        return


select_columns_for_obs_conditions = {
    "original": {
        "raw_ch1": [
            "ch1",
        ],
        "raw_ecg": [
            "ch1",
            "ch2",
        ],
        "raw_eeg": [
            "ch3",
            "ch4",
        ],
        "raw_ch4": [
            "ch4",
        ],
        "raw_all": [
            "ch1",
            "ch2",
            "ch3",
            "ch4",
        ],
    },
    "coarsed": {
        "ch4_relative_powers": [
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "relative_delta_power"),
            ("ch4", "relative_theta_power"),
            ("ch4", "relative_gamma_low_power"),
            ("ch4", "relative_gamma_mid_power"),
            ("ch4", "total_power"),
        ],
        "ch4_3_vars": [
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "total_power"),
        ],
        "ch4_alpha": [("ch4", "relative_alpha_power")],
        "all_ch_relative_powers": [
            ("ch1", "relative_alpha_power"),
            ("ch1", "relative_beta_power"),
            ("ch1", "relative_delta_power"),
            ("ch1", "relative_theta_power"),
            ("ch1", "relative_gamma_low_power"),
            ("ch1", "relative_gamma_mid_power"),
            ("ch1", "total_power"),
            ("ch2", "relative_alpha_power"),
            ("ch2", "relative_beta_power"),
            ("ch2", "relative_delta_power"),
            ("ch2", "relative_theta_power"),
            ("ch2", "relative_gamma_low_power"),
            ("ch2", "relative_gamma_mid_power"),
            ("ch2", "total_power"),
            ("ch3", "relative_alpha_power"),
            ("ch3", "relative_beta_power"),
            ("ch3", "relative_delta_power"),
            ("ch3", "relative_theta_power"),
            ("ch3", "relative_gamma_low_power"),
            ("ch3", "relative_gamma_mid_power"),
            ("ch3", "total_power"),
            ("ch4", "relative_alpha_power"),
            ("ch4", "relative_beta_power"),
            ("ch4", "relative_delta_power"),
            ("ch4", "relative_theta_power"),
            ("ch4", "relative_gamma_low_power"),
            ("ch4", "relative_gamma_mid_power"),
            ("ch4", "total_power"),
        ],
    },
}


def select_observation_window_slice(dataset_exp_label, the_sequence):
    """
    Slices the data based on the dataset experiment label to select a specific time window.

    :param dataset_exp_label: The label for the dataset (e.g., "XHRO_04_XH-15").
    :param the_sequence: The pandas DataFrame with a datetime index.
    :return: The sliced DataFrame.
    """
    if dataset_exp_label == "XHRO_01_XH006":
        # Slice from Oct 6 2023 15:53 to 16:03
        start_time = pd.Timestamp("2023-10-06 15:53:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-10-06 16:03:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    elif dataset_exp_label == "XHRO_04_XH057":
        # Slice from Nov 10 2023 14:52 to 15:02
        start_time = pd.Timestamp("2023-11-10 14:52:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-11-10 15:02:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    elif dataset_exp_label == "XHRO_02_XH070":
        # Slice from Oct 29 2023 22:12 to 22:22
        start_time = pd.Timestamp("2023-10-29 22:12:00").tz_localize("Asia/Tokyo")
        end_time = pd.Timestamp("2023-10-29 22:22:00").tz_localize("Asia/Tokyo")
        # Filter rows where 'datetime' is within the range
        mask = (the_sequence["datetime"] >= start_time) & (
            the_sequence["datetime"] <= end_time
        )
        return the_sequence[mask]
    # Add more cases here for other labels if needed
    # elif dataset_exp_label == "another_label":
    #     start_time = pd.Timestamp('...')
    #     end_time = pd.Timestamp('...')
    #     return the_sequence.loc[start_time:end_time]
    else:
        # For other labels, return the full sequence
        return the_sequence
