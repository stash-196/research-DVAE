#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils
import pickle


# define a class for DampedSimpleHarmonicOscillator dataset in the same style as the above HumanPoseXYZ, dataset
class DampedSimpleHarmonicOscillator(Dataset):
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
        with_nan,
        shuffle=True,
        **kwargs,
    ):
        """
        :param path_to_data: path to the data folder
        :param split: train, test or val
        :param seq_len: length of the sequence
        :param sample_rate: downsampling rate
        :param skip_rate: the skip length to get example, only used for train and test
        :param val_indices: the number of slices used for validation
        :param observation_process: the observation process to apply
        :param device: the device to use
        :param overlap: whether to use overlapping windows
        :param with_nan: whether to use nan in the data
        :param shuffle: whether to shuffle the data
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
        self.with_nan = with_nan

        dataset_label = (
            self.dataset_label
            if self.dataset_label != None and self.dataset_label != "None"
            else "omegas2pi,pi_gammas0.5,0.2_inst100_N1k_dt0.01"
        )

        self.true_alphas = None

        if split == "test":
            complete_data_filename = (
                "{0}/damped_sho/data/{1}/complete_dataset_test.pkl".format(
                    self.path_to_data, dataset_label
                )
            )
        else:
            complete_data_filename = (
                "{0}/damped_sho/data/{1}/complete_dataset_train.pkl".format(
                    self.path_to_data, dataset_label
                )
            )

        # load the complete dataset
        with open(complete_data_filename, "rb") as f:
            complete_sequence = np.array(pickle.load(f))

        # load the mask
        if (
            self.mask_label != None
            and self.mask_label != "None"
            and self.mask_label != ""
        ):
            # mask_label should be {distribution}_{rate}
            mask_label = self.mask_label.split("_")
            distribution = mask_label[0]
            rate = mask_label[-1]
            if distribution == "Markov":
                average_burst_length = mask_label[1]
                distribution_label = f"{distribution}_{average_burst_length}"
            else:
                distribution_label = f"{distribution}"

            if split == "test":
                mask_filename = f"{self.path_to_data}/damped_sho/data/{dataset_label}/mask_{distribution_label}_pnan{rate}_test.pkl"
            else:
                mask_filename = f"{self.path_to_data}/damped_sho/data/{dataset_label}/mask_{distribution_label}_pnan{rate}_train.pkl"

            with open(mask_filename, "rb") as f:
                mask_sequence = np.array(pickle.load(f))

            # Apply the mask to the sequence
            the_sequence = np.where(mask_sequence, np.nan, complete_sequence)
        else:
            the_sequence = complete_sequence

        # Store the full sequence before applying any observation process
        self.full_sequence = complete_sequence

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # the_sequence should be squeezed before this takes place
        the_sequence = the_sequence.squeeze()

        # Special handling for damped_sho datasets
        self.is_damped_sho = "damped_sho" in self.dataset_label.lower()

        # For damped SHO, treat as multiple 1D instances
        self.N, self.n_instances = the_sequence.shape
        self.x_dim = 1

        # Generate sequences with or without overlap
        self.is_segmented_1d = False
        # Now the_sequence is a 2D tensor
        self.seq = the_sequence

        self.update_sequence_length(self.seq_len)

    def apply_observation_process(self, sequence) -> torch.Tensor:
        """
        Applies an observation process to the sequence data.
        """
        # random seed for reproducibility
        np.random.seed(42)
        if self.observation_process == "mixed_1d":
            sequence = sequence
        elif self.observation_process == "mixed_1d_w_noise":
            sequence = sequence + np.random.normal(0, 0.2, sequence.shape)
        else:
            raise ValueError("Observation process not recognized.")
        return torch.tensor(sequence, dtype=torch.float32)

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
        Converts a 1D time series into a 2D array of overlapping sequences.
        """
        return np.lib.stride_tricks.sliding_window_view(
            sequence, window_shape=window_size
        )

    def split_dataset(self, indices, val_indices):
        """
        Splits the dataset into training and validation sets.
        """
        # only shuffle when self.shuffle is True
        if self.shuffle:
            np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_indices))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, index):
        instance, start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, self.N)
        return self.seq[start_frame:end_frame, instance].unsqueeze(-1)

    def update_sequence_length(self, new_seq_len=None):
        if new_seq_len is not None:
            self.seq_len = new_seq_len
        else:
            self.seq_len = self.N

        # Special handling for damped_sho: windows per instance
        all_indices = []
        for instance in range(self.n_instances):
            local_num_samples = self.N // self.seq_len
            if local_num_samples == 0:
                local_num_samples = 1

            if local_num_samples == 1:
                start = np.random.randint(0, max(1, self.N - self.seq_len + 1))
                local_indices = [start]
            else:
                # Evenly spaced windows
                step = (self.N - self.seq_len) // (local_num_samples - 1)
                local_indices = [i * step for i in range(local_num_samples)]

            for local_idx in local_indices:
                all_indices.append((instance, local_idx))

        if self.split == "test":
            self.data_idx = all_indices
        else:
            train_indices, validation_indices = self.split_dataset(
                all_indices, self.val_indices
            )
            if self.split == "train":
                self.data_idx = train_indices
            elif self.split == "valid":
                self.data_idx = validation_indices
            else:
                raise ValueError("Split not recognized.")
