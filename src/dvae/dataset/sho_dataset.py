#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" """
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils
import pickle


# define a class for SimpleHarmonicOscillator dataset in the same style as the above HumanPoseXYZ, dataset
class SimpleHarmonicOscillator(Dataset):
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
            else "amp1,2,0.5_freq1,0.5,20_phas0,piD2,0_N10000_dt0.01"
        )

        if split == "test":
            complete_data_filename = (
                "{0}/sho/data/{1}/complete_dataset_test.pkl".format(
                    self.path_to_data, dataset_label
                )
            )
        else:
            complete_data_filename = (
                "{0}/sho/data/{1}/complete_dataset_train.pkl".format(
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
                mask_filename = f"{self.path_to_data}/sho/data/amp1,2,0.5_freq1,0.5,20_phas0,piD2,0_N10000_dt0.01/mask_{distribution_label}_pnan{rate}_test.pkl"
            else:
                mask_filename = f"{self.path_to_data}/sho/data/amp1,2,0.5_freq1,0.5,20_phas0,piD2,0_N10000_dt0.01/mask_{distribution_label}_pnan{rate}_train.pkl"

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
        # Now the_sequence is a 2D tensor
        self.seq = the_sequence
        #

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
            sequence = sequence + np.random.normal(0, 0.2, sequence.shape[0])
        else:
            raise ValueError("Observation process not recognized.")
        return torch.tensor(sequence, dtype=torch.float32)

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
        start_frame = self.data_idx[index]
        end_frame = min(start_frame + self.seq_len, len(self.seq))
        return self.seq[start_frame:end_frame]

    def update_sequence_length(self, new_seq_len=None):
        # Only one index representing the start of each sequence
        if new_seq_len is not None:
            self.seq_len = new_seq_len
            # Recalculate data_idx based on the new sequence length
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
        else:
            # Use entire sequence if seq_len is None
            self.data_idx = [0]
