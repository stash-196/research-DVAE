#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils
import pickle


# Define a build_dataloader function for lorenz63 dataset following the style of the above data_builder
def build_dataloader(cfg, device, sequence_len):

    # Load dataset params for Lorenz63
    data_dir = cfg.get("User", "data_dir")
    x_dim = cfg.getint("Network", "x_dim")
    dataset_label = cfg.get("DataFrame", "dataset_label", fallback=None)
    shuffle = cfg.getboolean("DataFrame", "shuffle")
    batch_size = cfg.getint("DataFrame", "batch_size")
    num_workers = cfg.getint("DataFrame", "num_workers")
    sample_rate = cfg.getint("DataFrame", "sample_rate")
    skip_rate = cfg.getint("DataFrame", "skip_rate")
    val_indices = cfg.getfloat("DataFrame", "val_indices")
    observation_process = cfg.get("DataFrame", "observation_process")
    overlap = cfg.getboolean("DataFrame", "overlap")
    with_nan = cfg.getboolean("DataFrame", "with_nan", fallback=False)
    data_cfgs = {}
    # define long as a boolean if field exists
    if cfg.has_option("DataFrame", "long"):
        long = cfg.getboolean("DataFrame", "long")
    else:
        long = False

    if cfg.has_option("DataFrame", "s_dim"):
        data_cfgs["s_dim"] = cfg.getint("DataFrame", "s_dim")
    else:
        data_cfgs["s_dim"] = False

    # Load dataset
    train_dataset = Lorenz63(
        path_to_data=data_dir,
        dataset_label=dataset_label,
        split="train",
        seq_len=sequence_len,
        x_dim=x_dim,
        sample_rate=sample_rate,
        skip_rate=skip_rate,
        val_indices=val_indices,
        observation_process=observation_process,
        device=device,
        overlap=overlap,
        with_nan=with_nan,
    )
    val_dataset = Lorenz63(
        path_to_data=data_dir,
        dataset_label=dataset_label,
        split="valid",
        seq_len=sequence_len,
        x_dim=x_dim,
        sample_rate=sample_rate,
        skip_rate=skip_rate,
        val_indices=val_indices,
        observation_process=observation_process,
        device=device,
        overlap=overlap,
        with_nan=with_nan,
    )

    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()

    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Lorenz63 dataset in the same style as the above HumanPoseXYZ, dataset
class Lorenz63(Dataset):
    def __init__(
        self,
        path_to_data,
        dataset_label,
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
    ):
        """
        :param path_to_data: path to the data folder
        :param split: train, test or val
        :param seq_len: length of the sequence
        :param sample_rate: downsampling rate
        :param skip_rate: the skip length to get example, only used for train and test
        :param val_indices: the number of slices used for validation
        """

        self.path_to_data = path_to_data
        self.dataset_label = dataset_label
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

        # read motion data from pickle file
        if self.dataset_label == None or self.dataset_label == "None":
            if split == "test":
                filename = "{0}/lorenz63/dataset_test.pkl".format(self.path_to_data)
            else:
                filename = "{0}/lorenz63/dataset_train.pkl".format(self.path_to_data)
        else:
            if split == "test":
                filename = f"{self.path_to_data}/lorenz63/dataset_test_{self.dataset_label}.pkl"
            else:
                filename = f"{self.path_to_data}/lorenz63/dataset_train_{self.dataset_label}.pkl"

        with open(filename, "rb") as f:
            the_sequence = np.array(pickle.load(f))

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # Generate sequences with or without overlap
        if self.overlap:
            the_sequence = self.create_moving_window_sequences(the_sequence, self.x_dim)
        else:  # Remove the last sequence if it is not the correct length
            the_sequence = np.array(
                [
                    the_sequence[i : i + x_dim]
                    for i in range(0, len(the_sequence), x_dim)
                    if i + x_dim <= len(the_sequence)
                ]
            )

        self.seq = torch.from_numpy(the_sequence).float()

        # Use entire sequence if seq_len is None
        if seq_len is None:
            self.seq_len = len(self.seq)
            # Only one index representing the start of the entire sequence
            self.data_idx = [0]
        else:
            # Determine indices for training and validation sets
            num_frames = self.seq.shape[0]
            all_indices = data_utils.find_indices(
                num_frames, self.seq_len, num_frames // self.seq_len
            )
            train_indices, validation_indices = self.split_dataset(
                all_indices, self.val_indices
            )
            # Select appropriate indices based on the split
            if self.split == "train":  # for train and test
                valid_frames = train_indices
            else:  # for validation
                valid_frames = validation_indices

            self.data_idx = list(valid_frames)

    def apply_observation_process(self, sequence):
        """
        Applies an observation process to the sequence data.
        """
        if self.observation_process == "3dto3d":
            pass  # For a 3D to 3D observation process
        elif self.observation_process == "3dto3d_w_noise":
            pass  # For a 3D to 3D observation process with noise
        elif self.observation_process == "3dto1d":
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v  # Vector product to convert 3D to 1D
        elif self.observation_process == "3dto1d_w_noise":
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v + np.random.normal(
                0, 5.7, sequence.shape[0]
            )  # Add Gaussian noise
        elif self.observation_process == "only_x":
            # observe only x out of xyz dimensions
            sequence = sequence[:, 0]
        elif self.observation_process == "only_x_w_noise":
            sequence = sequence[:, 0] + np.random.normal(0, 5.7, sequence.shape[0])
        else:
            raise ValueError("Observation process not recognized.")
        return sequence

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

    def update_sequence_length(self, new_seq_len):
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
