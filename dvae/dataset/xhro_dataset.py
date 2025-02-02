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
    train_dataset = Xhro(
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
    )
    val_dataset = Xhro(
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


class Xhro(Dataset):
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
        shuffle=True,
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

        # Read data from file
        filename = (
            f"{self.path_to_data}/xhro/{self.dataset_label}/preprocessed_data.npy"
        )
        the_sequence = np.load(filename, allow_pickle=True)

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

    def apply_observation_process(self, sequence):
        """
        Applies an observation process to the sequence data.
        """
        if self.observation_process == "only_ch1_normalize":
            # Extract the desired channel (e.g., column 3)
            data = sequence[:, 3].astype(np.float32)  # Convert to float
            # Reshape to (num_samples, 1)
            data = data.reshape(-1, 1)
            return self.normalize(data)
        elif self.observation_process == "select_columns_normalize":
            # For example, select columns 3 to 9
            data = sequence[:, 3:9].astype(np.float32)
            return self.normalize(data)
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
