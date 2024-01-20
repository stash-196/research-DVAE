#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils
import pickle


# Define a build_dataloader function for Sinusoid dataset following the style of the above data_builder
def build_dataloader(cfg, device):

    # Load dataset params for Sinusoid
    data_dir = cfg.get('User', 'data_dir')
    x_dim = cfg.getint('Network', 'x_dim')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    sample_rate = cfg.getint('DataFrame', 'sample_rate')
    skip_rate = cfg.getint('DataFrame', 'skip_rate')
    val_indices = cfg.getfloat('DataFrame', 'val_indices')
    observation_process = cfg.get('DataFrame', 'observation_process')
    overlap = cfg.getboolean('DataFrame', 'overlap')
    
    data_cfgs = {}
    # define long as a boolean if field exists
    if cfg.has_option('DataFrame', 'long'):
        long = cfg.getboolean('DataFrame', 'long')
    else:
        long = False

    if cfg.has_option('DataFrame', 's_dim'):
        data_cfgs['s_dim'] = cfg.getint('DataFrame', 's_dim')
    else:
        data_cfgs['s_dim'] = False
  
    # Load dataset
    train_dataset = Sinusoid(path_to_data=data_dir, split=0, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap, long=long, data_cfgs=data_cfgs)
    val_dataset = Sinusoid(path_to_data=data_dir, split=2, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap, long=long, data_cfgs=data_cfgs)


    train_num = train_dataset.__len__()    
    val_num = val_dataset.__len__()
    
    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Sinusoid dataset
class Sinusoid(Dataset):
    def __init__(self, path_to_data, split, seq_len, x_dim, sample_rate, skip_rate, val_indices, observation_process, device, overlap, long, data_cfgs):
        """
        Initializes the Sinusoid dataset object.
        :param path_to_data: path to the data folder
        :param split: train, test, or val
        :param seq_len: length of the sequence
        :param x_dim: the dimension of the data points
        :param sample_rate: downsampling rate
        :param skip_rate: the skip length to get example, only used for train and test
        :param val_indices: the number of indices/samples used for validation
        :param observation_process: the type of observation process applied to the data
        :param device: the device to load the data onto
        :param overlap: boolean indicating whether the sequences should overlap
        """
        
        self.path_to_data = path_to_data
        self.x_dim = x_dim
        self.seq_len = seq_len
        self.split = split
        self.sample_rate = sample_rate
        self.skip_rate = skip_rate
        self.val_indices = val_indices
        self.observation_process = observation_process
        self.overlap = overlap
        self.long = long
        self.data_cfgs = data_cfgs
        
        # Load and process data
        if self.long:
            filename = f'{self.path_to_data}/sinusoid/dataset_long.pkl'
        elif self.data_cfgs['s_dim'] == 1:
            filename = f'{self.path_to_data}/sinusoid/dataset_1d.pkl'
        else:
            filename = f'{self.path_to_data}/sinusoid/dataset.pkl'
        with open(filename, 'rb') as f:
            the_sequence = np.array(pickle.load(f))
        
        # Process the sequence based on the observation process
        the_sequence = self.apply_observation_process(the_sequence)

        # Generate sequences with or without overlap
        if self.overlap:
            the_sequence = self.create_moving_window_sequences(the_sequence, self.x_dim)
        else: # Remove the last sequence if it is not the correct length
            the_sequence = np.array([the_sequence[i:i+x_dim] for i in range(0, len(the_sequence), x_dim) if i+x_dim <= len(the_sequence)])

        self.seq = torch.from_numpy(the_sequence).float().to(device)
        
        # Determine indices for training and validation sets
        num_frames = self.seq.shape[0]
        all_indices = data_utils.find_indices(num_frames, self.seq_len, num_frames // self.seq_len)
        train_indices, validation_indices = self.split_dataset(all_indices, self.val_indices)

        # Select appropriate indices based on the split
        if self.split <= 1: # for train and test
            valid_frames = train_indices
        else: # for validation
            valid_frames = validation_indices

        # Set the indices for the dataset
        self.data_idx = list(valid_frames)

    def apply_observation_process(self, sequence):
        """
        Applies an observation process to the sequence data.
        """
        if self.observation_process == '3dto3d':
            pass  # For a 3D to 3D observation process
        elif self.observation_process == '3dto3d_w_noise':
            pass  # For a 3D to 3D observation process with noise
        elif self.observation_process == '3dto1d':
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v  # Vector product to convert 3D to 1D
        elif self.observation_process == '3dto1d_w_noise':
            v = np.ones(sequence.shape[-1])
            sequence = sequence @ v + np.random.normal(0, 0.3, sequence.shape[0])  # Add Gaussian noise
        elif self.observation_process == '1dto1d':
            sequence = np.squeeze(sequence)
        elif self.observation_process == '1dto1d_w_noise':
            sequence = np.squeeze(sequence) + np.random.normal(0, 0.3, sequence.shape[0])
        else:
            raise ValueError('Observation process not recognized.')
        return sequence

    @staticmethod
    def create_moving_window_sequences(sequence, window_size):
        """
        Converts a 1D time series into a 2D array of overlapping sequences.
        """
        return np.lib.stride_tricks.sliding_window_view(sequence, window_shape=window_size)

    @staticmethod
    def split_dataset(indices, val_indices):
        """
        Splits the dataset into training and validation sets.
        """
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - val_indices))
        return indices[:split_point], indices[split_point:]

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, index):
        start_frame = self.data_idx[index]
        return self.seq[start_frame:start_frame + self.seq_len]

