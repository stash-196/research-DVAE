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
def build_dataloader(cfg, device):

    # Load dataset params for Lorenz63
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
    train_dataset = Lorenz63(path_to_data=data_dir, split=0, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap, long=long, data_cfgs=data_cfgs)
    val_dataset = Lorenz63(path_to_data=data_dir, split=0, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap, long=long, data_cfgs=data_cfgs)


    train_num = train_dataset.__len__()    
    val_num = val_dataset.__len__()
    
    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Lorenz63 dataset in the same style as the above HumanPoseXYZ, dataset
class Lorenz63(Dataset):
    def __init__(self, path_to_data, split, seq_len, x_dim, sample_rate, skip_rate, val_indices, observation_process, device, overlap, long, data_cfgs):
        """
        :param path_to_data: path to the data folder
        :param split: train, test or val
        :param seq_len: length of the sequence
        :param sample_rate: downsampling rate
        :param skip_rate: the skip length to get example, only used for train and test
        :param val_indices: the number of slices used for validation
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
        
        self.seq = {}
        self.data_idx = []

        self.device = device
        
        # read motion data from pickle file
        filename = '{0}/lorenz63/dataset.pkl'.format(self.path_to_data)
        with open(filename, 'rb') as f:
            the_sequence = np.array(pickle.load(f))
        
        if self.observation_process == '3dto3d':
            pass
        elif self.observation_process == '3dto3d_noisy':
            pass
        elif self.observation_process == '3dto1d':
            # v = np.random.normal(0, 1, the_sequence.shape[-1])
            v = np.ones(the_sequence.shape[-1])
            the_sequence = the_sequence @ v  # Perform vector product to convert 3D to 1D
            the_sequence = np.array([the_sequence[i:i+x_dim] for i in range(0, len(the_sequence), x_dim) if i+x_dim <= len(the_sequence)])
        elif self.observation_process == '3dto1d_w_noise':
            v = np.ones(the_sequence.shape[-1])
            the_sequence = the_sequence @ v + np.random.normal(0, 5.7, the_sequence.shape[0])  # Add Gaussian noise
            the_sequence = np.array([the_sequence[i:i+x_dim] for i in range(0, len(the_sequence), x_dim) if i+x_dim <= len(the_sequence)])

            
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
            sequence = sequence @ v + np.random.normal(0, 5.7, sequence.shape[0])  # Add Gaussian noise
        elif self.observation_process == 'only_x':
            # observe only x out of xyz dimensions
            sequence = sequence[:,0]
        elif self.observation_process == 'only_x_w_noise':
            sequence = sequence[:,0] + np.random.normal(0, 5.7, sequence.shape[0])
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

