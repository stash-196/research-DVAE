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
    val_indices = cfg.getint('DataFrame', 'val_indices')
    observation_process = cfg.get('DataFrame', 'observation_process')
  
    # Load dataset
    train_dataset = Sinusoid(path_to_data=data_dir, split=0, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device)
    val_dataset = Sinusoid(path_to_data=data_dir, split=2, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device)


    train_num = train_dataset.__len__()    
    val_num = val_dataset.__len__()
    
    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Sinusoid dataset
class Sinusoid(Dataset):
    def __init__(self, path_to_data, split, seq_len, x_dim, sample_rate, skip_rate, val_indices, observation_process, device, overlap):
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
        
        self.seq = {}
        self.data_idx = []

        self.device = device
        
        # read motion data from pickle file
        filename = '{0}/dataset.pkl'.format(self.path_to_data)
        with open(filename, 'rb') as f:
            the_sequence = np.array(pickle.load(f))
        
        if self.observation_process == '3dto3d':
            pass
        elif self.observation_process == '3dto3d_w_noise':
            pass
        elif self.observation_process == '3dto1d':
            # v = np.random.normal(0, 1, the_sequence.shape[-1])
            v = np.ones(the_sequence.shape[-1])
            the_sequence = the_sequence @ v  # Perform vector product to convert 3D to 1D
        elif self.observation_process == '3dto1d_w_noise':
            v = np.ones(the_sequence.shape[-1])
            the_sequence = the_sequence @ v + np.random.normal(0, 0.3, the_sequence.shape[0])  # Add Gaussian noise

        if self.overlap:
            the_sequence = self.create_moving_window_sequences(the_sequence, x_dim)
        else:
            the_sequence = np.array([the_sequence[i:i+x_dim] for i in range(0, len(the_sequence), x_dim) if i+x_dim <= len(the_sequence)])
            
        # Convert to torch tensor and send to device
        self.seq = torch.from_numpy(the_sequence).float().to(self.device)
        
        # save valid start frames, based on skip_rate
        num_frames = self.seq.shape[0] # Number of complete sequences in the data


        all_indices = data_utils.find_indices(num_frames, self.seq_len, num_frames // self.seq_len)
        train_indices, validation_indices = self.split_dataset(all_indices, self.val_indices)


        if self.split <= 1: # for train and test
            valid_frames = train_indices
        else: # for validation
            valid_frames = validation_indices
        
        self.data_idx = list(valid_frames)
    
    def create_moving_window_sequences(self, sequence, window_size):
        """
        Converts a 1D time series into a sequence of vectors representing a moving window,
        overlapping by one time step at each step.

        :param sequence: The original 1D time series.
        :param window_size: The size of the moving window (equivalent to x_dim).
        :return: A 2D numpy array where each row is a windowed sequence.
        """
        sequence_length = len(sequence)
        num_sequences = sequence_length - window_size + 1
        windowed_sequences = np.zeros((num_sequences, window_size))

        for i in range(num_sequences):
            windowed_sequences[i] = sequence[i:i + window_size]

        return windowed_sequences

    def split_dataset(self, indices, val_indices):

        np.random.seed(42)  # Set a random seed for reproducibility
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Compute the split point
        split_point = len(indices) - val_indices
        
        # Split indices into training and validation sets
        train_indices = indices[:split_point]
        validation_indices = indices[split_point:]
        
        return train_indices, validation_indices

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, item):
        start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len)
        return self.seq[fs]
