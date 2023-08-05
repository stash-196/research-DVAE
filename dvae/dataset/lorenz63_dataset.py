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
    val_indices = cfg.getint('DataFrame', 'val_indices')
    observation_process = cfg.get('DataFrame', 'observation_process')
  
    # Load dataset
    train_dataset = Lorenz63(path_to_data=data_dir, split=0, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device)
    val_dataset = Lorenz63(path_to_data=data_dir, split=2, seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate, skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device)


    train_num = train_dataset.__len__()    
    val_num = val_dataset.__len__()
    
    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Lorenz63 dataset in the same style as the above HumanPoseXYZ, dataset
class Lorenz63(Dataset):
    def __init__(self, path_to_data, split, seq_len, x_dim, sample_rate, skip_rate, val_indices, observation_process, device):
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
        
        self.seq = {}
        self.data_idx = []

        self.device = device
        
        # read motion data from pickle file
        filename = '{0}/dataset.pkl'.format(self.path_to_data)
        with open(filename, 'rb') as f:
            the_sequence = np.array(pickle.load(f))
        
        if self.observation_process == '3dto3d':
            pass
        elif self.observation_process == '3dto3d_noisy':
            pass
        elif self.observation_process == '3dto1d':
            v = np.random.normal(0, 1, the_sequence.shape[-1])
            the_sequence = the_sequence @ v  # Perform vector product to convert 3D to 1D
            the_sequence = np.array([the_sequence[i:i+x_dim] for i in range(0, len(the_sequence), x_dim) if i+x_dim <= len(the_sequence)])
  # Divide 1D time series into frames of x_dim
            # the_sequence = np.array([the_sequence[i:i+seq_len] for i in range(0, len(the_sequence), seq_len) if i+seq_len <= len(the_sequence)])  # Divide frames into sequences of num_seq
        elif self.observation_process == '3dto1d_noisy':
            pass
            
        
        # Convert to torch tensor and send to device
        self.seq = torch.from_numpy(the_sequence).float().to(self.device)
        
        # save valid start frames, based on skip_rate
        num_frames = self.seq.shape[0] # Number of complete sequences in the data
        
        if self.split <= 1: # for train and test
            valid_frames = np.arange(0, num_frames - self.seq_len + 1, self.skip_rate)
        else: # for validation
            valid_frames = data_utils.find_indices(num_frames, self.seq_len, self.val_indices)
        
        self.data_idx = list(valid_frames)
        
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, item):
        start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len)
        return self.seq[fs]
