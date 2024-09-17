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
    data_dir = cfg.get('User', 'data_dir')
    x_dim = cfg.getint('Network', 'x_dim')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
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
    train_dataset = Xhro(path_to_data=data_dir, split='train', seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate,
                             skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap)
    val_dataset = Xhro(path_to_data=data_dir, split='valid', seq_len=sequence_len, x_dim=x_dim, sample_rate=sample_rate,
                           skip_rate=skip_rate, val_indices=val_indices, observation_process=observation_process, device=device, overlap=overlap)

    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()

    # Build dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, train_num, val_num


# define a class for Lorenz63 dataset in the same style as the above HumanPoseXYZ, dataset
class Xhro(Dataset):
    def __init__(self, path_to_data, split, seq_len, x_dim, sample_rate, skip_rate, val_indices, observation_process, device, overlap, shuffle=True):
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
        self.shuffle = shuffle
        self.device = device

        
        # read data from pickle file
        if split == 'test':
            filename = '{0}/Xhro/dataset_test.pkl'.format(
                self.path_to_data)
        else:
            filename = '{0}/Xhro/dataset.pkl'.format(self.path_to_data)

        with open(filename, 'rb') as f:
            the_sequence = np.array(pickle.load(f))

        # Store the full sequence before applying any observation process
        self.full_sequence = the_sequence

        # Apply conditional observation process
        if self.observation_process == 'raw':
            self.sequence = the_sequence
        elif self.observation_process == '':
            self.sequence = the_sequence