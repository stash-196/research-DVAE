#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.dataset import sinusoid_dataset
from dvae.utils import EvalMetrics
from torch.nn.functional import mse_loss

torch.manual_seed(0)
np.random.seed(0)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

params = Options().get_params()

if params['ss']:
    learning_algo = LearningAlgorithm_ss(params=params)
    learning_algo.build_model()
    dvae = learning_algo.model
    dvae.out_mean = True
else:
    learning_algo = LearningAlgorithm(params=params)
    learning_algo.build_model()
    dvae = learning_algo.model
dvae.load_state_dict(torch.load(params['saved_dict'], map_location='cpu'))
eval_metrics = EvalMetrics(metric='all')
dvae.eval()
cfg = learning_algo.cfg
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))

# Use the sinusoid dataset
train_dataloader, val_dataloader, train_num, val_num = sinusoid_dataset.build_dataloader(cfg)
test_num = len(val_dataloader.dataset)
print('Test samples: {}'.format(test_num))

RMSE = 0
for _, batch_data in tqdm(enumerate(val_dataloader)):

    batch_data = batch_data.to('cuda')
    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
    batch_data = batch_data.permute(1, 0, 2)
    recon_batch_data = dvae(batch_data)
    
    loss_recon = torch.sqrt(mse_loss(batch_data, recon_batch_data))  # Compute RMSE
    seq_len = batch_data.shape[0]
    RMSE += loss_recon.item() / seq_len

RMSE = RMSE / test_num
print('RMSE: {:.2f}'.format(RMSE))
