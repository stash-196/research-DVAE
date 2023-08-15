#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt

https://github.com/stash-196/research-DVAE

# Evaluation on speech data
python eval_wsj.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT
python eval_wsj.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT # schedule sampling

# Evaluation on human motion data
python eval_h36m.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT
python eval_h36m.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT # schedule sampling
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

def visualize_sequences(true_data, recon_data, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data, label='True Sequence', color='blue')
    plt.plot(recon_data, label='Predicted Sequence', color='red')
    plt.legend()
    plt.title('Comparison of True and Predicted Sequences')
    plt.xlabel('Time steps')
    plt.ylabel('Value')
    plt.grid(True)
    fig_file = os.path.join(save_dir, 'vis_pred_true_series.png')
    plt.savefig(fig_file)
    plt.close()

def spectral_analysis(true_data, recon_data, save_dir):
    true_fft = np.fft.fft(true_data)
    recon_fft = np.fft.fft(recon_data)

    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(true_fft), label='True Spectrum', color='blue')
    plt.plot(np.abs(recon_fft), label='Predicted Spectrum', color='red')
    plt.legend()
    plt.title('Spectral Analysis')
    plt.xlabel('Frequency components')
    plt.ylabel('Magnitude')
    plt.grid(True)
    fig_file = os.path.join(save_dir, 'vis_pred_true_spectrums.png')
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)


    params = Options().get_params()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    train_dataloader, val_dataloader, train_num, val_num = sinusoid_dataset.build_dataloader(cfg, device)
    test_num = len(val_dataloader.dataset)
    print('Test samples: {}'.format(test_num))

    RMSE = 0
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(val_dataloader)):

            batch_data = batch_data.to(device)
            # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
            batch_data = batch_data.permute(1, 0, 2)
            recon_batch_data = dvae(batch_data)
            
            loss_recon = torch.sqrt(mse_loss(batch_data, recon_batch_data))  # Compute RMSE
            seq_len = batch_data.shape[0]
            RMSE += loss_recon.item() / seq_len

            if i == 0:
                true_series = batch_data[0:10, 0, :].reshape(-1).cpu().numpy()
                recon_series = recon_batch_data[0:10, 0, :].reshape(-1).cpu().numpy()

                # Plot the reconstruction vs true sequence
                visualize_sequences(true_series, recon_series, os.path.dirname(params['saved_dict']))
                # Plot the spectral analysis
                spectral_analysis(true_series, recon_series, os.path.dirname(params['saved_dict']))



    RMSE = RMSE / test_num
    print('RMSE: {:.2f}'.format(RMSE))

    results_path = os.path.join(os.path.dirname(params['saved_dict']), 'results.txt')
    with open(results_path, 'w') as f:
        f.write('RMSE: {:.2f}\n'.format(RMSE))

    # Plot the reconstruction
