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
from dvae.dataset import sinusoid_dataset, lorenz63_dataset
from dvae.utils import EvalMetrics, loss_MSE, create_mode_selector, visualize_variable_evolution, visualize_sequences, visualize_spectral_analysis
from torch.nn.functional import mse_loss
import plotly.graph_objects as go
import plotly.express as px


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

        # If cfg is not provided, set it based on the directory of saved_dict
        if params['cfg'] is None:
            params['cfg'] = os.path.join(os.path.dirname(params['saved_dict']), 'config.ini')
        
        return params


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

    if cfg['DataFrame']["dataset_name"] == "Sinusoid":
        train_dataloader, val_dataloader, train_num, val_num = sinusoid_dataset.build_dataloader(cfg, device)
    elif cfg['DataFrame']["dataset_name"] == "Lorenz63":
        train_dataloader, val_dataloader, train_num, val_num = lorenz63_dataset.build_dataloader(cfg, device)
    else:
        raise ValueError("Unsupported dataset_name in configuration file.")

    overlap = cfg['DataFrame'].getboolean('overlap')

    test_num = len(val_dataloader.dataset)
    print('Test samples: {}'.format(test_num))

    # Check if "alpha" exists in the config.ini under the [Network] section
    alphas_per_unit = None
    if 'Network' in cfg and 'alphas' in cfg['Network']:
        alphas_per_unit = dvae.alphas_per_unit()

    RMSE = 0
    MSE = 0
    MY_MSE = 0
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(val_dataloader)):

            batch_data = batch_data.to(device)
            # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
            batch_data = batch_data.permute(1, 0, 2)
            recon_batch_data = dvae(batch_data)
            
            rmse_recon = torch.sqrt(mse_loss(batch_data, recon_batch_data))  # Compute RMSE
            mse_recon = mse_loss(batch_data, recon_batch_data)  # Compute MSE
            MY_MSE = loss_MSE(batch_data, recon_batch_data)  # Compute MSE

            seq_len, batch_size, x_dim = batch_data.shape
            RMSE += rmse_recon.item() / (seq_len * batch_size)
            MSE += mse_recon.item() / (seq_len * batch_size)
            MY_MSE += MY_MSE.item() / (seq_len * batch_size)

            if i == 0:
                if overlap:
                    true_series_overlapping = batch_data[:, 0, :].reshape(-1).cpu().numpy()

                    recon_series = recon_batch_data[:, 0, :].reshape(-1).cpu().numpy()
                else:        
                    true_series = batch_data[:, 0, :].reshape(-1).cpu().numpy()
                    recon_series = recon_batch_data[:, 0, :].reshape(-1).cpu().numpy()
                # Plot the spectral analysis
                visualize_spectral_analysis(true_series, recon_series, os.path.dirname(params['saved_dict']))

                # visualize the hidden states
                visualize_variable_evolution(dvae.h, os.path.dirname(params['saved_dict']), variable_name='hidden', alphas=alphas_per_unit)

                # visualize the x_features
                visualize_variable_evolution(dvae.feature_x, os.path.dirname(params['saved_dict']), variable_name='x_features')

                # Check if the model has a z variable
                if hasattr(dvae, 'z_mean'):
                    # visualize the latent states
                    visualize_variable_evolution(dvae.z_mean, os.path.dirname(params['saved_dict']), variable_name='z_mean_posterior')
                    visualize_variable_evolution(dvae.z_logvar, os.path.dirname(params['saved_dict']), variable_name='z_logvar_posterior')
                    visualize_variable_evolution(dvae.z_mean_p, os.path.dirname(params['saved_dict']), variable_name='z_mean_prior')
                    visualize_variable_evolution(dvae.z_logvar_p, os.path.dirname(params['saved_dict']), variable_name='z_logvar_prior')


                n_seq = int(1000/x_dim)
                # portion of sequences to reconstruct
                n_gen_portion = 0.5
                recon_len = n_seq - int(n_seq * n_gen_portion)

                mode_selector = create_mode_selector(n_seq, 'half_half')

                # reconstruct the first n_seq sequences
                recon_batch_data = dvae(batch_data[0:n_seq, 0:1, :], mode_selector=mode_selector).detach().clone()

                # generate next n_seq sequences
                # generate_batch_data = dvae(batch_data[recon_len:n_seq, 0:1, :], autonomous=True, initialize_states=False)

                true_series = batch_data[0:n_seq, 0, :].reshape(-1).cpu().numpy()

                # Plot the reconstruction vs true sequence
                visualize_sequences(true_series, recon_batch_data[0:recon_len,0,:], recon_batch_data[recon_len:n_seq,0,:], os.path.dirname(params['saved_dict']))







    RMSE = RMSE / test_num
    print('RMSE: {:.2f}'.format(RMSE))
    print('MSE: {:.2f}'.format(MSE))
    print('MY_MSE: {:.2f}'.format(MY_MSE))

    results_path = os.path.join(os.path.dirname(params['saved_dict']), 'results.txt')
    with open(results_path, 'w') as f:
        f.write('RMSE: {:.2f}\n'.format(RMSE))
        f.write('MSE: {:.2f}\n'.format(MSE))
        f.write('MY_MSE: {:.2f}\n'.format(MY_MSE))

    # Plot the reconstruction
