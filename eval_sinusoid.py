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
from dvae.utils import EvalMetrics, loss_MSE
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

        # If cfg is not provided, set it based on the directory of saved_dict
        if params['cfg'] is None:
            params['cfg'] = os.path.join(os.path.dirname(params['saved_dict']), 'config.ini')
        
        return params

def visualize_sequences(true_data, recon_data, save_dir, n_gen_portion, name=''):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data, label='True Sequence', color='blue')

    recon_length = len(recon_data) - int(len(recon_data) * n_gen_portion)

    # Plotting the reconstructed part in red
    plt.plot(recon_data[:recon_length], label='Reconstructed Sequence', color='red')
    
    # Plotting the self-generated part in green
    plt.plot(range(recon_length, len(recon_data)), recon_data[recon_length:], label='Self-Generated Sequence', color='green')
    
    plt.legend()
    plt.title('Comparison of True and Predicted Sequences')
    plt.xlabel('Time steps')
    plt.ylabel('Value')
    plt.grid(True)
    fig_file = os.path.join(save_dir, f'vis_pred_true_series{name}.png')
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

def visualize_variable_evolution(states, save_dir, variable_name='h', alphas=None):
    plt.figure(figsize=(12, 8))
    
    num_dims = states.shape[2]
    
    # If alphas are provided, determine unique colors based on unique alphas.
    if alphas is not None:
        colors = [plt.cm.viridis(alpha.item()) for alpha in alphas]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_dims))
  
    # Given h_states is of shape (seq_len, batch_size, dim)
    # For this example, we are plotting for batch 0 and all dimensions
    for dim in range(num_dims):
        plt.plot(states[:, 0, dim].cpu().numpy(), label=f'Dim {dim}', color=colors[dim])
    
    str_alphas = ' α:' + str(set(alphas.numpy())) if alphas is not None else ''

    plt.title(f'Evolution of {variable_name} States over Time' + str_alphas)
    plt.xlabel('Time steps')
    plt.ylabel(f'{variable_name} state value')
    if num_dims <= 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True)

    fig_file = os.path.join(save_dir, f'vis_{variable_name}_state_evolution.png')
    plt.savefig(fig_file, bbox_inches='tight')
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

    # Check if "alpha" exists in the config.ini under the [Network] section
    alphas_per_unit = None
    if 'Network' in cfg and 'alphas' in cfg['Network']:
        alphas_per_unit = dvae.alphas_per_unit

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
                n_seq = 20
                n_gen_portion = 0.5
                recon_len = n_seq - int(n_seq * n_gen_portion)
                # reconstruct the first n_seq sequences
                recon_batch_data = dvae(batch_data[0:recon_len, :, :], autonomous=False)
                # generate next n_seq sequences
                generate_batch_data = dvae(batch_data[recon_len:n_seq, :, :], autonomous=True)

                true_series = batch_data[0:n_seq, 0, :].reshape(-1).cpu().numpy()
                # concatinate reconstructions and generations
                recon_series = torch.cat((recon_batch_data[:, 0, :], generate_batch_data[:, 0, :]), dim=0).reshape(-1).cpu().numpy()

                # Plot the reconstruction vs true sequence
                visualize_sequences(true_series, recon_series, os.path.dirname(params['saved_dict']), n_gen_portion)
                # Plot the spectral analysis
                spectral_analysis(true_series, recon_series, os.path.dirname(params['saved_dict']))

                # visualize the hidden states
                visualize_variable_evolution(dvae.h, os.path.dirname(params['saved_dict']), variable_name='hidden', alphas=alphas_per_unit)

                # Check if the model has a z variable
                if hasattr(dvae, 'z_mean'):
                    # visualize the latent states
                    visualize_variable_evolution(dvae.z_mean, os.path.dirname(params['saved_dict']), variable_name='z_mean', alphas=alphas_per_unit)
                    visualize_variable_evolution(dvae.z_logvar, os.path.dirname(params['saved_dict']), variable_name='z_logvar', alphas=alphas_per_unit)





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
