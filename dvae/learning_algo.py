#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""


import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, loss_ISD, loss_KLD, loss_MPJPE, loss_MSE, create_autonomous_mode_selector, visualize_model_parameters, visualize_combined_parameters, visualize_teacherforcing_2_autonomous, profile_execution
from .dataset import h36m_dataset, speech_dataset, lorenz63_dataset, sinusoid_dataset
from .model import build_VAE, build_DKF, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE, build_RNN, build_MT_RNN, build_MT_VRNN_pp
import subprocess


class LearningAlgorithm():
    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """
    def __init__(self, params):
        # Load config parser
        self.params = params
        self.job_id = self.params['job_id']
        self.config_file = self.params['cfg']
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.experiment_name = self.cfg.get('User', 'experiment_name')
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')
        self.sequence_len = self.cfg.getint('DataFrame', 'sequence_len')

        self.sampling_method = self.cfg.get('Training', 'sampling_method')
        self.sampling_ratio = self.cfg.getfloat('Training', 'sampling_ratio')

        # Get host name and date
        self.hostname = socket.gethostname()
        # Get current date
        self.datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.time_str = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'


        # if optimize_alphas is not '', turn into boolian
        try:
            # Try to fetch the boolean value from the configuration.
            self.optimize_alphas = self.cfg.getboolean('Training', 'optimize_alphas')
        except ValueError:
            # If there is a ValueError, likely because the value is empty or invalid,
            # set self.optimize_alphas to False.
            self.optimize_alphas = None
        
        if self.optimize_alphas is not None:
            self.alphas = [float(i) for i in self.cfg.get('Network', 'alphas').split(',') if i != '']




    def build_model(self):
        if self.model_name == 'VRNN':
            self.model = build_VRNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'RNN':
            self.model = build_RNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'MT_RNN':
            self.model = build_MT_RNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'MT_VRNN':
            self.model = build_MT_VRNN_pp(cfg=self.cfg, device=self.device)

        

    def init_optimizer(self):
        optimization = self.cfg.get('Training', 'optimization')
        self.lr = self.cfg.getfloat('Training', 'lr')
        if self.optimize_alphas:
            self.alpha_lr = self.cfg.getfloat('Training', 'alpha_lr', fallback=self.lr)  # defaults to the same as lr if not present
            params = [
                {'params': self.model.base_parameters(), 'lr': self.lr},
                {'params': [self.model.sigmas], 'lr': self.alpha_lr}  # include sigma values
            ]
        else:
            params = self.model.parameters()

        if optimization == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr)
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)  # fallback to Adam if optimization method is not recognized

        return optimizer



    def get_basic_info(self):
        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.datetime_str)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        
        return basic_info

    # @profile_execution
    def train(self, profiler=None):
        ############
        ### Init ###
        ############

        # Build model
        self.build_model()

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        if not self.params['reload']:
            saved_root = self.cfg.get('User', 'saved_root')
            tag = self.cfg.get('Network', 'tag')

            if self.optimize_alphas:
                filename = "{}_{}_{}_{}_SM-{}_α-{}".format(self.job_id, self.dataset_name, self.datetime_str, tag, self.sampling_method, self.alphas)
            else:
                filename = "{}_{}_{}_{}_SM-{}".format(self.job_id, self.dataset_name, self.datetime_str, tag, self.sampling_method)

            if self.job_id is not None:
                save_dir = os.path.join(saved_root, self.date_str, f"{self.experiment_name}", filename)
            else:
                save_dir = os.path.join(saved_root, self.date_str, f"{self.experiment_name}", filename)
                
            print(f"Saving to: {save_dir}")
            try:
                os.makedirs(save_dir)
            except FileExistsError:
                # The directory already exists, so you can either pass or handle it as needed
                print(f"Directory already exists: {save_dir}")
        else:
            tag = self.cfg.get('Network', 'tag')
            save_dir = self.params['model_dir']
            

        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

            
        # Init optimizer
        optimizer = self.init_optimizer()

        # Create data loader
        if self.dataset_name == "Lorenz63":
            train_dataloader, val_dataloader, train_num, val_num = lorenz63_dataset.build_dataloader(self.cfg, device=self.device)
        elif self.dataset_name == "Sinusoid":
            train_dataloader, val_dataloader, train_num, val_num = sinusoid_dataset.build_dataloader(self.cfg, device=self.device)
        else:
            logger.error('Unknown datset!')
        logger.info('Training samples: {}'.format(train_num))
        logger.info('Validation samples: {}'.format(val_num))

        ######################
        ### Batch Training ###
        ######################

        # Load training parameters  
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')
        beta = self.cfg.getfloat('Training', 'beta')
        kl_warm = 0

        # Create python list for loss
        if not self.params['reload']:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_kl = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_kl = np.zeros((epochs,))
            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = epochs
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            start_epoch = -1
            if self.optimize_alphas:
                alphas_init = [float(i) for i in self.cfg.get('Network', 'alphas').split(',') if i != '']
                sigmas_history = np.zeros((len(alphas_init), epochs))
                # set initial values of sigmas_history with alphas_init
                sigmas_history[:, 0] = sigmoid_reverse(alphas_init)
        else:
            cp_file = os.path.join(save_dir, '{}_checkpoint.pt'.format(self.model_name))
            checkpoint = torch.load(cp_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            start_epoch = checkpoint['epoch']
            loss_log = checkpoint['loss_log']
            logger.info('Resuming trainning: epoch: {}'.format(start_epoch))
            train_loss = np.pad(loss_log['train_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_loss = np.pad(loss_log['val_loss'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_recon = np.pad(loss_log['train_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            train_kl = np.pad(loss_log['train_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_recon = np.pad(loss_log['val_recon'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            val_kl = np.pad(loss_log['val_kl'], (0, epochs-start_epoch), mode='constant', constant_values=0)
            best_val_loss = checkpoint['best_val_loss']
            cpt_patience = 0
            cur_best_epoch = start_epoch
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            if self.optimize_alphas:
                alphas_init = [float(i) for i in self.cfg.get('Network', 'alphas').split(',') if i != '']
                sigmas_history = np.zeros((len(alphas_init), epochs))
                # set initial values of sigmas_history with alphas_init
                sigmas_history[:, 0] = sigmoid_reverse(alphas_init)

        kl_warm_epochs = [0]
        kl_warm_values = [0]
        best_state_epochs = []
        cpt_patience_epochs = []
        delta_per_epoch = []

        self.model.to(self.device)

        # Train with mini-batch SGD
        for epoch in range(start_epoch+1, epochs):
            
            start_time = datetime.datetime.now()
            
            if self.sampling_method == 'ss':
                model_mode_selector = create_autonomous_mode_selector(self.sequence_len, mode='bernoulli_sampling', autonomous_ratio=kl_warm*self.sampling_ratio)
            elif self.sampling_method == 'ptf':
                model_mode_selector = create_autonomous_mode_selector(self.sequence_len, mode='bernoulli_sampling', autonomous_ratio=self.sampling_ratio)
            elif self.sampling_method == 'mtf':
                model_mode_selector = create_autonomous_mode_selector(self.sequence_len, mode='mix_sampling', autonomous_ratio=self.sampling_ratio)
            else:  # error
                logger.error('Unknown sampling method')
                break




            # Batch training
            for _, batch_data in enumerate(train_dataloader):
                batch_data = batch_data.to(self.device)
                print(f'sent to device: {self.device}')
                autonomous_ratio = kl_warm * 0.8
                
                if self.dataset_name == 'WSJ0':
                    # (batch_size, x_dim, seq_len) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(2, 0, 1)
                    recon_batch_data = torch.exp(self.model(batch_data, mode_selector=model_mode_selector)) # output log-variance
                    loss_recon = loss_ISD(batch_data, recon_batch_data)
                elif self.dataset_name == 'H36M':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2) / 1000 # normalize to meters
                    recon_batch_data = self.model(batch_data, mode_selector=model_mode_selector)
                    loss_recon = loss_MPJPE(batch_data*1000, recon_batch_data*1000)
                elif self.dataset_name == 'Lorenz63' or self.dataset_name == 'Sinusoid':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    recon_batch_data = self.model(batch_data, mode_selector=model_mode_selector)
                    loss_recon = loss_MSE(batch_data, recon_batch_data)
                else:
                    logger.error('Unknown datset')
                seq_len, bs, _ = self.model.y.shape # Sequence Length and Batch Size
                loss_recon_avg = loss_recon / (seq_len * bs) # Average Reconstruction Loss

                if self.model_name == 'DSAE':
                    loss_kl_z = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                    loss_kl_v = loss_KLD(self.model.v_mean, self.model.v_logvar, self.model.v_mean_p, self.model.v_logvar_p)
                    loss_kl = loss_kl_z + loss_kl_v
                elif self.model_name == 'RNN' or self.model_name == 'MT_RNN':
                    loss_kl = torch.zeros(1).to(self.device)
                else:
                    loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl_avg = kl_warm * beta * loss_kl / (seq_len * bs) # Average KL Divergence
                
                # Print device of loss
                print(f'loss_recon.device: {loss_recon.device}, loss_kl.device: {loss_kl.device}')
                loss_tot_avg = loss_recon_avg + loss_kl_avg
                optimizer.zero_grad()
                loss_tot_avg.backward()

                # Gradient clipping
                # check if model has model.type_rnn, if True, and if type_rnn=='RNN', then clip the gradient
                if hasattr(self.model, 'type_rnn'):
                    if self.model.type_rnn == 'RNN':
                        self.gradient_clip = self.cfg.getfloat('Training', 'gradient_clip')
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                optimizer.step()
                # profiler.step()

                train_loss[epoch] += loss_tot_avg.item() * bs
                train_recon[epoch] += loss_recon_avg.item() * bs
                train_kl[epoch] += loss_kl_avg.item() * bs
                if self.optimize_alphas:
                    sigmas_history[:, epoch] = self.model.sigmas.detach().cpu().numpy()


                
            # Validation
            for _, batch_data in enumerate(val_dataloader):

                batch_data = batch_data.to(self.device)             

                if self.dataset_name == 'WSJ0':
                    # (batch_size, x_dim, seq_len) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(2, 0, 1)
                    recon_batch_data = torch.exp(self.model(batch_data, mode_selector=model_mode_selector)) # output log-variance
                    loss_recon = loss_ISD(batch_data, recon_batch_data)
                elif self.dataset_name == 'H36M':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2) / 1000 # normalize to meters
                    recon_batch_data = self.model(batch_data, mode_selector=model_mode_selector)
                    loss_recon = loss_MPJPE(batch_data*1000, recon_batch_data*1000)
                elif self.dataset_name == 'Lorenz63' or self.dataset_name == 'Sinusoid':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    recon_batch_data = self.model(batch_data, mode_selector=model_mode_selector)
                    loss_recon = loss_MSE(batch_data, recon_batch_data)
                seq_len, bs, _ = self.model.y.shape
                loss_recon_avg = loss_recon / (seq_len * bs)
                
                if self.model_name == 'DSAE':
                    loss_kl_z = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                    loss_kl_v = loss_KLD(self.model.v_mean, self.model.v_logvar, self.model.v_mean_p, self.model.v_logvar_p)
                    loss_kl = loss_kl_z + loss_kl_v
                elif self.model_name == 'RNN' or self.model_name == 'MT_RNN':
                    loss_kl = torch.zeros(1).to(self.device)
                else:
                    loss_kl = loss_KLD(self.model.z_mean, self.model.z_logvar, self.model.z_mean_p, self.model.z_logvar_p)
                loss_kl_avg = kl_warm * beta * loss_kl / (seq_len * bs)

                print(f'loss_recon.device: {loss_recon.device}, loss_kl.device: {loss_kl.device}')
                loss_tot_avg = loss_recon_avg + loss_kl_avg

                val_loss[epoch] += loss_tot_avg.item() * bs
                val_recon[epoch] += loss_recon_avg.item() * bs
                val_kl[epoch] += loss_kl_avg.item() * bs

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_kl[epoch] = train_kl[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_kl[epoch] = val_kl[epoch] / val_num

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            logger.info('Epoch: {} training time {:.2f}m'.format(epoch, interval))
            logger.info('Train => tot: {:.2f} recon {:.2f} KL {:.2f} Val => tot: {:.2f} recon {:.2f} KL {:.2f}'.format(train_loss[epoch], train_recon[epoch], train_kl[epoch], val_loss[epoch], val_recon[epoch], val_kl[epoch]))


            delta = val_loss[epoch] - best_val_loss
            delta_per_epoch.append(delta)

            # Early stop patiance
            if delta < -1e-4:
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            cpt_patience_epochs.append(cpt_patience)
            best_state_epochs.append(epoch)

            # KL warm-up
            if epoch % early_stop_patience == 0 and kl_warm < 1 and epoch > 0:
                kl_warm += 0.2 
                kl_warm_epochs += [epoch]
                kl_warm_values += [kl_warm]
                logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
                # reset early stop patience and best_val_loss
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch

            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience:
                if kl_warm >= 1.0:
                    logger.info('Early stop patience achieved')
                    break
                else:
                    # increase kl_warm
                    kl_warm += 0.2 
                    kl_warm_epochs += [epoch]
                    kl_warm_values += [kl_warm]
                    logger.info('KL warm-up, anneal coeff: {}'.format(kl_warm))
                    cpt_patience = 0
                    best_val_loss = val_loss[epoch]
                    best_state_dict = self.model.state_dict()
                    best_optim_dict = optimizer.state_dict()
                    cur_best_epoch = epoch
            
            

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                loss_log = {'train_loss': train_loss[:cur_best_epoch+1],
                            'val_loss': val_loss[:cur_best_epoch+1],
                            'train_recon': train_recon[:cur_best_epoch+1],
                            'train_kl': train_kl[:cur_best_epoch+1], 
                            'val_recon': val_recon[:cur_best_epoch+1], 
                            'val_kl': val_kl[:cur_best_epoch+1]}
                save_file = os.path.join(save_dir, self.model_name + '_checkpoint.pt')
                torch.save({'epoch': cur_best_epoch,
                            'best_val_loss': best_val_loss,
                            'cpt_patience': cpt_patience,
                            'model_state_dict': best_state_dict,
                            'optim_state_dict': best_optim_dict,
                            'loss_log': loss_log
                        }, save_file)
                logger.info('Epoch: {} ===> checkpoint stored with current best epoch: {}'.format(epoch, cur_best_epoch))

                # Save the loss figure
                save_figures_dir = os.path.join(save_dir, 'vis_during_training')
                if not os.path.exists(save_figures_dir):
                    os.makedirs(save_figures_dir)

                plt.clf()
                fig = plt.figure(figsize=(8,6))
                plt.rcParams['font.size'] = 12
                # add vertical line for each epoch where KL warm-up is increased
                for kl_warm_epoch in kl_warm_epochs:
                    plt.axvline(x=kl_warm_epoch, color='r', linestyle='--')
                plt.plot(train_loss[:epoch+1], label='training loss')
                plt.plot(val_loss[:epoch+1], label='validation loss')
                plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
                plt.xlabel('epochs', fontdict={'size':16})
                plt.ylabel('loss', fontdict={'size':16})
                fig_file = os.path.join(save_figures_dir, 'vis_training_loss_{}.png'.format(tag))
                plt.savefig(fig_file)
                plt.close(fig)

                plt.clf()
                fig = plt.figure(figsize=(8,6))
                plt.rcParams['font.size'] = 12
                for kl_warm_epoch in kl_warm_epochs:
                    plt.axvline(x=kl_warm_epoch, color='r', linestyle='--')
                plt.plot(train_recon[:epoch+1], label='Training')
                plt.plot(val_recon[:epoch+1], label='Validation')
                plt.legend(fontsize=16, title='{}: Recon. Loss'.format(self.model_name), title_fontsize=20)
                plt.xlabel('epochs', fontdict={'size':16})
                plt.ylabel('loss', fontdict={'size':16})
                fig_file = os.path.join(save_figures_dir, 'vis_training_loss_recon_{}.png'.format(tag))
                plt.savefig(fig_file) 
                plt.close(fig)

                plt.clf()
                fig = plt.figure(figsize=(8,6))
                plt.rcParams['font.size'] = 12
                for kl_warm_epoch in kl_warm_epochs:
                    plt.axvline(x=kl_warm_epoch, color='r', linestyle='--')
                plt.plot(train_kl[:epoch+1], label='Training')
                plt.plot(val_kl[:epoch+1], label='Validation')
                plt.legend(fontsize=16, title='{}: KL Divergence'.format(self.model_name), title_fontsize=20)
                plt.xlabel('epochs', fontdict={'size':16})
                plt.ylabel('loss', fontdict={'size':16})
                fig_file = os.path.join(save_figures_dir, 'vis_training_loss_KLD_{}.png'.format(tag))
                plt.savefig(fig_file)
                plt.close(fig)


                # Save visualization of delta_per_epoch, kl_warm_values, cpt_patience_epochs, best_state_epochs with subplots
                plt.clf()
                fig, axs = plt.subplots(4, 1, figsize=(12, 12))
                plt.rcParams['font.size'] = 12
                axs[0].plot(delta_per_epoch[:epoch+1], label='delta')
                axs[0].legend(fontsize=16, title='delta', title_fontsize=20)
                axs[0].set_xlabel('epochs', fontdict={'size':16})
                axs[0].set_ylabel('delta', fontdict={'size':16})
                axs[1].step(kl_warm_epochs[:epoch+1], kl_warm_values[:epoch+1], label='kl_warm')
                axs[1].legend(fontsize=16, title='kl_warm', title_fontsize=20)
                axs[1].set_xlabel('epochs', fontdict={'size':16})
                axs[1].set_ylabel('kl_warm', fontdict={'size':16})
                axs[2].plot(cpt_patience_epochs[:epoch+1], label='cpt_patience')
                axs[2].legend(fontsize=16, title='cpt_patience', title_fontsize=20)
                axs[2].set_xlabel('epochs', fontdict={'size':16})
                axs[2].set_ylabel('cpt_patience', fontdict={'size':16})
                axs[3].step(range(epoch+1), best_state_epochs, label='best_state')
                axs[3].legend(fontsize=16, title='best_state', title_fontsize=20)
                axs[3].set_xlabel('epochs', fontdict={'size':16})
                axs[3].set_ylabel('best_state', fontdict={'size':16})
                fig_file = os.path.join(save_figures_dir, 'vis_training_delta_kl_cpt_best_state_{}.png'.format(tag))
                plt.savefig(fig_file)
                plt.close(fig)


                # Save hisotry of sigma and alpha
                if self.optimize_alphas:
                    plt.clf()
                    fig = plt.figure(figsize=(8,6))
                    plt.rcParams['font.size'] = 12
                    for kl_warm_epoch in kl_warm_epochs:
                        plt.axvline(x=kl_warm_epoch, color='r', linestyle='--')
                    for i in range(sigmas_history.shape[0]):
                        plt.plot(sigmas_history[i, :epoch+1], label='Sigma {}'.format(i+1))
                    plt.legend(fontsize=16, title='Sigma values', title_fontsize=20)
                    plt.xlabel('epochs', fontdict={'size':16})
                    plt.ylabel('sigma', fontdict={'size':16})
                    fig_file = os.path.join(save_figures_dir, 'vis_training_history_of_sigma_{}.png'.format(tag))
                    plt.savefig(fig_file)
                    plt.close(fig)

                    plt.clf()
                    fig = plt.figure(figsize=(8,6))
                    plt.rcParams['font.size'] = 12
                    for kl_warm_epoch in kl_warm_epochs:
                        plt.axvline(x=kl_warm_epoch, color='r', linestyle='--')
                    for i in range(sigmas_history.shape[0]):
                        alphas = 1 / (1 + np.exp(-sigmas_history[i, :]))
                        plt.plot(alphas[:epoch+1], label='Alpha {}'.format(i+1))
                    plt.legend(fontsize=16, title='Alpha values', title_fontsize=20)
                    plt.xlabel('epochs', fontdict={'size':16})
                    plt.ylabel('alpha', fontdict={'size':16})
                    plt.yscale('log')  # Set y-axis to logarithmic scale
                    fig_file = os.path.join(save_figures_dir, 'vis_training_history_of_alpha_{}.png'.format(tag))
                    plt.savefig(fig_file)
                    plt.close(fig)

                visualize_combined_parameters(self.model, explain='epoch_{}'.format(epoch), save_path=save_figures_dir)

                visualize_teacherforcing_2_autonomous(batch_data, self.model, mode_selector=model_mode_selector, save_path=save_figures_dir, explain='epoch:{}_klwarm:{}'.format(epoch, kl_warm), inference_mode=True)

                if self.optimize_alphas:
                    alphas = 1 / (1 + np.exp(-sigmas_history[:, epoch]))
                    logger.info('alphas: {}'.format([f'{alpha:.5f}' for alpha in alphas]))


        # Save the final weights of network with the best validation loss
        save_file = os.path.join(save_dir, self.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_kl = train_kl[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_kl = val_kl[:epoch+1]
        if self.optimize_alphas:
            sigmas_history = sigmas_history[:, :epoch+1]
        loss_file = os.path.join(save_dir, 'loss_model.pckl')

        # create dictionary to save(pickle) the loss
        pickle_dict = {'train_loss': train_loss, 'val_loss': val_loss, 'train_recon': train_recon, 'train_kl': train_kl, 'val_recon': val_recon, 'val_kl': val_kl, 'kl_warm_epochs': kl_warm_epochs}
        if self.optimize_alphas:
            pickle_dict['sigmas_history'] = sigmas_history

        with open(loss_file, 'wb') as f:
            # if self.optimize_alphas:
            #     pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl, sigmas_history], f)
            # else:
            #     pickle.dump([train_loss, val_loss, train_recon, train_kl, val_recon, val_kl], f)
            pickle.dump(pickle_dict, f)

            print('Loss saved in: {}'.format(loss_file))
            


        # run evaluation script
        subprocess.run(["python", "eval_sinusoid.py", "--saved_dict", save_file])
        

        
# sigmoid function for arrays
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

# reverse of sigmoid function for arrays
def sigmoid_reverse(x):
    x = np.array(x)
    return np.log(x / (1 - x))
