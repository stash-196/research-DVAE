import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_total_loss(train_loss, val_loss, kl_warm_epochs, auto_warm_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        plt.axvline(x=auto_warm_epoch, color='r', linestyle=':')
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend(
        fontsize=16, title=f'{model_name}: Total Loss', title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size': 16})
    plt.ylabel('loss', fontdict={'size': 16})
    fig_file = os.path.join(save_figures_dir, f'vis_training_loss_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_recon_loss(train_recon, val_recon, kl_warm_epochs, auto_warm_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        plt.axvline(x=auto_warm_epoch, color='r', linestyle=':')
    plt.plot(train_recon, label='Training')
    plt.plot(val_recon, label='Validation')
    plt.legend(
        fontsize=16, title=f'{model_name}: Recon. Loss', title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size': 16})
    plt.ylabel('loss', fontdict={'size': 16})
    fig_file = os.path.join(
        save_figures_dir, f'vis_training_loss_recon_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_kld_loss(train_kl, val_kl, kl_warm_epochs, auto_warm_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        plt.axvline(x=auto_warm_epoch, color='r', linestyle=':')
    plt.plot(train_kl, label='Training')
    plt.plot(val_kl, label='Validation')
    plt.legend(
        fontsize=16, title=f'{model_name}: KL Div. Loss', title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size': 16})
    plt.ylabel('loss', fontdict={'size': 16})
    fig_file = os.path.join(
        save_figures_dir, f'vis_training_loss_KLD_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_combined_metrics(delta_per_epoch, kl_warm_epochs, auto_warm_epochs, kl_warm_values, auto_warm_values, cpt_patience_epochs, best_state_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    plt.rcParams['font.size'] = 12
    axs[0].plot(delta_per_epoch, label='delta')
    axs[0].legend(fontsize=16, title='delta', title_fontsize=20)
    axs[0].set_xlabel('epochs', fontdict={'size': 16})
    axs[0].set_ylabel('delta', fontdict={'size': 16})
    axs[1].step(np.arange(len(kl_warm_values)),
                kl_warm_values, label='kl_warm', color='c')
    axs[1].step(np.arange(len(auto_warm_values)),
                auto_warm_values, label='auto_warm', color='r')
    axs[1].legend(fontsize=16, title='warm values', title_fontsize=20)
    axs[1].set_xlabel('epochs', fontdict={'size': 16})
    axs[1].set_ylabel('warm values', fontdict={'size': 16})
    axs[2].plot(cpt_patience_epochs, label='cpt_patience')
    axs[2].legend(fontsize=16, title='cpt_patience', title_fontsize=20)
    axs[2].set_xlabel('epochs', fontdict={'size': 16})
    axs[2].set_ylabel('cpt_patience', fontdict={'size': 16})
    axs[3].step(range(len(best_state_epochs)),
                best_state_epochs, label='best_state')
    axs[3].legend(fontsize=16, title='best_state', title_fontsize=20)
    axs[3].set_xlabel('epochs', fontdict={'size': 16})
    axs[3].set_ylabel('best_state', fontdict={'size': 16})
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            axs[0].axvline(x=kl_warm_epoch, color='c', linestyle='--')
            axs[1].axvline(x=kl_warm_epoch, color='c', linestyle='--')
            axs[2].axvline(x=kl_warm_epoch, color='c', linestyle='--')
            axs[3].axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        axs[0].axvline(x=auto_warm_epoch, color='r', linestyle=':')
        axs[1].axvline(x=auto_warm_epoch, color='r', linestyle=':')
        axs[2].axvline(x=auto_warm_epoch, color='r', linestyle=':')
        axs[3].axvline(x=auto_warm_epoch, color='r', linestyle=':')
    fig_file = os.path.join(
        save_figures_dir, f'vis_training_delta_kl_cpt_best_state_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_sigma_history(sigmas_history, kl_warm_epochs, auto_warm_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        plt.axvline(x=auto_warm_epoch, color='r', linestyle=':')
    for i in range(sigmas_history.shape[0]):
        plt.plot(sigmas_history[i], label=f'Sigma {i+1}')
    plt.legend(fontsize=16, title='Sigma values', title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size': 16})
    plt.ylabel('sigma', fontdict={'size': 16})
    fig_file = os.path.join(
        save_figures_dir, f'vis_training_history_of_sigma_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)


def visualize_alpha_history(sigmas_history, kl_warm_epochs, auto_warm_epochs, model_name, save_figures_dir, tag):
    plt.clf()
    fig = plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 12
    if model_name in ['VRNN', 'MT_VRNN']:
        for kl_warm_epoch in kl_warm_epochs:
            plt.axvline(x=kl_warm_epoch, color='c', linestyle='--')
    for auto_warm_epoch in auto_warm_epochs:
        plt.axvline(x=auto_warm_epoch, color='r', linestyle=':')
    for i in range(sigmas_history.shape[0]):
        alphas = 1 / (1 + np.exp(-sigmas_history[i]))
        plt.plot(alphas, label=f'Alpha {i+1}')
    plt.legend(fontsize=16, title='Alpha values', title_fontsize=20)
    plt.xlabel('epochs', fontdict={'size': 16})
    plt.ylabel('alpha', fontdict={'size': 16})
    plt.yscale('log')
    fig_file = os.path.join(
        save_figures_dir, f'vis_training_history_of_alpha_{tag}.png')
    plt.savefig(fig_file)
    plt.close(fig)
