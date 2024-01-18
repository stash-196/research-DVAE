import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from dvae.utils import create_autonomous_mode_selector

def visualize_model_parameters(model, explain, save_path=None, fixed_scale=None):
    if save_path:
        dir_path = os.path.join(save_path, explain)
        os.makedirs(dir_path, exist_ok=True)  # Create a new directory

    # If a fixed scale is not provided, determine the max and min across all parameters
    if not fixed_scale:
        all_values = [param.detach().cpu().numpy().flatten() for _, param in model.named_parameters()]
        min_val = min(map(min, all_values))
        max_val = max(map(max, all_values))
        fixed_scale = (min_val, max_val)

    for name, param in model.named_parameters():
        plt.figure(figsize=(10, 5))
        param_data = param.detach().cpu().numpy()
        
        # Reshape if one-dimensional
        if param.ndim == 1:
            param_data = param_data.reshape(-1, 1)

        sns.heatmap(param_data, annot=False, cmap="RdBu", center=0, vmin=fixed_scale[0], vmax=fixed_scale[1])
        plt.title(f'{name} at {explain}')
        plt.xlabel('Parameters')
        plt.ylabel('Values')

        if save_path:
            plt.savefig(os.path.join(dir_path, f'{name}_{explain}.png'))
        plt.close()


def visualize_combined_parameters(model, explain, save_path=None, fixed_scale=None):
    if fixed_scale is None:
        # Compute the scale if not provided
        all_values = [param.detach().cpu().numpy().flatten() for _, param in model.named_parameters()]
        min_val = min(map(min, all_values))
        max_val = max(map(max, all_values))
        fixed_scale = (min_val, max_val)

    params = list(model.named_parameters())
    n = len(params)
    fig, axs = plt.subplots(nrows=n, figsize=(10, 5 * n))

    for i, (name, param) in enumerate(params):
        ax = axs[i] if n > 1 else axs
        param_data = param.detach().cpu().numpy()

        # Reshape if one-dimensional
        if param.ndim == 1:
            param_data = param_data.reshape(-1, 1)

        sns.heatmap(param_data, annot=False, cmap="RdBu", center=0, vmin=fixed_scale[0], vmax=fixed_scale[1], ax=ax)
        ax.set_title(f'{name} at {explain}')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f'all_parameters_{explain}.png'))
    plt.close()



import matplotlib.pyplot as plt
import os

def visualize_sequences(true_data, recon_data, mode_selector, save_dir, name=''):
    plt.figure(figsize=(10, 6))
    
    # Plotting the true sequence in blue
    plt.plot(true_data, label='True Sequence', color='blue')

    for i in range(len(mode_selector)):
        # Choose color based on mode
        if mode_selector[i] == 0:
            color = 'green'  # Teacher-forced
        elif mode_selector[i] == 1:
            color = 'red'    # Autonomous
        else:
            # Gradient between green (0) and red (1)
            red_intensity = mode_selector[i]
            green_intensity = 1 - mode_selector[i]
            color = (red_intensity, green_intensity, 0)  # RGB color

        # Plotting the segment
        if i < len(recon_data) - 1:
            plt.plot([i, i + 1], recon_data[i:i + 2], color=color)

    # Creating custom legend
    plt.plot([], [], color='green', label='Teacher-Forced Sequence')
    plt.plot([], [], color='red', label='Autonomous Sequence')

    plt.legend()

    plt.title('Comparison of True and Predicted Sequences')
    plt.xlabel('Time steps')
    plt.ylabel('Value')
    plt.grid(True)
    
    fig_file = os.path.join(save_dir, f'vis_pred_true_series_{name}.png')
    plt.savefig(fig_file)
    plt.close()

# def visualize_sequences(true_data, recon_data, save_dir, n_gen_portion, name=''):
#     recon_length = len(recon_data) - int(len(recon_data) * n_gen_portion)
#     time_steps = list(range(len(true_data)))

#     fig = go.Figure()

#     # True Sequence
#     fig.add_trace(go.Scatter(x=time_steps, y=true_data, mode='lines', name='True Sequence'))

#     # Reconstructed Sequence
#     fig.add_trace(go.Scatter(x=time_steps[:recon_length], y=recon_data[:recon_length], mode='lines', name='Reconstructed Sequence'))

#     # Self-Generated Sequence
#     fig.add_trace(go.Scatter(x=time_steps[recon_length:], y=recon_data[recon_length:], mode='lines', name='Self-Generated Sequence'))

#     fig.update_layout(title='Comparison of True and Predicted Sequences', xaxis_title='Time steps', yaxis_title='Value')

#     fig.write_image(os.path.join(save_dir, f'vis_pred_true_series{name}.svg'), format='svg')


def visualize_spectral_analysis(true_data, recon_data, save_dir, sampling_rate=0.25, explain=''):  # 1 sample every 4 seconds = 0.25 Hz
    true_fft = np.fft.fft(true_data)
    recon_fft = np.fft.fft(recon_data)

    freqs = np.fft.fftfreq(len(true_data), 1/sampling_rate)  # Get proper frequency axis based on sampling rate
    periods = np.zeros_like(freqs)
    periods[1:] = 1 / freqs[1:]  # Get periods corresponding to frequencies
    
    # Filter out zero frequencies (to avoid log(0) issues)
    nonzero_indices = np.where(freqs > 0)

    # Power spectrum (magnitude squared)
    true_power = np.abs(true_fft[nonzero_indices]) ** 2
    recon_power = np.abs(recon_fft[nonzero_indices]) ** 2
    
    # Phase spectrum
    true_phase = np.angle(true_fft[nonzero_indices])
    recon_phase = np.angle(recon_fft[nonzero_indices])

    # Power spectral plots
    plt.figure(figsize=(10, 6))
    plt.loglog(periods[nonzero_indices], true_power, label='True Sequence Power Spectrum', color='blue')
    plt.loglog(periods[nonzero_indices], recon_power, label='Predicted Sequence Power Spectrum', color='red')
    plt.legend()
    plt.title('Power Spectral Analysis {}'.format(explain))
    plt.xlabel('Period (seconds)')
    plt.ylabel('Power')
    plt.grid(True)
    fig_file = os.path.join(save_dir, 'vis_pred_true_power_spectrums_loglog_{}.png'.format(explain))
    plt.savefig(fig_file)
    plt.close()

    # Phase spectral plots
    plt.figure(figsize=(10, 6))
    plt.semilogx(periods[nonzero_indices], recon_phase, label='Predicted Sequence Phase Spectrum', color='red')
    plt.semilogx(periods[nonzero_indices], true_phase, label='True Sequence Phase Spectrum', color='blue')
    plt.legend()
    plt.title('Phase Spectral Analysis {}'.format(explain))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    fig_file = os.path.join(save_dir, 'vis_pred_true_phase_spectrums_semilogx_{}.png'.format(explain))
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


def visualize_teacherforcing_2_autonomous(batch_data, dvae, mode_selector, save_path, explain=''):
    seq_len, batch_size, x_dim = batch_data.shape
    n_seq = seq_len  # Use the full sequence length for visualization

    # Reconstruct the first n_seq sequences
    recon_batch_data = dvae(batch_data[:n_seq, :1, :], mode_selector=mode_selector).detach().clone()

    # Flattening the time series for true and reconstructed data
    true_series = batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()
    recon_series = recon_batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()

    # Expanding the mode selector to match the flattened time series
    expanded_mode_selector = np.repeat(mode_selector[:n_seq], x_dim)

    # Plot the reconstruction vs true sequence
    visualize_sequences(true_series, recon_series, expanded_mode_selector, save_path, explain)
