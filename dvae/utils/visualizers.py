import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from dvae.utils import create_autonomous_mode_selector
import torch

import time

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

def visualize_sequences(true_data, recon_data, mode_selector, save_dir, explain=''):
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

    plt.title('Comparison of True and Predicted Sequences {}'.format(explain))
    plt.xlabel('Time steps')
    plt.ylabel('Value')
    plt.grid(True)
    
    fig_file = os.path.join(save_dir, f'vis_pred_true_series_{explain}.png')
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


def visualize_spectral_analysis(data_lst, name_lst, colors_lst, save_dir, sampling_rate, explain='', max_sequences=None):
    if max_sequences is not None and len(data_lst) > max_sequences:
        data_lst = data_lst[:max_sequences]
        name_lst = name_lst[:max_sequences]

    max_length = max(len(data) for data in data_lst)

    num_datasets = len(data_lst)
    fig, axs = plt.subplots(num_datasets, 2, figsize=(10, 3 * num_datasets))

    power_spectrum_list = []
    min_power, max_power = float('inf'), float('-inf')  # Initialize min and max power

    # First pass: Compute the power spectrum and update min/max values
    for data in data_lst:
        padded_data = np.pad(data, (0, max_length - len(data)), mode='constant')
        fft = np.fft.fft(padded_data)
        freqs = np.fft.fftfreq(max_length, 1/sampling_rate)
        nonzero_indices = np.where(freqs > 0)
        
        power_spectrum = np.abs(fft[nonzero_indices]) ** 2
        power_spectrum_list.append(power_spectrum)

        # Update global min and max power spectrum values
        min_power = min(min_power, power_spectrum.min())
        max_power = max(max_power, power_spectrum.max())
        # Calculate the limits for the frequency axis
        freq_min, freq_max = freqs[nonzero_indices].min(), freqs[nonzero_indices].max()

        # Convert these frequency limits to period limits for the secondary axis
        period_max, period_min = 1 / freq_min, 1 / freq_max  # Note the inversion here


    # Second pass: Plotting with unified y-axis range
    for idx, (data, power_spectrum) in enumerate(zip(data_lst, power_spectrum_list)):
        padded_data = np.pad(data, (0, max_length - len(data)), mode='constant')
        fft = np.fft.fft(padded_data)
        freqs = np.fft.fftfreq(max_length, 1/sampling_rate)
        nonzero_indices = np.where(freqs > 0)
        freqs_nonzero = freqs[nonzero_indices]
        periods_nonzero = 1 / freqs_nonzero

        dataset_color = colors_lst[idx]

        # Power spectral plot with unified y-axis
        axs[idx, 0].loglog(freqs_nonzero, power_spectrum, label=f'{name_lst[idx]} Power Spectrum', color=dataset_color)
        axs[idx, 0].set_ylim(min_power, max_power)  # Set the same y-axis range for all plots
        axs[idx, 0].set_xlim(freq_min, freq_max)
        axs[idx, 0].set_title(f'{name_lst[idx]} Power Spectrum')
        axs[idx, 0].set_xlabel('Frequency (Hz)')
        axs[idx, 0].set_ylabel('Power')
        axs[idx, 0].grid(True)

        # Adding secondary x-axis for periods
        ax_new = axs[idx, 0].twiny()
        ax_new.loglog(periods_nonzero, power_spectrum, alpha=0)  # Invisible plot to generate secondary x-axis
        ax_new.set_xlabel('Period (seconds)')
        ax_new.set_xscale('log')
        ax_new.set_xlim(period_min, period_max)

        # Phase spectral plot
        phase_spectrum = np.angle(fft[nonzero_indices])
        axs[idx, 1].semilogx(freqs_nonzero, phase_spectrum, label=f'{name_lst[idx]} Phase Spectrum', color=dataset_color)
        axs[idx, 1].set_title(f'{name_lst[idx]} Phase Spectrum')
        axs[idx, 1].set_xlabel('Frequency (Hz)')
        axs[idx, 1].set_ylabel('Phase (radians)')
        axs[idx, 1].grid(True)

    plt.tight_layout()
    fig_file = os.path.join(save_dir, f'vis_spectral_analysis_{explain}.png')
    plt.savefig(fig_file)
    plt.close()

    print(f"Spectral analysis plots saved at: {fig_file}")

    # Returning only for the first dataset for simplicity
    return power_spectrum_list, freqs_nonzero, periods_nonzero



def visualize_variable_evolution(states, batch_data, save_dir, variable_name='h', alphas=None, add_lines_lst=[]):
    seq_len, batch_size, x_dim = batch_data.shape
    reshaped_batch_data = batch_data[:, 0, :].reshape(-1).cpu().numpy()  # Assuming batch size is at least 1

    num_dims = states.shape[2]

    # Prepare colors for the plots
    if alphas is not None:
        colors = [plt.cm.viridis(alpha.item()) for alpha in alphas]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_dims))

    # Create a figure with two subplots, making the bottom subplot smaller
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plotting the variables
    for dim in range(num_dims):
        axs[0].plot(states[:, 0, dim].cpu().numpy(), label=f'Dim {dim}', color=colors[dim])

    # Add vertical lines at specified epochs
    for t in add_lines_lst:
        axs[0].axvline(x=t, color='r', linestyle='--')
        axs[1].axvline(x=t * x_dim, color='r', linestyle='--')

    str_alphas = ' α:' + str(set(alphas.numpy())) if alphas is not None else ''
    axs[0].set_title(f'Evolution of {variable_name} States over Time' + str_alphas)
    axs[0].set_xlabel('Time steps')
    axs[0].set_ylabel(f'{variable_name} state value')
    if num_dims <= 10:
        axs[0].legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    axs[0].grid(True)

    # Plotting the true signal
    axs[1].plot(reshaped_batch_data, label='True Signal', color='blue')
    axs[1].set_title(f'True Signal for {variable_name}')
    axs[1].set_xlabel('Time steps')
    axs[1].set_ylabel('Signal value')
    axs[1].legend()
    axs[1].grid(True)

    # Save the figure
    fig_file = os.path.join(save_dir, f'vis_evolution_of_{variable_name}_state.png')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close(fig)


def visualize_teacherforcing_2_autonomous(batch_data, dvae, mode_selector, save_path, explain='', inference_mode=False):
    seq_len, batch_size, x_dim = batch_data.shape
    n_seq = seq_len  # Use the full sequence length for visualization

    # Reconstruct the first n_seq sequences
    recon_batch_data = dvae(batch_data[:n_seq, :1, :], mode_selector=mode_selector, inference_mode=inference_mode).detach().clone()

    # Flattening the time series for true and reconstructed data
    true_series = batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()
    recon_series = recon_batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()

    # Expanding the mode selector to match the flattened time series
    expanded_mode_selector = np.repeat(mode_selector[:n_seq], x_dim)

    # Plot the reconstruction vs true sequence
    visualize_sequences(true_series, recon_series, expanded_mode_selector, save_path, explain)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from sklearn.decomposition import PCA, FastICA, NMF, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS, Isomap, SpectralEmbedding
from umap import UMAP

def reduce_dimensions(embeddings, technique='pca', n_components=3):
    """
    Reduce dimensions of embeddings using specified technique.

    Args:
    - embeddings: A numpy array of shape (num_samples, embedding_dim).
    - technique: Dimensionality reduction technique.
    - n_components: Number of dimensions to reduce to.

    Returns:
    - reduced_embeddings: Embeddings in the reduced dimension space.
    """
    technique = technique.lower()
    if technique == 'pca':
        reducer = PCA(n_components=n_components)
    elif technique == 'tsne':
        reducer = TSNE(n_components=n_components, n_iter=300)
    elif technique == 'lle':
        reducer = LocallyLinearEmbedding(n_components=n_components)
    elif technique == 'umap':
        reducer = UMAP(n_components=n_components)
    elif technique == 'ica':
        reducer = FastICA(n_components=n_components)
    elif technique == 'mds':
        reducer = MDS(n_components=n_components)
    elif technique == 'nmf':
        # Shift data to be non-negative if using NMF
        embeddings_min = embeddings.min()
        if embeddings_min < 0:
            embeddings = embeddings - embeddings_min
        reducer = NMF(n_components=n_components, init='random', random_state=0)
    elif technique == 'isomap':
        reducer = Isomap(n_components=n_components)
    elif technique == 'laplacian':
        reducer = SpectralEmbedding(n_components=n_components)
    elif technique == 'kernel_pca':
        reducer = KernelPCA(n_components=n_components, kernel='rbf')
    else:
        raise ValueError(f"Unsupported dimensionality reduction technique: {technique}")

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


from matplotlib.animation import PillowWriter, FuncAnimation


def visualize_embedding_space(states_list, save_dir, variable_name, condition_names, explain='', technique='pca', rotation_speed=5, total_rotation=360, base_colors=['Blues']):
    """
    Visualize a list of embedding states in separate 3D subplots, each with its own animation.

    Args:
    - states_list: List of states numpy arrays.
    - save_dir: Directory to save the plots.
    - variable_name_list: List of names for each variable corresponding to states.
    - explain: Explanation text to add to the plot title.
    - technique: Dimensionality reduction technique to use.
    - rotation_speed: Degrees to rotate for each frame of the animation.
    - total_rotation: Total degrees of rotation.
    - base_color_list: List of base colors for each subplot.
    """
    num_plots = len(states_list)
    fig = plt.figure(figsize=(10, 10 * num_plots))
    
    for i, (states, condition_name, base_color) in enumerate(zip(states_list, condition_names, base_colors)):
        ax = fig.add_subplot(num_plots, 1, i + 1, projection='3d')
        reduced_embeddings = reduce_dimensions(states, technique=technique)
        zs = reduced_embeddings[:, 2]
        cmap = plt.get_cmap(base_color)
        z_min, z_max = zs.min(), zs.max()

        for j in range(1, len(reduced_embeddings)):
            x, y, z = reduced_embeddings[j-1:j+1, 0], reduced_embeddings[j-1:j+1, 1], reduced_embeddings[j-1:j+1, 2]
            color_value = (z.mean() - z_min) / (z_max - z_min)
            ax.plot(x, y, z, color=cmap(color_value))
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f'Trajectory of {variable_name.capitalize()} {condition_name.capitalize()} in {explain} {technique} space', fontsize=16)

    def update(frame):
        for ax in fig.axes:
            ax.view_init(30, frame)
        return fig,

    anim = FuncAnimation(fig, update, frames=np.arange(0, total_rotation, rotation_speed), interval=200, blit=True)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Construct a unique filename to avoid overwriting
    fig_file = os.path.join(save_dir, f'vis_trajectory_of_{variable_name}_space_{technique}.gif')
    print(f"Saving animation at: {fig_file}")

    start_time = time.time()
    anim.save(fig_file, writer=PillowWriter(fps=10))
    end_time = time.time()
    print(f"Animation saved in {end_time - start_time:.2f} seconds")
    plt.close(fig)






def visualize_accuracy_over_time(accuracy_values, variance_values, save_dir, measure, num_batches, num_iter, explain, autonomous_mode_selector):
    """
    Visualize and save the accuracy over time, with different colors for teacher-forced and autonomous modes.

    Args:
    - accuracy_values: A tensor of accuracy values.
    - save_dir: Directory to save the plot.
    - measure: The name of the accuracy measure (e.g., 'rmse').
    - num_samples: Number of samples used in the evaluation.
    - explain: Additional explanation for the plot title.
    - autonomous_mode_selector: Array indicating the mode (0 for teacher-forced, 1 for autonomous).
    """
    plt.figure(figsize=(10, 6))
    
    time_steps = range(len(accuracy_values))

    # Plot the average accuracy line
    for i in range(len(time_steps) - 1):
        color = 'green' if autonomous_mode_selector[i] == 0 else 'red'
        plt.plot(time_steps[i:i + 2], accuracy_values.cpu().numpy()[i:i + 2], color=color)

    # Shading the variance
    std_dev = torch.sqrt(variance_values).cpu().numpy()
    plt.fill_between(time_steps, (accuracy_values - std_dev).cpu().numpy(), (accuracy_values + std_dev).cpu().numpy(), color='gray', alpha=0.3)

    plt.xlabel('Time Steps')
    plt.ylabel(measure.upper())
    title = f'Expected {measure.upper()} Over Time - {explain.capitalize()} (Nbatch={num_batches}, Niter={num_iter})'
    plt.title(title)
    plt.grid(True)

    # Custom legend
    plt.plot([], [], color='green', label='Teacher-Forced Sequence')
    plt.plot([], [], color='red', label='Autonomous Sequence')
    plt.plot([], [], color='gray', alpha=0.3, label='Variance')
    plt.legend()

    fig_file = os.path.join(save_dir, f'vis_accuracy_{measure}_{explain.replace(" ", "_")}_btch{num_batches}_iter{num_iter}.png')
    plt.savefig(fig_file)
    print(f"{measure} plot saved at: {fig_file}")
    plt.close()


def visualize_delay_embedding(observation, delay, dimensions, save_dir, variable_name, explain='', base_color='Blues', rotation_speed=5, total_rotation=360):
    n = len(observation)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError("Delay and dimensions are too large for the length of the observation array.")

    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay: i * delay + embedding_length]

    if dimensions != 3:
        raise NotImplementedError("Rotation and color gradient for dimensions other than 3 is not implemented.")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap(base_color)
    zs = embedded[:, 2]
    z_min, z_max = zs.min(), zs.max()

    for i in range(1, len(embedded)):
        x = embedded[i-1:i+1, 0]
        y = embedded[i-1:i+1, 1]
        z = embedded[i-1:i+1, 2]
        color_value = (z.mean() - z_min) / (z_max - z_min)
        ax.plot(x, y, z, color=cmap(color_value))

    ax.set_xlabel('X(t)')
    ax.set_ylabel('X(t + delay)')
    ax.set_zlabel('X(t + 2 * delay)')
    title = f'Delay Embedding of {variable_name.capitalize()} {explain} (Delay: {delay})'
    plt.title(title, fontsize=16)

    def update(frame):
        ax.view_init(30, frame)
        return fig,

    anim = FuncAnimation(fig, update, frames=np.arange(0, total_rotation, rotation_speed), interval=200, blit=True)
    fig_file = os.path.join(save_dir, f'vis_delay_embedding_of_{variable_name}_τ{delay}_{explain}.gif')

    print(f"Saving animation at: {fig_file}")
    start_time = time.time()
    anim.save(fig_file, writer=PillowWriter(fps=10))
    end_time = time.time()

    print(f"Animation saved in {end_time - start_time:.2f} seconds")
    
    plt.close()


def visualize_alpha_history(sigmas_history, power_spectrum_lst, spectrum_color_lst,spectrum_name_lst, frequencies, save_dir, dt, kl_warm_epochs=None, explain='', true_alphas=[]):
    plt.clf()
    periods = 1 / frequencies
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.rcParams['font.size'] = 12

    # Example color scheme for alpha curves
    alpha_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']


    alphas_from_periods = dt / periods
    periods_max, periods_min = np.max(periods), np.min(periods)
    alpha_range_min = 1 / (1 + np.exp(-np.max(sigmas_history))) / 10
    alpha_range_max = 1.1
    ylim_max = max(dt/alpha_range_min, periods_max)
    ylim_min = min(dt/alpha_range_max, periods_min)

    if kl_warm_epochs:
        for kl_warm_epoch in kl_warm_epochs:
            ax1.axvline(x=kl_warm_epoch, color='r', linestyle='--', label='KL & SS Warm-up' if kl_warm_epoch == kl_warm_epochs[0] else "")

    # Plot and annotate true alphas with a specific color
    true_alpha_color = 'black'
    for idx, true_alpha in enumerate(true_alphas, start=1):
        line_y = dt/true_alpha
        ax1.axhline(y=line_y, color=true_alpha_color, linestyle='--', label=f"\"True\" α" if idx == 1 else "")
        ax2.axhline(y=true_alpha/dt, color=true_alpha_color, linestyle='--')
        ax1.text(0.5, line_y, f"α_true: {true_alpha:.{4}g}", verticalalignment='bottom', horizontalalignment='center', transform=ax1.get_yaxis_transform(), color=true_alpha_color, fontsize=15)

    num_alphas = sigmas_history.shape[0]
    for i in range(min(num_alphas, len(alpha_colors))):  # Ensure we don't exceed the number of predefined colors
        alphas = 1 / (1 + np.exp(-sigmas_history[i]))
        periods_from_alpha = dt / alphas
        curve_color = alpha_colors[i]  # Get color for this curve
        ax1.plot(periods_from_alpha, label=f'α {i+1}', color=curve_color)
        # Annotate the last alpha value
        last_alpha_period = periods_from_alpha[-1]
        ax1.text(len(sigmas_history[i])-1, last_alpha_period, f"α_end: {alphas[-1]:.{4}g}", verticalalignment='bottom', horizontalalignment='right', color=curve_color, fontsize=15)
        # Optionally, annotate the initial alpha value
        init_alpha_period = periods_from_alpha[0]
        ax1.text(0, init_alpha_period, f"α_0: {alphas[0]:.{4}g}", verticalalignment='bottom', horizontalalignment='left', color=curve_color, fontsize=15)


    ax1.set_title('α values during training', fontsize=18)
    ax1.legend(fontsize=16, loc='upper left')
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Period (sec)', fontsize=16)
    ax1.set_yscale('log')
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.grid(True)

    ax1_right = ax1.twinx()
    ax1_right.set_ylabel('α =Δt/T', fontsize=16)
    ax1_right.set_yscale('log')
    ax1_right.set_ylim(dt/ylim_max, dt/ylim_min)
    ax1_right.invert_yaxis()

    for power_spectrum, color, label in zip(power_spectrum_lst, spectrum_color_lst, spectrum_name_lst):
        ax2.loglog(power_spectrum, frequencies, color=color, alpha=0.5, label=label)
    
    ax2.set_title('Lorenz63\nPower Spectrum', fontsize=18)
    ax2.legend(fontsize=16, loc='upper right')
    ax2.set_xlabel('Amplitude', fontsize=16)
    ax2.set_ylabel('Frequency (Hz)', fontsize=16)
    ax2.set_ylim(1/ylim_min, 1/ylim_max)
    ax2.grid(True)
    ax2.invert_xaxis()

    ax2_right = ax2.twinx()
    ax2_right.set_ylabel('Periods (s)', fontsize=16)
    ax2_right.set_yscale('log')
    ax2_right.set_ylim(ylim_max, ylim_min)
    ax2_right.invert_yaxis()

    plt.suptitle(f'α during training and Power Spectrum {explain}', fontsize=20, x=0.5, y=1.02)
    plt.tight_layout()
    fig_file = os.path.join(save_dir, f'vis_alpha_vs_power_spectrum_{explain}.png')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close()
    print(f"Alpha vs Power Spectrum plot saved at: {fig_file}")