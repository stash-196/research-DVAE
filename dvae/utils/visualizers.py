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


def visualize_spectral_analysis(data_lst, name_lst, colors_lst, save_dir, sampling_rate=0.25, explain='', max_sequences=None):
    # Limit the number of sequences if max_sequences is specified
    if max_sequences is not None and len(data_lst) > max_sequences:
        data_lst = data_lst[:max_sequences]
        name_lst = name_lst[:max_sequences]

    max_length = max(len(data) for data in data_lst)  # Find the length of the longest signal

    # Preparing for subplots
    num_datasets = len(data_lst)
    fig, axs = plt.subplots(num_datasets, 2, figsize=(20, 6 * num_datasets))  # 2 columns for Power and Phase spectra

    for idx, data in enumerate(data_lst):
        padded_data = np.pad(data, (0, max_length - len(data)), mode='constant')  # Pad the signal
        fft = np.fft.fft(padded_data)

        freqs = np.fft.fftfreq(max_length, 1/sampling_rate)
        periods = np.zeros_like(freqs)
        periods[1:] = 1 / freqs[1:]  # Get periods corresponding to frequencies

        nonzero_indices = np.where(freqs > 0)

        power_spectrum = np.abs(fft[nonzero_indices]) ** 2  # Power spectrum
        phase_spectrum = np.angle(fft[nonzero_indices])  # Phase spectrum

        # Color for the current dataset
        dataset_color = colors_lst[idx]

        # Power spectral plot
        axs[idx, 0].loglog(periods[nonzero_indices], power_spectrum, label=f'{name_lst[idx]} Power Spectrum', color=dataset_color)
        axs[idx, 0].set_title(f'{name_lst[idx]} Power Spectrum')
        axs[idx, 0].set_xlabel('Period (seconds)')
        axs[idx, 0].set_ylabel('Power')
        axs[idx, 0].grid(True)

        # Phase spectral plot
        axs[idx, 1].semilogx(periods[nonzero_indices], phase_spectrum, label=f'{name_lst[idx]} Phase Spectrum', color=dataset_color)
        axs[idx, 1].set_title(f'{name_lst[idx]} Phase Spectrum')
        axs[idx, 1].set_xlabel('Frequency (Hz)')
        axs[idx, 1].set_ylabel('Phase (radians)')
        axs[idx, 1].grid(True)

    plt.tight_layout()
    fig_file = os.path.join(save_dir, f'vis_spectral_analysis_{explain}.png')
    plt.savefig(fig_file)
    plt.close()

    print(f"Spectral analysis plots saved at: {fig_file}")

    # return power spectra, phase spectra, and periods
    return 


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


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def reduce_dimensions(embeddings, technique='pca', n_components=3):
    """
    Reduce dimensions of embeddings using specified technique (PCA or t-SNE).

    Args:
    - embeddings: A numpy array of shape (num_samples, embedding_dim).
    - technique: Dimensionality reduction technique ('pca' or 'tsne').
    - n_components: Number of dimensions to reduce to.

    Returns:
    - reduced_embeddings: Embeddings in the reduced dimension space.
    """
    if technique.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif technique.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, n_iter=300)
    else:
        raise ValueError("Unsupported dimensionality reduction technique. Choose 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

from matplotlib.animation import PillowWriter, FuncAnimation

def visualize_embedding_space(states, save_dir, variable_name='embedding', explain='', technique='pca', rotation_speed=5, total_rotation=360, base_color='Blues'):
    reduced_embeddings = reduce_dimensions(states, technique=technique)

    # Get the z-values for color mapping
    zs = reduced_embeddings[:, 2]

    # Get the color map
    cmap = plt.get_cmap(base_color)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine min and max z-values for coloring
    z_min, z_max = zs.min(), zs.max()

    # Plot each segment with color based on z-value
    for i in range(1, len(reduced_embeddings)):
        x = reduced_embeddings[i-1:i+1, 0]
        y = reduced_embeddings[i-1:i+1, 1]
        z = reduced_embeddings[i-1:i+1, 2]
        color_value = (z.mean() - z_min) / (z_max - z_min)  # Normalize color value within the range
        ax.plot(x, y, z, color=cmap(color_value))

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.title('Trajectory of {} in {} {} space'.format(variable_name.replace('_', ' '), explain.upper(), technique.upper()))

    # Function to update the plot for each frame
    def update(frame):
        ax.view_init(30, frame)
        return fig,

    # Creating animation
    anim = FuncAnimation(fig, update, frames=np.arange(0, total_rotation, rotation_speed), interval=200, blit=True)
    fig_file = os.path.join(save_dir, f'vis_trajectory_of_{variable_name}_space_{technique}.gif')

    print(f"Saving animation at: {fig_file}")
    # Measure the time taken to save the GIF
    start_time = time.time()
    anim.save(fig_file, writer=PillowWriter(fps=10))
    end_time = time.time()

    # Print the duration
    print(f"Animation saved in {end_time - start_time:.2f} seconds")

    
    plt.close()





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