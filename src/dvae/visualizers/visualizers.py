import textwrap
from matplotlib.animation import PillowWriter, FuncAnimation
from umap import UMAP
from sklearn.manifold import (
    TSNE,
    LocallyLinearEmbedding,
    MDS,
    Isomap,
    SpectralEmbedding,
)
from sklearn.decomposition import PCA, FastICA, NMF, KernelPCA
from mpl_toolkits.mplot3d import Axes3D
import time
import torch
from dvae.utils import create_autonomous_mode_selector, power_spectrum_error
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg' to avoid GUI issues
# plt.rcParams["font.family"] = "Liberation Serif"


import torch


def get_plot_config(paper_ready=True):
    """
    Updates Matplotlib's global plotting settings for exploratory or publication-quality figures,
    and returns a dictionary of additional settings.

    Args:
        paper_ready (bool, optional): If True, applies settings for publication-quality figures.
                                      Defaults to False.

    Returns:
        dict: A dictionary containing additional plotting settings, including 'show_title'.
    """
    config = {}
    if paper_ready:
        # Settings for a publication-quality figure ✒️
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
                "axes.labelsize": 24,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize": 20,
                "lines.linewidth": 2,
                "lines.markersize": 8,
            }
        )
        config["show_title"] = (
            False  # Command to remove/disable titles for paper-ready plots
        )
    else:
        # Default settings for exploratory plotting
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "axes.labelsize": "medium",  # Matplotlib default
                "xtick.labelsize": "medium",
                "ytick.labelsize": "medium",
                "legend.fontsize": "medium",
                "lines.linewidth": 1.5,
                "lines.markersize": 6,
            }
        )
        config["show_title"] = True  # Keep titles for exploratory plots
    return config


def visualize_model_parameters(model, explain, save_path=None, fixed_scale=None):
    if save_path:
        dir_path = os.path.join(save_path, explain)
        os.makedirs(dir_path, exist_ok=True)  # Create a new directory

    # If a fixed scale is not provided, determine the max and min across all parameters
    if not fixed_scale:
        all_values_flattened = [
            param.detach().cpu().numpy().flatten()
            for _, param in model.named_parameters()
        ]
        min_val = min(map(min, all_values_flattened))
        max_val = max(map(max, all_values_flattened))
        fixed_scale = (min_val, max_val)

    for name, param in model.named_parameters():
        plt.figure(figsize=(10, 5))
        param_data = param.detach().cpu().numpy()

        # Reshape if one-dimensional
        if param.ndim == 1:
            param_data = param_data.reshape(-1, 1)

        sns.heatmap(
            param_data,
            annot=False,
            cmap="RdBu",
            center=0,
            vmin=fixed_scale[0],
            vmax=fixed_scale[1],
        )
        plt.title(f"{name} at {explain}")
        plt.xlabel("Parameters")
        plt.ylabel("Values")

        if save_path:
            plt.savefig(os.path.join(dir_path, f"{name}_{explain}.png"))
        plt.close()


def visualize_combined_parameters(
    name_values,
    explain,
    save_path=None,
    fixed_scale=None,
    matrix_or_line="matrix",
    showing_gradient=False,
    gradient_clip_value=None,
):
    if fixed_scale is None:
        # Compute the scale if not provided
        all_values_flattened = [param.flatten() for _, param in name_values]
        min_val = min(map(min, all_values_flattened))
        max_val = max(map(max, all_values_flattened))
        fixed_scale = (min_val, max_val)

    n = len(name_values)
    if matrix_or_line == "matrix":
        fig, axs = plt.subplots(nrows=n, figsize=(10, 5 * n))

        for i, (name, param) in enumerate(name_values):
            ax = axs[i] if n > 1 else axs
            param_data = param

            # Reshape if one-dimensional
            if param.ndim == 1:
                param_data = param_data.reshape(-1, 1)

            sns.heatmap(
                param_data,
                annot=False,
                cmap="RdBu",
                center=0,
                vmin=fixed_scale[0],
                vmax=fixed_scale[1],
                ax=ax,
            )
            if showing_gradient:
                ax.set_title(f"Gradient of {name} at {explain}")
            else:
                ax.set_title(f"{name} at {explain}")

        plt.tight_layout()

    elif matrix_or_line == "line":
        # Plotting all the parameters flattened in one continuous line in one plot
        plt.figure(figsize=(10, 5))
        if gradient_clip_value is not None:  # draw a line at the gradient clip value
            plt.axhline(y=gradient_clip_value, color="r", linestyle="--")
            plt.axhline(y=-gradient_clip_value, color="r", linestyle="--")

        cumulative_length = 0
        for name, param in name_values:
            range = np.arange(
                cumulative_length, cumulative_length + len(param.flatten())
            )
            plt.plot(range, param.flatten(), label=name)
            cumulative_length += len(param.flatten())
            # set title
            if showing_gradient:
                plt.title(f"Gradient of {name} at {explain}")
            else:
                plt.title(f"{name} at {explain}")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.yscale("symlog")
        plt.xlabel("Parameters")
        plt.ylabel("Values")
        plt.tight_layout()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"all_parameters_{explain}.png"))
    plt.close()


def visualize_sequences(
    sequences, mode_selector, save_dir, explain="", missing_mask=None
):
    linewidth = 2
    true_data = sequences[0]["data"]  # Assuming the first entry is the true sequence
    recon_data = sequences[1][
        "data"
    ]  # Assuming the second entry is the reconstructed sequence
    noise_data = (
        sequences[2]["data"] if len(sequences) > 2 else None
    )  # Optional noise sequence
    # Add offset for noise data if present
    min_val = min(np.min(true_data), np.min(recon_data))
    if noise_data is not None:
        offset = 1.1 * min_val - np.max(noise_data)
        noise_data = noise_data + offset

    PAPER_READY = True
    # Apply paper-ready style BEFORE creating axes so ticks/labels inherit it.
    get_plot_config(paper_ready=PAPER_READY)
    plt.rcParams["lines.linewidth"] = linewidth

    # Determine number of dimensions
    if true_data.ndim == 1:
        true_data = true_data[:, np.newaxis]  # Reshape to (n_seq, 1)
        recon_data = recon_data[:, np.newaxis]
        mode_selector = mode_selector[:, np.newaxis]
        if noise_data is not None:
            noise_data = noise_data[:, np.newaxis]

    n_seq, x_dim = true_data.shape

    # Reshape missing_mask if provided to match dimensions
    if missing_mask is not None:
        if missing_mask.ndim == 1:
            missing_mask = missing_mask[:, np.newaxis]
        # Ensure missing_mask has same time dimension as true_data
        if missing_mask.shape[0] != n_seq:
            if missing_mask.shape[0] == n_seq:
                pass  # Already correct
            else:
                # Warn if shape mismatch and zero it out
                print(
                    f"[Warning] missing_mask shape {missing_mask.shape} doesn't match data shape {true_data.shape}"
                )
                missing_mask = None

    # Create subplots with constrained_layout to handle spacing
    fig, axes = plt.subplots(
        x_dim, 1, figsize=(10, 4 * x_dim), sharex=True, constrained_layout=True
    )

    # If x_dim == 1, axes is a single Axes object; make it iterable
    if x_dim == 1:
        axes = [axes]

    # Plot each dimension in its own subplot
    for dim in range(x_dim):
        ax = axes[dim]

        # Add gray background shading for missing regions
        if missing_mask is not None:
            missing_dim = (
                missing_mask
                if missing_mask.ndim == 1
                else missing_mask[:, min(dim, missing_mask.shape[1] - 1)]
            )
            # Find contiguous missing regions and shade them
            in_missing = False
            start_missing = None
            for i in range(len(missing_dim)):
                if missing_dim[i] and not in_missing:
                    # Start of a missing region
                    start_missing = i - 0.5
                    in_missing = True
                elif not missing_dim[i] and in_missing:
                    # End of a missing region
                    ax.axvspan(
                        start_missing,
                        i - 0.5,
                        alpha=0.15,
                        color="gray",
                        label="Missing Data" if i == 1 else "",
                    )
                    in_missing = False
            # Handle case where sequence ends with missing data
            if in_missing:
                ax.axvspan(
                    start_missing,
                    len(missing_dim) - 0.5,
                    alpha=0.15,
                    color="gray",
                    label="Missing Data" if dim == 0 else "",
                )

        # Plot added noise if present
        if noise_data is not None:
            ax.plot(noise_data[:, dim], label="Added Noise", color="purple")
        # Plot true sequence
        ax.plot(true_data[:, dim], label="True Sequence", color="blue")

        # Plot reconstructed sequence with mode-based coloring
        for i in range(n_seq - 1):
            if np.isnan(true_data[i, dim]):
                color = "red"
            elif mode_selector[i, dim] == 0:
                color = "green"  # Teacher-forced
            elif mode_selector[i, dim] == 1:
                color = "red"  # Autonomous
            else:
                red_intensity = mode_selector[i, dim]
                green_intensity = 1 - mode_selector[i, dim]
                color = (red_intensity, green_intensity, 0)
            ax.plot([i, i + 1], recon_data[i : i + 2, dim], color=color)

        # Customize subplot
        if x_dim > 1:  # Only set subplot title if x_dim > 1
            ax.set_title(f"Input Dimension {dim}")
        ax.set_ylabel("Value")
        if dim == x_dim - 1:  # Only label x-axis on the bottom subplot
            ax.set_xlabel("Time steps")
        ax.grid(True)

    # Add a shared legend for the entire figure
    handles = [
        plt.Line2D([0], [0], color="blue", label="True Sequence"),
        plt.Line2D([0], [0], color="green", label="Teacher-Forced"),
        plt.Line2D([0], [0], color="red", label="Autonomous"),
    ]
    if noise_data is not None:
        handles.append(plt.Line2D([0], [0], color="purple", label="Added Noise"))
    if missing_mask is not None:
        handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, fc="gray", alpha=0.15, label="Missing Data (Original)"
            )
        )
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.2))

    # Add a super title with adjusted y-position
    if not PAPER_READY:
        fig.suptitle(
            textwrap.fill(
                f"Comparison of True and Predicted Sequences {explain}",
                width=50,
                break_long_words=True,
            ),
            y=0.98,  # Adjust position to prevent cutoff
        )

    # Adjust the top margin to ensure the super title is not cut off
    # plt.subplots_adjust(top=0.9)

    # Save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_file = os.path.join(save_dir, f"vis_pred_true_series_{explain}.png")
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close()


def visualize_spectral_analysis(
    data_lst,
    name_lst,
    colors_lst,
    save_dir,
    sampling_rate,
    explain="",
    max_sequences=None,
    use_log_scale=False,
):
    if max_sequences is not None and len(data_lst) > max_sequences:
        data_lst = data_lst[:max_sequences]
        name_lst = name_lst[:max_sequences]

    max_length = max(len(data) for data in data_lst)

    num_datasets = len(data_lst)
    fig, axs = plt.subplots(num_datasets, 2, figsize=(10, 3 * num_datasets))

    power_spectrum_list = []
    min_power, max_power = float("inf"), float("-inf")  # Initialize min and max power

    # First pass: Compute the power spectrum and update min/max values
    for data in data_lst:
        padded_data = np.pad(data, (0, max_length - len(data)), mode="constant")
        fft = np.fft.fft(padded_data)
        freqs = np.fft.fftfreq(max_length, 1 / sampling_rate)
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
        padded_data = np.pad(data, (0, max_length - len(data)), mode="constant")
        fft = np.fft.fft(padded_data)
        freqs = np.fft.fftfreq(max_length, 1 / sampling_rate)
        nonzero_indices = np.where(freqs > 0)
        freqs_nonzero = freqs[nonzero_indices]
        periods_nonzero = 1 / freqs_nonzero

        dataset_color = colors_lst[idx]

        # Power spectral plot with unified y-axis
        if use_log_scale:
            axs[idx, 0].loglog(
                freqs_nonzero,
                power_spectrum,
                color=dataset_color,
                label=f"{name_lst[idx]} Power Spectrum",
            )
            axs[idx, 0].set_xlim(freq_min, freq_max)
            axs[idx, 0].set_ylim(min_power, max_power)
        else:
            axs[idx, 0].plot(
                freqs_nonzero,
                power_spectrum,
                color=dataset_color,
                label=f"{name_lst[idx]} Power Spectrum",
            )
            axs[idx, 0].set_xscale("linear")
            axs[idx, 0].set_yscale("linear")
        axs[idx, 0].set_title(
            textwrap.fill(
                f"{name_lst[idx]} Power Spectrum",
                width=50,
                break_long_words=True,
            )
        )
        axs[idx, 0].set_xlabel("Frequency (Hz)")
        axs[idx, 0].set_ylabel("Power")
        axs[idx, 0].grid(True)

        # Adding secondary x-axis for periods
        ax_new = axs[idx, 0].twiny()
        # Invisible plot to generate secondary x-axis
        ax_new.loglog(periods_nonzero, power_spectrum, alpha=0)
        ax_new.set_xscale("log")
        ax_new.set_xlabel("Period (seconds)")
        ax_new.set_xlim(period_min, period_max)

        # Phase spectral plot
        phase_spectrum = np.angle(fft[nonzero_indices])
        if use_log_scale:
            axs[idx, 1].semilogx(
                freqs_nonzero,
                phase_spectrum,
                color=dataset_color,
                label=f"{name_lst[idx]} Phase Spectrum",
            )
        else:
            axs[idx, 1].plot(
                freqs_nonzero,
                phase_spectrum,
                color=dataset_color,
                label=f"{name_lst[idx]} Phase Spectrum",
            )
        axs[idx, 1].set_title(
            textwrap.fill(
                f"{name_lst[idx]} Phase Spectrum",
                width=50,
                break_long_words=True,
            )
        )
        axs[idx, 1].set_xlabel("Frequency (Hz)")
        axs[idx, 1].set_ylabel("Phase (radians)")
        axs[idx, 1].grid(True)

    plt.tight_layout()
    fig_file = os.path.join(
        save_dir, f"vis_spectral_analysis_{explain}_log{use_log_scale}.png"
    )
    plt.savefig(fig_file)
    plt.close()

    print(f"Spectral analysis plots saved at: {fig_file}")

    # Returning only for the first dataset for simplicity
    return power_spectrum_list, freqs_nonzero, periods_nonzero


def visualize_variable_evolution(
    states,
    batch_data,
    save_dir,
    variable_name="h",
    alphas=None,
    add_lines_lst=[],
    is_segmented_1d=False,
):
    seq_len, batch_size, x_dim = batch_data.shape

    num_dims = states.shape[2]

    # Prepare colors for the plots
    if alphas is not None:
        colors = [plt.cm.viridis(alpha.item()) for alpha in alphas]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_dims))

    # Determine the layout based on is_segmented_1d
    if is_segmented_1d:
        # For segmented 1D data, use 2 rows: upper for state evolution, lower for reconstructed 1D signal
        fig, axs = plt.subplots(
            2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        axs = np.atleast_1d(axs)  # Ensure axs is always iterable
    else:
        # For true multi-dimensional data, use 1 row for state evolution and multiple rows for signal dimensions
        num_signal_dims = min(x_dim, 4)  # Cap at 4 dimensions for readability
        total_rows = 1 + num_signal_dims
        fig, axs = plt.subplots(
            total_rows,
            1,
            figsize=(16, 4 * total_rows),
            gridspec_kw={"height_ratios": [3] + [1] * num_signal_dims},
        )
        axs = np.atleast_1d(axs)  # Ensure axs is iterable

    # Plotting the variables on the first subplot
    for dim in range(num_dims):
        axs[0].plot(
            states[:, 0, dim].cpu().numpy(), label=f"Dim {dim}", color=colors[dim]
        )

    # Adjust layout before adding colorbar
    plt.tight_layout()
    # Add more vertical space between subplots
    plt.subplots_adjust(hspace=0.3, right=0.85)  # Leave space on the right for colorbar

    # Add colorbar if alphas are provided
    if alphas is not None:
        import matplotlib.cm as cm

        alphas_np = alphas.cpu().numpy()
        sm = cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=np.min(alphas_np), vmax=np.max(alphas_np)),
        )
        sm.set_array([])
        # Position colorbar manually next to the first subplot
        cax = fig.add_axes(
            [0.87, 0.55, 0.02, 0.3]
        )  # [left, bottom, width, height] in figure coordinates
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Alpha values")

    # Add vertical lines at specified epochs
    for t in add_lines_lst:
        axs[0].axvline(x=t, color="r", linestyle="--")
        for ax_idx in range(1, len(axs)):
            axs[ax_idx].axvline(x=t, color="r", linestyle="--")

    str_alphas = (
        f" α: {[round(float(a), 4) for a in sorted(set(alphas.cpu().detach().numpy()))]}"
        if alphas is not None
        else ""
    )
    axs[0].set_title(
        textwrap.fill(
            f"Evolution of {variable_name} States over Time" + str_alphas,
            width=50,
            break_long_words=True,
        )
    )

    axs[0].set_xlabel("Time steps")
    axs[0].set_ylabel(f"{variable_name} state value")
    if num_dims <= 10:
        axs[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    axs[0].grid(True)

    # Plotting the true signal(s) with synchronized time axis
    if is_segmented_1d:
        # Reconstruct 1D signal from segmented chunks
        batch_data_np = batch_data[:, 0, :].cpu().numpy()
        reshaped_batch_data = batch_data_np.reshape(-1)
        time_steps = np.arange(len(reshaped_batch_data)) / x_dim
        axs[1].plot(time_steps, reshaped_batch_data, label="True Signal", color="blue")
        axs[1].set_title(
            textwrap.fill(
                f"True Signal for {variable_name} (reconstructed from {x_dim}-D chunks)",
                width=50,
                break_long_words=True,
            )
        )
        axs[1].set_xlabel("Time steps (original 1D)")
        axs[1].set_ylabel("Signal value")
        axs[1].legend()
        axs[1].grid(True)
    else:
        # Plot each input dimension separately
        batch_data_np = batch_data[:, 0, :].cpu().numpy()  # Shape: (seq_len, x_dim)
        time_steps = np.arange(batch_data_np.shape[0])

        # Plot each dimension in its own subplot
        dim_colors = plt.cm.Set2(np.linspace(0, 1, x_dim))
        for dim in range(min(x_dim, 4)):  # Limit to 4 dimensions for readability
            ax_idx = 1 + dim
            ax = axs[ax_idx] if len(axs) > ax_idx else axs[-1]
            ax.plot(
                time_steps,
                batch_data_np[:, dim],
                label=f"Dim {dim}",
                color=dim_colors[dim],
            )
            ax.set_title(
                textwrap.fill(
                    f"True Signal - Input Dimension {dim}",
                    width=50,
                    break_long_words=True,
                )
            )
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)

        # If x_dim > 4, add a note on the last subplot
        if x_dim > 4:
            axs[-1].text(
                0.5,
                0.5,
                f"(Showing 4 out of {x_dim} dimensions)",
                ha="center",
                va="center",
                transform=axs[-1].transAxes,
                fontsize=12,
                style="italic",
                color="gray",
            )

    # Save the figure
    fig_file = os.path.join(save_dir, f"vis_evolution_of_{variable_name}_state.png")
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)


def visualize_teacherforcing_2_autonomous(
    batch_data,
    dvae,
    auto_mode_selector,
    save_path,
    explain="",
    inference_mode=False,
    seq_len=None,
    is_segmented_1d=False,
    noise_selector=None,
    missing_mask=None,
    hide_mask_output=False,
):
    # Get sequence length and dimensions
    seq_len_total, batch_size, x_dim = batch_data.shape
    n_seq = seq_len_total if seq_len is None else min(seq_len, seq_len_total)

    if noise_selector is None:
        noise_selector = None
    else:
        noise_selector = noise_selector[:n_seq, :1, :]

    # Move data to device and generate reconstruction
    batch_data = batch_data.to(dvae.device)
    recon_batch_data = (
        dvae(
            batch_data[:n_seq, :1, :],
            mode_selector=auto_mode_selector[:n_seq, :1, :],
            noise_selector=noise_selector,
            inference_mode=inference_mode,
        )
        .detach()
        .clone()
    )

    if is_segmented_1d:
        # Scenario 1: Flatten segmented 1D data
        true_series = batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()
        recon_series = recon_batch_data[:n_seq, 0, :].reshape(-1).cpu().numpy()
        # Expand mode_selector to match flattened length
        expanded_mode_selector = (
            auto_mode_selector[:n_seq, 0, :].reshape(-1).cpu().numpy()
        )
        # Flatten missing_mask to match flattened data
        expanded_missing_mask = None
        if missing_mask is not None:
            expanded_missing_mask = missing_mask[:n_seq, 0, :].reshape(-1)
        # If requested, hide mask output channel (e.g., only_x_indicate)
        if hide_mask_output:
            # When flattened, keep only the first channel
            if recon_series.ndim > 1:
                recon_series = recon_series[:, :1].reshape(-1)
            if true_series.ndim > 1:
                true_series = true_series[:, :1].reshape(-1)
            if expanded_mode_selector is not None:
                expanded_mode_selector = (
                    auto_mode_selector[:n_seq, 0, :1].reshape(-1).cpu().numpy()
                )
            if expanded_missing_mask is not None:
                # expanded_missing_mask already flattened per channel; take first channel
                expanded_missing_mask = expanded_missing_mask
        if noise_selector is not None:
            nv = None
            if hasattr(dvae, "noise_values"):
                nv = getattr(dvae, "noise_values")
            elif hasattr(dvae, "noise_vals"):
                nv = getattr(dvae, "noise_vals")

            if nv is not None:
                if torch.is_tensor(nv):
                    tmp = nv[:n_seq, 0, :].reshape(-1)
                    if getattr(tmp, "requires_grad", False):
                        tmp = tmp.detach()
                    noise_series = tmp.cpu().numpy()
                else:
                    noise_series = np.array(nv[:n_seq, 0, :].reshape(-1))
            else:
                noise_series = None
        else:
            noise_series = None
    else:
        # Scenario 2: Keep high-dimensional data as 2D
        true_series = batch_data[:n_seq, 0, :].cpu().numpy()  # Shape: (n_seq, x_dim)
        recon_series = (
            recon_batch_data[:n_seq, 0, :].cpu().numpy()
        )  # Shape: (n_seq, x_dim)
        expanded_mode_selector = auto_mode_selector[:n_seq, 0, :].cpu().numpy()
        # Extract missing_mask slice
        expanded_missing_mask = None
        if missing_mask is not None:
            expanded_missing_mask = missing_mask[:n_seq, 0, :]
        # Optionally hide the mask output dimension from plots
        if hide_mask_output and true_series.ndim > 1 and true_series.shape[1] > 1:
            true_series = true_series[:, :1]
            recon_series = recon_series[:, :1]
            expanded_mode_selector = auto_mode_selector[:n_seq, 0, :1].cpu().numpy()
            if expanded_missing_mask is not None:
                expanded_missing_mask = expanded_missing_mask[:, :1]
        if noise_selector is not None:
            nv = None
            if hasattr(dvae, "noise_values"):
                nv = getattr(dvae, "noise_values")
            elif hasattr(dvae, "noise_vals"):
                nv = getattr(dvae, "noise_vals")

            if nv is not None:
                if torch.is_tensor(nv):
                    tmp = nv[:n_seq, 0, :]
                    if getattr(tmp, "requires_grad", False):
                        tmp = tmp.detach()
                    noise_series = tmp.cpu().numpy()
                else:
                    noise_series = np.array(nv[:n_seq, 0, :])
            else:
                noise_series = None
        else:
            noise_series = None

    recon_tf_color_intensity = auto_mode_selector[:n_seq, 0, :].cpu().numpy()
    recon_auto_color_intensity = 1 - auto_mode_selector[:n_seq, 0, :].cpu().numpy()
    recon_color = (recon_tf_color_intensity, recon_auto_color_intensity, 0)
    sequences = [
        {"title": "True Series", "data": true_series, "color": "blue"},
        {"title": "Reconstructed Series", "data": recon_series, "color": recon_color},
    ]
    if noise_series is not None:
        sequences.append(
            {
                "title": "Added Noise",
                "data": noise_series,
                "color": "purple",
            }
        )
    # Call visualization function with missing_mask
    visualize_sequences(
        sequences,
        expanded_mode_selector,
        save_path,
        explain,
        missing_mask=expanded_missing_mask,
    )


def reduce_dimensions(embeddings, technique="pca", n_components=3):
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
    if technique == "pca":
        reducer = PCA(n_components=n_components)
    elif technique == "tsne":
        reducer = TSNE(n_components=n_components, n_iter=300)
    elif technique == "lle":
        reducer = LocallyLinearEmbedding(
            n_components=n_components, eigen_solver="dense"
        )
    elif technique == "umap":
        reducer = UMAP(n_components=n_components)
    elif technique == "ica":
        reducer = FastICA(n_components=n_components)
    elif technique == "mds":
        reducer = MDS(n_components=n_components)
    elif technique == "nmf":
        # Shift data to be non-negative if using NMF
        embeddings_min = embeddings.min()
        if embeddings_min < 0:
            embeddings = embeddings - embeddings_min
        reducer = NMF(n_components=n_components, init="random", random_state=0)
    elif technique == "isomap":
        reducer = Isomap(n_components=n_components)
    elif technique == "laplacian":
        reducer = SpectralEmbedding(n_components=n_components)
    elif technique == "kernel_pca":
        reducer = KernelPCA(n_components=n_components, kernel="rbf")
    else:
        raise ValueError(f"Unsupported dimensionality reduction technique: {technique}")

    # Perform the dimensionality reduction
    # show error if it fails
    try:
        reduced_embeddings = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"Failed to reduce dimensions using {technique}. Error: {e}")
        return None

    return reduced_embeddings


def visualize_embedding_space(
    states_list,
    save_dir,
    variable_name,
    condition_names,
    explain="",
    technique="pca",
    rotation_speed=5,
    total_rotation=360,
    base_colors=["Blues"],
):
    """
    Visualize a list of embedding states in separate 3D subplots, each with its own animation.

    Args:
    - states_list: List of states numpy arrays.
    - save_dir: Directory to save the plots.
    - variable_name: Name of the variable corresponding to states.
    - condition_names: List of condition names for each set of states.
    - explain: Explanation text to add to the plot title.
    - technique: Dimensionality reduction technique.
    - rotation_speed: Degrees to rotate for each frame of the animation.
    - total_rotation: Total degrees of rotation.
    - base_colors: List of base colors for each subplot.
    """

    # Apply publication-ready formatting
    plot_config = get_plot_config(paper_ready=True)

    num_plots = len(states_list)
    fig = plt.figure(figsize=(14, 10 * num_plots))

    for i, (states, condition_name, base_color) in enumerate(
        zip(states_list, condition_names, base_colors)
    ):
        ax = fig.add_subplot(num_plots, 1, i + 1, projection="3d")
        reduced_embeddings = reduce_dimensions(states.cpu(), technique=technique)
        if reduced_embeddings is None:
            continue  # Skip this iteration if dimensionality reduction failed
        zs = reduced_embeddings[:, 2]
        cmap = plt.get_cmap(base_color)
        z_min, z_max = zs.min(), zs.max()

        for j in range(1, len(reduced_embeddings)):
            x, y, z = (
                reduced_embeddings[j - 1 : j + 1, 0],
                reduced_embeddings[j - 1 : j + 1, 1],
                reduced_embeddings[j - 1 : j + 1, 2],
            )
            color_value = (z.mean() - z_min) / (z_max - z_min)
            ax.plot(x, y, z, color=cmap(color_value))

        # Set axis labels with publication-ready font sizes
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

        # Only show title if not paper-ready mode
        if plot_config.get("show_title", True):
            ax.set_title(
                textwrap.fill(
                    f"Trajectory of {variable_name.capitalize()} {condition_name.capitalize()} in {explain} {technique.upper()} space",
                    width=50,
                    break_long_words=True,
                )
            )

    def update(frame):
        for ax in fig.axes:
            ax.view_init(30, frame)
        return fig.axes

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, total_rotation, rotation_speed),
        interval=200,
        blit=False,
    )

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct a unique filename to avoid overwriting
    fig_file = os.path.join(
        save_dir, f"vis_trajectory_of_{variable_name}_space_{technique}.gif"
    )
    print(f"Saving animation at: {fig_file}")

    start_time = time.time()
    anim.save(fig_file, writer=PillowWriter(fps=10))
    end_time = time.time()
    print(f"Animation saved in {end_time - start_time:.2f} seconds")
    plt.close(fig)


def visualize_accuracy_over_time(
    accuracy_values,
    variance_values,
    save_dir,
    measure,
    num_batches,
    num_iter,
    explain,
    autonomous_mode_selector,
):
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
        color = "green" if autonomous_mode_selector[i] == 0 else "red"
        plt.plot(
            time_steps[i : i + 2], accuracy_values.cpu().numpy()[i : i + 2], color=color
        )

    # Shading the variance
    std_dev = torch.sqrt(variance_values).cpu().numpy()
    plt.fill_between(
        time_steps,
        (accuracy_values - std_dev).cpu().numpy(),
        (accuracy_values + std_dev).cpu().numpy(),
        color="gray",
        alpha=0.3,
    )

    plt.xlabel("Time Steps")
    plt.ylabel(measure.upper())
    title = f"Expected {measure.upper()} Over Time - {explain.capitalize()} (Nbatch={num_batches}, Niter={num_iter})"
    plt.title(textwrap.fill(title, width=50, break_long_words=True))

    plt.grid(True)

    # Custom legend
    plt.plot([], [], color="green", label="Teacher-Forced Sequence")
    plt.plot([], [], color="red", label="Autonomous Sequence")
    plt.plot([], [], color="gray", alpha=0.3, label="Variance")
    plt.legend()

    fig_file = os.path.join(
        save_dir,
        f'vis_accuracy_{measure}_{explain.replace(" ", "_")}_btch{num_batches}_iter{num_iter}.png',
    )
    plt.savefig(fig_file)
    print(f"{measure} plot saved at: {fig_file}")
    plt.close()


def visualize_delay_embedding(
    embedded,
    save_dir,
    variable_name,
    explain="",
    base_color="Blues",
    rotation_speed=5,
    total_rotation=360,
    handle_nan="mask",  # Only for viz if embedded has NaNs
):
    """
    Visualize the pre-computed 3D delay embedding as a rotating GIF.

    :param embedded: Pre-computed embedding array (N, 3) from compute_delay_embedding.
    :param save_dir: Directory to save the GIF.
    :param variable_name: Name for title/filename.
    :param explain: Additional title text.
    :param base_color: Colormap base (e.g., 'Blues').
    :param rotation_speed: Degrees per frame.
    :param total_rotation: Total rotation degrees.
    :param handle_nan: If embedded has NaNs, 'mask' skips invalid segments.
    :return: Path to saved GIF.
    """
    if not isinstance(embedded, np.ndarray) or embedded.shape[1] != 3:
        raise ValueError("embedded must be (N, 3) NumPy array.")

    def normalize_filename_token(token):
        return token.replace("-_", "-")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Safe min/max/mean with NaNs
    zs = embedded[:, 2]
    z_min = np.nanmin(zs)
    z_max = np.nanmax(zs)
    if np.isnan(z_min) or np.isnan(z_max):
        raise ValueError("All z values are NaN; cannot visualize.")

    cmap = plt.get_cmap(base_color)

    # Plot loop: Handle NaNs by finding contiguous non-NaN segments
    i = 1
    while i < len(embedded):
        # Find next contiguous non-NaN segment
        start = i - 1
        while i < len(embedded) and not np.any(np.isnan(embedded[i])):
            i += 1
        end = i

        if end - start >= 2:  # Need at least 2 points for a line
            segment = embedded[start:end]
            for j in range(1, len(segment)):
                # Get the two consecutive points for this sub-segment
                x_seg = segment[j - 1 : j + 1, 0]
                y_seg = segment[j - 1 : j + 1, 1]
                z_seg = segment[j - 1 : j + 1, 2]

                # Use the z-value of the starting point (or average if preferred)
                z_val = segment[j - 1, 2]
                color_value = (z_val - z_min) / (z_max - z_min)

                # Plot this small segment with its specific color
                ax.plot(x_seg, y_seg, z_seg, color=cmap(color_value))

        i += 1  # Skip to next potential segment

    font_scale = 3
    title = textwrap.fill(
        f"Delay Embedding of {variable_name.capitalize()} {explain}",
        width=50,
        break_long_words=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=12 * font_scale)

    def update(frame):
        ax.view_init(30, frame)
        return (fig,)

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, total_rotation, rotation_speed),
        interval=200,
        blit=True,
    )

    fig_file = os.path.join(
        save_dir,
        f"vis_delay_embedding_of_{normalize_filename_token(variable_name)}_{normalize_filename_token(explain)}.gif",
    )
    print(f"Saving animation at: {fig_file}")
    start_time = time.time()
    anim.save(fig_file, writer="pillow", fps=10)
    end_time = time.time()
    print(f"Animation saved in {end_time - start_time:.2f} seconds")

    plt.close()
    return fig_file


# def visualize_delay_embedding(
#     observation,
#     delay,
#     dimensions,
#     save_dir,
#     variable_name,
#     explain="",
#     base_color="Blues",
#     rotation_speed=5,
#     total_rotation=360,
# ):
#     n = len(observation)
#     embedding_length = n - (dimensions - 1) * delay
#     if embedding_length <= 0:
#         raise ValueError(
#             "Delay and dimensions are too large for the length of the observation array."
#         )

#     embedded = np.empty((embedding_length, dimensions))
#     for i in range(dimensions):
#         embedded[:, i] = observation[i * delay : i * delay + embedding_length]

#     if dimensions != 3:
#         raise NotImplementedError(
#             "Rotation and color gradient for dimensions other than 3 is not implemented."
#         )

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     cmap = plt.get_cmap(base_color)
#     zs = embedded[:, 2]
#     z_min, z_max = zs.min(), zs.max()

#     for i in range(1, len(embedded)):
#         x = embedded[i - 1 : i + 1, 0]
#         y = embedded[i - 1 : i + 1, 1]
#         z = embedded[i - 1 : i + 1, 2]
#         color_value = (z.mean() - z_min) / (z_max - z_min)
#         ax.plot(x, y, z, color=cmap(color_value))

#     font_scale = 3

#     # ax.set_xlabel("X(t)", fontsize=16 * font_scale)
#     # ax.set_ylabel("X(t + delay)", fontsize=16 * font_scale)
#     # ax.set_zlabel("X(t + 2 * delay)", fontsize=16 * font_scale)
#     title = textwrap.fill(
#         f"Delay Embedding of {variable_name.capitalize()} {explain} (Delay: {delay})",
#         width=50,
#         break_long_words=True,
#     )
#     # plt.title(title, fontsize=16 * font_scale)

#     # Adjust tick label font sizes
#     ax.tick_params(axis="both", which="major", labelsize=12 * font_scale)

#     def update(frame):
#         ax.view_init(30, frame)
#         return (fig,)

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=np.arange(0, total_rotation, rotation_speed),
#         interval=200,
#         blit=True,
#     )
#     fig_file = os.path.join(
#         save_dir, f"vis_delay_embedding_of_{variable_name}_τ{delay}_{explain}.gif"
#     )

#     print(f"Saving animation at: {fig_file}")
#     start_time = time.time()
#     anim.save(fig_file, writer=PillowWriter(fps=10))
#     end_time = time.time()

#     print(f"Animation saved in {end_time - start_time:.2f} seconds")

#     plt.close()
#     # Returning the embedded data for further analysis
#     return embedded


def visualize_alpha_history_and_spectrums(
    sigmas_history,
    power_spectrum_lst,
    spectrum_color_lst,
    spectrum_name_lst,
    frequencies,
    save_dir,
    dt,
    kl_warm_epochs=None,
    explain="",
    true_alphas=[],
):
    plt.clf()
    periods = 1 / frequencies
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.rcParams["font.size"] = 12

    # Example color scheme for alpha curves
    alpha_colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

    alphas_from_periods = dt / periods
    periods_max, periods_min = np.max(periods), np.min(periods)
    alpha_range_min = 1 / (1 + np.exp(-np.max(sigmas_history))) / 10
    alpha_range_max = 1.1
    ylim_max = max(dt / alpha_range_min, periods_max)
    ylim_min = min(dt / alpha_range_max, periods_min)

    if kl_warm_epochs is not None:
        for kl_warm_epoch in kl_warm_epochs:
            ax1.axvline(
                x=kl_warm_epoch,
                color="r",
                linestyle="--",
                label="KL & SS Warm-up" if kl_warm_epoch == kl_warm_epochs[0] else "",
            )

    # Plot and annotate true alphas with a specific color
    true_alpha_color = "black"
    for idx, true_alpha in enumerate(true_alphas, start=1):
        line_y = dt / true_alpha
        ax1.axhline(
            y=line_y,
            color=true_alpha_color,
            linestyle="--",
            label=f'"True" α' if idx == 1 else "",
        )
        ax2.axhline(y=true_alpha / dt, color=true_alpha_color, linestyle="--")
        ax1.text(
            0.5,
            line_y,
            f"α_true: {true_alpha:.{4}g}",
            verticalalignment="bottom",
            horizontalalignment="center",
            transform=ax1.get_yaxis_transform(),
            color=true_alpha_color,
            fontsize=15,
        )

    num_alphas = sigmas_history.shape[0]
    # Ensure we don't exceed the number of predefined colors
    for i in range(min(num_alphas, len(alpha_colors))):
        alphas = 1 / (1 + np.exp(-sigmas_history[i]))
        periods_from_alpha = dt / alphas
        curve_color = alpha_colors[i]  # Get color for this curve
        ax1.plot(periods_from_alpha, label=f"α {i+1}", color=curve_color)
        # Annotate the last alpha value
        last_alpha_period = periods_from_alpha[-1]
        ax1.text(
            len(sigmas_history[i]) - 1,
            last_alpha_period,
            f"α_end: {alphas[-1]:.{4}g}",
            verticalalignment="bottom",
            horizontalalignment="right",
            color=curve_color,
            fontsize=15,
        )
        # Optionally, annotate the initial alpha value
        init_alpha_period = periods_from_alpha[0]
        ax1.text(
            0,
            init_alpha_period,
            f"α_0: {alphas[0]:.{4}g}",
            verticalalignment="bottom",
            horizontalalignment="left",
            color=curve_color,
            fontsize=15,
        )

    ax1.set_title(
        textwrap.fill("α values during training", width=50, break_long_words=True),
        fontsize=18,
    )
    ax1.legend(fontsize=16, loc="upper left")
    ax1.set_xlabel("Epochs", fontsize=16)
    ax1.set_ylabel("Period (sec)", fontsize=16)
    ax1.set_yscale("log")
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.grid(True)

    ax1_right = ax1.twinx()
    ax1_right.set_ylabel("α =Δt/T", fontsize=16)
    ax1_right.set_yscale("log")
    ax1_right.set_ylim(dt / ylim_max, dt / ylim_min)
    ax1_right.invert_yaxis()

    for power_spectrum, color, label in zip(
        power_spectrum_lst, spectrum_color_lst, spectrum_name_lst
    ):
        ax2.loglog(power_spectrum, frequencies, color=color, alpha=0.5, label=label)

    ax2.set_title("Lorenz63\nPower Spectrum", fontsize=18)
    ax2.legend(fontsize=16, loc="upper right")
    ax2.set_xlabel("Amplitude", fontsize=16)
    ax2.set_ylabel("Frequency (Hz)", fontsize=16)
    ax2.set_ylim(1 / ylim_min, 1 / ylim_max)
    ax2.grid(True)
    ax2.invert_xaxis()

    ax2_right = ax2.twinx()
    ax2_right.set_ylabel("Periods (s)", fontsize=16)
    ax2_right.set_yscale("log")
    ax2_right.set_ylim(ylim_max, ylim_min)
    ax2_right.invert_yaxis()

    plt.suptitle(
        textwrap.fill(
            f"α during training and Power Spectrum {explain}",
            width=50,
            break_long_words=True,
        ),
        fontsize=20,
        x=0.5,
        y=1.02,
    )
    plt.tight_layout()
    fig_file = os.path.join(save_dir, f"vis_alpha_vs_power_spectrum_{explain}.png")
    plt.savefig(fig_file, bbox_inches="tight")
    plt.close()
    print(f"Alpha vs Power Spectrum plot saved at: {fig_file}")


# Visualize the errors from an error list of any error, using a bar graph. x-axis is the subjects of the error (name_lst), y-axis is the error value.
# true_signal_index is the index of the true signal in the error_lst.
def visualize_errors_from_lst(
    error_lst, name_lst, error_unit, colors, save_dir, explain, true_signal_index=None
):
    plt.figure(figsize=(12, 6))
    plt.bar(name_lst, error_lst, color=colors)
    plt.xlabel("Signals")
    plt.ylabel(f"{error_unit}")
    plt.title(
        textwrap.fill(
            f"{explain} Error against True Signal", width=50, break_long_words=True
        )
    )
    plt.grid(True)
    fig_file = os.path.join(save_dir, f"vis_errors_{explain}.png")
    plt.savefig(fig_file)
    plt.close()
    print(f"Errors plot saved at: {fig_file}")
