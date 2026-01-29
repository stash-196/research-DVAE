import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label  # For segment lengths


def make_log_bins(data, num_bins=100, log=False):
    """
    Create symlog-spaced bins for a given data array.
    """
    # Return default number of bins if data is empty, or if log is disabled
    if len(data) == 0 or not log:
        return num_bins

    # Separate positive and negative values
    pos_data = data[data > 0]
    neg_data = data[data < 0]

    # Generate log-spaced bins for positive values
    if len(pos_data) > 0:
        pos_bins = np.logspace(
            np.log10(max(pos_data.min(), 1e-10)),
            np.log10(pos_data.max()),
            num_bins // 2,
        )
    else:
        pos_bins = np.array([])

    # Generate log-spaced bins for negative values
    if len(neg_data) > 0:
        neg_bins = -np.logspace(
            np.log10(abs(neg_data.max())),
            np.log10(max(abs(neg_data.min()), 1e-10)),
            num_bins // 2,
        )
    else:
        neg_bins = np.array([])

    # Combine negative, zero, and positive bins
    bins = np.concatenate((neg_bins[::-1], [0], pos_bins))

    return bins


def plot_nan_heatmap(data, channels, save_root=None, explain="", show_plot=False):
    """
    Heatmap of NaN patterns (rows=channels, columns=time; white=NaN, black=valid).
    """
    import matplotlib.dates as mdates  # For datetime formatting if needed

    nan_df = data[channels].isna().astype(int)  # 1=NaN, 0=valid
    fig, ax = plt.subplots(figsize=(12, len(channels) * 1.5))
    sns.heatmap(
        nan_df.T, cbar=False, cmap="binary", ax=ax
    )  # Transpose for channels as rows
    ax.set_title(f"NaN Heatmap: {explain}")
    # ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Channels")
    ax.set_yticklabels(channels, rotation=0)

    # Set x-axis to datetime (subsample ticks to ~10 for readability)
    n_ticks = 10
    step = max(1, len(data) // n_ticks)
    xtick_positions = range(0, len(data), step)
    xtick_labels = [
        # data["datetime"].iloc[i].strftime("%Y-%m-%d %H:%M:%S") for i in xtick_positions
        data["datetime"].iloc[i].strftime("%H:%M:%S")
        for i in xtick_positions
    ]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_xlabel("Time (datetime)")

    # Annotation if no NaNs
    if nan_df.sum().sum() == 0:
        ax.text(
            0.5,
            0.5,
            "No NaNs Detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )

    if save_root:
        filepath = os.path.join(save_root, f"nan_heatmap_{explain}.png")
        plt.savefig(filepath)
        print(f"[INFO] NaN heatmap saved to: {filepath}")

    if show_plot:
        plt.show()
    plt.close()


def plot_segment_length_hist(
    data,
    channels,
    save_root=None,
    explain="",
    bins=50,
    show_plot=False,
    non_or_nonnan="nan",
):
    """
    Histogram of non-NaN segment lengths per channel.
    """
    fig, axes = plt.subplots(
        len(channels), 1, figsize=(10, 4 * len(channels)), sharex=True
    )
    axes = np.atleast_1d(axes)  # Handle single channel

    for i, ch in enumerate(channels):
        if non_or_nonnan == "nan":
            mask = data[ch].isna().values  # 1D numpy array
        elif non_or_nonnan == "nonnan":
            mask = ~data[ch].isna().values  # 1D numpy array
        else:
            raise ValueError("non_or_nonnan must be either 'nan' or 'nonnan'")
        structure = np.array([1, 1, 1])  # 1D connectivity: left, center, right

        labeled, num_segments = label(mask, structure)
        seg_lengths = [np.sum(labeled == j) for j in range(1, num_segments + 1)]

        if seg_lengths:
            bin_edges = make_log_bins(np.array(seg_lengths), bins, log=True)
            if non_or_nonnan == "nan":
                axes[i].hist(seg_lengths, bins=bin_edges, color="red", alpha=0.7)
            else:
                axes[i].hist(seg_lengths, bins=bin_edges, color="blue", alpha=0.7)
        else:
            axes[i].text(
                0.5,
                0.5,
                "No Non-NaN Segments (All Valid or All NaN)",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
                fontsize=10,
                color="red",
            )
        axes[i].set_title(f"{non_or_nonnan} Segment Lengths: {ch} ({explain})")
        axes[i].set_xlabel("Segment Length (samples)")
        axes[i].set_ylabel("Frequency")
        # axes[i].set_yscale("symlog")
        axes[i].set_xscale("symlog")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_root:
        filepath = os.path.join(
            save_root, f"segment_hist_{explain}_{non_or_nonnan}.png"
        )
        plt.savefig(filepath)
        print(f"[INFO] Segment histogram saved to: {filepath}")

    if show_plot:
        plt.show()
    plt.close()
