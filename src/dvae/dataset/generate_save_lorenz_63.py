# %%
import textwrap
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.io as pio
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pickle
from fractions import Fraction

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
import os
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go

from dvae.utils import find_project_root

project_root = find_project_root(__file__)

# Handy function stolen from the fast.ai library
# Get location of current files directory
file_dir = os.path.dirname(os.path.realpath(__file__))


# %%


def human_format(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.0f}M"
    elif num >= 1_000:
        return f"{num/1_000:.0f}k"
    else:
        return str(num)


def V(x, requires_grad=False, gpu=False):
    t = torch.FloatTensor(np.atleast_1d(x).astype(np.float32))
    if gpu:
        t = t.cuda()
    return Variable(t, requires_grad=requires_grad)


class L63:
    def __init__(self, sigma, rho, beta, init, dt):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.x, self.y, self.z = init
        self.dt = dt
        self.hist = [init]

    def step(self):
        self.x += (self.sigma * (self.y - self.x)) * self.dt
        self.y += (self.x * (self.rho - self.z)) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt
        self.hist.append([self.x, self.y, self.z])

    def integrate(self, n_steps):
        for n in range(n_steps):
            self.step()


def plot_attractor_plotly(hists, save_dir=None, explain=None, format="pdf"):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = go.Figure()
    for h in hists:
        fig.add_trace(
            go.Scatter3d(
                x=h[:, 0], y=h[:, 1], z=h[:, 2], mode="lines", line=dict(color="blue")
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        title=textwrap.fill(
            f"Attractor Plot for {explain.replace('_', ' ')} set",
            width=50,
            break_long_words=True,
        ),
    )
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"attractor_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_attractor_subplots(hists, explain, save_dir=None, format="pdf"):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]

    # Create subplots: one row for each of x, y and z
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("X Timeseries", "Y Timeseries", "Z Timeseries"),
    )

    for h in hists:
        # X timeseries
        fig.add_trace(
            go.Scatter(y=h[:, 0], mode="lines", line=dict(color="blue")), row=1, col=1
        )
        # Y timeseries
        fig.add_trace(
            go.Scatter(y=h[:, 1], mode="lines", line=dict(color="red")), row=2, col=1
        )
        # Z timeseries
        fig.add_trace(
            go.Scatter(y=h[:, 2], mode="lines", line=dict(color="green")), row=3, col=1
        )

    fig.update_layout(
        title_text="Timeseries Subplots for X, Y, and Z for {} set".format(
            explain.replace("_", " ")
        )
    )
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"timeseries_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_components_vs_time_plotly(
    time_series, time_step, explain, save_dir=None, format="pdf"
):
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x", line=dict(color="blue")))
    fig.add_trace(
        go.Scatter(x=t, y=y, mode="lines", name="y", line=dict(color="green"))
    )
    fig.add_trace(go.Scatter(x=t, y=z, mode="lines", name="z", line=dict(color="red")))

    fig.update_layout(
        title=textwrap.fill(
            f"Components of Lorenz63 System vs. Time for {explain.replace('_', ' ')} set",
            width=50,
            break_long_words=True,
        ),
        xaxis_title="Time",
        yaxis_title="Values",
        showlegend=True,
        template="plotly_white",
    )

    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"components_vs_time_{explain}.{format}")
        pio.write_image(fig, save_path)


# %%
def calculate_power_spectrum(time_series, sampling_rate):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series), 1 / sampling_rate)
    time_periods = np.zeros_like(freqs)
    # Get periods corresponding to frequencies
    time_periods[1:] = 1 / freqs[1:]

    nonzero_indices = np.where(freqs > 0)

    # Compute the power spectrum: the square of the absolute value of the FFT
    power_spectrum = np.abs(fft_result[nonzero_indices]) ** 2
    phases = np.angle(fft_result[nonzero_indices])

    # Compute the frequencies corresponding to the values in the power spectrum
    frequencies = np.fft.fftfreq(len(time_series))[nonzero_indices]

    return time_periods, frequencies, power_spectrum


def plot_power_spectrum_plotly(
    time_series, sampling_rate, explain, save_dir=None, format="pdf"
):
    # Create a figure
    fig = go.Figure()

    # Iterate over each component and plot its power spectrum
    for i, component in enumerate(["x", "y", "z"]):
        series = np.array(time_series)[:, i]
        time_periods, frequencies, spectrum = calculate_power_spectrum(
            series, sampling_rate
        )

        # Skip the zero frequency
        time_periods, frequencies, spectrum = (
            time_periods[1:],
            frequencies[1:],
            spectrum[1:],
        )

        fig.add_trace(
            go.Scatter(
                x=time_periods,
                y=spectrum,
                mode="lines",
                name=f"{component} power spectrum",
            )
        )

    # Set the x-axis to be log scale
    fig.update_layout(
        title="Power Spectrum of Lorenz63 System for {} set".format(explain),
        xaxis=dict(type="log", title="Time Periods"),
        # yaxis=dict(type="log", title="Power"),
        yaxis=dict(title="Power"),
        template="plotly_white",
    )

    # Display the figure
    fig.show()

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_power_spectrum_subplots_loglog(
    time_series, sampling_rate, explain, component_labels, save_dir=None, format="pdf"
):
    """
    Plots the power spectrum of each component in the input time series data on a log-log scale,
    including peak detection. It works for any input time series data, making it suitable for
    both original data and PCA components.

    :param time_series: Input time series data as a numpy array with shape (n_samples, n_components).
    :param sampling_rate: The sampling rate of the time series data.
    :param explain: Description or label for the dataset being analyzed.
    :param component_labels: List of strings representing the labels for each component in the time series.
    :param save_dir: Directory to save the plot image. If None, the plot is not saved.
    :param format: The format for saving the plot image (e.g., 'pdf', 'png').
    """
    fig = make_subplots(
        rows=len(component_labels),
        cols=1,
        subplot_titles=[f"{label} Power Spectrum" for label in component_labels],
    )

    for i, label in enumerate(component_labels):
        series = time_series[:, i]
        fft_result = np.fft.fft(series)
        frequencies = np.fft.fftfreq(len(series), 1 / sampling_rate)
        power_spectrum = np.abs(fft_result) ** 2

        # Focus on the positive frequencies
        positive_freqs = frequencies > 0
        frequencies = frequencies[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]

        # Find peaks
        peaks, properties = find_peaks(
            power_spectrum, height=0.5, distance=100, prominence=0.05
        )
        peak_heights = properties["peak_heights"]

        # Select the indices of the largest 3 peaks based on their height
        # Get indices of the largest 3 peaks
        largest_peaks_indices = np.argsort(peak_heights)[-3:]

        # Get the frequency indices of the largest peaks
        peaks = peaks[largest_peaks_indices]

        # Plotting
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=power_spectrum,
                mode="lines",
                name=f"{label} Power Spectrum",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frequencies[peaks],
                y=power_spectrum[peaks],
                mode="markers",
                marker=dict(color="red", size=8),
                name=f"{label} Peaks",
            ),
            row=i + 1,
            col=1,
        )

    # Update the layout for log-log scale
    fig.update_layout(
        height=600,
        width=800,
        title_text=f"Power Spectrum for {explain} (Log-Log Scale)",
    )
    fig.update_xaxes(type="log", title="Frequency")
    fig.update_yaxes(type="log", title="Power Spectrum")

    fig.show()

    if save_dir is not None and format:
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)

    return peaks, frequencies, power_spectrum


def generate_fixed_burst_mask(length, p_missing, burst_length=50):
    mask = np.zeros(length, dtype=int)
    num_bursts = int(p_missing * length / burst_length)
    starts = np.random.choice(length - burst_length, num_bursts, replace=False)
    for start in starts:
        mask[start : start + burst_length] = 1
    return mask


def generate_markovian_burst_mask(shape, pi_1, exp_burst_length):
    """
    Generate a Markovian burst mask for missing data.

    Parameters:
    - length (int): Length of the mask.
    - pi_1 (float): Desired long-run proportion of missing data (between 0 and 1).
    - p_01 (float): Probability of transitioning from missing (1) to present (0).

    Returns:
    - numpy array: Binary mask where 1 means missing, 0 means present.
    """
    length = shape[0]
    dim = shape[1] if len(shape) > 1 else 1

    exp_burst_length = exp_burst_length - 1

    pi_0 = 1 - pi_1
    p_11 = exp_burst_length / (1 + exp_burst_length)
    p_10 = 1 - p_11
    p_00 = 1 / pi_0 * ((1 + p_10) * pi_0 - p_10)
    p_01 = 1 - p_00

    P = np.array([[p_00, p_01], [p_10, p_11]])

    # print(f"p_00: {p_00}, p_01: {p_01}, p_10: {p_10}, p_11: {p_11}")
    # print(P)

    # Start with data present (state 0)
    masks = []
    # Generate the sequence
    for _ in range(dim):
        state = 0
        mask = []
        for _ in range(length):
            mask.append(state)
            state = np.random.choice([0, 1], p=P[state])
        masks.append(mask)
    return np.array(masks).T


# %%
# Define the default parameters values

save_dir_plots = os.path.join(project_root, "data/lorenz63", "plots")
save_dir_data = os.path.join(project_root, "data/lorenz63", "data")
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)
if not os.path.exists(save_dir_data):
    os.makedirs(save_dir_data)

# %%
# ================== For Dataset Generation ==================
sigma = 10
rho = 40
beta = 8 / 3
N = 15 * 60 * 24 * 5  # 5 days worth of data of frequency 0.25Hz (4s period)
# N = 50000  # 50k samples, 5 days worth of data of frequency 0.25Hz (4s period)
dt = 1e-2


beta_frac = Fraction(beta).limit_denominator()
beta_str = f"{beta_frac.numerator}d{beta_frac.denominator}"


parameter_str = f"sigma{sigma}_rho{rho}_beta{beta_str}_N{human_format(N)}_dt{dt}"
save_dir_specific_data = os.path.join(
    save_dir_data,
    parameter_str,
)
save_dir_specific_plots = os.path.join(
    save_dir_plots,
    parameter_str,
)
if not os.path.exists(save_dir_specific_data):
    os.makedirs(save_dir_specific_data)
if not os.path.exists(save_dir_specific_plots):
    os.makedirs(save_dir_specific_plots)


l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=dt)
l2 = L63(sigma, rho, beta, init=[10, 1, 2], dt=dt)
l1.integrate(int(N * 0.9))
l2.integrate(int(N * 0.1))

plot_attractor_plotly(
    [l1.hist], save_dir=save_dir_specific_plots, explain=f"{parameter_str}_train"
)
plot_attractor_plotly(
    [l2.hist], save_dir=save_dir_specific_plots, explain=f"{parameter_str}_test"
)


plot_attractor_subplots(
    [l1.hist], save_dir=save_dir_specific_plots, explain=f"{parameter_str}_train"
)
plot_attractor_subplots(
    [l2.hist], save_dir=save_dir_specific_plots, explain=f"{parameter_str}_test"
)

plot_components_vs_time_plotly(
    np.array(l1.hist),
    time_step=dt,
    explain=f"{parameter_str}_train",
    save_dir=save_dir_specific_plots,
)
plot_components_vs_time_plotly(
    np.array(l2.hist),
    time_step=dt,
    explain=f"{parameter_str}_test",
    save_dir=save_dir_specific_plots,
)


# store l.hist as pickle data for later use in pytorch dataloader
def save_pickle(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved data to {path}")


save_pickle(l1.hist, os.path.join(save_dir_specific_data, "complete_dataset_train.pkl"))
save_pickle(l2.hist, os.path.join(save_dir_specific_data, "complete_dataset_test.pkl"))
# %%
# ================== Generate Masks ==================
missing_ratios = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.92,
    0.95,
    0.99,
    1.0,
]
# Generate masks
np.random.seed(42)
# Do the same for the list of missing data
for p_nan in missing_ratios:
    mask_train = np.random.binomial(1, p_nan, size=(len(l1.hist), 3))
    mask_test = np.random.binomial(1, p_nan, size=(len(l2.hist), 3))

    save_pickle(
        mask_train,
        os.path.join(save_dir_specific_data, f"mask_Bernoulli_pnan{p_nan}_train.pkl"),
    )
    save_pickle(
        mask_test,
        os.path.join(save_dir_specific_data, f"mask_Bernoulli_pnan{p_nan}_test.pkl"),
    )

    markovian_expected_burst_length = 15
    if markovian_expected_burst_length * (1 - p_nan) >= p_nan:
        mask_train = generate_markovian_burst_mask(
            np.array(l1.hist).shape,
            pi_1=p_nan,
            exp_burst_length=markovian_expected_burst_length,
        )
        mask_test = generate_markovian_burst_mask(
            np.array(l2.hist).shape,
            pi_1=p_nan,
            exp_burst_length=markovian_expected_burst_length,
        )
        save_pickle(
            mask_train,
            os.path.join(
                save_dir_specific_data,
                f"mask_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_train.pkl",
            ),
        )
        save_pickle(
            mask_test,
            os.path.join(
                save_dir_specific_data,
                f"mask_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_test.pkl",
            ),
        )


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# Example: Load data with p_nan = 0.1
p_nan = 0.1
# load train data
complete_train = load_pickle(
    os.path.join(save_dir_specific_data, "complete_dataset_train.pkl")
)
mask_Bernoulli_train = load_pickle(
    os.path.join(save_dir_specific_data, f"mask_Bernoulli_pnan{p_nan}_train.pkl")
)
train_data_with_Bernoulli_nan = np.where(mask_Bernoulli_train, np.nan, complete_train)

mask_Markovian_train = load_pickle(
    os.path.join(
        save_dir_specific_data,
        f"mask_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_train.pkl",
    )
)
train_data_with_Markovian_nan = np.where(mask_Markovian_train, np.nan, complete_train)

# Load test data
complete_test = load_pickle(
    os.path.join(save_dir_specific_data, "complete_dataset_test.pkl")
)
mask_Bernoulli_test = load_pickle(
    os.path.join(save_dir_specific_data, f"mask_Bernoulli_pnan{p_nan}_test.pkl")
)
test_data_with_Bernoulli_nan = np.where(mask_Bernoulli_test, np.nan, complete_test)

mask_Markovian_test = load_pickle(
    os.path.join(
        save_dir_specific_data,
        f"mask_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_test.pkl",
    )
)
test_data_with_Markovian_nan = np.where(mask_Markovian_test, np.nan, complete_test)

plot_components_vs_time_plotly(
    np.array(train_data_with_Bernoulli_nan),
    time_step=dt,
    explain=f"s{sigma}_r{rho}_b{beta_str}_Bernoulli_pnan{p_nan}_train",
    save_dir=save_dir_plots,
)
plot_components_vs_time_plotly(
    np.array(test_data_with_Bernoulli_nan),
    time_step=dt,
    explain=f"s{sigma}_r{rho}_b{beta_str}_Bernoulli_pnan{p_nan}_test",
    save_dir=save_dir_plots,
)
plot_components_vs_time_plotly(
    np.array(train_data_with_Markovian_nan),
    time_step=dt,
    explain=f"s{sigma}_r{rho}_b{beta_str}_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_train",
    save_dir=save_dir_plots,
)
plot_components_vs_time_plotly(
    np.array(test_data_with_Markovian_nan),
    time_step=dt,
    explain=f"s{sigma}_r{rho}_b{beta_str}_Markov_AvgLen{markovian_expected_burst_length}_pnan{p_nan}_test",
    save_dir=save_dir_plots,
)
# %%

# calculate sampling rate
sampling_rate = 1 / dt

# Call the function to display the plot
plot_power_spectrum_plotly(
    np.array(masked_train_data),
    sampling_rate,
    explain=f"s{sigma}_r{rho}_b{beta_str}_train",
    save_dir="temp_save/lorenz63",
)
plot_power_spectrum_plotly(
    np.array(l2.hist),
    sampling_rate,
    explain=f"s{sigma}_r{rho}_b{beta_str}_test",
    save_dir="temp_save/lorenz63",
)


# Call the function to display the plots
# %%
# Example usage for original XYZ data
component_labels_xyz = ["X", "Y", "Z"]
peaks_train, frequenciest_train, power_spectrum_train = (
    plot_power_spectrum_subplots_loglog(
        np.array(l1.hist),
        sampling_rate,
        "s10_r28_b8d3_train_xyz",
        component_labels_xyz,
        save_dir="temp_save/lorenz63",
    )
)
peaks_test, frequencies_test, power_spectrum_test = plot_power_spectrum_subplots_loglog(
    np.array(l2.hist),
    sampling_rate,
    "s10_r28_b8d3_train_xyz",
    component_labels_xyz,
    save_dir="temp_save/lorenz63",
)

# Example usage for PCA components

# # calculate PCs from train and test data
# pca = PCA(n_components=3)
# PCs_train = pca.fit_transform(np.array(l1.hist))
# PCs_test = pca.transform(np.array(l2.hist))

# component_labels_pca = ["PC1", "PC2", "PC3"]
# plot_power_spectrum_subplots_loglog(PCs_train, sampling_rate, 's10_r28_b8d3_train_pca', component_labels_pca, save_dir='temp_save/lorenz63')
# plot_power_spectrum_subplots_loglog(PCs_test, sampling_rate, 's10_r28_b8d3_test_pca', component_labels_pca, save_dir='temp_save/lorenz63')
alphas = dt * frequencies_test[peaks_test]
# %%


def plot_delay_embedding(observation, delay, dimensions):
    """
    Plots the delay embedding of a 1D observation with lines.

    :param observation: 1D array of observations.
    :param delay: Time delay for embedding.
    :param dimensions: Number of embedding dimensions.
    """
    n = len(observation)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError(
            "Delay and dimensions are too large for the length of the observation array."
        )

    # Create the delay-embedded matrix
    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay : i * delay + embedding_length]

    # Plotting
    if dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(x=embedded[:, 0], y=embedded[:, 1], mode="lines")
        )
        fig.update_layout(
            title="2D Delay Embedding", xaxis_title="X(t)", yaxis_title="X(t + delay)"
        )
    elif dimensions == 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], mode="lines"
            )
        )
        fig.update_layout(
            title="3D Delay Embedding",
            scene=dict(
                xaxis_title="X(t)",
                yaxis_title="X(t + delay)",
                zaxis_title="X(t + 2 * delay)",
            ),
        )
    else:
        raise NotImplementedError(
            "Plotting for dimensions higher than 3 is not implemented."
        )

    fig.show()


# # Example usage
# # Assuming you have an array `x` from the Lorenz system:
# x = np.array(l1.hist)[:, 0]
# plot_delay_embedding(x, delay=10, dimensions=3)

# x_nan = np.array(l1_nan.hist)[:, 0]
# plot_delay_embedding(x_nan, delay=10, dimensions=3)

# %%


# %% ================== For Presentation ==================
sigma = 10
rho = 28
beta = 8 / 3
N = 1000
dt = 1e-2
l_present = L63(sigma, rho, beta, init=[1, 10, 20], dt=dt)
l_present.integrate(N)

# mask 10-20, 30-50, 70-80
L = len(l_present.hist)
mask_burst = np.zeros(L)
mask_burst[int(L * 0.1) : int(L * 0.2)] = 1
mask_burst[int(L * 0.3) : int(L * 0.4)] = 1
mask_burst[int(L * 0.6) : int(L * 0.8)] = 1


mask_burst_inv = mask_burst * -1 + 1

mask_bernoulli = np.random.binomial(1, 0.5, size=L)
maskb_inv = mask_bernoulli * -1 + 1

x_signal = np.array(l_present.hist)[:, 0]
# create x_signal copy masked with NaN
x_signal_data_remain = np.where(mask_burst, np.nan, x_signal)
x_signal_data_lack = np.where(mask_burst_inv, np.nan, x_signal)

x_signal_auto_sample = np.where(mask_bernoulli, np.nan, x_signal)
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 20,  # General font size
        "axes.titlesize": 22,  # Title size
        "axes.labelsize": 20,  # X and Y label size
        "xtick.labelsize": 18,  # X tick labels
        "ytick.labelsize": 18,  # Y tick labels
        "legend.fontsize": 18,  # Legend font size
        # font times new roman -> Liberation Serif
        "font.family": "serif",
        "font.serif": ["Liberation Serif"],
        "font.sans-serif": ["Liberation Serif"],
    }
)

# Plot Scheduled Sampling Learning
linewidth = 6
plt.figure(figsize=(10, 7.5))
offset = 20
(line1,) = plt.plot(
    x_signal + offset, color="blue", label="True Signal", linewidth=linewidth
)
(line2,) = plt.plot(
    x_signal - offset, color="green", label="Teacher-Forced Output", linewidth=linewidth
)
(line3,) = plt.plot(
    x_signal_auto_sample - offset,
    color="red",
    label="Autonomous Output",
    linewidth=linewidth,
)
line_auto_sample = mlines.Line2D(
    [], [], color="red", linestyle="--", label="Autonomous Output", linewidth=linewidth
)

# Combine all legend handles
handles = [line1, line2, line_auto_sample]
# plt.legend(handles=handles, loc="upper right")  # Single legend box
plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
# plt.title("Scheduled Sampling Learning for Lorenz63 System")
plt.xlabel("Time")
plt.ylabel("x variable of Lorenz63")


# save to this file directory
plt.savefig(os.path.join(file_dir, "Scheduled_Sampling_Learning.png"))
plt.show()


# Plot Opportunistic Teacher-Forced Learning
plt.figure(figsize=(10, 7.5))
offset = 20
(line1,) = plt.plot(
    x_signal_data_remain + offset,
    color="blue",
    label="True Signal",
    linewidth=linewidth,
)
(line2,) = plt.plot(
    x_signal_data_remain - offset,
    color="green",
    label="Teacher-Forced Output",
    linewidth=linewidth,
)
(line3,) = plt.plot(
    x_signal_data_lack - offset,
    color="red",
    label="Autonomous Output filling",
    linewidth=linewidth,
)
(line4,) = plt.plot(x_signal_auto_sample - offset, color="red", linewidth=linewidth)

# plt.title("Opportunistic Teacher-Forced Learning for Lorenz63 System")
plt.xlabel("Time")
plt.ylabel("x variable of Lorenz63")

mask_for_loss = mask_burst_inv * mask_bernoulli
mask_for_loss_inv = mask_for_loss * -1 + 1

shade_mask = mask_burst
# Add shade for masked regions
shade_mask_start = 0
for i in range(1, L):  # Iterate through the mask
    # if mask_burst[i] == 1 and mask_burst[i - 1] == 0:
    #     start = i
    # if mask_burst[i] == 0 and mask_burst[i - 1] == 1:
    #     end = i
    #     plt.axvspan(start, end, color="gray", alpha=0.3)  # Shade the region
    # yellow shade for loss mask
    if shade_mask[i] == 1 and shade_mask[i - 1] == 0:
        shade_mask_start = i
    if shade_mask[i] == 0 and shade_mask[i - 1] == 1:
        if shade_mask_start != 0:
            mask_loss_end = i
            plt.axvspan(shade_mask_start, mask_loss_end, color="gray", alpha=0.3)

# Create legend items
gray_patch = mpatches.Patch(color="gray", alpha=0.3, label="Missing Data")
line_auto_sample = mlines.Line2D(
    [], [], color="red", linestyle="--", label="Autonomous Output", linewidth=linewidth
)

# Combine all legend handles
handles = [line1, line2, line3, line_auto_sample, gray_patch]
# plt.legend(handles=handles, loc="upper right")  # Single legend box
# put legend outside the box
plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

plt.savefig(os.path.join(file_dir, "Opportunistic_Teacher_Forced_Learning.png"))
plt.show()
# save to this file directory

# Plot Generalized Teacher-Forced Learning

plt.figure(figsize=(10, 7.5))
offset = 20
(line1,) = plt.plot(
    x_signal + offset, color="blue", label="True Signal", linewidth=linewidth
)
(line2,) = plt.plot(
    x_signal - offset, color="brown", label="GTF Output", linewidth=linewidth
)


# Combine all legend handles
handles = [line1, line2, line_auto_sample]
# plt.legend(handles=handles, loc="upper right")  # Single legend box
plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))
# plt.title("Scheduled Sampling Learning for Lorenz63 System")
plt.xlabel("Time")
plt.ylabel("x variable of Lorenz63")


# save to this file directory
plt.savefig(os.path.join(file_dir, "Generalized_Teacher_Forced_Learning.png"))
plt.show()


# ================== END ==================

# %%
