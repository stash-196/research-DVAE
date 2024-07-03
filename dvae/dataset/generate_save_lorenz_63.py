# %%
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.io as pio
import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
import os
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go

# Handy function stolen from the fast.ai library


def V(x, requires_grad=False, gpu=False):
    t = torch.FloatTensor(np.atleast_1d(x).astype(np.float32))
    if gpu:
        t = t.cuda()
    return Variable(t, requires_grad=requires_grad)


class L63():
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


def plot_attractor_plotly(hists, save_dir=None, explain=None, format='pdf'):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = go.Figure()
    for h in hists:
        fig.add_trace(go.Scatter3d(
            x=h[:, 0], y=h[:, 1], z=h[:, 2], mode='lines', line=dict(color='blue')))
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
    ), title=f'Attractor Plot for {explain} set')
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'attractor_{explain}.{format}')
        pio.write_image(fig, save_path)


def plot_attractor_subplots(hists, explain, save_dir=None, format='pdf'):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]

    # Create subplots: one row for each of x, y and z
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('X Timeseries', 'Y Timeseries', 'Z Timeseries'))

    for h in hists:
        # X timeseries
        fig.add_trace(go.Scatter(
            y=h[:, 0], mode='lines', line=dict(color='blue')), row=1, col=1)
        # Y timeseries
        fig.add_trace(go.Scatter(
            y=h[:, 1], mode='lines', line=dict(color='red')), row=2, col=1)
        # Z timeseries
        fig.add_trace(go.Scatter(
            y=h[:, 2], mode='lines', line=dict(color='green')), row=3, col=1)

    fig.update_layout(
        title_text="Timeseries Subplots for X, Y, and Z for {} set".format(explain))
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'timeseries_{explain}.{format}')
        pio.write_image(fig, save_path)


def plot_components_vs_time_plotly(time_series, time_step, explain, save_dir=None, format='pdf'):
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=x, mode='lines',
                  name='x', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines',
                  name='y', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=t, y=z, mode='lines',
                  name='z', line=dict(color='red')))

    fig.update_layout(title='Components of Lorenz63 System vs. Time for {} set'.format(explain),
                      xaxis_title='Time',
                      yaxis_title='Values',
                      showlegend=True,
                      template='plotly_white')

    fig.show()
    if save_dir is not None:
        save_path = os.path.join(
            save_dir, f'components_vs_time_{explain}.{format}')
        pio.write_image(fig, save_path)


# %%


def calculate_power_spectrum(time_series, sampling_rate):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series), 1/sampling_rate)
    time_periods = np.zeros_like(freqs)
    # Get periods corresponding to frequencies
    time_periods[1:] = 1 / freqs[1:]

    nonzero_indices = np.where(freqs > 0)

    # Compute the power spectrum: the square of the absolute value of the FFT
    power_spectrum = np.abs(fft_result[nonzero_indices])**2
    phases = np.angle(fft_result[nonzero_indices])

    # Compute the frequencies corresponding to the values in the power spectrum
    frequencies = np.fft.fftfreq(len(time_series))[nonzero_indices]

    return time_periods, frequencies, power_spectrum


def plot_power_spectrum_plotly(time_series, sampling_rate, explain, save_dir=None, format='pdf'):
    # Create a figure
    fig = go.Figure()

    # Iterate over each component and plot its power spectrum
    for i, component in enumerate(["x", "y", "z"]):
        series = np.array(time_series)[:, i]
        time_periods, frequencies, spectrum = calculate_power_spectrum(
            series, sampling_rate)

        # Skip the zero frequency
        time_periods, frequencies, spectrum = time_periods[1:], frequencies[1:], spectrum[1:]

        fig.add_trace(go.Scatter(x=time_periods, y=spectrum,
                      mode='lines', name=f'{component} power spectrum'))

    # Set the x-axis to be log scale
    fig.update_layout(
        title="Power Spectrum of Lorenz63 System for {} set".format(explain),
        xaxis=dict(type="log", title="Time Periods"),
        # yaxis=dict(type="log", title="Power"),
        yaxis=dict(title="Power"),
        template="plotly_white"
    )

    # Display the figure
    fig.show()

    if save_dir is not None:
        save_path = os.path.join(
            save_dir, f'power_spectrum_{explain}.{format}')
        pio.write_image(fig, save_path)


def plot_power_spectrum_subplots_loglog(time_series, sampling_rate, explain, component_labels, save_dir=None, format='pdf'):
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
    fig = make_subplots(rows=len(component_labels), cols=1, subplot_titles=[
                        f'{label} Power Spectrum' for label in component_labels])

    for i, label in enumerate(component_labels):
        series = time_series[:, i]
        fft_result = np.fft.fft(series)
        frequencies = np.fft.fftfreq(len(series), 1/sampling_rate)
        power_spectrum = np.abs(fft_result)**2

        # Focus on the positive frequencies
        positive_freqs = frequencies > 0
        frequencies = frequencies[positive_freqs]
        power_spectrum = power_spectrum[positive_freqs]

        # Find peaks
        peaks, properties = find_peaks(
            power_spectrum, height=0.5, distance=100, prominence=0.05)
        peak_heights = properties["peak_heights"]

        # Select the indices of the largest 3 peaks based on their height
        # Get indices of the largest 3 peaks
        largest_peaks_indices = np.argsort(peak_heights)[-3:]

        # Get the frequency indices of the largest peaks
        peaks = peaks[largest_peaks_indices]

        # Plotting
        fig.add_trace(go.Scatter(x=frequencies, y=power_spectrum,
                      mode='lines', name=f'{label} Power Spectrum'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=frequencies[peaks], y=power_spectrum[peaks], mode='markers', marker=dict(
            color='red', size=8), name=f'{label} Peaks'), row=i+1, col=1)

    # Update the layout for log-log scale
    fig.update_layout(height=600, width=800,
                      title_text=f"Power Spectrum for {explain} (Log-Log Scale)")
    fig.update_xaxes(type="log", title="Frequency")
    fig.update_yaxes(type="log", title="Power Spectrum")

    fig.show()

    if save_dir is not None and format:
        save_path = os.path.join(
            save_dir, f'power_spectrum_{explain}.{format}')
        pio.write_image(fig, save_path)

    return peaks, frequencies, power_spectrum


# %%
# Define the default parameters values
sigma = 10
rho = 28
beta = 8/3
N = 15*60*24*5
dt = 1e-2
l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=1e-2)
l2 = L63(sigma, rho, beta, init=[10, 1, 2], dt=1e-2)


l1.integrate(N)
l2.integrate(int(N*0.1))

plot_attractor_plotly(
    [l1.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_train')
plot_attractor_plotly(
    [l2.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_test')

plot_attractor_subplots(
    [l1.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_train')
plot_attractor_subplots(
    [l2.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_test')

plot_components_vs_time_plotly(np.array(
    l1.hist), time_step=1e-2, explain='s10_r28_b8d3_train', save_dir='temp_save/lorenz63')
plot_components_vs_time_plotly(np.array(
    l2.hist), time_step=1e-2, explain='s10_r28_b8d3_test', save_dir='temp_save/lorenz63')


# store l.hist as pickle data for later use in pytorch dataloader
def save_pickle(data, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


save_pickle(l1.hist, 'temp_save/lorenz63/dataset_train.pkl')
save_pickle(l2.hist, 'temp_save/lorenz63/dataset_test.pkl')

# calculate sampling rate
sampling_rate = 1 / dt

# Call the function to display the plot
plot_power_spectrum_plotly(np.array(l1.hist), sampling_rate,
                           explain='s10_r28_b8d3_train', save_dir='temp_save/lorenz63')
plot_power_spectrum_plotly(np.array(l2.hist), sampling_rate,
                           explain='s10_r28_b8d3_test', save_dir='temp_save/lorenz63')

# Call the function to display the plots
# %%
# Example usage for original XYZ data
component_labels_xyz = ["X", "Y", "Z"]
peaks_train, frequenciest_train, power_spectrum_train = plot_power_spectrum_subplots_loglog(np.array(
    l1.hist), sampling_rate, 's10_r28_b8d3_train_xyz', component_labels_xyz, save_dir='temp_save/lorenz63')
peaks_test, frequencies_test, power_spectrum_test = plot_power_spectrum_subplots_loglog(np.array(
    l2.hist), sampling_rate, 's10_r28_b8d3_train_xyz', component_labels_xyz, save_dir='temp_save/lorenz63')

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
            "Delay and dimensions are too large for the length of the observation array.")

    # Create the delay-embedded matrix
    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = observation[i * delay: i * delay + embedding_length]

    # Plotting
    if dimensions == 2:
        fig = go.Figure(data=go.Scatter(
            x=embedded[:, 0], y=embedded[:, 1], mode='lines'))
        fig.update_layout(title='2D Delay Embedding',
                          xaxis_title='X(t)', yaxis_title='X(t + delay)')
    elif dimensions == 3:
        fig = go.Figure(data=go.Scatter3d(
            x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], mode='lines'))
        fig.update_layout(title='3D Delay Embedding', scene=dict(
            xaxis_title='X(t)', yaxis_title='X(t + delay)', zaxis_title='X(t + 2 * delay)'))
    else:
        raise NotImplementedError(
            "Plotting for dimensions higher than 3 is not implemented.")

    fig.show()


# Example usage
# Assuming you have an array `x` from the Lorenz system:
x = np.array(l1.hist)[:, 0]
plot_delay_embedding(x, delay=20, dimensions=3)

# %%
