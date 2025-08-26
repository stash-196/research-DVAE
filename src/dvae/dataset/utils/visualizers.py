# import textwrap
# from sklearn.decomposition import PCA
# from plotly.subplots import make_subplots
# import plotly.io as pio
# import numpy as np
# import plotly.graph_objects as go
# from scipy.signal import find_peaks
# import os


# def plot_attractor_plotly(hists, save_dir=None, explain=None, format="pdf"):
#     if np.array(hists).ndim == 2:
#         hists = [hists]
#     hists = [np.array(h) for h in hists]
#     fig = go.Figure()
#     for h in hists:
#         fig.add_trace(
#             go.Scatter3d(
#                 x=h[:, 0], y=h[:, 1], z=h[:, 2], mode="lines", line=dict(color="blue")
#             )
#         )
#     fig.update_layout(
#         scene=dict(
#             xaxis_title="x",
#             yaxis_title="y",
#             zaxis_title="z",
#         ),
#         title=textwrap.fill(
#             f"Attractor Plot for {explain.replace('_', ' ')} set",
#             width=50,
#             break_long_words=True,
#         ),
#     )
#     fig.show()
#     if save_dir is not None:
#         save_path = os.path.join(save_dir, f"attractor_{explain}.{format}")
#         pio.write_image(fig, save_path)


# def plot_attractor_subplots(hists, explain, save_dir=None, format="pdf"):
#     if np.array(hists).ndim == 2:
#         hists = [hists]
#     hists = [np.array(h) for h in hists]

#     # Create subplots: one row for each of x, y and z
#     fig = make_subplots(
#         rows=3,
#         cols=1,
#         shared_xaxes=True,
#         subplot_titles=("X Timeseries", "Y Timeseries", "Z Timeseries"),
#     )

#     for h in hists:
#         # X timeseries
#         fig.add_trace(
#             go.Scatter(y=h[:, 0], mode="lines", line=dict(color="blue")), row=1, col=1
#         )
#         # Y timeseries
#         fig.add_trace(
#             go.Scatter(y=h[:, 1], mode="lines", line=dict(color="red")), row=2, col=1
#         )
#         # Z timeseries
#         fig.add_trace(
#             go.Scatter(y=h[:, 2], mode="lines", line=dict(color="green")), row=3, col=1
#         )

#     fig.update_layout(
#         title_text="Timeseries Subplots for X, Y, and Z for {} set".format(
#             explain.replace("_", " ")
#         )
#     )
#     fig.show()
#     if save_dir is not None:
#         save_path = os.path.join(save_dir, f"timeseries_{explain}.{format}")
#         pio.write_image(fig, save_path)


# def plot_components_vs_time_plotly(
#     time_series, time_step, explain, save_dir=None, format="pdf"
# ):
#     t = np.arange(0, len(time_series) * time_step, time_step)
#     x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x", line=dict(color="blue")))
#     fig.add_trace(
#         go.Scatter(x=t, y=y, mode="lines", name="y", line=dict(color="green"))
#     )
#     fig.add_trace(go.Scatter(x=t, y=z, mode="lines", name="z", line=dict(color="red")))

#     fig.update_layout(
#         title=textwrap.fill(
#             f"Components of Lorenz63 System vs. Time for {explain.replace('_', ' ')} set",
#             width=50,
#             break_long_words=True,
#         ),
#         xaxis_title="Time",
#         yaxis_title="Values",
#         showlegend=True,
#         template="plotly_white",
#     )

#     fig.show()
#     if save_dir is not None:
#         save_path = os.path.join(save_dir, f"components_vs_time_{explain}.{format}")
#         pio.write_image(fig, save_path)


# # %%
# def calculate_power_spectrum(time_series, sampling_rate):
#     # Compute the Fast Fourier Transform (FFT)
#     fft_result = np.fft.fft(time_series)
#     freqs = np.fft.fftfreq(len(time_series), 1 / sampling_rate)
#     time_periods = np.zeros_like(freqs)
#     # Get periods corresponding to frequencies
#     time_periods[1:] = 1 / freqs[1:]

#     nonzero_indices = np.where(freqs > 0)

#     # Compute the power spectrum: the square of the absolute value of the FFT
#     power_spectrum = np.abs(fft_result[nonzero_indices]) ** 2
#     phases = np.angle(fft_result[nonzero_indices])

#     # Compute the frequencies corresponding to the values in the power spectrum
#     frequencies = np.fft.fftfreq(len(time_series))[nonzero_indices]

#     return time_periods, frequencies, power_spectrum


# def plot_power_spectrum_plotly(
#     time_series, sampling_rate, explain, save_dir=None, format="pdf"
# ):
#     # Create a figure
#     fig = go.Figure()

#     # Iterate over each component and plot its power spectrum
#     for i, component in enumerate(["x", "y", "z"]):
#         series = np.array(time_series)[:, i]
#         time_periods, frequencies, spectrum = calculate_power_spectrum(
#             series, sampling_rate
#         )

#         # Skip the zero frequency
#         time_periods, frequencies, spectrum = (
#             time_periods[1:],
#             frequencies[1:],
#             spectrum[1:],
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=time_periods,
#                 y=spectrum,
#                 mode="lines",
#                 name=f"{component} power spectrum",
#             )
#         )

#     # Set the x-axis to be log scale
#     fig.update_layout(
#         title="Power Spectrum of Lorenz63 System for {} set".format(explain),
#         xaxis=dict(type="log", title="Time Periods"),
#         # yaxis=dict(type="log", title="Power"),
#         yaxis=dict(title="Power"),
#         template="plotly_white",
#     )

#     # Display the figure
#     fig.show()

#     if save_dir is not None:
#         save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
#         pio.write_image(fig, save_path)


# def plot_power_spectrum_subplots_loglog(
#     time_series, sampling_rate, explain, component_labels, save_dir=None, format="pdf"
# ):
#     """
#     Plots the power spectrum of each component in the input time series data on a log-log scale,
#     including peak detection. It works for any input time series data, making it suitable for
#     both original data and PCA components.

#     :param time_series: Input time series data as a numpy array with shape (n_samples, n_components).
#     :param sampling_rate: The sampling rate of the time series data.
#     :param explain: Description or label for the dataset being analyzed.
#     :param component_labels: List of strings representing the labels for each component in the time series.
#     :param save_dir: Directory to save the plot image. If None, the plot is not saved.
#     :param format: The format for saving the plot image (e.g., 'pdf', 'png').
#     """
#     fig = make_subplots(
#         rows=len(component_labels),
#         cols=1,
#         subplot_titles=[f"{label} Power Spectrum" for label in component_labels],
#     )

#     for i, label in enumerate(component_labels):
#         series = time_series[:, i]
#         fft_result = np.fft.fft(series)
#         frequencies = np.fft.fftfreq(len(series), 1 / sampling_rate)
#         power_spectrum = np.abs(fft_result) ** 2

#         # Focus on the positive frequencies
#         positive_freqs = frequencies > 0
#         frequencies = frequencies[positive_freqs]
#         power_spectrum = power_spectrum[positive_freqs]

#         # Find peaks
#         peaks, properties = find_peaks(
#             power_spectrum, height=0.5, distance=100, prominence=0.05
#         )
#         peak_heights = properties["peak_heights"]

#         # Select the indices of the largest 3 peaks based on their height
#         # Get indices of the largest 3 peaks
#         largest_peaks_indices = np.argsort(peak_heights)[-3:]

#         # Get the frequency indices of the largest peaks
#         peaks = peaks[largest_peaks_indices]

#         # Plotting
#         fig.add_trace(
#             go.Scatter(
#                 x=frequencies,
#                 y=power_spectrum,
#                 mode="lines",
#                 name=f"{label} Power Spectrum",
#             ),
#             row=i + 1,
#             col=1,
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=frequencies[peaks],
#                 y=power_spectrum[peaks],
#                 mode="markers",
#                 marker=dict(color="red", size=8),
#                 name=f"{label} Peaks",
#             ),
#             row=i + 1,
#             col=1,
#         )

#     # Update the layout for log-log scale
#     fig.update_layout(
#         height=600,
#         width=800,
#         title_text=f"Power Spectrum for {explain} (Log-Log Scale)",
#     )
#     fig.update_xaxes(type="log", title="Frequency")
#     fig.update_yaxes(type="log", title="Power Spectrum")

#     fig.show()

#     if save_dir is not None and format:
#         save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
#         pio.write_image(fig, save_path)

#     return peaks, frequencies, power_spectrum


# def plot_delay_embedding(observation, delay, dimensions):
#     """
#     Plots the delay embedding of a 1D observation with lines.

#     :param observation: 1D array of observations.
#     :param delay: Time delay for embedding.
#     :param dimensions: Number of embedding dimensions.
#     """
#     n = len(observation)
#     embedding_length = n - (dimensions - 1) * delay
#     if embedding_length <= 0:
#         raise ValueError(
#             "Delay and dimensions are too large for the length of the observation array."
#         )

#     # Create the delay-embedded matrix
#     embedded = np.empty((embedding_length, dimensions))
#     for i in range(dimensions):
#         embedded[:, i] = observation[i * delay : i * delay + embedding_length]

#     # Plotting
#     if dimensions == 2:
#         fig = go.Figure(
#             data=go.Scatter(x=embedded[:, 0], y=embedded[:, 1], mode="lines")
#         )
#         fig.update_layout(
#             title="2D Delay Embedding", xaxis_title="X(t)", yaxis_title="X(t + delay)"
#         )
#     elif dimensions == 3:
#         fig = go.Figure(
#             data=go.Scatter3d(
#                 x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], mode="lines"
#             )
#         )
#         fig.update_layout(
#             title="3D Delay Embedding",
#             scene=dict(
#                 xaxis_title="X(t)",
#                 yaxis_title="X(t + delay)",
#                 zaxis_title="X(t + 2 * delay)",
#             ),
#         )
#     else:
#         raise NotImplementedError(
#             "Plotting for dimensions higher than 3 is not implemented."
#         )

#     fig.show()

import textwrap
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
import os


def plot_attractor_plotly(hists, save_dir=None, explain=None, format="pdf"):
    # Ensure hists is a list of NumPy arrays
    hists = np.asarray(hists)  # Convert input to NumPy array
    if hists.ndim == 2:
        hists = [hists]  # Wrap single trajectory in a list
    elif hists.ndim == 1:
        hists = [hists[:, np.newaxis]]  # Reshape 1D to (n_steps, 1)

    # Determine the dimension from the first history
    dim = hists[0].shape[1] if hists[0].ndim > 1 else 1
    if dim < 2:
        raise ValueError(
            "Attractor plots require at least 2D data; 1D data is not supported. "
            "Use plot_components_vs_time_plotly or plot_power_spectrum_plotly for 1D data."
        )

    fig = go.Figure()

    for h in hists:
        if dim == 2:
            fig.add_trace(
                go.Scatter(x=h[:, 0], y=h[:, 1], mode="lines", line=dict(color="blue"))
            )
            fig.update_layout(
                xaxis_title="dim1",
                yaxis_title="dim2",
                title=textwrap.fill(
                    f"Attractor Plot for {explain.replace('_', ' ')} set",
                    width=50,
                    break_long_words=True,
                ),
            )
        elif dim == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=h[:, 0],
                    y=h[:, 1],
                    z=h[:, 2],
                    mode="lines",
                    line=dict(color="blue"),
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="dim1",
                    yaxis_title="dim2",
                    zaxis_title="dim3",
                ),
                title=textwrap.fill(
                    f"Attractor Plot for {explain.replace('_', ' ')} set",
                    width=50,
                    break_long_words=True,
                ),
            )
        else:
            # For dim > 3, project to 3D using PCA
            pca = PCA(n_components=3)
            h_proj = pca.fit_transform(h)
            fig.add_trace(
                go.Scatter3d(
                    x=h_proj[:, 0],
                    y=h_proj[:, 1],
                    z=h_proj[:, 2],
                    mode="lines",
                    line=dict(color="blue"),
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3",
                ),
                title=textwrap.fill(
                    f"Projected Attractor Plot (PCA to 3D) for {explain.replace('_', ' ')} set",
                    width=50,
                    break_long_words=True,
                ),
            )

    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"attractor_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_attractor_subplots(hists, explain, save_dir=None, format="pdf"):
    # Ensure hists is a list of NumPy arrays
    hists = np.asarray(hists)  # Convert input to NumPy array
    if hists.ndim == 2:
        hists = [hists]  # Wrap single trajectory in a list
    elif hists.ndim == 1:
        hists = [hists[:, np.newaxis]]  # Reshape 1D to (n_steps, 1)

    # Determine the dimension from the first history
    dim = hists[0].shape[1] if hists[0].ndim > 1 else 1

    # Create subplots: one row for each dimension
    fig = make_subplots(
        rows=dim,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Dim {i+1} Timeseries" for i in range(dim)],
    )

    for h in hists:
        for i in range(dim):
            fig.add_trace(
                go.Scatter(
                    y=h[:, i],
                    mode="lines",
                    line=dict(color=f"rgb({i*50 % 255}, {i*100 % 255}, {i*150 % 255})"),
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        title_text="Timeseries Subplots for All Dimensions for {} set".format(
            explain.replace("_", " ")
        ),
        height=200 * dim,  # Adjust height based on number of dimensions
    )
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"timeseries_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_components_vs_time_plotly(
    time_series, time_step, explain, save_dir=None, format="pdf"
):
    # Convert input to NumPy array and handle 1D/2D cases
    time_series = np.asarray(time_series)
    if time_series.ndim == 1:
        time_series = time_series[:, np.newaxis]  # Reshape 1D to (n_steps, 1)

    # Determine the dimension
    dim = time_series.shape[1]
    t = np.arange(0, len(time_series) * time_step, time_step)

    fig = go.Figure()

    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]
    for i in range(dim):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=time_series[:, i],
                mode="lines",
                name=f"dim{i+1}",
                line=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title=textwrap.fill(
            f"Components of System vs. Time for {explain.replace('_', ' ')} set",
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
    # Convert input to NumPy array and handle 1D/2D cases
    time_series = np.asarray(time_series)
    if time_series.ndim == 1:
        time_series = time_series[:, np.newaxis]  # Reshape 1D to (n_steps, 1)

    # Determine the dimension
    dim = time_series.shape[1]

    # Create a figure
    fig = go.Figure()

    # Iterate over each component and plot its power spectrum
    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]
    for i in range(dim):
        series = time_series[:, i]
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
                name=f"dim{i+1} power spectrum",
                line=dict(color=colors[i % len(colors)]),
            )
        )

    # Set the x-axis to be log scale
    fig.update_layout(
        title="Power Spectrum of System for {} set".format(explain),
        xaxis=dict(type="log", title="Time Periods"),
        yaxis=dict(title="Power"),
        template="plotly_white",
    )

    # Display the figure
    fig.show()

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_power_spectrum_subplots_loglog(
    time_series,
    sampling_rate,
    explain,
    component_labels=None,
    save_dir=None,
    format="pdf",
):
    """
    Plots the power spectrum of each component in the input time series data on a log-log scale,
    including peak detection. It works for any input time series data, making it suitable for
    both original data and PCA components.

    :param time_series: Input time series data as a numpy array with shape (n_samples, n_components).
    :param sampling_rate: The sampling rate of the time series data.
    :param explain: Description or label for the dataset being analyzed.
    :param component_labels: List of strings representing the labels for each component in the time series.
                             If None, defaults to ['dim1', 'dim2', ..., 'dimN'].
    :param save_dir: Directory to save the plot image. If None, the plot is not saved.
    :param format: The format for saving the plot image (e.g., 'pdf', 'png').
    """
    # Convert input to NumPy array and handle 1D/2D cases
    time_series = np.asarray(time_series)
    if time_series.ndim == 1:
        time_series = time_series[:, np.newaxis]  # Reshape 1D to (n_steps, 1)

    # Determine the dimension
    dim = time_series.shape[1]

    if component_labels is None:
        component_labels = [f"dim{i+1}" for i in range(dim)]
    elif len(component_labels) != dim:
        raise ValueError(
            "Number of component_labels must match the number of dimensions."
        )

    fig = make_subplots(
        rows=dim,
        cols=1,
        subplot_titles=[f"{label} Power Spectrum" for label in component_labels],
    )

    peaks_list = []
    frequencies_list = []
    power_spectrum_list = []

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

        peaks_list.append(peaks)
        frequencies_list.append(frequencies)
        power_spectrum_list.append(power_spectrum)

    # Update the layout for log-log scale
    fig.update_layout(
        height=200 * dim,  # Adjust height based on number of dims
        width=800,
        title_text=f"Power Spectrum for {explain} (Log-Log Scale)",
    )
    fig.update_xaxes(type="log", title="Frequency")
    fig.update_yaxes(type="log", title="Power Spectrum")

    fig.show()

    if save_dir is not None and format:
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)

    return peaks_list, frequencies_list, power_spectrum_list


def plot_delay_embedding(observation, delay=1, dimensions=3):
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
