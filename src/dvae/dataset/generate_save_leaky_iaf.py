# %%
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go
from scipy.signal import find_peaks
from collections import defaultdict
import os
import pickle


# Define the Integrate-and-Fire neuron class
class IntegrateAndFireNeuron:
    def __init__(self, tau, threshold, reset_value, init_potential, dt):
        """
        Initializes an integrate-and-fire neuron model.

        :param tau: Time constant (ms).
        :param threshold: Firing threshold potential.
        :param reset_value: Reset potential after firing.
        :param init_potential: Initial membrane potential.
        :param dt: Time step size.
        """
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.membrane_potential = init_potential
        self.dt = dt
        self.hist = [init_potential]

    def step(self, input_current):
        """
        Performs one time step of the leaky integrate-and-fire model.

        :param input_current: Input current at this time step.
        """
        # Update membrane potential with a leakage term
        self.membrane_potential += (input_current - self.membrane_potential) * (
            self.dt / self.tau
        )
        self.membrane_potential -= self.membrane_potential * (
            self.dt / self.tau
        )  # Leak term

        # Check if the neuron fires
        if self.membrane_potential >= self.threshold:
            self.hist.append(self.threshold)  # Record the spike
            self.membrane_potential = self.reset_value  # Reset the potential
        else:
            self.hist.append(self.membrane_potential)

    def integrate(self, input_current_series):
        """
        Integrates the neuron model over a series of input currents.

        :param input_current_series: Array of input currents over time.
        """
        for input_current in input_current_series:
            self.step(input_current)


# Plotting functions


def plot_input_and_potential_vs_time(
    input_current_series,
    potential_series,
    time_step,
    explain,
    save_dir=None,
    format="pdf",
):
    """
    Plots the input current and membrane potential of the neuron over time using Plotly.

    :param input_current_series: List or array of input currents.
    :param potential_series: List or array of membrane potentials.
    :param time_step: Time step size in milliseconds.
    :param explain: Description or label for the plot.
    :param save_dir: Directory to save the plot image. If None, the plot is not saved.
    :param format: The format for saving the plot image (e.g., 'pdf', 'png').
    """
    t = np.arange(0, len(potential_series) * time_step, time_step)

    fig = go.Figure()

    # Plot input current
    fig.add_trace(
        go.Scatter(
            x=t, y=input_current_series[: len(t)], mode="lines", name="Input Current"
        )
    )

    # Plot membrane potential
    fig.add_trace(
        go.Scatter(x=t, y=potential_series, mode="lines", name="Membrane Potential")
    )

    fig.update_layout(
        title="Input Current and Neuron Membrane Potential vs. Time for {}".format(
            explain
        ),
        xaxis_title="Time (ms)",
        yaxis_title="Value",
        showlegend=True,
        template="plotly_white",
    )
    fig.show()

    if save_dir is not None:
        save_path = os.path.join(
            save_dir, f"input_and_potential_vs_time_{explain}.{format}"
        )
        pio.write_image(fig, save_path)


def calculate_power_spectrum(time_series, sampling_rate):
    fft_result = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(len(time_series), 1 / sampling_rate)
    nonzero_indices = np.where(freqs > 0)

    power_spectrum = np.abs(fft_result[nonzero_indices]) ** 2
    frequencies = freqs[nonzero_indices]

    return frequencies, power_spectrum


def plot_power_spectrum_plotly(
    time_series, sampling_rate, explain, save_dir=None, format="pdf"
):
    frequencies, spectrum = calculate_power_spectrum(time_series, sampling_rate)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frequencies, y=spectrum, mode="lines", name="Power Spectrum")
    )
    fig.update_layout(
        title="Power Spectrum of Integrate-and-Fire Neuron for {}".format(explain),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        showlegend=True,
        template="plotly_white",
    )
    fig.show()

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"power_spectrum_{explain}.{format}")
        pio.write_image(fig, save_path)


def plot_neuron_delay_embedding(
    potential_series, delay, dimensions, explain, save_dir=None, format="pdf"
):
    """
    Plots the delay embedding of a neuron's membrane potential.

    :param potential_series: 1D array of membrane potentials.
    :param delay: Time delay for embedding.
    :param dimensions: Number of embedding dimensions.
    :param explain: Description or label for the dataset being analyzed.
    :param save_dir: Directory to save the plot image. If None, the plot is not saved.
    :param format: The format for saving the plot image (e.g., 'pdf', 'png').
    """
    n = len(potential_series)
    embedding_length = n - (dimensions - 1) * delay
    if embedding_length <= 0:
        raise ValueError(
            "Delay and dimensions are too large for the length of the observation array."
        )

    # Create the delay-embedded matrix
    embedded = np.empty((embedding_length, dimensions))
    for i in range(dimensions):
        embedded[:, i] = potential_series[i * delay : i * delay + embedding_length]

    # Plotting
    if dimensions == 2:
        fig = go.Figure(
            data=go.Scatter(x=embedded[:, 0], y=embedded[:, 1], mode="lines")
        )
        fig.update_layout(
            title="2D Delay Embedding for {}".format(explain),
            xaxis_title="X(t)",
            yaxis_title="X(t + delay)",
        )
    elif dimensions == 3:
        fig = go.Figure(
            data=go.Scatter3d(
                x=embedded[:, 0], y=embedded[:, 1], z=embedded[:, 2], mode="lines"
            )
        )
        fig.update_layout(
            title="3D Delay Embedding for {}".format(explain),
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

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"neuron_delay_embedding_{explain}.{format}")
        pio.write_image(fig, save_path)


# %%
# %%
# %%
import numpy as np
import os

# Define default parameters for the integrate-and-fire neuron
tau = 20.0  # Time constant (ms)
threshold = 1.0  # Firing threshold potential
reset_value = 0.0  # Reset potential after firing
init_potential = 0.0  # Initial membrane potential
f = 1000  # Frequency of the input current (Hz)
dt = 1000 / f  # Time step size (ms)
duration = 1000 * 60  # Total duration of the simulation (ms)

# Parameters for the noisy input current
mean_current = 2  # Mean input current
noise_std = 0.1  # Standard deviation of the noise

# Generate input current series for training with Gaussian noise
np.random.seed(42)  # Seed for reproducibility
input_current_series_train = mean_current + noise_std * np.random.normal(
    size=int(duration / dt)
)

# Generate input current series for testing with Gaussian noise
np.random.seed(24)  # Different seed for testing data
input_current_series_test = mean_current + noise_std * np.random.normal(
    size=int(duration / dt * 0.1)
)

# Create an instance of the IntegrateAndFireNeuron
neuron_train = IntegrateAndFireNeuron(tau, threshold, reset_value, init_potential, dt)
neuron_test = IntegrateAndFireNeuron(tau, threshold, reset_value, init_potential, dt)

save_dir = "temp_save/leaky_iaf"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Integrate the neuron model over time for training data
neuron_train.integrate(input_current_series_train)

# Integrate the neuron model over time for testing data
neuron_test.integrate(input_current_series_test)

# Truncate the output to match the input length
truncated_membrane_potential_train = np.array(neuron_train.hist[:len(input_current_series_train)])
truncated_membrane_potential_test = np.array(neuron_test.hist[:len(input_current_series_test)])

# Stack input current and membrane potential together for training data
combined_data_train = np.column_stack((input_current_series_train, truncated_membrane_potential_train))

# Stack input current and membrane potential together for testing data
combined_data_test = np.column_stack((input_current_series_test, truncated_membrane_potential_test))

# Save the combined data for training
np.save(os.path.join(save_dir, "dataset_train.npy"), combined_data_train)

# Save the combined data for testing
np.save(os.path.join(save_dir, "dataset_test.npy"), combined_data_test)

# Plot the neuron's membrane potential over time for training data
plot_input_and_potential_vs_time(
    input_current_series_train, truncated_membrane_potential_train, dt, explain="noisy_input_train", save_dir=save_dir
)

# Plot the neuron's membrane potential over time for testing data
plot_input_and_potential_vs_time(
    input_current_series_test, truncated_membrane_potential_test, dt, explain="noisy_input_test", save_dir=save_dir
)

# Calculate the sampling rate for the power spectrum
sampling_rate = 1 / dt

# Plot the power spectrum of the neuron's membrane potential for training data
plot_power_spectrum_plotly(
    truncated_membrane_potential_train, sampling_rate, explain="noisy_input_train", save_dir=save_dir
)

# Plot the power spectrum of the neuron's membrane potential for testing data
plot_power_spectrum_plotly(
    truncated_membrane_potential_test, sampling_rate, explain="noisy_input_test", save_dir=save_dir
)

# Plot delay embedding for training data
plot_neuron_delay_embedding(
    np.array(truncated_membrane_potential_train[:10000]),
    delay=10,
    dimensions=3,
    explain="noisy_input_train",
    save_dir=save_dir,
)

# Plot delay embedding for testing data
plot_neuron_delay_embedding(
    np.array(truncated_membrane_potential_test[:10000]),
    delay=10,
    dimensions=3,
    explain="noisy_input_test",
    save_dir=save_dir,
)

# %%
