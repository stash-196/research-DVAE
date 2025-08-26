# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pickle
from fractions import Fraction
import os

# %matplotlib inline
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm

from dvae.utils import (
    find_project_root,
    num2str,
    phase2str,
    lst2str,
    phaselst2str,
)

from dvae.dataset.utils.visualizers import (
    plot_attractor_plotly,
    plot_attractor_subplots,
    plot_components_vs_time_plotly,
    plot_power_spectrum_plotly,
    plot_power_spectrum_subplots_loglog,
    plot_delay_embedding,
)
from dvae.dataset.utils.data_utils import save_pickle
from dvae.dataset.utils.masking_funcs import (
    generate_markovian_burst_mask,
    generate_fixed_burst_mask,
)

project_root = find_project_root(__file__)
file_dir = os.path.dirname(os.path.realpath(__file__))


class DampedSHO:
    def __init__(self, omegas, gammas, n_instances, dt, init_range=(-1, 1)):
        """
        Initialize a mixture of damped harmonic oscillators with multiple instances.

        Parameters:
        - omegas: List of natural frequencies (rad/s).
        - gammas: List of damping coefficients (1/s), same length as omegas.
        - n_instances: Number of independent trajectories (summed oscillators).
        - dt: Time step for integration.
        - init_range: Tuple (min, max) for random initial conditions [x0, v0] per oscillator.
        """
        if len(omegas) != len(gammas):
            raise ValueError("omegas and gammas must have the same length")
        self.omegas = np.array(omegas)
        self.gammas = np.array(gammas)
        self.n_oscillators = len(omegas)
        self.n_instances = n_instances
        self.dt = dt
        # Generate random initial conditions: (n_instances, n_oscillators, 2) for [x0, v0]
        self.init_conditions = np.random.uniform(
            init_range[0], init_range[1], (n_instances, self.n_oscillators, 2)
        )
        self.hist = None  # Will store (n_steps, n_instances) summed position data

    def step(self, x, omega, gamma):
        """Euler step for dx/dt = [x2, -omega^2 * x1 - 2*gamma * x2]."""
        x1, x2 = x
        dx1 = x2
        dx2 = -(omega**2) * x1 - 2 * gamma * x2
        return x + np.array([dx1, dx2]) * self.dt

    def integrate(self, n_steps):
        """Integrate n_instances trajectories, summing positions of n_oscillators."""
        trajectories = []
        for instance_idx in range(self.n_instances):
            # Initialize state for all oscillators in this instance
            states = self.init_conditions[
                instance_idx
            ].copy()  # Shape: (n_oscillators, 2)
            trajectory = [np.sum(states[:, 0])]  # Sum initial positions (x1)
            for _ in range(n_steps - 1):
                # Update each oscillator
                for osc_idx in range(self.n_oscillators):
                    states[osc_idx] = self.step(
                        states[osc_idx], self.omegas[osc_idx], self.gammas[osc_idx]
                    )
                trajectory.append(np.sum(states[:, 0]))  # Sum positions (x1)
            trajectories.append(trajectory)
        self.hist = np.array(trajectories).T  # Shape: (n_steps, n_instances)


# %%
# Define parameters
omegas = [2 * np.pi * 1, 2 * np.pi * 0.5]  # Natural frequencies (1 Hz, 0.5 Hz)
gammas = [0.5, 0.2]  # Damping coefficients (tau = 2 s, 5 s)
n_instances = 80  # Number of trajectories
dt = 0.01  # Time step
N = 1000  # Steps per trajectory

parameter_str = f"omegas{phaselst2str(omegas)}_gammas{lst2str(gammas)}_inst{num2str(n_instances)}_N{num2str(N)}_dt{num2str(dt)}"
save_dir_data = os.path.join(project_root, "data/damped_sho", "data", parameter_str)
save_dir_plots = os.path.join(project_root, "data/damped_sho", "plots", parameter_str)

if not os.path.exists(save_dir_data):
    os.makedirs(save_dir_data)
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)
# %%
# Generate dataset
data1 = DampedSHO(omegas=omegas, gammas=gammas, n_instances=n_instances, dt=dt)
data1.integrate(N)  # Shape: (1000, 80)

# Save dataset
save_pickle(data1.hist, os.path.join(save_dir_data, "complete_dataset_train.pkl"))

# %%
# Visualize (select first 3 instances for plotting to avoid clutter)
plot_subset = data1.hist[:, :3]  # Shape: (1000, 3)

plot_components_vs_time_plotly(
    plot_subset,
    time_step=dt,
    explain=f"{parameter_str}_train_subset",
    save_dir=save_dir_plots,
)

sampling_rate = 1 / dt
plot_power_spectrum_plotly(
    plot_subset,
    sampling_rate=sampling_rate,
    explain=f"{parameter_str}_train_subset",
    save_dir=save_dir_plots,
)

plot_power_spectrum_subplots_loglog(
    plot_subset,
    sampling_rate=sampling_rate,
    explain=f"{parameter_str}_train_subset",
    save_dir=save_dir_plots,
)

# Delay embedding for first instance (1D input)
plot_delay_embedding(
    data1.hist[:, 0],  # First trajectory
    delay=5,
    dimensions=3,
)


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved data to {path}")


# %%
