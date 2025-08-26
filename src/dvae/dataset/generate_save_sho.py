# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pickle

# %matplotlib inline
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from dvae.dataset.utils.data_utils import save_pickle

import os
import numpy as np

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
    plot_delay_embedding,
    plot_power_spectrum_plotly,
    plot_power_spectrum_subplots_loglog,
)
from dvae.dataset.utils.masking_funcs import (
    generate_markovian_burst_mask,
    generate_fixed_burst_mask,
)

project_root = find_project_root(__file__)

# Handy function stolen from the fast.ai library
# Get location of current files directory
file_dir = os.path.dirname(os.path.realpath(__file__))


class MixedSHO:
    def __init__(self, amplitudes, frequencies, phases, dt):
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phases = phases
        self.dt = dt
        self.t = 0.0
        self.hist = [self._compute()]

    def _compute(self):
        return sum(
            A * np.sin(2 * np.pi * f * self.t + phi)
            for A, f, phi in zip(self.amplitudes, self.frequencies, self.phases)
        )

    def step(self):
        self.t += self.dt
        self.hist.append(self._compute())

    def integrate(self, n_steps):
        for _ in range(n_steps):
            self.step()


# %%
# Define the default parameters values

save_dir_plots = os.path.join(project_root, "data/sho", "plots")
save_dir_data = os.path.join(project_root, "data/sho", "data")
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)
if not os.path.exists(save_dir_data):
    os.makedirs(save_dir_data)

# %%
# ================== For Dataset Generation ==================
# Parameters for two sinusoids
amplitudes = [1, 2, 0.5]
frequencies = [1, 0.5, 20]
phases = [0, np.pi / 2, 0]  # Starting phases

dt = 0.01  # Time step for the generation
N = 10000  # Number of time steps to generate


parameter_str = f"amp{lst2str(amplitudes)}_freq{lst2str(frequencies)}_phas{phaselst2str(phases)}_N{num2str(N)}_dt{dt}"
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


data1 = MixedSHO(amplitudes, frequencies, phases, dt=dt)
data2 = MixedSHO(amplitudes, frequencies, phases, dt=dt)  # Same params, but could vary
data1.integrate(int(N * 0.9))
data2.integrate(int(N * 0.1))

# %%


plot_attractor_subplots(
    np.array(data1.hist),
    save_dir=save_dir_specific_plots,
    explain=f"{parameter_str}_train",
)
plot_attractor_subplots(
    np.array(data2.hist),
    save_dir=save_dir_specific_plots,
    explain=f"{parameter_str}_test",
)

plot_components_vs_time_plotly(
    np.array(data1.hist),
    time_step=dt,
    explain=f"{parameter_str}_train",
    save_dir=save_dir_specific_plots,
)
plot_components_vs_time_plotly(
    np.array(data2.hist),
    time_step=dt,
    explain=f"{parameter_str}_test",
    save_dir=save_dir_specific_plots,
)

sampling_rate = 1 / dt
plot_power_spectrum_plotly(
    np.array(data1.hist),
    sampling_rate,
    explain=f"{parameter_str}_train",
    save_dir=save_dir_specific_plots,
)

plot_power_spectrum_plotly(
    np.array(data2.hist),
    sampling_rate,
    explain=f"{parameter_str}_test",
    save_dir=save_dir_specific_plots,
)

plot_delay_embedding(
    np.array(data1.hist),
    delay=5,
    dimensions=3,
    # explain=f"{parameter_str}_train",
    # save_dir=save_dir_specific_plots,
)

plot_delay_embedding(
    np.array(data2.hist),
    delay=5,
    dimensions=3,
    # explain=f"{parameter_str}_test",
    # save_dir=save_dir_specific_plots,
)


save_pickle(
    data1.hist, os.path.join(save_dir_specific_data, "complete_dataset_train.pkl")
)
save_pickle(
    data2.hist, os.path.join(save_dir_specific_data, "complete_dataset_test.pkl")
)
# %%
