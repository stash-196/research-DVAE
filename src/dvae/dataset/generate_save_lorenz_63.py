# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pickle
from fractions import Fraction

# %matplotlib inline
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm

import os
import numpy as np


from dvae.utils import find_project_root
from dvae.dataset.utils.visualizers import (
    plot_attractor_plotly,
    plot_attractor_subplots,
    plot_components_vs_time_plotly,
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


# %%
def human_format(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.0f}M"
    elif num >= 1_000:
        return f"{num/1_000:.0f}k"
    else:
        return str(num)


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
# %%
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
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.85,
    # 0.9,
    # 0.92,
    # 0.95,
    # 0.99,
    # 1.0,
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
    save_dir=save_dir_specific_plots,
)
plot_power_spectrum_plotly(
    np.array(l2.hist),
    sampling_rate,
    explain=parameter_str,
    save_dir=save_dir_specific_plots,
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
