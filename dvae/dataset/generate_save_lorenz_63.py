# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
import os

# Handy function stolen from the fast.ai library
def V(x, requires_grad=False, gpu=False):
    t = torch.FloatTensor(np.atleast_1d(x).astype(np.float32))
    if gpu: t = t.cuda()
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
        for n in range(n_steps): self.step()

import plotly.graph_objects as go
import plotly.io as pio

def plot_attractor_plotly(hists, save_dir=None, explain=None, format='pdf'):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = go.Figure()
    for h in hists:
        fig.add_trace(go.Scatter3d(x=h[:, 0], y=h[:, 1], z=h[:, 2], mode='lines', line=dict(color='blue')))
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z', 
    ), title=f'Attractor Plot for {explain} set')
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'attractor_{explain}.{format}')
        pio.write_image(fig, save_path)

    


from plotly.subplots import make_subplots

def plot_attractor_subplots(hists, explain, save_dir=None, format='pdf'):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]

    # Create subplots: one row for each of x, y and z
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('X Timeseries', 'Y Timeseries', 'Z Timeseries'))

    for h in hists:
        # X timeseries
        fig.add_trace(go.Scatter(y=h[:, 0], mode='lines', line=dict(color='blue')), row=1, col=1)
        # Y timeseries
        fig.add_trace(go.Scatter(y=h[:, 1], mode='lines', line=dict(color='red')), row=2, col=1)
        # Z timeseries
        fig.add_trace(go.Scatter(y=h[:, 2], mode='lines', line=dict(color='green')), row=3, col=1)

    fig.update_layout(title_text="Timeseries Subplots for X, Y, and Z for {} set".format(explain))
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'timeseries_{explain}.{format}')
        pio.write_image(fig, save_path)


import plotly.graph_objects as go

def plot_components_vs_time_plotly(time_series, time_step, explain, save_dir=None, format='pdf'):
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=t, y=z, mode='lines', name='z', line=dict(color='red')))
    
    fig.update_layout(title='Components of Lorenz63 System vs. Time for {} set'.format(explain),
                      xaxis_title='Time',
                      yaxis_title='Values',
                      showlegend=True,
                      template='plotly_white')
    
    fig.show()
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'components_vs_time_{explain}.{format}')
        pio.write_image(fig, save_path)


#%%
import numpy as np

def power_spectrum(time_series):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(time_series)
    
    # Compute the power spectrum: the square of the absolute value of the FFT
    power_spectrum = np.abs(fft_result)**2
    phases = np.angle(fft_result)    

    n_pos_freq = len(fft_result)//2
    # Since the power spectrum is symmetric, we only need to return the first half
    power_spectrum = power_spectrum[:n_pos_freq]
    phases = phases[:n_pos_freq]
    
    # Compute the frequencies corresponding to the values in the power spectrum
    frequencies = np.fft.fftfreq(len(time_series))[:n_pos_freq]

    time_periods = np.zeros_like(frequencies)
    time_periods[1:] = 1 / frequencies[1:]
    
    return time_periods, frequencies, power_spectrum


import plotly.graph_objects as go

def plot_power_spectrum_plotly(time_series, explain, save_dir=None, format='pdf'):
    # Create a figure
    fig = go.Figure()

    # Iterate over each component and plot its power spectrum
    for i, component in enumerate(["x", "y", "z"]):
        series = np.array(time_series)[:, i]
        time_periods, frequencies, spectrum = power_spectrum(series)
        
        # Skip the zero frequency
        time_periods, frequencies, spectrum = time_periods[1:], frequencies[1:], spectrum[1:]
        
        fig.add_trace(go.Scatter(x=time_periods, y=spectrum, mode='lines', name=f'{component} power spectrum'))

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
        save_path = os.path.join(save_dir, f'power_spectrum_{explain}.{format}')
        pio.write_image(fig, save_path)

from plotly.subplots import make_subplots

def plot_power_spectrum_subplots_loglog(time_series, explain, save_dir=None, format='pdf'):
    # Create a subplots figure with 3 rows and 1 column
    fig = make_subplots(rows=3, cols=1, subplot_titles=('X Power Spectrum', 'Y Power Spectrum', 'Z Power Spectrum'))
    
    # Iterate over each component and plot its power spectrum in a separate subplot
    for i, component in enumerate(["x", "y", "z"]):
        series = np.array(time_series)[:, i]
        time_periods, frequencies, spectrum = power_spectrum(series)
        
        # Skip the zero frequency
        time_periods, frequencies, spectrum = time_periods[1:], frequencies[1:], spectrum[1:]
        
        fig.add_trace(go.Scatter(x=time_periods, y=spectrum, mode='lines', name=f'{component} power spectrum'), row=i+1, col=1)

    # Update the layout with log-log axes
    fig.update_layout(title="Power Spectrum of Lorenz63 Components for {} set".format(explain))
    fig.update_xaxes(type="log", title="Time Periods")
    # fig.update_yaxes(type="log", title="Power")
    fig.update_yaxes(title="Power")
    
    # Display the figure
    fig.show()


# %%
# Define the default parameters values
sigma = 10
rho = 28
beta = 8/3
N = 15*60*24*5
l1 = L63(sigma, rho, beta, init=[1, 10, 20], dt=1e-2)
l2 = L63(sigma, rho, beta, init=[10, 1, 2], dt=1e-2)


l1.integrate(N)
l2.integrate(N)

plot_attractor_plotly([l1.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_train')
plot_attractor_plotly([l2.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_test')

plot_attractor_subplots([l1.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_train')
plot_attractor_subplots([l2.hist], save_dir='temp_save/lorenz63', explain='s10_r28_b8d3_test')

plot_components_vs_time_plotly(np.array(l1.hist), time_step=1e-2, explain='s10_r28_b8d3_train', save_dir='temp_save/lorenz63')
plot_components_vs_time_plotly(np.array(l2.hist), time_step=1e-2, explain='s10_r28_b8d3_test', save_dir='temp_save/lorenz63')


# store l.hist as pickle data for later use in pytorch dataloader
def save_pickle(data, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

save_pickle(l1.hist, 'temp_save/lorenz63/dataset_train.pkl')
save_pickle(l2.hist, 'temp_save/lorenz63/dataset_test.pkl')

# Call the function to display the plot
plot_power_spectrum_plotly(np.array(l1.hist), explain='s10_r28_b8d3_train', save_dir='temp_save/lorenz63')
plot_power_spectrum_plotly(np.array(l2.hist), explain='s10_r28_b8d3_test', save_dir='temp_save/lorenz63')

# Call the function to display the plots

plot_power_spectrum_subplots_loglog(np.array(l1.hist), explain='s10_r28_b8d3_train', save_dir='temp_save/lorenz63')
plot_power_spectrum_subplots_loglog(np.array(l2.hist), explain='s10_r28_b8d3_test', save_dir='temp_save/lorenz63')


# %%
