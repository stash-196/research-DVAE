# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict

# %%
# Handy function stolen from the fast.ai library
def V(x, requires_grad=False, gpu=False):
    t = torch.FloatTensor(np.atleast_1d(x).astype(np.float32))
    if gpu: t = t.cuda()
    return Variable(t, requires_grad=requires_grad)

# %%
# Define the default parameters values
sigma = 10
rho = 28
beta = 8/3
N = 15*60*24*5

# %%
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


# %%
l = L63(sigma, rho, beta, init=[1, 10, 20], dt=1e-2)

# %%
l.integrate(N)

# %%
l2 = L63(sigma, rho, beta, init=[1.1, 10, 20], dt=1e-2)

# %%
l2.integrate(N)

# %%
import plotly.graph_objects as go

def plot_attractor_plotly(hists):
    if np.array(hists).ndim == 2:
        hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = go.Figure()
    for h in hists:
        fig.add_trace(go.Scatter3d(x=h[:, 0], y=h[:, 1], z=h[:, 2], mode='lines', line=dict(color='blue')))
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ))
    fig.show()

# %%
plot_attractor_plotly([l.hist, l2.hist])

# %%
print(len(l.hist))

#%%
import plotly.graph_objects as go

def plot_components_vs_time_plotly(time_series, time_step):
    t = np.arange(0, len(time_series) * time_step, time_step)
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='y', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=t, y=z, mode='lines', name='z', line=dict(color='red')))
    
    fig.update_layout(title='Components of Lorenz63 System vs. Time',
                      xaxis_title='Time',
                      yaxis_title='Values',
                      showlegend=True,
                      template='plotly_white')
    
    fig.show()
plot_components_vs_time_plotly(np.array(l.hist), time_step=1e-2)


# %%
# store l.hist as pickle data for later use in pytorch dataloader
def save_pickle(data, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# %%
save_pickle(l.hist, 'lorenz63.pkl')

# %%
save_pickle(l.hist, 'lorenz63_2.pkl')



#%%
import numpy as np

def power_spectrum(time_series):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(time_series)
    
    # Compute the power spectrum: the square of the absolute value of the FFT
    power_spectrum = np.abs(fft_result)**2
    
    # Since the power spectrum is symmetric, we only need to return the first half
    power_spectrum = power_spectrum[:len(power_spectrum) // 2]
    
    # Compute the frequencies corresponding to the values in the power spectrum
    frequencies = np.fft.fftfreq(len(time_series))[:len(power_spectrum)]
    
    return frequencies, power_spectrum
# Compute and plot the power spectrum for each component of the Lorenz 63 system
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

for i, component in enumerate(["x", "y", "z"]):
    time_series = np.array(l.hist)[:, i]
    frequencies, spectrum = power_spectrum(time_series)
    
    # Skip the zero frequency
    frequencies, spectrum = frequencies[1:], spectrum[1:]
    
    axes[i].semilogx(frequencies, spectrum)
    axes[i].set_title(f"Power Spectrum of {component}-component")
    axes[i].set_xlabel("Frequency (log scale)")
    axes[i].set_ylabel("Power")

plt.tight_layout()
plt.show()

# %%
