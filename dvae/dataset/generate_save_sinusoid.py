#%%
import numpy as np
import pickle
import plotly.graph_objects as go

class DirectSinusoid():
    def __init__(self, amplitudes, frequencies, phases, dt):
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phases = phases
        self.dt = dt
        self.values = []

    def generate(self, n_steps):
        t_values = np.arange(0, n_steps*self.dt, self.dt)
        for t in t_values:
            y = [A * np.sin(2 * np.pi * f * t + phi) for A, f, phi in zip(self.amplitudes, self.frequencies, self.phases)]
            self.values.append(y)

def plot_sinusoids_plotly(time_series, time_step):
    t = np.arange(0, len(time_series) * time_step, time_step)
    time_series = np.array(time_series)

    fig = go.Figure()
    for i in range(time_series.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=time_series[:, i], mode='lines', name=f'Sinusoid {i+1}'))
    
    fig.update_layout(title='Sinusoids vs. Time',
                      xaxis_title='Time',
                      yaxis_title='Value',
                      showlegend=True,
                      template='plotly_white')
    
    fig.show()
#%%
# Parameters for two sinusoids
amplitudes = [1]
frequencies = [1]  # Frequency in Hz
phases = [0]   # Starting phases
time_step = 0.01        # Time step for the generation
n_steps = 1000000 #15*60*24*5          # Number of time steps to generate

s = DirectSinusoid(amplitudes, frequencies, phases, time_step)
s.generate(n_steps)
plot_sinusoids_plotly(s.values, time_step)

#%%
# To save the generated data
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

save_pickle(s.values, 'dataset_1d.pkl')

# %%
