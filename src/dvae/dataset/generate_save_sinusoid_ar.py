#%%
import numpy as np
import pickle
import plotly.graph_objects as go

class DESinusoid():
    def __init__(self, amplitudes, frequencies, phases, dt):
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.phases = phases
        self.dt = dt
        
        # Initial values
        self.values = [np.array(amplitudes) * np.sin(np.array(phases))]
        self.velocities = [np.array(amplitudes) * np.cos(np.array(phases))]

    def step(self):
        new_values = []
        new_velocities = []
        for i in range(len(self.amplitudes)):
            omega = 2 * np.pi * self.frequencies[i]
            
            # Compute using Euler's method
            new_val = self.values[-1][i] + self.velocities[-1][i] * self.dt
            new_velocity = self.velocities[-1][i] - omega**2 * self.values[-1][i] * self.dt
            
            new_values.append(new_val)
            new_velocities.append(new_velocity)
        
        self.values.append(new_values)
        self.velocities.append(new_velocities)
    
    def generate(self, n_steps):
        for _ in range(n_steps): 
            self.step()

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
amplitudes = [1, 2]
frequencies = [1, 0.5] # Frequency in Hz
phases = [0, np.pi/2] # Starting phases
time_step = 0.01 # Time step for the generation
n_steps = 1000 # Number of time steps to generate

s = DESinusoid(amplitudes, frequencies, phases, time_step)
s.generate(n_steps)
plot_sinusoids_plotly(s.values, time_step)
#%%
# To save the generated data
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

save_pickle(s.values, 'desinusoids.pkl')
