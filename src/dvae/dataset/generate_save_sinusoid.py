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

def plot_sinusoids_plotly(time_series, time_step, name):
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
    
    # fig.show()
    # save fig
    fig.write_image(f"temp_save/sinusoid/sinusoids_{name}.png")

#%%
# Parameters for two sinusoids
amplitudes = [1, 1, 1]
frequencies = [1, 1/100, 1/1000]  # Frequency in Hz
phases = [0, 0, 0]   # Starting phases
time_step = 1e-2        # Time step for the generation
n_steps = 100000 #15*60*24*5          # Number of time steps to generate

s_train = DirectSinusoid(amplitudes, frequencies, phases, time_step)
s_train.generate(n_steps)
name_train = f'{len(amplitudes)}d_train'
plot_sinusoids_plotly(s_train.values, time_step, name_train)

s_test = DirectSinusoid(amplitudes, frequencies, phases, time_step)
s_test.generate(n_steps//10)
name_test = f'{len(amplitudes)}d_test'
plot_sinusoids_plotly(s_test.values, time_step, name_test)



#%%
# To save the generated data
def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

save_pickle(s_train.values, f'temp_save/sinusoid/dataset_{name_train}.pkl')
save_pickle(s_test.values, f'temp_save/sinusoid/dataset_{name_test}.pkl')

# %%
