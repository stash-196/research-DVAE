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
def plot_attractor(hists):
    if np.array(hists).ndim == 2: hists = [hists]
    hists = [np.array(h) for h in hists]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    [ax.plot(h[:,0], h[:,1], h[:,2]) for h in hists]

# %%
plot_attractor([l.hist, l2.hist])

# %%
print(len(l.hist))

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



