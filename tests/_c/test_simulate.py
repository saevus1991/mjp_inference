import os
import numpy as np
# import torch
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# from pymbvi.models.observation.tasep_obs_model import LognormGauss
# from pymbvi.util import num_derivative
from time import time as stopwatch
# import seaborn as sns
from scipy.linalg import expm
import scipy.sparse as sp
import mjp_inference as mjpi
from numba import cfunc, types
from scipy import LowLevelCallable


TransformationFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))
SampleFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))

np.random.seed(2103110956)

# set up model
num_sites = 12
num_collect = 10-1
# set up model
model = mjpi.MJP("Tasep")
# add species
for i in range(num_sites-1):
    model.add_species(name=f'X_{i}')
model.add_species(name=f'X_{num_sites-1}', upper=num_collect)
# add rates
model.add_rate('k_init', 0.1)
model.add_rate('k_elong', 0.3)
model.add_rate('k_termin', 0.01)
# add init
model.add_event(mjpi.Event(name='Initiation', input_species=['X_0'], output_species=['X_0'] , rate='k_init', propensity=lambda x: (1-x[0]), change_vec=[1]))
# add add hops
for i in range(1, num_sites-1):
    input_species = [f'X_{i-1}', f'X_{i}']
    model.add_event(mjpi.Event(name=f'Hop {i}', input_species=input_species, output_species=input_species, rate='k_elong', propensity=lambda x: x[0]*(1-x[1]), change_vec=[-1, 1]))
input_species = [f'X_{num_sites-2}', f'X_{num_sites-1}']
def prop(x):
    if (x[1] < num_collect):
        return(x[0])
    else:
        return(0.0)
model.add_event(mjpi.Event(name=f'Hop {num_sites-1}', input_species=input_species, output_species=input_species, rate='k_elong', propensity=prop, change_vec=[-1, 1]))
# add termination
model.add_event(mjpi.Event(name='Termination', input_species=[f'X_{num_sites-1}'], output_species=[f'X_{num_sites-1}'], rate='k_termin', propensity=lambda x: x[0], change_vec=[-1]))
model.build()

# get initial value
seed = np.random.randint(2**16)
initial = np.zeros(num_sites)
simulator = mjpi.Simulator(model, initial, np.array([0.0, 1000.0]), seed)
trajectory = simulator.simulate()
initial = trajectory['states'][-1]

# obs model hyper params
alpha = np.array([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24], dtype=np.float64)

# set up up obs model
obs_model = mjpi.ObservationModel(model, noise_type="normal")
# register parameters
obs_model.add_param(name='b0', value=5.0)
obs_model.add_param(name='b1', value=3.0)
obs_model.add_param(name='lamb', value=0.0001)
obs_model.add_param(name='gamma', value=1.1)
obs_model.add_param(name='sigma', value=15)


# spot intensity computation
def intensity(time, state, param, transformed):
    # parse parameters
    b0 = param[0]
    b1 = param[1]
    lambd = param[2]
    gamma = param[3]
    # evaluate stems
    n_stem = 0
    for i in range(num_sites):
        n_stem += alpha[i] * state[i]
    # evaluate intensity
    transformed[0] = b0 + (b1 + gamma * n_stem) * np.exp(-lambd*time)


# extract noise parameter
def sigma(time, state, param, transformed):
    transformed[0] = param[4]


# add transforms to obs model
obs_model.add_transform(mjpi.Transform('mu', intensity))
obs_model.add_transform(mjpi.Transform('sigma', sigma))
obs_model.build()

# prepare simulation
seed = np.random.randint(2**16)
tspan = np.array([0.0, 200.0])
t_obs = np.arange(tspan[0], tspan[1], 3.0)

# simulate
obs = mjpi.simulate(initial, model, obs_model, t_obs, seed)

# plot
fig, axs = plt.subplots(1, 1)
axs.plot(t_obs, obs, '--o', color='tab:red')
plt.show()