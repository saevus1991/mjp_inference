import numpy as np
import mjp_inference as mjpi
from mjp_inference.util.diff import num_derivative
# import torch
# from transcription.compiled import transcription
# from transcription.simulation.models.collecting_tasep import CollectingTASEP
# from pymbvi.models.observation.tasep_obs_model import LognormGauss
# from pyssa import ssa
# import pyro
# from pyro.distributions import Gamma, Uniform, Normal
# from transcription.pyro.cthmm import CTHMM
# import matplotlib.pyplot as plt
# import time

np.random.seed(2103110956)

# set up model
num_sites = 12
num_collect = 9
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

# set up up obs model
alpha = np.array([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24], dtype=np.float64)
obs_model = mjpi.ObservationModel(model, noise_type="lognormal")
# register parameters
obs_model.add_param(name='b0', value=5.0)
obs_model.add_param(name='b1', value=3.0)
obs_model.add_param(name='lamb', value=0.0001)
obs_model.add_param(name='gamma', value=1.1)
obs_model.add_param(name='sigma', value=15)


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


def intensity_backward(time, state, param, grad_output, grad):
    # evaluate stems
    n_stem = 0
    for i in range(num_sites):
        n_stem += alpha[i] * state[i]
    # compute gradient
    grad[0] = grad_output[0]
    grad[1] = np.exp(-param[2]*time) * grad_output[0]
    grad[2] = -np.exp(-param[2]*time)*(param[1] + param[3]*n_stem)*time * grad_output[0]
    grad[3] = np.exp(-param[2]*time)*n_stem*grad_output[0]


def sigma(time, state, param, transformed):
    transformed[0] = param[4]


def sigma_backward(time, state, param, grad_output, grad):
    grad[4] = grad_output[0]

# add transforms to obs model
obs_model.add_transform(mjpi.Transform('mu', intensity, transform_grad=intensity_backward))
obs_model.add_transform(mjpi.Transform('sigma', sigma, transform_grad=sigma_backward))
obs_model.build()

# create simulated data
tspan = np.array([0.0, 100])
seed = np.random.randint(2**18)
trajectory = mjpi.simulate(initial, model, tspan, seed)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)

# produce observations 
delta_t = 3
t_obs = np.arange(tspan[0]+delta_t, tspan[1]-delta_t, delta_t)
observations = mjpi.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

# create initial distribution
initial = np.zeros(model.num_species)
ind = model.state2ind(initial)
initial_dist = np.zeros(model.num_states)
initial_dist[ind] = 1.0

# stack of obs_params
num_samples = 10
obs_param = np.tile(np.array(obs_model.param_array), [num_samples, 1])
tmp = 1.0 + 0.1*np.random.randn(num_samples, 2)
obs_param[:, :2] *= tmp

# set up master equation
master_equation = mjpi.MEInference(model)
rates = model.rate_array

# evaluate log prob
log_prob, initial_grad, rates_grad, obs_param_grad = mjpi.batched_filter(initial_dist, rates, master_equation, obs_model, t_obs, observations, obs_param, get_gradient=True, num_workers=10)

print(log_prob.mean())

# compute numerical rates gradient
def fun(x):
    log_prob = mjpi.batched_filter(initial_dist, x, master_equation, obs_model, t_obs, observations, obs_param, get_gradient=False, num_workers=10)[0]
    return(log_prob.sum(keepdims=True))

rates_grad_num = num_derivative(fun, rates, h=1e-7).squeeze()

print("rates_grad")
print(np.stack([rates_grad.sum(axis=0), rates_grad_num], axis=1))

# compute numerical obs_param gradient
def fun(x):
    log_prob = mjpi.batched_filter(initial_dist, rates, master_equation, obs_model, t_obs, observations, x.reshape(obs_param.shape), get_gradient=False, num_workers=10)[0]
    return(log_prob.sum(keepdims=True))

# obs_param_grad_num = num_derivative(fun, obs_param.squeeze()).reshape(obs_param.shape)