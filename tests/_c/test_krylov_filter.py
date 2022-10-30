import numpy as np
# import torch
# from pathlib import Path
import mjp_inference as mjpi
# import matplotlib.pyplot as plt
# from pymbvi.util import num_derivative
from mjp_inference.util.diff import num_derivative
# from transcription.autograd.transitio n import markov_transition
import time as stopwatch

np.random.seed(2201251626)

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

# set up initial
initial = np.zeros(num_sites)
ind = model.state2ind(initial)
initial_dist = np.zeros(model.num_states)
initial_dist[ind] = 1.0

# preparations
alpha = np.array([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24], dtype=np.float64)
time = 28.0

# set up up obs model
obs_model = mjpi.ObservationModel(model, noise_type="normal")
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
# get observations
seed = np.random.randint(2**16)
t_obs = np.arange(3.0, 94.0, 3.0)
observations = mjpi.simulate(initial, model, obs_model, t_obs, seed)

# set up master equation
master_equation = mjpi.MEInference(model)
rates = model.rate_array
obs_param = obs_model.param_array

# run filter
start_time = stopwatch.time()
filt = mjpi.KrylovFilter(master_equation, obs_model, t_obs, observations, initial_dist, rates, obs_param)
llh_compiled = filt.log_prob()
filt.log_prob_backward()
filt.compute_rates_grad()
end_time = stopwatch.time()
print(llh_compiled)
print(f"Computing log prob via Krylov required {end_time-start_time} seconds")
initial_grad = filt.get_initial_grad()
rates_grad = filt.get_rates_grad()
obs_param_grad = filt.get_obs_param_grad()


# # plt.plot(observations)
# # plt.show()

# # # propagate
# # t_eval = np.arange(3.0, 100.0, 15.0)
# # solver = transcription.KrylovSolver(model, initial_dist, rates, t_eval)
# # states = solver.forward()
# # print(states.shape, np.sum(states, axis=1))

# # # compute gradients
# # out = np.random.rand(len(t_eval), num_states)
# # solver.backward(5, out)
# # solver.compute_rates_grad()
# # rates_grad = solver.get_rates_grad()
# # initial_grad = solver.get_initial_grad()
# # print(rates_grad)
# # print(initial_grad.shape)


# numerical rates gradient
def fun(x):
    filt = mjpi.KrylovFilter(master_equation, obs_model, t_obs, observations, initial_dist, x, obs_param)
    llh = filt.log_prob()
    return(np.array([llh]))
rates_grad_num = num_derivative(fun, rates, 1e-5)
print(rates_grad)
print(rates_grad_num)

ind = np.random.choice(np.arange(model.num_states), 5)

# numerical initial gradient 
def fun(x):
    initial = initial_dist.copy()
    initial[ind] = x
    filt = mjpi.KrylovFilter(master_equation, obs_model, t_obs, observations, initial, rates, obs_param)
    llh = filt.log_prob()
    return(np.array([llh]))
initial_grad_num = num_derivative(fun, initial_dist[ind])
print(initial_grad[ind])
print(initial_grad_num)
# check = np.linalg.norm(initial_grad_num-initial_grad)
# print('initial grad check', check)

# numerical obs param gradient
def fun(x):
    filt = mjpi.KrylovFilter(master_equation, obs_model, t_obs, observations, initial_dist, rates, x)
    llh = filt.log_prob()
    return(np.array([llh]))
obs_param_grad_num = num_derivative(fun, obs_param, 1e-5)
print(obs_param_grad)
print(obs_param_grad_num)