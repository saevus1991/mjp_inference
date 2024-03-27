import numpy as np
import mjp_inference as mjpi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# import torch
# from transcription.compiled import transcription
# from transcription.simulation.models.collecting_tasep import CollectingTASEP
# from pymbvi.models.observation.tasep_obs_model import LognormGauss
# from pyssa import ssa
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# import time
# from pymbvi.forward_backward import ForwardBackward, FilterObsModel

# # np.random.seed(2106301102)
# torch.set_default_tensor_type(torch.DoubleTensor)

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

# preparations
alpha = np.array([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24], dtype=np.float64)

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

# get initial value
seed = np.random.randint(2**16)
initial = np.zeros(num_sites)
simulator = mjpi.Simulator(model, initial, np.array([0.0, 1000.0]), seed)
trajectory = simulator.simulate()
initial = trajectory['states'][-1]

# simulate trajctory 
tspan = np.array([0.0, 100.0])
delta_t = 3
t_plot = np.linspace(tspan[0], tspan[1], 100)
t_obs = np.arange(tspan[0]+delta_t, tspan[1]-delta_t, delta_t)
seed = np.random.randint(2**16)
simulator = mjpi.Simulator(model, initial, tspan, seed)
trajectory = simulator.simulate()
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)
observations = mjpi.discretize_trajectory(trajectory, t_obs, obs_model)

# create initial distribtion
initial_dist = np.zeros(model.num_states)
ind = model.state2ind(initial)
initial_dist[ind] = 1.0
master_equation = mjpi.MEInference(model)
rates = model.rate_array
initial_dist = solve_ivp(master_equation.forward, np.array([0.0, 1000.0]), initial_dist)['y'][:, -1]
print(observations.shape)

# run backward filter
t_post = np.linspace(tspan[0], tspan[1], 100*len(t_obs))
obs_param = obs_model.param_array
filt = mjpi.KrylovBackwardFilter(master_equation, obs_model, t_obs, observations, initial_dist, rates, obs_param, tspan)
filt.forward_filter()
print("ran forward filter")
filt.backward_filter()
print(filt.log_prob())
backward = filt.eval_backward_filter(t_post)
print(backward.shape)

post_simulator = mjpi.PosteriorSimulator(model, initial, tspan, seed)
trajectory = post_simulator.simulate(t_post, backward)
print(trajectory)