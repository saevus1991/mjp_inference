import numpy as np
import torch
import mjp_inference as mjpi
import mjp_inference.torch as mjpt
from mjp_inference.util.diff import num_derivative
from time import time as stopwatch


torch.manual_seed(2201251626)
torch.set_default_tensor_type(torch.DoubleTensor)

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

# set up master equation
time = 100.0
rates = model.rate_array
master_equation = mjpi.MEInference(model)

# set up initial
initial = np.zeros(num_sites)
ind = model.state2ind(initial)
initial_dist = np.zeros(model.num_states)
initial_dist[ind] = 1.0
initial_dist = mjpi.KrylovPropagator(master_equation, initial_dist, rates, 20.0).propagate()
initial_dist[initial_dist < 0] = 0.0
initial = model.ind2state(torch.distributions.Categorical(torch.from_numpy(initial_dist)).sample().item())

# set up up obs model
alpha = np.array([0, 4, 8, 12, 16, 20, 24, 24, 24, 24, 24, 24], dtype=np.float64)
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

# create simulated data
tspan = np.array([0.0, 100])
seed = torch.randint(2**16, (1,)).item()
trajectory = mjpi.simulate(initial, model, tspan, seed)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = mjpi.discretize_trajectory(trajectory, t_plot)

# produce observations 
obs_param = obs_model.param_array.copy()
delta_t = 3
t_obs = np.arange(tspan[0]+delta_t, tspan[1]-delta_t, delta_t)
observations = mjpi.discretize_trajectory(trajectory, t_obs, obs_model)

# run filter
start_time = stopwatch()
rates_torch = torch.from_numpy(rates)
rates_torch.requires_grad = True
obs_param_torch = torch.from_numpy(obs_param)
obs_param_torch.requires_grad = True
log_prob = mjpt.filter(torch.from_numpy(initial_dist), rates_torch, master_equation, obs_model, torch.from_numpy(t_obs), torch.from_numpy(observations), obs_param_torch)
print('log_prob', log_prob)
log_prob.backward()
end_time = stopwatch()
print(f'Computing log prob via Krylov required {end_time-start_time} seconds')
rates_grad = rates_torch.grad.numpy()
obs_param_grad = obs_param_torch.grad.numpy()

# compute rates gradients numerically
def fun(x):
    log_prob = mjpt.filter(torch.from_numpy(initial_dist), torch.from_numpy(x), master_equation, obs_model, torch.from_numpy(t_obs), torch.from_numpy(observations), torch.from_numpy(obs_param)).numpy()
    return(log_prob)
rates_grad_num = num_derivative(fun, rates, 1e-5).squeeze()
print('rates gradient')
print(rates_grad)
print(rates_grad_num)

# compute obs_param gradients numerically
def fun(x):
    log_prob = mjpt.filter(torch.from_numpy(initial_dist), torch.from_numpy(rates), master_equation, obs_model, torch.from_numpy(t_obs), torch.from_numpy(observations), torch.from_numpy(x)).numpy()
    return(log_prob)
obs_param_grad_num = num_derivative(fun, obs_param, 1e-5).squeeze()
print('obs_param gradient')
print(obs_param_grad)
print(obs_param_grad_num)
