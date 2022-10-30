import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mjp_inference.util.diff import num_derivative
import mjp_inference as mjpi
from numba import cfunc, types

TransformationFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))
SampleFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))

np.random.seed(2103110956)
# torch.manual_seed(2103170804)
# torch.set_default_tensor_type(torch.DoubleTensor)

# # set up model
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

# prepare grad computation
param = obs_model.param_array
time = 9.0
obs = np.array([305.2])
# seed = np.random.randint(2**16)
# tspan = np.array([0.0, 200.0])
# t_obs = np.arange(tspan[0], tspan[1], 3.0)

log_prob_grad = obs_model.log_prob_grad(time, initial, param, obs)

def fun(x):
    return(np.array([obs_model.log_prob(time, initial, x, obs)]))

log_prob_grad_num = num_derivative(fun, param)


print(log_prob_grad)
print(log_prob_grad_num)
