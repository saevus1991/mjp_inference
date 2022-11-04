import numpy as np
import mjp_inference as mjpi
import mjp_inference.wrapper.pyro as mjpp
import torch
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO


pyro.set_rng_seed(2211041122)

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

# set up master equation
rates = torch.from_numpy(model.rate_array)
master_equation = mjpi.MEInference(model)

# get initial dist
initial_dist = torch.zeros(model.num_states)
ind = model.state2ind(np.zeros(model.num_species))
initial_dist[ind] = 1.0
initial_dist = mjpp.markov_transition(initial_dist, rates, torch.tensor([0.0, 1000.0]), master_equation)

# prepare observations
obs_param = torch.from_numpy(obs_model.param_array)
tspan = np.array([0.0, 100])
delta_t = 3
t_obs = np.arange(tspan[0]+delta_t, tspan[1]-delta_t, delta_t)

# set parameter prior
rates_mean = rates
cv = torch.tensor([0.5, 0.5, 0.5])*2
a = 1/cv**2
b = 1/(cv**2*rates_mean)


# define a model
def pyro_model(obs=None):
    # draw parameter prior
    rates = pyro.sample('rates', dist.Gamma(a, b).to_event(1))
    # sample observations
    with pyro.plate('cells', 5):
        obs = pyro.sample('obs', mjpp.CTHMM(master_equation, obs_model, torch.from_numpy(t_obs), initial_dist, rates, obs_param), obs=obs)
    return(obs)


# trace = pyro.poutine.trace(pyro_model).get_trace()
# trace.compute_log_prob()
# print(trace.nodes['rates'])
# quit()

# create observations
observations = pyro_model()


# define a guide
def guide(obs=None):
    # define parameters
    a_post = pyro.param("a_post", a, constraint=dist.constraints.positive)
    b_post = pyro.param("b_post", b, constraint=dist.constraints.positive)
    # sample rates
    rates = pyro.sample('rates', dist.Gamma(a_post, b_post).to_event(1))



# set up optimizer
optimizer = Adam({'lr': 1e-1})
# set up svi stuff
svi = SVI(pyro_model, guide, optimizer, loss=Trace_ELBO())
# run inference
svi_steps = 10
losses = torch.zeros(svi_steps)
for i in range(svi_steps):
    losses[i] = svi.step(obs=observations)
    print("step", i, "loss:", losses[i], "avg loss:", losses[:i+1].mean(), flush=True)