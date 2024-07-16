# import packges
import numpy as np
import mjp_inference as mjpi
import matplotlib.pyplot as plt
from mjp_inference.util.conv import discretize_trajectory
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value
import mjp_inference.wrapper.pyro as mjpp
from mjp_inference.util.diff import num_derivative
from pathlib import Path
import logging
from scipy.sparse import linalg as slinalg
import matplotlib.pyplot as plt

# configure logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='switching_poissong.log',
                    filemode='a')


# set up model for first part
model1 = mjpi.MJP(name='poisson 1')
# add species
model1.add_species(name='Count', upper=150)
# add an event
model1.add_event(mjpi.Event(name='birth', input_species=['Count'], output_species=['Count'], rate=1.0, propensity=lambda x: 1, change_vec=[1]))
model1.build()

# set up model for second part
model2 = mjpi.MJP(name='poisson 2')
# add species
model2.add_species(name='Count', upper=150)
# add an event
model2.add_event(mjpi.Event(name='birth', input_species=['Count'], output_species=['Count'], rate=3.0, propensity=lambda x: 1, change_vec=[1]))
model2.build()

# set up an observation model
sigma = 0.25
obs_model = mjpi.NormalObs(model1, sigma, observed_species='Count')

# set initial conditions
np.random.seed(2401022301)
tspan1 = np.array([0.0, 20.0])
tspan2 = np.array([20.0, 40.0])
initial = np.array([0.0])

# simulate first trajectory part
seed = np.random.randint(2**18)
simulator = mjpi.Simulator(model1, initial, tspan1, seed)
tr1 = simulator.simulate()

# simulate second trajectory part
initial2 = tr1['states'][-1]
seed = np.random.randint(2**18)
simulator = mjpi.Simulator(model2, initial2, tspan2, seed)
tr2 = simulator.simulate()

# create a combined trajectory
trajectory = {
    'initial': tr1['initial'],
    'tspan': np.array([tspan1[0], tspan2[1]]),
    'times': np.concatenate([tr1['times'], tr2['times']]),
    'events': np.concatenate([tr1['events'], tr2['events']]),
    'states': np.concatenate([tr1['states'], tr2['states']]),
}

# create observations
delta_t = 3
t_obs = np.arange(delta_t, trajectory['tspan'][1], delta_t)
obs = discretize_trajectory(trajectory, t_obs, obs_model)

# extend states and time by first
t_plot = np.concatenate([trajectory['tspan'][[0]], trajectory['times']])
states_plot = np.concatenate([trajectory['initial'][:, None], trajectory['states']])

# # plot simulated data
# fig, axs = plt.subplots()
# # plt.plot(t_eval, states, '--r')
# axs.step(t_plot, states_plot, '--', color='tab:red', where='pre')
# axs.plot(t_obs, obs, 'xk')
# plt.show()

# torch converstions
rates = torch.from_numpy(model1.rate_array)
obs_param = torch.from_numpy(obs_model.param_array)
observations = torch.from_numpy(obs)

# set up a master equation
master_equation = mjpi.MEInference(model1)

# get initial dist
initial_dist = torch.zeros(model1.num_states)
ind = model1.state2ind(np.zeros(model1.num_species))
initial_dist[ind] = 1.0

# create pyro model
def pyro_model(obs=None):
    # draw parameter prior
    #rates = pyro.sample('rates', Gamma(a, b))
    rates = pyro.sample('rates', dist.Uniform(0.0, 5.0))
    # sample observations
    obs = pyro.sample('obs', mjpp.CTHMM(master_equation, obs_model, torch.from_numpy(t_obs), initial_dist, rates, obs_param), obs=obs)
    return(obs)

# load data
workdir = Path(__file__).parent
save_path = Path(workdir).joinpath('switching_poisson', 'hmc.pt')
posterior_samples = torch.load(save_path)

# # simulate posterior paths
# traces_per_sample = 1
# num_samples = 1
# tspan = trajectory['tspan']
# post_trajectories = []
# for i in range(num_samples):
#     rate = posterior_samples['rates'][i].numpy()
#     t_post = np.linspace(tspan[0], tspan[1], 100*len(t_obs))
#     print(i)
#     print(tspan)
#     print(t_obs)
#     quit()
#     post_trajectory = mjpi.simulate_posterior_batched(master_equation, obs_model, initial_dist.numpy(), rates, obs_param.numpy(), tspan, t_obs, observations.numpy(), t_post, seed, num_samples=traces_per_sample, num_workers=1)
#     post_trajectories.append(post_trajectory)

# master_equation.update_generator(np.array([2.0]))
# prop = master_equation.generator.T @ initial_dist
# proj = prop.T @ initial_dist.numpy()
# print(initial_dist)
# print(prop)
# print(np.linalg.norm(prop))
# print(np.linalg.norm(proj))
# quit()

# initial = np.ones(model1.num_states)
# test1 = np.linalg.norm(initial)
# master_equation.update_generator(np.array([2.0]))
# test2 = slinalg.norm(master_equation.generator)
# test3 = np.linalg.norm(master_equation.generator @ initial)
# # test
# print("state", test1)
# print("generator", test2)
# print("propagated", test3)
# quit()

# run backward filter
tspan = trajectory['tspan']
t_post = np.linspace(tspan[0], tspan[1], 100*len(t_obs))
# obs_param = obs_model.param_array
rates = np.array([2.0])
print(obs_param, observations.shape)
filt = mjpi.KrylovBackwardFilter(master_equation, obs_model, t_obs, observations.numpy(), initial_dist.numpy(), rates, obs_param.numpy(), tspan)
print('set up filter')
filt.forward_filter()
print('ran forward filter')
filt.backward_filter()
print('ran backward filter')
print(filt.log_prob())
forward = filt.eval_forward_filter(t_post)
print("evaluated forward")
backward = filt.eval_backward_filter(t_post)
print('evaluated backward')
smoothed = forward * backward
smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)

# plot
fig, axs = plt.subplots()
axs.imshow(smoothed)
plt.show()