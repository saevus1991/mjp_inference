# import os
# os.environ["OMP_NUM_THREADS"] = "1" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1" 
# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import numpy as np
import mjp_inference as mjpi
import mjp_inference.wrapper.pyro as mjpp
import torch
# from pathlib import Path
# from transcription.compiled import transcription
# from transcription.simulation.models.collecting_tasep import CollectingTASEP
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# # from pymbvi.forward_backward import ForwardBackward, FilterObsModel
# # from pymbvi.variational_engine import FBVariationalEngine
# from pymbvi.models.observation.tasep_obs_model import LognormGauss
# from pymbvi.util import num_derivative
# #from pymbvi.forward_backward import ForwardBackward, FilterObsModel
# from pyssa.ssa import Simulator, discretize_trajectory
# import time as clock
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value
# import seaborn as sns
# from transcription.pyro.cthmm import CTHMM

pyro.set_rng_seed(2211041122)

torch.set_default_tensor_type(torch.DoubleTensor)


# # class TasepDist(TorchDistribution):

# #     def __init__(self, transition_model, obs_model, sim_model, obs_model_sim, obs_times, initial_dist, rates, obs_param, validate_args=False):
# #         self.transition_model = transition_model
# #         self.obs_model = obs_model
# #         self.sim_model = sim_model
# #         self.obs_model_sim = obs_model_sim
# #         self.obs_times = obs_times
# #         self.initial_dist = initial_dist
# #         self.rates = rates
# #         self.obs_param = obs_param
# #         self.num_intervals = len(obs_times)
# #         # required for parent class
# #         #arg_constraints =
# #         super().__init__(event_shape=(self.num_intervals, 1), validate_args=validate_args)

# #     def sample(self, sample_shape=torch.Size()):
# #         if len(sample_shape) > 0:
# #             observations = torch.Tensor(sample_shape+self.event_shape)
# #             for i in range(sample_shape[0]):
# #                 # sample initial
# #                 ind = torch.distributions.categorical.Categorical(self.initial_dist).sample()
# #                 initial = self.transition_model.ind2state(ind.item())
# #                 # set up simulator
# #                 simulator = Simulator(self.sim_model, initial)
# #                 # perform simulation
# #                 tspan = np.array([0.0, self.obs_times[-1]])
# #                 trajectory = simulator.simulate(initial, tspan, get_states=True)
# #                 # get oservations
# #                 observations[i] = torch.from_numpy(discretize_trajectory(trajectory, self.obs_times.numpy(), obs_model=self.obs_model_sim))
# #             return(observations)
# #         else:
# #             # sample initial
# #             ind = torch.distributions.categorical.Categorical(self.initial_dist).sample()
# #             initial = self.transition_model.ind2state(ind.item())
# #             # set up simulator
# #             simulator = Simulator(self.sim_model, initial)
# #             # perform simulation
# #             tspan = np.array([0.0, self.obs_times[-1]])
# #             trajectory = simulator.simulate(initial, tspan, get_states=True)
# #             # get oservations
# #             observations = discretize_trajectory(trajectory, self.obs_times.numpy(), obs_model=self.obs_model_sim)
# #             return(torch.from_numpy(observations))

# #     def log_prob(self, obs_data):
# #         """
# #         Takes one vector of observations as input and computes log_prob
# #         """
# #         log_prob = filter_mp(self.initial_dist.clone(), self.rates, self.transition_model, self.obs_model, self.obs_times, obs_data, self.obs_param, num_workers=5)
# #         return(log_prob)


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

# # get initial value
# seed = np.random.randint(2**16)
# initial = np.zeros(num_sites)
# simulator = mjpi.Simulator(model, initial, np.array([0.0, 1000.0]), seed)
# trajectory = simulator.simulate()
# initial = trajectory['states'][-1]


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
tspan = np.array([0.0, 100])
delta_t = 3
t_obs = np.arange(tspan[0]+delta_t, tspan[1]-delta_t, delta_t)

# # create initial distribution
# num_states = model.get_num_states()
# ind = model.state2ind(initial)
# initial_model = np.zeros(num_states)
# initial_model[ind] = 1.0

# # set up vi obs_model
# obs_model = transcription.LognormObs(obs_param, alpha, model)

# # # create pecomputed obs llhs 
# # filter_obs = FilterObsModel(model, obs_model, t_obs, observations)

# # # set parameter prior
# # rates_mean = torch.tensor(rates)#torch.ones(len(rates))
# # cv = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])*2
# # a = 1/cv**2
# # b = 1/(cv**2*rates_mean)
# # # gamma = Gamma(a, b)

rates_mean = rates
obs_param = torch.from_numpy(obs_model.param_array)
lower = torch.Tensor([0.01, 0.01, 0.001])
upper = torch.Tensor([1.0, 1.0, 0.1])

# # rates_tmp = torch.tensor([0.1530, 0.4724, 0.2606, 0.0656, 0.1256], requires_grad=True)
# # dist = TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial_model), rates_tmp, torch.from_numpy(obs_param))
# # llh = dist.log_prob(torch.from_numpy(observations))
# # llh.backward()
# # print(llh)
# # print(rates_tmp.grad)

# # def fun(rates):
# #     dist = TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial_model), torch.from_numpy(rates), torch.from_numpy(obs_param))
# #     res = dist.log_prob(torch.from_numpy(observations))
# #     return(np.array([res.item()]))

# # rates_grad_num = num_derivative(fun, rates_tmp.detach().numpy(), 1e-3)
# # print(rates_grad_num)


# # test = pyro.sample('test', TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial), torch.from_numpy(rates)))

# define a model
def pyro_model(obs=None):
    # draw parameter prior
    #rates = pyro.sample('rates', Gamma(a, b))
    rates = pyro.sample('rates', dist.Uniform(lower, upper))
    # sample observations
    with pyro.plate('cells', 5):
        obs = pyro.sample('obs', mjpp.CTHMM(master_equation, obs_model, torch.from_numpy(t_obs), initial_dist, rates, obs_param), obs=obs)
    return(obs)


observations = pyro_model()
print(observations.shape)


# # rates_tmp = torch.from_numpy(rates)
# # dist = TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial_model), rates_tmp, torch.from_numpy(obs_param))
# # log_prob = dist.log_prob(obs)
# # print(log_prob)

# # print(obs.shape)
# # quit()

# # def pyro_model():
# #     # draw parameter prior
# #     rates = pyro.sample('rates', Gamma(a, b))
# #     # sample observations
# #     #obs = pyro.sample('obs', TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial_model), rates, torch.from_numpy(obs_param)), obs=torch.from_numpy(observations))
# #     return(rates)


# set up mcmc
init_fn = init_to_value(values={'rates': rates_mean})
kernel = NUTS(pyro_model, step_size=1.0, init_strategy=init_fn)
mcmc = MCMC(kernel, num_samples=1000, warmup_steps=100)
mcmc.run(observations)
# # get hmc samples
# data = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# data['true_rates'] = rates
# data['lower'] = lower
# data['upper'] = upper

# # save
# save_path = Path(__file__).with_suffix('.pt')
# torch.save(data, save_path)


# # # plot
# # fig, axs = plt.subplots(3, 2)
# # for i in range(5):
# #     ax = axs.flatten()[i]
# #     sns.histplot(hmc_samples['rates'][:, i], ax=ax, stat='density')
# #     gamma = Gamma(a[i], b[i])
# #     def fun(x): return(gamma.log_prob(x).exp())
# #     x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
# #     y = fun(x)
# #     ax.plot(x, y, '-r')
# # plt.show()

# #obs = pyro.sample("obs", TasepDist(model, obs_model, sim_model, obs_model_sim, torch.from_numpy(t_obs), torch.from_numpy(initial), torch.from_numpy(rates)))
# #print(obs)


# # # set up likelihood model
# # llh_model = DataLikelihood(model, obs_model, t_obs, observations)

# # start_time = time.time()
# # initial_torch = torch.from_numpy(initial_model)
# # initial_torch.requires_grad = True
# # rates_torch = torch.from_numpy(rates)
# # rates_torch.requires_grad = True
# # log_prob = llh_model.forward(initial_torch, rates_torch)
# # end_time = time.time()
# # print(f"Computing log_prob required {end_time-start_time}")
# # print(log_prob)

# # log_prob.backward()
# # print(rates_torch.grad)

# # def fun(rates):
# #     res = llh_model.forward(torch.from_numpy(initial_model), torch.from_numpy(rates)).reshape(1,)
# #     return(res.numpy())

# # rates_grad_num = num_derivative(fun, rates, 1e-3)
# # print(rates_grad_num)

# # # set up vi model
# # param = np.concatenate([rates, np.log(rates)])
# # model = bt_full.VITransitionModel(num_sites, param)




# # # def zerofun(t):
# # #     return(np.zeros(len(initial_model)))

# # # # set up ode function
# # # def odefun(t, x): 
# # #     backward = zerofun(t)
# # #     return(model.forward_reduced(t, x, backward, param))


# # # terminal = obs_model.get_terminal(initial_model, observations[-1], t_obs[-1], obs_param)
# # # print(terminal.shape)
# # # print(np.max(terminal))
# # # test = model.backward_reduced(t_obs[-1], terminal, state, rates)
# # # print(test.sum())
# # #quit()


# # # create variational engine
# # options = {'solver_method': 'RK45'}
# # vi_engine = FBVariationalEngine(initial_model, model, obs_model, t_obs, observations, tspan, options=options)

# # # # solve forward
# # # start_time = time.time()
# # # sol = solve_ivp(odefun, tspan, initial_model)
# # # end_time = time.time()
# # # print(f'Plain solution required {end_time-start_time} seconds')

# # # test update
# # start_time = time.time()
# # vi_engine.backward_update()
# # end_time = time.time()
# # print(f'Backward update required {end_time-start_time} seconds')
# # start_time = time.time()
# # vi_engine.forward_update()
# # end_time = time.time()
# # print(f'Forward update required {end_time-start_time} seconds')

# # # def odefun(time, forward):
# # #     return(model.forward(time, forward, rates))

# # # create filer obs
# # filter_obs = FilterObsModel(model, obs_model_sim, t_obs, observations)

# # # set up forward backward engine fr comparison
# # fb_engine = ForwardBackward(initial_model, model_fb, filter_obs, t_obs, observations, subsample=100, tspan=tspan)

# # # run filter
# # start_time = time.time()
# # fb_engine.forward_update()
# # end_time = time.time()
# # print(f'Forward filter required {end_time-start_time} seconds')
# # start_time = time.time()
# # fb_engine.backward_update()
# # end_time = time.time()
# # print(f'Backward filter required {end_time-start_time} seconds')

# # # get smoothing data
# # t_smooth = fb_engine.get_time()
# # p_smooth = fb_engine.get_smoothed()
# # marginal_smooth = model.marginal_moments(p_smooth)
# # pol_smooth = marginal_smooth[:, 1:num_sites].sum(axis=1)
# # stem_smooth = marginal_smooth@alpha
# # intensity_smooth = obs_model.intensity(marginal_smooth, t_smooth, obs_param)

# # # get vi data
# # t_post = t_plot
# # states_post = vi_engine.get_forward(t_post)
# # marginal_post = model.marginal_moments(states_post)

# # # extract plotting stuff
# # pol_post = marginal_post[:, 1:num_sites].sum(axis=1)
# # pol_plot = states_plot[:, 1:].sum(axis=1)
# # stem_post = marginal_post@alpha
# # stem_plot = states_plot@alpha
# # intensity_post = obs_model.intensity(marginal_post, t_post, obs_param)
# # intensity_plot = obs_model.intensity(states_plot, t_plot, obs_param)

# # # plotting 
# # plt.subplot(2, 2, 1)
# # plt.plot(t_plot, states_plot[:, 0], '-r')
# # plt.plot(t_post, marginal_post[:, 0], '-m')
# # plt.plot(t_smooth, marginal_smooth[:, 0], '-b')
# # plt.ylabel('Gene Activity')

# # plt.subplot(2, 2, 2)
# # plt.plot(t_plot, pol_plot, '-r')
# # plt.plot(t_post, pol_post, '-m')
# # plt.plot(t_smooth, pol_smooth, '-b')
# # plt.ylabel('Polymerases')

# # plt.subplot(2, 2, 3)
# # plt.plot(t_plot, stem_plot, '-r')
# # plt.plot(t_post, stem_post, '-m')
# # plt.plot(t_smooth, stem_smooth, '-b')
# # plt.ylabel('Stemloops')

# # plt.subplot(2, 2, 4)
# # plt.plot(t_plot, intensity_plot, '-r')
# # plt.plot(t_post, intensity_post, '-m')
# # plt.plot(t_smooth, intensity_smooth, '-b')
# # plt.plot(t_obs, observations, 'kx')
# # plt.ylabel('Intensity')

# # plt.show()


# # backward = vi_engine.get_backward(t_post)

# # for i in range(num_sites):
# #     plt.subplot(5, 4, i+1) 
# #     plt.plot(t_post, backward[:, i], '-b')    
# #     plt.ylabel(f'X_{i}')
# #     #plt.ylim(0.0, 1.0)
# # plt.show()

