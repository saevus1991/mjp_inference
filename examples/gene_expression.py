import mjp_inference as mjpi
import numpy as np
import matplotlib.pyplot as plt
from mjp_inference.util.conv import discretize_trajectory
from pathlib import Path
import torch

np.random.seed(2209141018)

# set up model
model = mjpi.MJP("Simple gene expression model")
# add species
model.add_species(name='G0', default_value=1)
model.add_species(name='G1')
model.add_species(name='mRNA', upper=100)
model.add_species(name='Protein', upper=500)
# add events
model.add_event(mjpi.MassAction(name='Activation', reaction='1 G0 -> 1 G1', rate=0.001))
model.add_event(mjpi.MassAction(name='Deactivation', reaction='1 G1 -> 1 G0', rate=0.001))
model.add_event(mjpi.MassAction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06))
model.add_event(mjpi.MassAction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.001))
model.add_event(mjpi.MassAction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein', rate=0.01))
model.add_event(mjpi.MassAction(name='Protein Decay', reaction='1 Protein -> 0 Protein', rate=0.0009))
model.build()

# prepare simulation
initial = model.default_state
tspan = np.array([0.0, 10000.0])
t_eval = np.linspace(tspan[0], tspan[1], 100)
seed = np.random.randint(2**16)

# set up simulator
simulator = mjpi.Simulator(model, initial, tspan, seed)
trajectory = simulator.simulate()
states = mjpi.discretize_trajectory(trajectory, t_eval)

# set up obs model   
sigma = 25.0
obs_model = mjpi.NormalObs(model, sigma, observed_species='Protein')

# discrete observations
t_obs = np.arange(200, tspan[1], 300)
obs = discretize_trajectory(trajectory, t_obs, obs_model)

# plot
ylabels = ['G1', 'mRNA', 'Protein']
fig, axs = plt.subplots(3, 1)
for i in range(3):
    axs[i].plot(t_eval, states[:, i+1], '--r')
    axs[i].set_ylabel(ylabels[i])
axs[2].plot(t_obs, obs, 'xk')
axs[2].set_xlabel('Time (s)')
plt.show()

# set up deterministic initial distribution
initial_dist = np.zeros(model.num_states)
ind = model.state2ind(initial)
initial_dist[ind] = 1.0

# set up master equation
obs_param = obs_model.param_array
rates = model.rate_array
master_equation = mjpi.MEInference(model)

save_path = Path(__file__).with_suffix('.pt')
if not save_path.exists():
    # set up filter
    filt = mjpi.KrylovBackwardFilter(master_equation, obs_model, t_obs, obs, initial_dist, rates, obs_param, tspan)
    filt.forward_filter()
    filt.backward_filter()
    log_prob = filt.log_prob()
    # extract stuff
    prob_filt = filt.eval_forward_filter(t_eval)
    prob_smoothed = filt.eval_smoothed(t_eval)
    marginals_filt = mjpi.eval_marginals(model, prob_filt)
    marginals_smoothed = mjpi.eval_marginals(model, prob_smoothed)
    means_filt = mjpi.marginals2means(model, marginals_filt)
    means_smoothed = mjpi.marginals2means(model, marginals_smoothed)
    # save
    data = {'marginals_filt': marginals_filt, 'marginals_smoothed': marginals_smoothed, 'means_filt': means_filt, 'means_smoothed': means_smoothed}
    torch.save(data, save_path)
    print(log_prob)
else:
    data = torch.load(save_path)

# # plot
# species = ['G1', 'mRNA', 'Protein']
# fig, axs = plt.subplots(3, 1)
# for i in range(3):
#     axs[i].plot(t_eval, states[:, i+1], '--k')
#     axs[i].plot(t_eval, data['means_filt'][species[i]], '-b')
#     axs[i].plot(t_eval, data['means_smoothed'][species[i]], '-r')
#     axs[i].set_ylabel(species[i])
# axs[2].plot(t_obs, obs, 'xk')
axs[2].set_xlabel('Time (s)')
# plt.show()
