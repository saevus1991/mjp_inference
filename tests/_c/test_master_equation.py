import mjp_inference as mjpi
import numpy as np
import matplotlib.pyplot as plt
from time import time as stopwatch
from scipy.integrate import solve_ivp


np.random.seed(2209141017)

# set up model
model = mjpi.MJP("Simple gene expression model")
# add species
model.add_species(name='G0', default_value=1)
model.add_species(name='G1')
model.add_species(name='mRNA', upper=100)
model.add_species(name='Protein', upper=500)
# add events
model.add_event(mjpi.MassAction(name='Activation', reaction='1 G0 -> 1 G1', rate=0.001))
model.add_event(mjpi.Reaction(name='Deactivation', reaction='1 G1 -> 1 G0', rate=0.001, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.001, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein', rate=0.01, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='Protein Decay', reaction='1 Protein -> 0 Protein', rate=0.0009, propensity=lambda x: x[0]))
model.build()

# prepare simulation
initial = model.default_state
tspan = np.array([0.0, 100.0])
t_eval = np.linspace(tspan[0], tspan[1], 100)

# construct initial distribution
eps = 0.99
initial_dist = np.ones(model.num_states) * (1-eps) / (model.num_states-1)
ind = model.state2ind(initial)
initial_dist[ind] = eps

# set up master equation and run solver
me = mjpi.MasterEquation(model)
me.forward(0.0, initial_dist)

start_time = stopwatch()
sol = solve_ivp(me.forward, tspan, initial_dist, t_eval=t_eval)
end_time = stopwatch()
print(f'Solution required {end_time-start_time} seconds')

prob = sol['y'].T

# compute marginals and means
species = model.species_list
me_marginals = {}
for i, spec in enumerate(species):
    ind = tuple([j+1 for j in range(len(species)) if j != i])
    me_marginals[spec] = prob.reshape(prob.shape[0:1] + tuple(model.dims)).sum(axis=ind)
# compute means 
me_means = mjpi.marginals2means(model, me_marginals, species)

# plot
fig, axs = plt.subplots(3, 1)
species.pop(0)
for i, spec in enumerate(species):
    axs[i].plot(t_eval, me_means[spec], '-', color='tab:red', label='ME')
plt.show()