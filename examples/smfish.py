import numpy as np
import mjp_inference as mjpi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(2209141017)

# set up model
model = mjpi.MJP("Telegraph model")
# add species
model.add_species(name='G0')
model.add_species(name='G1')
model.add_species(name='mRNA', upper=200)
# add events
model.add_event(mjpi.MassAction(name='Activation', reaction='1 G0 -> 1 G1', rate=0.001))
model.add_event(mjpi.MassAction(name='Deactivation', reaction='1 G1 -> 1 G0', rate=0.001))
model.add_event(mjpi.MassAction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06))
model.add_event(mjpi.MassAction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.0011))
model.build()

# set initial conditions
tspan = np.array([0.0, 20000.0])
t_eval = np.linspace(tspan[0], tspan[1], 200)
initial = np.array([1.0, 0.0, 0.0])

# simulate
seed = np.random.randint(2**18)
simulator = mjpi.Simulator(model, initial, tspan, seed)
states = simulator.simulate(t_eval)

# # plot
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(t_eval/60, states[:, 1], '--r')
# axs[0].set_ylabel('Promoter')
# axs[1].plot(t_eval/60, states[:, 2], '--r')
# axs[1].set_ylabel('mRNA')
# axs[1].set_xlabel('Time (min)')
# plt.show()

# set up deterministic initial distribution
initial_dist = np.zeros(model.num_states)
ind = model.state2ind(initial)
initial_dist[ind] = 1.0

# solve dist
master_equation = mjpi.MasterEquation(model)
sol = solve_ivp(master_equation.forward, np.array([0.0, 50000.0]), initial_dist)
prob = sol['y'].T

# extract marinal
marginals = mjpi.eval_marginals(model, prob)

plt.plot(marginals['mRNA'][-1], '-r')
plt.show()