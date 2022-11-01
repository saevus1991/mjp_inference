import numpy as np
import mjp_inference as mjpi
import matplotlib.pyplot as plt

# fix seed
np.random.seed(2210310820)

# set up model
model = mjpi.MJP(name='birth death')
# add species
model.add_species(name='Protein')
# add an event
model.add_event(mjpi.Event(name='birth', input_species=['Protein'], output_species=['Protein'], rate=5.0, propensity=lambda x: 1, change_vec=[1]))
model.add_event(mjpi.Event(name='death', input_species=['Protein'], output_species=['Protein'], rate=0.1, propensity=lambda x: x[0], change_vec=[-1]))
# # model.add_event(mjpi.Reaction(name='death', reaction='1 Protein -> 0 Protein', rate=0.1, propensity=lambda x: x[0]))
# # model.add_event(mjpi.Reaction(name='death', reaction='1 Protein -> 0 Protein', rate=0.1, propensity=lambda x: x[0]))
model.add_event(mjpi.MassAction(name='death', reaction='1 Protein -> 0 Protein', rate=0.1))
model.build()

# set initial conditions
tspan = np.array([0.0, 100.0])
t_eval = np.linspace(tspan[0], tspan[1], 200)
initial = np.array([0.0])

# simulate
seed = np.random.randint(2**18)
simulator = mjpi.Simulator(model, initial, tspan, seed)
states = simulator.simulate(t_eval)

# plot
plt.plot(t_eval, states, '--r')
plt.show()
