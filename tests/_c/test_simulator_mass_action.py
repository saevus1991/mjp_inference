import mjp_inference as mjpi
import numpy as np
import matplotlib.pyplot as plt
from time import time as stopwatch


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
# model.add_event(mjpi.Reaction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06, propensity=lambda x: x[0]))
model.add_event(mjpi.MassAction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06))
model.add_event(mjpi.Reaction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.001, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein', rate=0.01, propensity=lambda x: x[0]))
model.add_event(mjpi.Reaction(name='Protein Decay', reaction='1 Protein -> 0 Protein', rate=0.0009, propensity=lambda x: x[0]))
model.build()

# prepare simulation
initial = model.default_state
tspan = np.array([0.0, 10000.0])
t_eval = np.linspace(tspan[0], tspan[1], 100)
seed = np.random.randint(2**16)

# set up simulator
simulator = mjpi.Simulator(model, initial, tspan, seed)

# get trajectory
num_samples = 1000
start_time = stopwatch()
for i in range(num_samples):
    states = simulator.simulate(t_eval)
end_time = stopwatch()
print(f'Required {end_time-start_time} seconds to simulate {num_samples} trajectories')

# plot
fig, axs = plt.subplots(3, 1)
for i in range(3):
    axs[i].plot(t_eval, states[:, i+1], '-r')
plt.show()

