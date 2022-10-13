import mjp_inference as mjpi
import numpy as np
import matplotlib.pyplot as plt
from time import time as stopwatch

# parameters
num_sites = 20
k_init = 0.1
k_elong = 0.5
k_termin = 0.05
# set up model
model = mjpi.MJP("Tasep")
# add species
for i in range(num_sites):
    model.add_species(name=f'X_{i}')
# add init
model.add_event(mjpi.Event(name='Initiation', input_species=['X_0'], output_species=['X_0'] , propensity=lambda x: k_init*(1-x[0]), change_vec=[1]))
# add add hops
for i in range(1, num_sites):
    input_species = [f'X_{i-1}', f'X_{i}']
    model.add_event(mjpi.Event(name=f'Hop {i}', input_species=input_species, output_species=input_species, propensity=lambda x: k_elong*x[0]*(1-x[1]), change_vec=[-1, 1]))
# add termination
model.add_event(mjpi.Event(name='Termination', input_species=[f'X_{num_sites-1}'], output_species=[f'X_{num_sites-1}'], propensity=lambda x: k_termin*x[0], change_vec=[-1]))
model.build()

# prepare simulation
initial = model.default_state
tspan = np.array([0.0, 500.0])
t_eval = np.linspace(tspan[0], tspan[1], 100)
seed = np.random.randint(2**16)

# set up simulator
simulator = mjpi.Simulator(model, initial, tspan, seed)

# get trajectory
num_samples = 1000
start_time = stopwatch()
states = np.zeros((num_samples, (len(t_eval)), model.num_species))
for i in range(num_samples):
    states[i] = simulator.simulate(t_eval)
end_time = stopwatch()
print(f'Required {end_time-start_time} seconds to simulate {num_samples} trajectories')

# plot
states_mean = states.mean(axis=0)
fig, ax = plt.subplots()
plt.imshow(states_mean.T, extent=[tspan[0], tspan[1], model.num_species-1, 0], aspect='auto', cmap='afmhot_r')
ax.invert_yaxis()
ax.set_ylabel('Sites')
ax.set_xlabel('Time (s)')
plt.colorbar()
plt.show()