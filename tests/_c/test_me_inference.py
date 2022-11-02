import numpy as np
import mjp_inference as mjpi
from scipy.integrate import solve_ivp
from time import time as stopwatch
import scipy.sparse.linalg as sla

# parameters
num_sites = 10
k_init = mjpi.Rate('k_init', 0.1)
k_elong = mjpi.Rate('k_elong', 0.5)
k_termin = mjpi.Rate('k_termin', 0.05)
# set upmodel
model = mjpi.MJP("Tasep")
# add species
for i in range(num_sites):
    model.add_species(name=f'X_{i}')
# add rates
model.add_rate('k_init', 0.1)
model.add_rate('k_elong', 0.5)
model.add_rate('k_termin', 0.05)
# add init
model.add_event(mjpi.Event(name='Initiation', input_species=['X_0'], output_species=['X_0'] , rate='k_init', propensity=lambda x: (1-x[0]), change_vec=[1]))
# add add hops
for i in range(1, num_sites):
    input_species = [f'X_{i-1}', f'X_{i}']
    model.add_event(mjpi.Event(name=f'Hop {i}', input_species=input_species, output_species=input_species, rate='k_elong', propensity=lambda x: x[0]*(1-x[1]), change_vec=[-1, 1]))
# add termination
model.add_event(mjpi.Event(name='Termination', input_species=[f'X_{num_sites-1}'], output_species=[f'X_{num_sites-1}'], rate='k_termin', propensity=lambda x: x[0], change_vec=[-1]))
model.build()

# prepare simulation
initial = model.default_state
tspan = np.array([0.0, 100.0])
t_eval = np.linspace(tspan[0], tspan[1], 100)


# set up inference engine
rates = model.rate_array
me = mjpi.MEInference(model)

# make initial dist
initial_dist = np.zeros(model.num_states)
ind = model.state2ind(initial)
initial_dist[ind] = 1.0

param_generators = me.param_generators
generator1 = me.generator
generator2 = rates[0] * param_generators[0]
for i in range(1, len(param_generators)):
    generator2 += rates[i] * param_generators[i]
check = sla.norm(generator1-generator2)
print("param gerator check", check)

# start_time = stopwatch()
# sol = solve_ivp(me.forward, tspan, initial_dist, t_eval=t_eval)
# end_time = stopwatch()
# print(f'Solution required {end_time-start_time} seconds')