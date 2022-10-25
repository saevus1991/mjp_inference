import numpy as np
from mjp_inference.util.diff import num_derivative
import mjp_inference as mjpi

np.random.seed(2201251626)

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

# create initial distribution
initial = np.zeros(num_sites)
ind = model.state2ind(initial)
initial_dist = np.zeros(model.num_states)
initial_dist[ind] = 1.0
time = 100.0
rates = model.rate_array

# krylov propagator with backward
master_equation = mjpi.MEInference(model)
propagator = mjpi.KrylovPropagator(master_equation, initial_dist, rates, time)
propagator.propagate()
out = np.random.rand(model.num_states)
propagator.backward(out)
propagator.compute_rates_grad()
rates_grad = propagator.get_rates_grad()
initial_grad = propagator.get_initial_grad()
time_grad = propagator.get_time_grad()

# numerical rates gradient
def fun(x):
    propagator = mjpi.KrylovPropagator(master_equation, initial_dist, x, time)
    forward = propagator.propagate()
    return(np.sum(out * forward, keepdims=True))
rates_grad_num = num_derivative(fun, rates, 1e-5)

print(rates_grad)
print(rates_grad_num)

# numerical initial gradient 
ind = np.random.choice(np.arange(model.num_states), 5)
def fun(x):
    initial = initial_dist.copy()
    initial[ind] = x
    propagator = mjpi.KrylovPropagator(master_equation, initial, rates, time)
    forward = propagator.propagate()
    return(np.sum(out * forward, keepdims=True))
initial_grad_num = num_derivative(fun, initial_dist[ind])

print(initial_grad[ind])
print(initial_grad_num)

# check time gradient
def fun(x):
    propagator = mjpi.KrylovPropagator(master_equation, initial_dist, rates, x[0])
    forward = propagator.propagate()
    return(np.sum(out * forward, keepdims=True))
time_grad_num = num_derivative(fun, np.array([time]), 1e-5)
print(time_grad)
print(time_grad_num)
