import numpy as np
import mjp_inference as mjpi
# import torch
# from pathlib import Path
# from transcription.compiled import transcription
# import matplotlib.pyplot as plt
# from pymbvi.util import num_derivative
# from transcription.autograd.transition import markov_transition
# import time as stopwatch

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
# time = 100.0
rates = model.rate_array

# start_time = stopwatch.time()

# solve forward
t_eval = np.arange(3.0, 100.0, 15.0)
master_equation = mjpi.MEInference(model)
print("set up master equation")
solver = mjpi.KrylovSolver(master_equation, initial_dist, rates, t_eval)
print("set sup solver")
print(solver)
states = solver.forward()
print(states.shape, np.sum(states, axis=1))

# # compute gradients
# out = np.random.rand(len(t_eval), num_states)
# solver.backward(5, out)
# solver.compute_rates_grad()
# rates_grad = solver.get_rates_grad()
# initial_grad = solver.get_initial_grad()
# print(rates_grad)
# print(initial_grad.shape)


# # numerical rates gradient
# def fun(x):
#     solver = transcription.KrylovSolver(model, initial_dist, x, t_eval)
#     forward = solver.forward()
#     return(np.sum(out * forward))
# rates_grad_num = num_derivative(fun, rates, 1e-5)
# print(rates_grad_num)

# ind = np.random.choice(np.arange(num_states), 5)

# # numerical initial gradient 
# def fun(x):
#     initial = initial_dist.copy()
#     initial[ind] = x
#     solver = transcription.KrylovSolver(model, initial, rates, t_eval)
#     forward = solver.forward()
#     return(np.sum(out * forward))
# initial_grad_num = num_derivative(fun, initial_dist[ind])
# print(initial_grad[ind])
# print(initial_grad_num)
# # check = np.linalg.norm(initial_grad_num-initial_grad)
# # print('initial grad check', check)

# end_time = stopwatch.time()
# print(f'Required {end_time-start_time} seconds')