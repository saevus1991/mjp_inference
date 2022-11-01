import numpy as np
import torch
import mjp_inference as mjpi
import mjp_inference.wrapper.torch as mjpt
from mjp_inference.util.diff import num_derivative
from time import time as stopwatch

# np.random.seed(2201251626)
torch.set_default_tensor_type(torch.DoubleTensor)

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

# set up master equation
time = 100.0
rates = model.rate_array
master_equation = mjpi.MEInference(model)

# set up initial
initial = np.zeros(num_sites)
ind = model.state2ind(initial)
initial_dist = np.zeros(model.num_states)
initial_dist[ind] = 1.0
initial_dist = mjpi.KrylovPropagator(master_equation, initial_dist, rates, 20.0).propagate()

# test torch propagator module
start_time = stopwatch()
initial_torch = torch.from_numpy(initial_dist)
rates_torch = torch.from_numpy(rates)
rates_torch.requires_grad = True
time_torch = torch.tensor([time])
time_torch.requires_grad = True
out = torch.rand(model.num_states)
res = mjpt.markov_transition(initial_torch, rates_torch, time_torch, master_equation)
res.backward(out)
rates_grad = rates_torch.grad.detach().numpy()
time_grad = time_torch.grad.detach().numpy()
end_time = stopwatch()

print(f'Required {end_time-start_time} seconds')

# compute rates gradients numerically
def fun(x):
    res = (out * mjpt.markov_transition(initial_torch.detach(), torch.from_numpy(x), time_torch.detach(), master_equation)).numpy().sum(keepdims=True)
    return(res)
rates_grad_num = num_derivative(fun, rates, 1e-5).squeeze()
print(rates_grad)
print(rates_grad_num)
