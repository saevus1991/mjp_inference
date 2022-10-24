import os
import numpy as np
# import torch
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# from pymbvi.models.observation.tasep_obs_model import LognormGauss
# from pymbvi.util import num_derivative
from time import time as stopwatch
# import seaborn as sns
from scipy.linalg import expm
import scipy.sparse as sp
import mjp_inference as mjpi


class Krylov():

    def __init__(self, mat, vec, order=1):
        self.mat = mat
        self.norm = np.linalg.norm(vec)
        self.q = vec / self.norm
        self.order = order
        self.span, self.proj = self.build(order)

    def build(self, order):
        # preparations
        span = np.zeros((order, len(self.q)))
        proj = np.zeros((order+1, order))
        # iterate
        for i in range(order):
            span[i] = self.q
            v = self.mat @ self.q
            for j in range(i+1):
                proj[j, i] = v @ span[j]
                v = v - proj[j, i] * span[j]
            proj[i+1, i] = np.linalg.norm(v)
            self.q = v / proj[i+1, i]
        # return basis and matrix
        return(span, proj)

    def expand(self, inc=1):
        # preparations
        new_order = self.order + inc
        span = np.zeros((new_order, len(self.q)))
        proj = np.zeros((new_order+1, new_order))
        # reuse existing stuff
        span[:self.order] = self.span
        proj[:self.order+1, :self.order] = self.proj
        # iterate 
        for i in range(self.order, new_order):
            span[i] = self.q
            v = self.mat @ self.q
            for j in range(i+1):
                proj[j, i] = v @ span[j]
                v = v - proj[j, i] * span[j]
            proj[i+1, i] = np.linalg.norm(v)
            self.q = v / proj[i+1, i]
        # update stored values
        self.order = new_order
        self.span = span
        self.proj = proj
        return

    def eval(self, t):
        # preparations
        initial = np.zeros(self.order+1)
        initial[0] = self.norm
        # mat_proj = self.proj[:-1] * t
        H = np.zeros((self.order+1, self.order+1))
        H[:, :-1] = self.proj[:, :] * t
        H[-1, -1] = 1.0
        # compute otuput
        res = expm(H) @ initial
        output = self.span.T @ res[:-1]
        err = res[-1]
        print("Error:", err)
        return(output)


np.random.seed(2103110956)
# torch.manual_seed(2103170804)
# torch.set_default_tensor_type(torch.DoubleTensor)

# # set up model
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

# get initial value
seed = np.random.randint(2**16)
initial = np.zeros(num_sites)
simulator = mjpi.Simulator(model, initial, np.array([0.0, 1000.0]), seed)
trajectory = simulator.simulate()
initial = trajectory['states'][-1]

# create initial distribution
ind = model.state2ind(initial)
initial_model = np.zeros(model.num_states)
initial_model[ind] = 1.0
time = 100.0

# get generator
me = mjpi.MasterEquation(model)
generator = me.generator.T

# set up python krylov
order = 120
start_time = stopwatch()
krylov = Krylov(generator, initial_model, order)
res1 = krylov.eval(time)
end_time = stopwatch()
print(f'solve via krylov required {end_time-start_time} seconds')

start_time = stopwatch()
krylov = mjpi.Krylov(generator, initial_model, order)
res2 = krylov.eval(time)
end_time = stopwatch()
print(f'solve via krylov required {end_time-start_time} seconds')

res3 = sp.linalg.expm(generator) @ initial_model

check = np.linalg.norm(res1-res3) + np.linalg.norm(res2-res3)
print(check)

start_time = stopwatch()
krylov = mjpi.Krylov(generator, initial_model, 100)
krylov.expand(20)
res4 = krylov.eval(time)
end_time = stopwatch()
print(f'solve with expansion required {end_time-start_time} seconds')