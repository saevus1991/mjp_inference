import mjp_inference as mjpi
from numba import cfunc, types, carray
from scipy import LowLevelCallable
import numpy as np


# set up species
mrna = mjpi.Species(name='mRNA', lower=0, upper=100, default=0)
species_dict = {'mRNA': mrna}

# define a hazard function
forward_sig = types.double(types.CPointer(types.double))
c1 = 0.2

def const_func(x):
    return(1.0)

def bimolecular(x):
    return(x[0]*x[1])


# set up event
birth = mjpi.Event("birth", input_species=['mRNA'], output_species=['mRNA'], propensity=const_func, change_vec=[1])

print(birth.name)
print(birth.input_species)
print(birth.output_species)
print(birth.change_vec)

# set up species
a = mjpi.Species(name='A', lower=0, upper=100, default=0)
b = mjpi.Species(name='B', lower=0, upper=100, default=0)
x = mjpi.Species(name='X', lower=0, upper=100, default=0)
y = mjpi.Species(name='Y', lower=0, upper=100, default=0)
species_dict = {'A': a, 'B': b, 'X': x, 'Y': y}
try:
    conversion = mjpi.Event("conversion", input_species=['A', 'B'], output_species=['A', 'X', 'B', 'Y'], propensity=bimolecular, change_vec=[-1, -1, 1, 1])
except ValueError:
    print("Captured wrong output species order")

# test more complicated event
conversion = mjpi.Event("conversion", input_species=['A', 'B'], output_species=['A', 'B', 'X', 'Y'], rate=0.5, propensity=bimolecular, change_vec=[-1, -1, 1, 1])

state = np.array([4, 5])
haz = conversion.hazard(state)
print("test full hazard", haz)

# test reaction formalism
conversion = mjpi.Reaction("conversion", reaction='1 A + 1 B -> 1 X + 1 Y', rate=0.5, propensity=bimolecular)

state = np.array([4, 5])
haz = conversion.hazard(state)
prop = conversion.propensity(state)
print("hazard", haz, "propensity", prop)

# test mass action
conversion = mjpi.MassAction("conversion", reaction='1 A + 1 B -> 1 X + 1 Y', rate=0.5)

state = np.array([4, 5])
haz = conversion.hazard(state)
prop = conversion.propensity(state)
print("hazard", haz, "propensity", prop)
