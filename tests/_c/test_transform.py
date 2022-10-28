import mjp_inference as mjpi
import numpy as np
from numba import cfunc, types
from scipy import LowLevelCallable

np.random.seed(2210281351)

# prepare
time = 10.0
input_dim = 20
output_dim = 2
param_dim = 5
state = np.arange(input_dim)
param = np.random.rand(param_dim)

# function
def transform_fun(time, state, param, transformed):
    state_sum = 0.0
    for i in range(input_dim):
        state_sum += state[i]
    transformed[0] = np.exp(-param[0]*state_sum)
    tmp = 0.0
    for i in range(param_dim):
        tmp += param[i]**2 
    transformed[1] = state_sum * tmp

def grad_state(time, state, param, grad_output, grad):
    state_sum = 0.0
    for i in range(input_dim):
        state_sum += state[i]
    transformed[0] = np.exp(-param[0]*state_sum)
    tmp = 0.0
    for i in range(param_dim):
        tmp += param[i]**2 
    transformed[1] = state_sum * tmp


# set up transformation
sig = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))
transform_cfunc = cfunc(sig)(transform_fun)
transform_callable = LowLevelCallable(transform_cfunc.ctypes)
transform = mjpi.Transform("test", transform_callable, output_dim)
test = transform.transform(time, state, param)
print(test)