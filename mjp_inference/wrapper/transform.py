from typing import Callable, Any
from mjp_inference._c.mjp_inference import Transform as _Transform
# import numpy as np
# import re
# import mjp_inference.util.functions as func
from numba import cfunc, types
from scipy import LowLevelCallable

__all__ = ['TransformFun', 'Transform']


# function signatures
TransformFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))


class Transform(_Transform):

    def __init__(self, name, transform, output_dim=1):
        transform_callable = self.parse_transform(transform)
        _Transform.__init__(self, name, transform_callable=transform_callable, output_dim=output_dim, grad_state_callable=(), grad_param_callable=())

    def parse_transform(self, transform):
        transform_compiled = cfunc(TransformFun)(transform)
        return(LowLevelCallable(transform_compiled.ctypes))
