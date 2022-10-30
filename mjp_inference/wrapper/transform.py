from typing import Callable, Any
from mjp_inference._c.mjp_inference import Transform as _Transform
from numba import cfunc, types
from scipy import LowLevelCallable
from mjp_inference.wrapper.compile import parse_function


__all__ = ['TransformFun', 'TransformGrad', 'Transform']


# function signatures
TransformFun = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))
TransformGrad = types.void(types.double, types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double), types.CPointer(types.double))


class Transform(_Transform):

    def __init__(self, name: str, transform: Callable, output_dim: int=1, transform_grad: Callable=None):
        # parse callables
        transform_callable = parse_function(transform, TransformFun)
        transform_grad_callable = parse_function(transform_grad, TransformGrad)
        # init c++ class
        _Transform.__init__(self, name, transform_callable=transform_callable, output_dim=output_dim, grad_callable=transform_grad_callable)

