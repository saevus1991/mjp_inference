from typing import Callable, Any
from numba import cfunc, types
from numba.core.typing.templates import Signature
from scipy import LowLevelCallable


def parse_function(function: Callable, sig: Signature):
    """
    Convert a python function to a c function pointer via numba and wrap as scipy LowLevelCallable
    """
    if function is not None:
        compiled = cfunc(sig)(function)
        return(LowLevelCallable(compiled.ctypes))
    else:
        return(())