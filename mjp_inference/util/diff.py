import numpy as np


def num_derivative(fun, x, h=1e-6):
    """
    Compute jacobian of fun at value x with symmetric finite differences
    """
    # set up output
    m = len(x)
    y = fun(x)
    if isinstance(y, np.ndarray):
        n = len(fun(x))
    elif isinstance(y, np.generic):
        n = 1
    else:
        raise ValueError('Unsupported output type {} of fun'.format(type(y)))
    jacobian = np.zeros((n, m))
    # evaluate
    for i in range(m):
        x_up = x.copy()
        x_up[i] += h
        x_low = x.copy()
        x_low[i] -= h
        jacobian[:, i] = fun(x_up)-fun(x_low)
    return(jacobian/(2*h))