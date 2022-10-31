import numpy as np


def falling_factorial(n, k):
    if (n < k):
        return(0)
    else:
        res = 1.0
        for i in range(int(k)):
            res *= n-i
        return(res)


def is_equal(x, y, size):
    equal = True
    for i in range(size):
        if np.abs(x[i]-y[i]) > 1e-12:
            equal = False
            break
    return(equal)