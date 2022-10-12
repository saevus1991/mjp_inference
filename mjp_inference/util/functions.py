import numpy as np


def falling_factorial(n, k):
    if (n < k):
        return(0)
    else:
        res = 1.0
        for i in range(int(k)):
            res *= n-i
        return(res)