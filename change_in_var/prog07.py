import numpy as np
import pandas as pd
from change_in_var import prog06

"""
This is a program for detecting a 2-sided change in the variance by a SR procedure with
constant unknown mean, known initial s.d. with representative post-change variance where
eta1>1 and eta2<1

input:
======
    :x: data
    :sigma: s.d.
    :eta1:
    :eta2:
    :cutoff:
"""


# def two_sided_change_in_var_unknown_mean(x: pd.DataFrame, sigma: float, eta1: float, eta2: float, cutoff: float):
def prog07(x: pd.DataFrame, sigma: float, eta1: float, eta2: float, cutoff: float):
    z = ( x - x[0] ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )
    r1 = prog06.calculate_r(z, t2, length, eta1, N)
    r2 = prog06.calculate_r(z, t2, length, eta2, N)
    r = ( r1 + r2 ) / 2
    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
