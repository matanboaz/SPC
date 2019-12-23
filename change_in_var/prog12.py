import pandas as pd
import numpy as np
from change_in_var import prog11

"""
This is a program for detecting a 2-sided change in the variance with known baseline mean but unknown baseline variance.
representatives eta1 and eta2 for the ratio of s.d.

input:
======
    :x: data
    :mu: mean
    :eta1: representative ratio of s.d.
    :eta2: representative ratio of s.d.
    :cutoff:
"""


# def two_sided_change_in_var_unknown_baseline_variance(x: pd.DataFrame, mu: float, eta1: float, eta2: float, cutoff: float):
def prog12(x: pd.DataFrame, mu: float, eta1: float, eta2: float, cutoff: float):
    z = ( x - mu ) / ( x[0] - mu )
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r1 = prog11.calculate_r(z, t2, length, eta1, N, N1)
    r2 = prog11.calculate_r(z, t2, length, eta2, N, N1)
    r = (r1 + r2)/2

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
