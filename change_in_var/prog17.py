import numpy as np
import pandas as pd
from change_in_var import prog16

"""
This is a program for detecting a 2-sided change in the variance with unknown baseline s.d. and unknown baseline mean.
Representatives eta1 and eta2 for the ratios of s.d.

input:
======
    :x: data
    :eta1: ratio of s.d.
    :eta2: ratio of s.d.
    :cutoff:
"""


# def one_sided_change_in_var_unknown_baseline_var_unknown_baseline_mean(x: pd.DataFrame, eta1: float, eta2: float, cutoff: float):
def prog17(x: pd.DataFrame, eta1: float, eta2: float, cutoff: float):
    z = ( x - x[0] ) / ( x[1] - x[0] )
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    # r1 = prog06.calculate_r(z, t2, length, eta1, N)
    # r2 = prog06.calculate_r(z, t2, length, eta2, N)
    r1 = prog16.calculate_r(z, t2, length, eta1, N, N1, N2, m)
    r2 = prog16.calculate_r(z, t2, length, eta2, N, N1, N2, m)
    r = (r1 + r2) / 2

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
