import numpy as np
import pandas as pd
from change_in_var import prog01

"""
This is a program for detecting a 2-sided change in the variance by a SR procedure 
with constant known mean, known initial s.d. with representatives for post-change variance,
where eta1>1 and eta2<1

input:
======
    :x: data
    :mu: mean
    :sigma: s.d.
    :eta1: eta1>1
    :eta2: eta2<1
    :cutoff:
"""


# def two_sided_change_in_var_known_mean(x: pd.DataFrame, mu: float, sigma: float, eta1: float, eta2: float, cutoff: float):
def prog02(x: pd.DataFrame, mu: float, sigma: float, eta1: float, eta2: float, cutoff: float):

    z = ( x - mu ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r1 = prog01.calculate_r(z, t2, length, eta1, N)
    r2 = prog01.calculate_r(z, t2, length, eta2, N)
    r = ( r1 + r2 ) / 2

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
