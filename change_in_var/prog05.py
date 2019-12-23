import pandas as pd
import numpy as np
from change_in_var import prog04, prog03

"""
This is a program for detecting a change in the variance by a SR procedure with
constant known mean, known initial s.d. and unknown post-change variance, with prior on 1/eta2
that is a 50-50 mixture of Gamma(alpha1,beta1) conditional on being <1 and Gamma(alpha2,beta2) conditional on being>1

input:
======
    :x: data
    :mu: mean
    :sigma: s.d.
    :alpha1:
    :beta1:
    :alpha2:
    :beta2:
    :cutoff:
"""

# def change_in_var_w_mix_50_50_known_mean(x: pd.DataFrame, mu: float, sigma: float, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float):
def prog05(x: pd.DataFrame, mu: float, sigma: float, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float):
    z = ( x - mu ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r1 = prog03.calculate_r(z, t2, length, alpha1, beta1, N)
    r2 = prog04.calculate_r(z, t2, length, alpha2, beta2, N)
    r = ( r1 + r2 ) / 2

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
