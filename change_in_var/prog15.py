import numpy as np
import pandas as pd
from scipy.stats import gamma
from change_in_var import prog13

"""
This is a program for detecting a 2-sided change in the variance with unknown baseline s.d. and known baseline mean,
by a mixture of discretized conditional Gamma.

input:
======
    :x: data
    :mu: mean
    :alpha1:
    :beta1:
    :alpha2:
    :beta2:
    :cutoff:
"""


# def two_sided_change_in_var_unknown_baseline_variance(x: pd.DataFrame, mu: float, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float, low=0.01, high=1.00, step=0.01):
def prog15(x: pd.DataFrame, mu: float, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float, low=0.01, high=1.00, step=0.01):
    # k1 = np.arange(low, high, step)
    k1 = np.arange(0.01, 1.00, 0.01)
    g1 = ( beta1**alpha1 ) / gamma(alpha1) * (k1**(alpha1 - 1))*np.exp(-beta1*k1)
    g1 = g1 / np.sum(g1)

    k2 = np.arange(low, high, step)
    g2 = ( beta2**alpha2 ) / gamma(alpha2) * (k2**(alpha2 - 1))*np.exp(-beta2*k2)
    g2 = g2 / np.sum(g2)

    g = np.c_[g1, g2]
    k = np.c_[k1, k2]
    _, m = k1.shape

    z = ( x - mu ) / (x[0] - mu)
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r = g * prog13.calculate_r(z, t2, length, N, N1, k, m)

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH