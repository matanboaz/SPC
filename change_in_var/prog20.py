from change_in_var import prog16
import pandas as pd
import numpy as np
from scipy.stats import gamma


"""
This is a program for detecting a 2-sided change in the variance with unknwon baseline s.d. and unknown baseline mean.
Mixture of two discretized conditional Gamma.

input:
======
    :x: data
    :alpha1:
    :beta1:
    :alpha2:
    :beta2:
    :cutoff:
"""


# def two_sided_change_in_var_unknown_baseline_var_unknown_baseline_mean(x: pd.DataFrame, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float, low=0.01, high=1.00, step=0.01):
def prog20(x: pd.DataFrame, alpha1: float, beta1: float, alpha2: float, beta2: float, cutoff: float, low=0.01, high=1.00, step=0.01):
    k1 = np.arange(low, high, step)
    # k1 = np.arange(0.01, 1.00, 0.01)
    g1 = ( beta1**alpha1 ) / gamma(alpha1) * (k1**(alpha1 - 1))*np.exp(-beta1*k1)
    g1 = g1 / np.sum(g1)

    k2 = np.arange(1.1, 40.1, 0.1)
    g2 = (beta2 ** alpha2) / gamma(alpha2) * (k2 ** (alpha2 - 1)) * np.exp(-beta2 * k2)
    g2 = g2 / np.sum(g2)

    _, m = k1.shape

    z = ( x - x[0] ) / ( x[1] - x[0] )
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r1 = g1 * prog16.calculate_r(z, t2, length, alpha1, beta1, N, N1, N2, m, k1)
    r2 = g2 * prog16.calculate_r(z, t2, length, alpha2, beta2, N, N1, N2, m, k2)
    r = (r1 + r2) / 2
    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
