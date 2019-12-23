import pandas as pd
import numpy as np
from scipy.special import gammaln, gammainc


"""
This is a pgorgram for detecting a 1-sided change in the variance with known baseline mean but unknown
baseline variance.
Representative s.d. ratio=eta

input:
======
    :x: data
    :mu: mean
    :eta: representative s.d. ratio
    :cutoff:
"""


def calculate_r(z, t2, length, eta, N, N1):
    x1 = np.power(z, 2)
    x1_ = np.cumsum(x1, axis=1)
    x1 = np.outer(x1_, t2)
    x2 = x1.transpose()
    x3 = np.zeros(length)
    x3 = np.c_[x3, x2[:, :-1]]
    x3 = np.tril( x3 )
    x4 = ( x1 - x3 )
    x5 = (1 / eta**2) * x4 + x3

    lmbda = np.tril(np.exp(0.5 * N1 * np.log(x1) - 0.5 * N1 * np.log(x5) - N * np.log(eta)))
    r = np.sum(lmbda.T, axis=0)
    return r

# def one_sided_change_in_var_unknown_baseline_variance(x: pd.DataFrame, mu: float, eta: float, cutoff: float):
def prog11(x: pd.DataFrame, mu: float, eta: float, cutoff: float):
    z = ( x - mu ) / ( x[0] - mu )
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r = calculate_r(z, t2, length, eta, N, N1)

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
