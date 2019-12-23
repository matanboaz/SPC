import numpy as np
import pandas as pd
from scipy.stats import gamma


"""
This is a program for detecting an increase in the variance with known baseline s.d., unknown baseline mean,
by a mixture of discretized conditional Gamma.

input:
======
    :x: data
    :sigma: s.d.
    :alpha:
    :beta:
    :cutoff:
"""


def calculate_r(z, t2, length, alpha, beta, N, N1, N2, m, k1):
    R = np.zeros(m, length)
    for i in range(m):
        eta = 1 / np.sqrt(k1[m])
        x1 = np.power(z, 2)
        x1_ = np.cumsum(x1, axis=1)
        x1 = np.outer(x1_, t2)
        x2 = x1.transpose()
        x3 = np.zeros(length)
        x3 = np.c_[x3, x2[:, :-1]]
        x3 = np.tril( x3 )
        x4 = ( x1 - x3 ) / 2
        x5 = (1 - 1 / eta**2) * x4 - N * np.log(eta)

        y1 = np.cumsum(z, axis=1)
        y1 = np.outer(y1, t2)
        y2 = y1.transpose()
        y3 = np.zeros(length)
        y3 = np.c_[y3, y2[:, :-1]]
        y4 = ( y1 - y3 ) / eta**2

        w1 = 0.5 * ( (y3 + y4)**2 ) / ( N2 - 1 + N / eta**2 )
        w2 = 0.5 * ( ( ( (np.cumsum(z))**2 ) / np.arange(1, length+1 ) )  )* t2
        w3 = 0.5 * np.log(N2 - 1 + N / eta**2)
        w4 = 0.5 * np.log(N1)
        lmbda = np.tril( np.exp( w1 - w2 + x5 - w3 + w4 ) )
        o = np.sum(lmbda.T, axis=0)
        R[i, :] = o

    return R


# def increase_in_var_unknown_mean(x: pd.DataFrame, sigma: float, alpha: float, beta: float, cutoff: float, low=0.01, high=1.00, step=0.01):
def prog08(x: pd.DataFrame, sigma: float, alpha: float, beta: float, cutoff: float, low=0.01, high=1.00, step=0.01):
    k1 = np.arange(low, high, step)
    # k1 = np.arange(0.01, 1.00, 0.01)
    g1 = ( beta**alpha ) / gamma(alpha) * (k1**(alpha - 1))*np.exp(-beta*k1)
    g1 = g1 / np.sum(g1)

    _, m = k1.shape

    z = ( x - x[0] ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    # r1 = prog06.calculate_r(z, t2, length, eta1, N)
    # r2 = prog06.calculate_r(z, t2, length, eta2, N)
    r = g1 * calculate_r(z, t2, length, alpha, beta, N, N1, N2, m, k1)
    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH