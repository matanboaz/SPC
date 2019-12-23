import numpy as np
import pandas as pd
from scipy.stats import gamma


"""
This is a program for detecting a 1-sided change in the variance with unknown baseline s.d. and unknown mean.
representative eta for the ratio of s.d.

input:
======
    :x: data
    :eta: representative for the ratio of s.d.
    :cutoff:
"""


def calculate_r(z, t2, length, eta, N, N1, N2, m):
    x1 = np.power(z, 2)
    x1_ = np.cumsum(x1, axis=1)
    x1 = np.outer(x1_, t2)
    x2 = x1.transpose()
    x3 = np.zeros(length)
    x3 = np.c_[x3, x2[:, :-1]]
    x3 = np.tril( x3 )
    x4 = ( x1 - x3 )

    y1 = np.cumsum(z, axis=1)
    y1 = np.outer(y1, t2)
    y2 = y1.transpose()
    y3 = np.zeros(length)
    y3 = np.c_[y3, y2[:, :-1]]
    y4 = ( y1 - y3 ) / eta**2

    w1eta = np.tril( ( (y3 + y4)**2 ) / ( ( N2 - 1 + N / eta**2 ) ) )
    w1eta2 = np.tril( ( ( y4[:, 1] )**2 ) / ( ( N1[:, 1] - 1 ) / ( eta**2 ) + 1 ) )
    w1one = np.tril( (y1**2) / N1 )
    w2eta = np.tril( x3 + x4 / eta**2 )
    w2one = np.tril(x1)
    w3eta = w2eta - w1eta
    w3eta2 = w2eta[:, 1] - w1eta2
    w3one = w2one - w1one
    w4eta = np.tril( ( N2 - 1 + N / eta**2 ) )

    oy = 0.5 * (N1 - 1) * (np.log(w3one) -np.log(w3eta) ) + 0.5 * np.log(N1) - N * np.log(eta) - 0.5 * np.log(w4eta)
    oy2 = 0.5 * (N1[:, 1] - 1) * (np.log(w3one[:, 1]) - np.log(w3eta2)) + 0.5*np.log(N1[:,1]) - N[:, 1]*np.log(eta) - 0.5*np.log(w4eta[:, 1])
    oy[:, 1] = oy2
    oy = np.tril(oy)
    lmbda = np.tril( np.exp( oy ) )
    lmbda[:, 1] = t2.transpose()
    r = np.sum(lmbda.T, axis=0)
    return r


# def one_sided_change_in_var_unknown_baseline_var_unknown_mean(x: pd.DataFrame, eta: float, cutoff: float):
def prog16(x: pd.DataFrame, eta: float, cutoff: float):
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
    r = calculate_r(z, t2, length, eta, N, N1, N2, m)
    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH