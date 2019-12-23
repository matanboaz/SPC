import pandas as pd
import numpy as np
from scipy.special import gammaln, gammainc


"""
This is a program for detecting a decrease in the variance by a SR procedure with constant known mean, known initial
s.d. and unkown post-change variance, with prior on 1/eta2 that is a Gamma(alpha,bbeta) dist. conditional on being >1

input:
======
    :mu: mean
    :sigma: s.d.
    :alpha:
    :beta:
    :cutoff:
"""


def calculate_r(z, t2, length, alpha, beta, N):
    x1 = np.power(z, 2)
    x1_ = np.cumsum(x1, axis=1)
    x1 = np.outer(x1_, t2)
    x2 = x1.transpose()
    x3 = np.zeros(length)
    x3 = np.c_[x3, x2[:, :-1]]
    x3 = np.tril( x3 )
    x4 = ( x1 - x3 ) / 2
    # x5 = (1 - 1 / eta**2) * x4 - N * np.log(eta)
    x5_1 = np.tril( np.log(1 - gammainc(beta + x4, alpha + N / 2)) )
    x5_2 = np.tril( gammaln(alpha + N / 2) )
    x5_3 = alpha * np.log(beta) - gammaln(alpha) - np.log(gammainc(beta, alpha))
    x5_4 = -(alpha + N / 2) * np.tril(np.log(beta + x4))
    x5 = x4 + x5_1 + x5_2 + x5_3 + x5_4
    lmbda = np.tril( np.exp( x5 ) )
    r = np.sum(lmbda.T, axis=0)
    return r


# def decrease_in_var_known_mean(x: pd.DataFrame, mu: float, sigma: float, alpha: float, beta: float, cutoff: float):
def prog04(x: pd.DataFrame, mu: float, sigma: float, alpha: float, beta: float, cutoff: float):
    z = ( x - mu ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )

    r = calculate_r(z, t2, length, alpha, beta, N)

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
