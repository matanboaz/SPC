import numpy as np
import pandas as pd


"""
This is a program for detecting a 1-sided change in the variance by a SR procedure with constant unkown mean,
known initial s.d. with representative post-change variance

input:
======
    :x: data
    :sigma: s.d.
    :eta:
    :cutoff:
"""


def calculate_r(z, t2, length, eta, N, N1, N2):
    x1 = np.power(z, 2)
    x1_ = np.cumsum(x1, axis=1)
    x1 = np.outer(x1_, t2)
    x2 = x1.transpose()
    x3 = np.zeros(length)
    x3 = np.c_[x3, x2[:, :-1]]
    x3 = np.tril( x3 )
    x4 = ( x1 - x3 ) / 2
    x5 = (1 - 1 / eta**2) * x4 - N * np.log(eta)

    y1_ = np.cumsum(z, axis=1)
    y1 = np.outer(y1_, t2)
    y2 = y1.transpose()
    y3 = np.zeros(length)
    y3 = np.c_[y3, y2[:, :-1]]
    y3 = np.tril( y3 )
    y4 = ( y1 - y3 ) / eta**2

    w1 = 0.5 * ( (y3 + y4)**2 ) / ( (N2 - 1 + N / eta**2) )
    w2 = 0.5 * ( ( ( ( np.cumsum(z) )**2 ) / np.arange(1,length+1) ) )*t2
    w3 = 0.5 * np.log(N2 - 1 + N / eta**2)
    w4 = 0.5 * np.log(N1)
    lmbda = np.tril( np.exp( w1 - w2 + x5 - w3 + w4 ) )
    r = np.sum(lmbda.T, axis=0)
    return r


# def one_sided_change_in_var_unknown_mean(x: pd.DataFrame, sigma: float, eta: float, cutoff: float):
def prog06(x: pd.DataFrame, sigma: float, eta: float, cutoff: float):
    z = ( x - x[0] ) / sigma
    _, length = z.shape

    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.T
    N = np.tril( N1 - N2 + 1 )
    r = calculate_r(z, t2, length, eta, N)

    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)

    return N, nummtLENGTH
