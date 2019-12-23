"""
This is a program for detecting an increase in mean with an |N(gamma, tau^2)| prior by a SR procedure
with known initial mean and constant known s.d.

input:
======
    :x: data
    :mu: mean
    :sigma: s.d.
    :gamma:
    :tau:
    :cutoff:
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_r(z, t2, length, gamma, tau, n3):
    zeta = norm.cdf( gamma / tau )
    x1 = np.outer( z.cumsum(), t2 )
    x2 = x1.transpose()

    if length > 1:
        x3 = np.c_[np.zeros(length), x2[:, :-1]]
    else:
        x3 = 0

    x4 = np.tril( x1 - x3 + gamma / tau**2 )

    y1 = np.sqrt(n3)

    y2 = np.tril( np.divide(x4, y1) )
    lmbda = np.tril( norm.cdf(y2) * np.exp(0.5 * y2**2 - 0.5*(gamma / tau)**2 / (y1 * tau * zeta)))
    r = lmbda.sum(axis=1)
    return r

# def increase_in_mean_known_mean_known_sd(x, mu, sigma, gamma, tau, cutoff):
def prog01(x, mu, sigma, gamma, tau, cutoff):
    r = 0
    z = (x - mu) / sigma # Standardization
    # _, length = z.shape
    length = len(z)

    zeta = norm.cdf(gamma / tau)
    w = 10**-10 * np.cumsum(np.ones(length))

    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1-N2+1+1/tau**2)

    r = calculate_r(z, t2, length, gamma, tau, N3)
    c = np.cumsum(np.maximum(r, cutoff) - cutoff) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    nummtLENGTH = np.maximum(N-length, 0)
    return N, nummtLENGTH


def main():
    observations = np.genfromtxt('random_normal.csv', delimiter=',')
    # x = pd.DataFrame(columns=np.arange(len(observations)))
    # x.loc[0] = list(observations)
    x = observations
    mu = 0
    sigma = 1
    gamma = 0.1
    tau = 0.5
    A = 30
    N, Nm = prog01(x, mu, sigma, gamma, tau, A)
    print(N)
    print(Nm)


if __name__ == '__main__':
    main()
    pass
