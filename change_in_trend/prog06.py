"""
Program for Rn for detecting a change in mean with a two-sided representative theta s.d.'s
SR procedure with unknown initial mean and known constant variance.

input:
======
    :x: data
    :sigma: s.d.
    :theta: representative
    :cutoff:
"""

import numpy as np
from scipy.stats import norm


def prog06(x, sigma, theta, A):
    z = x / sigma  # Standardization
    zm = -z
    length = len(z)

    w = 10 ** -10 * np.cumsum(np.ones(length))

    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1 - N2 + 1)
    N4 = N2 - 1

    X1 = np.outer(z.cumsum(), t2)
    X1m = np.outer(zm.cumsum(), t2)
    X2 = X1.transpose()
    X2m = X1m.transpose()
    X3 = np.zeros(length)
    X4 = np.c_[X3, X2[:, :-1]]
    X4m = np.c_[X3, X2m[:, :-1]]

    lambd = np.tril(np.exp(theta * (X1 * N4 / N1 - X4) - 0.5 * (theta ** 2) * N4 * N3 / N1))
    lambd_m = np.tril(np.exp(theta * (X1m * N4 / N1 - X4m) - 0.5 * (theta ** 2) * N4 * N3 / N1))
    rpos = lambd.sum(axis=1)
    rneg = lambd_m.sum(axis=1)
    rpos[0] = 1
    rneg[0] = 1
    r = (rpos + rneg) / 2
    c = np.cumsum(np.maximum(r, A) - A) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    NummtLENGTH = np.maximum(N - length, 0)

    return (N, NummtLENGTH)


def main():
    observations = np.genfromtxt('random_normal.csv', delimiter=',')
    x = observations
    sigma = 1
    theta = 4
    A = 10
    N, Nm = prog06(x, sigma, theta, A)
    print(N)
    print(Nm)


if __name__ == "__main__":
    main()

