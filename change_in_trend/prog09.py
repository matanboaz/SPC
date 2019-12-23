"""
Program for Rn for detecting an increase in mean with a Normal(gamma,tau2) prior by a
SR procedure with unknown initial mean and known constant variance

input:
======
    :x: data
    :sigma: s.d.
    :gamma:
    :tau:
    :cutoff:
"""

import numpy as np
from scipy.stats import norm


def prog09(x, sigma, gamma, tau, A):
    z = x / sigma
    length = len(z)
    w = 10 ** -10 * np.cumsum(np.ones(length))

    g1 = 1 / tau ** 2
    g2 = gamma * g1;
    g3 = gamma * g2;

    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1 - N2 + 1)
    N4 = N2 - 1

    X1 = np.outer(z.cumsum(), t2)
    X2 = X1.transpose()
    X3 = np.zeros(length)
    X4 = np.c_[X3, X2[:, :-1]]

    U = X1 * N4 / N1 - X4
    b = N4 * N3 / N1
    a = 0.5 * ((U + g2) ** 2 / (b + g1))
    lambd = np.tril(np.exp(a - (g3) / 2) / np.sqrt(b * tau ** 2 + 1))
    r = lambd.sum(axis=1)
    r[0] = 1
    c = np.cumsum(np.maximum(r, A) - A) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    NummtLENGTH = np.maximum(N - length, 0)

    return (N, NummtLENGTH)


def main():
    observations = np.genfromtxt('random_normal.csv', delimiter=',')
    x = observations
    mu = 0
    sigma = 0.1
    gamma = 0.5
    tau = 0.1
    A = 45
    N, Nm = prog09(x, sigma, gamma, tau, A)
    print(N)
    print(Nm)


if __name__ == "__main__":
    main()
