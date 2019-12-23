"""
This is a program for detecting a change in mean with a $\frac{1}{2}\cdot\left(N\left(\Gamma,\tau^2\right)\right)+\frac{1}{2}\cdot\left(N\left(-\Gamma,\tau^2\right)\right)$ prior
by a SR procedure with known initial mean and constant known s.d.

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
from scipy.stats import norm


# def change_in_mean
def prog04(x, mu, sigma, gamma, tau, A):
    z = (x - mu) / sigma  # Standardization
    zm = -z
    length = len(z)

    w = 10 ** -10 * np.cumsum(np.ones(length))

    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1 - N2 + 1 + 1 / tau ** 2)

    b1 = N3 * tau ** 2 + 1
    b2 = np.exp(-0.5 * (gamma ** 2) / (tau ** 2))

    X1 = np.outer(z.cumsum(), t2)
    X1m = np.outer(zm.cumsum(), t2)
    X2 = X1.transpose()
    X2m = X1.transpose()

    K1 = np.zeros(length)

    if length > 1:
        X3 = np.c_[np.zeros(length), X2[:, :-1]]
        X3m = np.c_[np.zeros(length), X2m[:, :-1]]
    else:
        X3 = K1
        X3m = K1

    X4 = np.tril(X1 - X3 + gamma / tau ** 2) ** 2
    X4m = np.tril(X1m - X3m + gamma / tau ** 2) ** 2
    lambd = b2 * (np.tril(np.exp(0.5 * (tau ** 2) * np.tril(X4 / b1))) / np.sqrt(b1))
    lambdm = b2 * (np.tril(np.exp(0.5 * (tau ** 2) * np.tril(X4m / b1))) / np.sqrt(b1))
    rpos = lambd.sum(axis=1)
    rneg = lambdm.sum(axis=1)

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
    mu = 0
    sigma = 1
    gamma = 0
    tau = 0.5
    A = 10
    N, Nm = prog4(x, mu, sigma, gamma, tau, A)
    print(N)
    print(Nm)


if __name__ == "__main__":
    main()

