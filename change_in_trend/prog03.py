"""
This is a program for detecting an increase with a N(Gamma,tau^2) prior, by a SR procedure with known
initial mean and constant known s.d.

input:
======
    :x: data
    :mu: mean
    :sigam: s.d.
    :gamma:
    :tau:
    :cutoff:
"""

import numpy as np
from scipy.stats import norm

# def increase
def prog3(x, mu, sigma, gamma, tau, A):
    z = (x - mu) / sigma  # Standardization
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
    X2 = X1.transpose()

    K1 = np.zeros(length)

    if length > 1:
        X3 = np.c_[np.zeros(length), X2[:, :-1]]
    else:
        X3 = K1

    X4 = np.tril(X1 - X3 + gamma / tau ** 2) ** 2

    lambd = b2 * (np.tril(np.exp(0.5 * (tau ** 2) * np.tril(X4 / b1))) / np.sqrt(b1))
    r = lambd.sum(axis=1)
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
    A = 70
    N, Nm = prog3(x, mu, sigma, gamma, tau, A)
    print(N)
    print(Nm)

if __name__ == "__main__":
    main()
