"""
This is a program for detecting a change in mean with a .5*abs(N(gamma,tau2))+.5*[-abs(N(gamma,tau2)] prior by a SR procedure
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

from change_in_trend import prog01


# def change_in_mean_known_mean_known_var(x, mu, sigma, gamma, tau, A):
def prog02(x, mu, sigma, gamma, tau, A):
    z = (x - mu) / sigma # Standardization
    z_pos = z
    z_neg = -z

    length = len(z)
    w = 10**-10 * np.cumsum(np.ones(length))
    t1 = np.arange(length) + 1
    t2 = np.ones(length)

    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1-N2+1+1/tau**2)

    r_pos = prog01.calculate_r(z_pos, t2, length, gamma, tau, N3)
    r_neg = prog01.calculate_r(z_neg, t2, length, gamma, tau, N3)

    r = (r_pos + r_neg) / 2
    c = np.cumsum(np.maximum(r, A) - A) - w
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
    sigma = 0.3
    gamma = 0
    tau = 0.5
    A = 30
    N, Nm = prog02(x, mu, sigma, gamma, tau, A)
    print(N)
    print(Nm)


if __name__ == '__main__':
    main()
    pass
