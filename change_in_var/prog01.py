import numpy as np
import pandas as pd


"""
This is a program for detecting a 1-sided change in the variance by a SR procedure
with constant known mean, known initial s.d. with representative post-change variance

input:
======
    :x: data
    :mu: mean
    :sigma: s.d.
    :eta: eta
    :cutoff: A
"""


def calculate_r(z, t2, length, eta, N):
    x1 = np.power(z, 2)
    x1_ = np.cumsum(x1, axis=1)
    x1 = np.outer(x1_, t2)
    x2 = x1.transpose()
    x3 = np.zeros(length)
    x3 = np.c_[x3, x2[:, :-1]]
    x3 = np.tril( x3 )
    x4 = ( x1 - x3 ) / 2
    x5 = (1 - 1 / eta**2) * x4 - N * np.log(eta)

    lmbda = np.tril( np.exp( x5 ) )
    r = np.sum(lmbda.T, axis=0)
    return r


# def one_sided_change_in_var_known_mean(x: pd.DataFrame, mu: float, sigma: float, eta: float, cutoff: float):
def prog01(x: pd.DataFrame, mu: float, sigma: float, eta: float, cutoff: float):
    z = ( x - mu ) / sigma
    _, length = z.shape
    # length = len(z)

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


def main():
    observations = np.genfromtxt('random_normal.csv', delimiter=',')
    x = pd.DataFrame(columns=np.arange(len(observations)))
    x.loc[0] = list(observations)
    mu = 0
    sigma = 1
    eta = 0.1
    A = 30

    N, Nm = prog01(x, mu, sigma, eta, A)
    print(N)
    print(Nm)


if __name__ == '__main__':
    main()
    pass
