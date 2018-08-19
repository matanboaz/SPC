
# coding: utf-8

# This is a program for detecting an increase in mean with an $\left|N\left(\gamma, \tau_2\right)\right|$ prior $\left(\gamma>0\right)$ by a SR procedure with known initial mean and constant known standard deviation $\left(=\sigma\right)$  
# 
# _Input:_  
# * $x$ - row vector of observations
# * $\mu$ - mean
# * $\sigma$ - standard deviation
# * $\gamma$
# * $\tau$
# * $A$



import numpy as np
from scipy.stats import norm


def prog1(x, mu, sigma, gamma, tau, A):
    r = 0
    z = (x - mu) / sigma # Standardization
    length = len(z)
    
    zeta = norm.cdf(gamma / tau)
    w = 10**-10 * np.cumsum(np.ones(length))
    
    t1 = np.arange(length) + 1
    t2 = np.ones(length)
    
    N1 = np.outer(t1, t2)
    N2 = N1.transpose()
    N3 = np.tril(N1-N2+1+1/tau**2)
    
    X1 = np.outer(z.cumsum(), t2)
    X2 = X1.transpose()
    
    if length > 1:
        X3 = np.c_[np.zeros(length), X2[:, :-1]]
    else:
        X3 = 0
    
    X4 = np.tril(X1 - X3 + gamma/tau**2)
    
    Y1 = np.sqrt(N3)
    Y2 = np.tril(X4 / Y1)

    lambd = np.tril(norm.cdf(Y2) * np.exp(0.5 * Y2**2 - 0.5*(gamma / tau)**2 / (Y1 * tau * zeta)))
    r = lambd.sum(axis = 1)
    c = np.cumsum(np.maximum(r, A) - A) - w
    d = c.min()
    I = np.argmin(c)
    N = I + 1
    N
    NummtLENGTH = np.maximum(N-length, 0)
    return (N, NummtLENGTH)


def main():
	observations = np.genfromtxt('.../random_normal.csv', delimiter=',')
	x = observations
	mu = 0
	sigma = 1
	gamma = 0
	tau = 0.5
	A = 100
	N, Nm = prog1(x, mu, sigma, gamma, tau, A)
	print(N)
	print(Nm)

if __name__ == "__main__":
	main()
	
