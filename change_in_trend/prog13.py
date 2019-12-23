import pandas as pd
import numpy as np

from change_in_trend import prog12


def calculate_bb(p, t):
    bb1 = np.ones( ( p, 1 ) )
    bb2 = np.reshape( t-1, (1, p) )
    bb3 = np.reshape( 1 / t, (p, 1) )
    bb4 = np.reshape( np.power(t-1, 2), (1, p) )
    bb = np.matmul( bb1, bb2 ) - np.matmul( bb3, bb4 )
    bb[:, 1] -= 0.5
    bb = np.tril( bb )
    return bb


def calculate_lambda(a, bb, c, d, e, gamma, lgpsi, p, t, tau):
    lmbda_1 = np.multiply(0.5 * d * gamma, a)
    lmbda_1 = np.divide( lmbda_1, e )
    lmbda_2 = 0.5 * bb * (gamma**2)
    lmbda_2 = np.divide( lmbda_2, e )

    lmbda_a = lmbda_1 - lmbda_2 + lgpsi
    lmbda_b = np.matmul( 0.5*np.diag(t-1), np.log(1+(tau**2)*c) )
    lmbda_c = np.matmul( 0.5*np.diag(t-2), np.log(e) )
    lmbda_d = np.exp(lmbda_a - lmbda_b + lmbda_c)
    lmbda = np.tril( lmbda_d )
    lmbda[:, 0:2] = np.ones((p, 1))

    return lmbda


def prog13(x: pd.DataFrame, cutoff: float, gamma, tau, lnpsi):
    _, p = x.shape
    t = np.arange( 1, p+1 )
    w = 10**-10 * t

    a, z1 = prog12.calculate_a(x, p, t, w)
    a = np.tril( a )

    bb = calculate_bb(p, t)
    c = bb - np.power(a, 2)
    d = ( gamma * a / (1 + c * tau**2) )
    e = 1 + ( tau**2 ) * bb

    lgpsi = prog12.calculate_lgpsi(d, p, lnpsi)

    lmbda = calculate_lambda(a, bb, c, d, e, gamma, lgpsi, p, t, tau)

    r = np.sum( lmbda.T, axis=0)
    c = np.cumsum( np.maximum( r, cutoff ) - cutoff ) - w
    d = np.min( c )
    I = np.argmin( c )
    N = I + 1
    y = r[min( N, p )]
    print(y)
    NUMmtLENGTH = max( N-p, 0 )
    print(NUMmtLENGTH)
    pass
