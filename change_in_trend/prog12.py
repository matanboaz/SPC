import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_a(x: pd.DataFrame, p: int, t: np.array, w: np.array):
    a   = np.zeros( ( p, p ) )
    y1a = ( x.values[0, 1:p] - np.divide(x.values[0:p-2].cumsum()[0:-1], t[0:p-1]) )
    y1b = np.sqrt( np.divide( t[0:p-1], t[1:p] ) )

    y1  = y1a * y1b
    y1  = np.hstack( ( 0, y1 ) )

    z1  = np.divide( y1, np.abs( y1[1] ) )

    v   = np.sqrt( np.cumsum( np.multiply( z1, z1 ) ) )

    ua  = np.hstack( ( 0, 0 ) )
    ub  = np.divide( z1[2:p], np.sqrt( np.multiply( t[2:p], t[1:-1] ) ) )

    u   = np.hstack( ( ua, ub ) )

    s   = np.vstack( ( np.ones( p-2 ) ) )
    s   = np.vstack( ( 0, 0, s ) )

    for n in range(3, p):
        a[n-1, 0:n] = (
                np.multiply(
                    np.sum(
                        np.tril(
                            np.multiply( s[0:n], u[0:n] ).transpose() ),
                        axis=0 ),
                    t[0:n]-1 )
                / v[n-1]
        )
    a[:, 1] = 0.5 * a[:, 2] + ( 2 * (z1[1] > 0) - 1 ) / v[n-1]
    return a, z1


def calculate_lgpsi(a: np.ndarray, p: int, lnpsi: np.ndarray):
    lgpsi = np.zeros( ( p, p ) )

    for n in range(3, p+1):
        for k in range(n):
            if np.abs( a[n-1, k] ) > 20:
                lgpsi[n-1, k] = a[n-1, k] * np.sqrt( n-2 ) - 0.2381 * np.power( ( a[n-1, k] ), 2 )
            else:
                if np.abs( a[n-1, k] ) <= 2:
                    b = 100 * a[n-1, k] + 251
                elif np.abs( np.abs( a[n-1, k] ) ) <= 5:
                    b = 10 * ( np.abs( a[n-1, k] ) - 2 ) * np.sign( a[n-1, k] ) + 251 + 200 * np.sign( a[n-1, k] )
                else:
                    b = ( a[n-1, k] - 5 ) * np.sign( a[n-1, k] ) + 251 + 230 * np.sign( a[n-1, k] )
                bf = int( np.floor( b ) )
                bc = bf + 1
                lgpsi[n-1, k] = lnpsi[bf-1, n-3] + ( b - bf ) * ( lnpsi[bc-1, n-3] - lnpsi[bf-1, n-3] )
    return lgpsi


def calculate_r(a: np.ndarray, lgpsi: np.ndarray, z1: np.ndarray, t: np.ndarray, p: int):
    lmbda = np.zeros( ( p, p ) )
    lmbda[:, 0] = np.ones( (p) )
    lmbda[1, 1] = 2 * norm.cdf(1) * ( z1[1] > 0 ) + 2 * ( 1 - norm.cdf(1) ) * ( z1[1] <= 0 )

    oa = -0.5 * ( 1 - 1 / t[2:p] )
    ob = 0.5 * np.power( ( a[2:p, 1] ), 2 )
    oc = lgpsi[2:p, 1]
    o  = np.exp(oa + ob + oc) * lmbda[1, 1]

    lmbda[2:p, 1] = 0

    A1 = np.reshape( np.ones( p-2 ), ( p-2, 1 ) )
    a1 = np.reshape( np.arange( 3, p+1 ), ( 1, p-2 ) )
    A1 = np.matmul( A1, a1 )
    B = A1-1 - (A1-1) * (A1-1) / A1.T
    o = np.tril( np.exp( -0.5 * B + 0.5 * np.power( a[2:p, 2:p], 2 ) + lgpsi[2:p, 2:p] ) )
    lmbda[2:p, 2:p] = o

    r = np.sum( lmbda.T, axis=0 )
    return r


def prog12(x: pd.DataFrame, cutoff: float, theta, lnpsi):
    _, p = x.shape
    x_pos = x
    x_neg = -x

    t = np.arange( 1, p+1 )
    w = 10**-10 * t

    a_pos, z1_pos = calculate_a(x_pos, p, t, w)
    a_neg, z1_neg = calculate_a(x_neg, p, t, w)

    lgpsi_pos = calculate_lgpsi(a_pos, p, lnpsi)
    lgpsi_neg = calculate_lgpsi(a_neg, p, lnpsi)

    r_pos = calculate_r(a_pos, lgpsi_pos, z1_pos, t, p)
    r_neg = calculate_r(a_neg, lgpsi_neg, z1_neg, t, p)

    r = ( r_pos + r_neg ) / 2
    c = np.cumsum( np.maximum( r, cutoff ) - cutoff ) - w

    d = np.min( c )
    I = np.argmin( c )
    N = I + 1
    y = r[min( N, p )]
    print( N )
    NUMmtLENGTH = max( N-p, 0 )
    print(NUMmtLENGTH)
