import pandas as pd
import numpy as np

from change_in_trend import prog12
from change_in_trend import prog13

def prog14(x: pd.DataFrame, cutoff: float, gamma, tau, lnpsi):
    _, p = x.shape
    x_pos = x
    x_neg = -x

    t = np.arange( 1, p+1 )
    w = 10**-10 * t

    a_pos, z1_pos = prog12.calculate_a(x_pos, p, t, w)
    a_pos = np.tril( a_pos )
    a_neg, z1_neg = prog12.calculate_a(x_neg, p, t, w)
    a_neg = np.tril( a_neg )

    bb = prog13.calculate_bb(p, t)

    c_pos = bb - np.power( a_pos, 2 )
    c_neg = bb - np.power( a_neg, 2 )

    d_pos = ( gamma * a_pos / (1 + c_pos * tau**2) )
    d_neg = ( gamma * a_neg / (1 + c_neg * tau**2) )

    e = 1 + ( tau**2 ) * bb

    lgpsi_pos = prog12.calculate_lgpsi(d_pos, p, lnpsi)
    lgpsi_neg = prog12.calculate_lgpsi(d_neg, p, lnpsi)

    lmbda_pos = prog13.calculate_lambda(a_pos, bb, c_pos, d_pos, e, gamma, lgpsi_pos, p, t, tau)
    lmbda_neg = prog13.calculate_lambda(a_neg, bb, c_neg, d_neg, e, gamma, lgpsi_neg, p, t, tau)

    r_pos = np.sum( lmbda_pos.T, axis=0)
    r_neg = np.sum( lmbda_neg.T, axis=0)

    r = ( r_pos + r_neg ) / 2
    c = np.cumsum( np.maximum( r, cutoff ) - cutoff ) - w
    d = np.min( c )
    I = np.argmin( c )
    N = I + 1
    y = r[min( N, p )]
    print(y)
    NUMmtLENGTH = max( N-p, 0 )
    print(NUMmtLENGTH)
    pass
