import numpy as np
import pandas as pd
from change_in_trend import prog12


def prog11(x: pd.DataFrame, cutoff: float, theta, lnpsi):
    _, p = x.shape

    t = np.arange( 1, p+1 )
    w = 10**-10 * t

    a, z1 = prog12.calculate_a(x, p, t, w)

    lgpsi = prog12.calculate_lgpsi(a, p, lnpsi)

    r = prog12.calculate_r(a, lgpsi, z1, t, p)

    c = np.cumsum( np.maximum( r, cutoff ) - cutoff ) - w

    d = np.min( c )
    I = np.argmin( c )
    N = I + 1
    y = r[min( N, p )]
    print( N )
    NUMmtLENGTH = max( N-p, 0 )
    print(NUMmtLENGTH)
    return N, NUMmtLENGTH