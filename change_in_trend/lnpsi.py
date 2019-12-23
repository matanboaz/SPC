import numpy as np
from scipy.special import gammaln


def lnpsi(q):
    xx_pos = np.hstack((np.arange(start=0.01, stop=2.01, step=0.01),
                        np.arange(start=2.1, stop=5.1, step=0.1),
                        np.arange(start=6.0, stop=26.0, step=1.0)))

    xx_neg = -np.flip(xx_pos)

    l = len(xx_pos)
    c1 = np.hstack( ( np.arange(1, q+1) ) )
    c2 = np.hstack( ( np.cumsum( np.log(c1) ) ) )
    c9 = np.power(-1, c1)

    s = (l, q)
    J = lnpsi_pos = mu_pos = np.zeros(s)

    # CALCULATING LNPSI FOR POSITIVE X'S
    for i, y in enumerate(xx_pos):
        for m in range(1, q+1):
            g1_0 = ( np.log( y * np.sqrt(2) ) ) * c1
            g1_1 = gammaln( 0.5 * (m + 1 + c1) )
            g1_2 = gammaln( 0.5 * (m + 1) )
            g1 = g1_0 + g1_1 - g1_2 - c2

            d = np.max(g1)
            I = np.argmax(g1)

            mu_pos[i, m-1] = d
            J[i, m-1] = I

            g2 = g1 - d
            g3 = np.sum( np.exp( g2 ) ) + np.exp( -d )
            g4 = np.log( g3 ) + d - 0.5 * (xx_pos[i])**2

            lnpsi_pos[i, m-1] = g4

    mu_neg = np.flipud( mu_pos )
    lnpsi_neg = np.zeros( s, dtype=complex )

    # CALCULATING LNPSI FOR NEGATIVE X'S
    for i, y in enumerate(xx_neg):
        y = np.abs( y )
        for m in range(1, q+1):
            d = mu_neg[i, m-1]
            c5 = ( m + 1 + c1 ) / 2
            c6 = gammaln( c5 )
            c8 = gammaln( ( m+1 ) / 2 )

            g1 = np.log( y * np.sqrt( 2 ) ) * c1
            g2 = c6 - c8 - c2
            g3 = g1 + g2
            g4 = g3 - d
            g5 = c9 * np.exp( g4 )
            g6 = np.sum( g5 ) + np.exp( -d )
            g7 = np.log( np.complex( g6 ) ) + d - 0.5 * y**2

            lnpsi_neg[i, m-1] = g7

    # FINDING THE m'S FOR WHICH LNPSI IS NOT IMAGINARY (FOR NEGATIVE x'S)
    w = np.abs( np.imag( lnpsi_neg ) )
    ltim = np.zeros( (1, l) )

    _, len_ltim = ltim.shape
    for i in range(len_ltim):
        val = np.cumsum( w[i, :] ) - ( ( 10**-10 ) * c1 )
        d = np.min( val )
        I = np.argmin( val )
        ltim[:, i] = I

    # A FIRST-ORDER APPROXIMATION FOR LNPSI OF NEGATIVE x'S
    v = np.zeros( s )
    for r in range(l):
        ko = xx_neg[r] * np.sqrt( c1 ) - 0.5 * (xx_neg[r])**2
        v[r, :] = ko

    # FINDING m'S FOR WHICH LNPSI DOES NOT WIGGLE (NEGATIVE x'S)
    tim = np.maximum( np.ceil( 0.7 * ltim ), 1 )

    # A SECOND-ORDER APPROXIMATION FOR LNPSI OF NEGATIVE x'S
    lnpsi_neg1 = np.zeros( lnpsi_neg.shape, dtype=complex )
    for i in range( l ):
        if i > 14:
            j = int( tim[:, i] )
            o1 = lnpsi_neg[i, np.arange( j )]
            o2 = v[i, np.arange( j, q )]
            o3 = np.hstack( (o1, o2) )
            lnpsi_neg1[i, :] = o3
        else:
            o1 = v[i, :] - 0.2381 * (xx_neg[i])**2
            lnpsi_neg1[i, :] = o1

    # CALCULATION OF LNPSI
    y = np.hstack( ( xx_neg, 0, xx_pos ) )
    z = np.zeros( ( 1, q ) )
    lnpsi = np.vstack( ( lnpsi_neg1, z, lnpsi_pos ) )
    return lnpsi
