import pandas as pd
from change_in_var import prog08

"""
This is a program for detecting a decrease in the variance with known baseline s.d., unkown baseline mean,
by a mixture of discretized conditional Gamma.

input:
======
    :x: data
    :sigma: s.d.
    :alpha:
    :beta:
    :cutoff:
"""


# def decrease_in_var_unknown_mean(x: pd.DataFrame, sigma: float, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
def prog09(x: pd.DataFrame, sigma: float, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
    N, numtLENGTH = prog08.prog08(x, sigma, alpha, beta, cutoff, low, high, step)
    return N, numtLENGTH
