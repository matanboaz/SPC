import pandas as pd
from change_in_var import prog13

"""
This is a program for detecting a decrease in the variance with unknown baseline s.d. and known baseline mean,
by a mixture of discretized conditional Gamma.

input:
======
    :x: data
    :mu: mean
    :alpha:
    :beta:
    :cutoff:
"""


# def decrease_in_var_unknown_baseline_variance(x: pd.DataFrame, mu: float, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
def prog14(x: pd.DataFrame, mu: float, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
    N, numtLENGTH = prog13.prog13(x, mu, alpha, beta, cutoff, low, high, step)
    return N, numtLENGTH
