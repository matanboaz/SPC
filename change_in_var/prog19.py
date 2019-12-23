import pandas as pd
from change_in_var import prog18

"""
This is a program for detecting a decrease in the variance with unknown baseline s.d. and unknown baseline mean.
Mixture of discretized conditional Gamma.

input:
======
    :x: data
    :alpha:
    :beta:
    :cutoff:
"""


# def decrease_in_var_unknown_baseline_var_unknown_baseline_mean(x: pd.DataFrame, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
def prog19(x: pd.DataFrame, alpha: float, beta: float, cutoff: float, low=1.1, high=40.1, step=0.1):
    N, numtLENGTH = prog18.prog18(x, alpha, beta, cutoff, low, high, step)
    return N, numtLENGTH
