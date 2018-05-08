"""
Utility functions for calculating efficiency uncertainties in a bayesian manner.

A good reference is probably:
http://home.fnal.gov/~paterno/images/effic.pdf

ADW: This should be moved into stats.py
"""

import scipy.special
import numpy as np
 
############################################################

def gammalnStirling(z):
    """
    Uses Stirling's approximation for the log-gamma function suitable for large arguments.
    """
    return (0.5 * (np.log(2. * np.pi) - np.log(z))) \
           + (z * (np.log(z + (1. / ((12. * z) - (1. / (10. * z))))) - 1.))

############################################################

def confidenceInterval(n, k, alpha = 0.68, errorbar=False):
    """
    Given n tests and k successes, return efficiency and confidence interval.
    """
    try:
        e = float(k) / float(n)
    except ZeroDivisionError:
        return np.nan, [np.nan, np.nan]

    bins = 1000001
    dx = 1. / bins

    efficiency = np.linspace(0, 1, bins)

    # MODIFIED FOR LARGE NUMBERS
    if n + 2 > 1000:
        a = gammalnStirling(n + 2)
    else:
        a = scipy.special.gammaln(n + 2)
    if k + 1 > 1000:
        b = gammalnStirling(k + 1)
    else:
        b = scipy.special.gammaln(k + 1)
    if n - k + 1 > 1000:
        c = gammalnStirling(n - k + 1)
    else:
        c = scipy.special.gammaln(n - k + 1)

    if k == 0:
        p = np.concatenate([[np.exp(a - b - c)],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [0.]])
    elif k == n:
        p = np.concatenate([[0.],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [np.exp(a - b - c)]])
    else:
        p = np.concatenate([[0.],
                               np.exp(a - b - c + (k * np.log(efficiency[1: -1])) + (n - k) * np.log(1. - efficiency[1: -1])),
                               [0.]])

    i = np.argsort(p)[::-1]
    p_i = np.take(p, i)

    s = i[np.cumsum(p_i * dx) < alpha]

    low = min(np.min(s) * dx, e)
    high = max(np.max(s) * dx, e)

    if not errorbar:
        return e, [low, high]
    else:
        return e, [e - low, high - e]

############################################################

bayesianInterval = confidenceInterval

############################################################

def binomialInterval(n, k, alpha = 0.68):
    """
    Given n tests and k successes, return efficiency and confidence interval.
    """
    e = float(k)/n
    delta_e = 1/float(n) * np.sqrt(e * (1 - e) * float(n)) * alpha/0.68
    return e, [e - delta_e, e + delta_e]

############################################################
