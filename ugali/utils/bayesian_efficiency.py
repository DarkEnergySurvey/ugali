"""
Documentation.
"""

import scipy.special
import numpy
import numpy as np
 
############################################################

def gammalnStirling(z):
    """
    Uses Stirling's approximation for the log-gamma function suitable for large arguments.
    """
    return (0.5 * (numpy.log(2. * numpy.pi) - numpy.log(z))) \
           + (z * (numpy.log(z + (1. / ((12. * z) - (1. / (10. * z))))) - 1.))

############################################################

def confidenceInterval(n, k, alpha = 0.68, errorbar=False):
    """
    Given n tests and k successes, return efficiency and confidence interval.
    """
    try:
        e = float(k) / float(n)
    except ZeroDivisionError:
        return numpy.nan, [numpy.nan, numpy.nan]

    bins = 1000001
    dx = 1. / bins

    efficiency = numpy.linspace(0, 1, bins)

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
        p = numpy.concatenate([[numpy.exp(a - b - c)],
                               numpy.exp(a - b - c + (k * numpy.log(efficiency[1: -1])) + (n - k) * numpy.log(1. - efficiency[1: -1])),
                               [0.]])
    elif k == n:
        p = numpy.concatenate([[0.],
                               numpy.exp(a - b - c + (k * numpy.log(efficiency[1: -1])) + (n - k) * numpy.log(1. - efficiency[1: -1])),
                               [numpy.exp(a - b - c)]])
    else:
        p = numpy.concatenate([[0.],
                               numpy.exp(a - b - c + (k * numpy.log(efficiency[1: -1])) + (n - k) * numpy.log(1. - efficiency[1: -1])),
                               [0.]])

    i = numpy.argsort(p)[::-1]
    p_i = numpy.take(p, i)

    s = i[numpy.cumsum(p_i * dx) < alpha]

    low = min(numpy.min(s) * dx, e)
    high = max(numpy.max(s) * dx, e)

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
    delta_e = 1/float(n) * numpy.sqrt(e * (1 - e) * float(n)) * alpha/0.68
    return e, [e - delta_e, e + delta_e]

############################################################
