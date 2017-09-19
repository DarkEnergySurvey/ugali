"""
Classes to handle initial mass functions (IMFs).
"""

import numpy

from ugali.utils.logger import logger

############################################################

class IMF:

    def __init__(self, type='chabrier'):
        """
        Initialize an instance of an initial mass function.
        """

        self.type = type

        if self.type == 'chabrier':
            self.pdf = chabrierIMF
        else:
            logger.warn('initial mass function type %s not recognized'%(self.type))

    def integrate(self, mass_min, mass_max, log_mode=True, weight=False, steps=10000):
        """
        Numerically integrate initial mass function.

        INPUTS:
            mass_min: minimum mass bound for integration (solar masses)
            mass_max: maximum mass bound for integration (solar masses)
            log_mode[True]: use logarithmic steps in stellar mass as oppose to linear
            weight[False]: weight the integral by stellar mass
            steps: number of numerical integration steps
        OUTPUT:
            result of integral
        """
        if log_mode:
            d_log_mass = (numpy.log10(mass_max) - numpy.log10(mass_min)) / float(steps)
            log_mass = numpy.linspace(numpy.log10(mass_min), numpy.log10(mass_max), steps)
            mass = 10.**log_mass

            if weight:
                return numpy.sum(mass * d_log_mass * self.pdf(mass, log_mode=True))
            else:
                return numpy.sum(d_log_mass * self.pdf(mass, log_mode=True))
        else:
            d_mass = (mass_max - mass_min) / float(steps)
            mass = numpy.linspace(mass_min, mass_max, steps)

            if weight:
                return numpy.sum(mass * d_mass * self.pdf(mass, log_mode=False))
            else:
                return numpy.sum(d_mass * self.pdf(mass, log_mode=False))

############################################################

def chabrierIMF(mass, log_mode=True, a=1.31357499301):
    """
    Chabrier initial mass function. Put the reference for the formula here.
    
    INPUTS:
        mass: stellar mass (solar masses)
        log_mode[True]: return number per logarithmic mass range, i.e., dN/dlog(M)
        a[1.31357499301]: normalization; normalized by default to the mass interval 0.1--100 solar masses
    OUTPUTS:
        number per (linear or logarithmic) mass range, i.e., dN/dM or dN/dlog(M) where mass unit is solar masses
    """
    log_mass = numpy.log10(mass)
    b = 0.279087531047 # Where did this hard-coded number come from??
    if log_mode:
        # Number per logarithmic mass range, i.e., dN/dlog(M)
        return ((log_mass <= 0.) * a * numpy.exp(-1. * (log_mass - numpy.log10(0.079))**2 / (2 * (0.69**2)))) + \
               ((log_mass > 0.) * a * b * mass**(-1.3))
    else:
        # Number per linear mass range, i.e., dN/dM
        return (((log_mass <= 0.) * a * numpy.exp(-1. * (log_mass - numpy.log10(0.079))**2 / (2 * (0.69**2)))) + \
                ((log_mass > 0.) * a * b * mass**(-1.3))) / (mass * numpy.log(10))

############################################################
