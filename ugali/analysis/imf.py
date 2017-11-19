"""
Classes to handle initial mass functions (IMFs).
"""

import numpy
import numpy as np
import scipy.interpolate

from ugali.utils.logger import logger

# ADW: TODO - This needs to be modernized
#      Also add Kroupa and Salpeter IMFs...

############################################################

class IMF:
    """
    Base class for initial mass functions (IMFs).
    """

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
            d_log_mass = (np.log10(mass_max) - np.log10(mass_min)) / float(steps)
            log_mass = np.linspace(np.log10(mass_min), np.log10(mass_max), steps)
            mass = 10.**log_mass

            if weight:
                return np.sum(mass * d_log_mass * self.pdf(mass, log_mode=True))
            else:
                return np.sum(d_log_mass * self.pdf(mass, log_mode=True))
        else:
            d_mass = (mass_max - mass_min) / float(steps)
            mass = np.linspace(mass_min, mass_max, steps)

            if weight:
                return np.sum(mass * d_mass * self.pdf(mass, log_mode=False))
            else:
                return np.sum(d_mass * self.pdf(mass, log_mode=False))

    def sample(self, n, mass_min=0.1, mass_max=10., steps=10000):
        """
        Sample n initial mass values between mass_min and mass_max, following the IMF distribution.
        """
        d_mass = (mass_max - mass_min) / float(steps)
        mass = numpy.linspace(mass_min, mass_max, steps)
        cdf = numpy.insert(numpy.cumsum(d_mass * self.pdf(mass[1:], log_mode=False)), 0, 0.)
        cdf = cdf / cdf[-1]
        f = scipy.interpolate.interp1d(cdf, mass)
        return f(numpy.random.uniform(size=n))

############################################################

def chabrierIMF(mass, log_mode=True, a=1.31357499301):
    """
    Chabrier initial mass function. 

    "Galactic Stellar and Substellar Initial Mass Function"
    Chabrier PASP 115:763-796 (2003)
    https://arxiv.org/abs/astro-ph/0304382    

    The form and coefficients are described in Equation 17 and Table 1:
      m <= 1 Msun: E(log m) = A1*exp(-(log m - log m_c)^2 / 2 sigma^2)
      m  > 1 Msun: E(log m) = A2 * m^-x

      A1 = 1.58 : normalization [ log(Msun)^-1 pc^-3]
      m_c = 0.079 [Msun]
      sigma = 0.69
      A2 = 4.43e-2
      x = 1.3
      
    We redefine a = A1, A2 = a * b;
      
    Chabrier set's his normalization 

    Parameters:
    -----------
    mass: stellar mass (solar masses)
    log_mode[True]: return number per logarithmic mass range, i.e., dN/dlog(M)
    a[1.31357499301]: normalization; normalized by default to the mass interval 0.1--100 solar masses
    
    Returns:
    --------
    number per (linear or log) mass bin, i.e., dN/dM or dN/dlog(M) where mass unit is solar masses
    """
    log_mass = np.log10(mass)
    b = 0.279087531047 # This value is required so that the two components match at 1 Msun
    if log_mode:
        # Number per logarithmic mass range, i.e., dN/dlog(M)
        return ((log_mass <= 0) * a * np.exp(-(log_mass - np.log10(0.079))**2 / (2 * (0.69**2)))) + \
               ((log_mass  > 0) * a * b * mass**(-1.3))
    else:
        # Number per linear mass range, i.e., dN/dM
        return (((log_mass <= 0) * a * np.exp(-(log_mass - np.log10(0.079))**2 / (2 * (0.69**2)))) + \
                ((log_mass  > 0) * a * b * mass**(-1.3))) / (mass * np.log(10))

############################################################
