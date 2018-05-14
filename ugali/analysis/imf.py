"""
Classes to handle initial mass functions (IMFs).
"""
from abc import abstractmethod
import numpy as np
import scipy.interpolate

from ugali.utils.logger import logger

# ADW: TODO - This needs to be modernized
#      Also add Kroupa and Salpeter IMFs...

############################################################

class IMF(object):
    """
    Base class for initial mass functions (IMFs).
    """

    def integrate(self, mass_min, mass_max, log_mode=True, weight=False, steps=1e4):
        """ Numerically integrate the IMF.

        Parameters:
        -----------
        mass_min: minimum mass bound for integration (solar masses)
        mass_max: maximum mass bound for integration (solar masses)
        log_mode[True]: use logarithmic steps in stellar mass as oppose to linear
            weight[False]: weight the integral by stellar mass
            steps: number of numerical integration steps

        Returns:
        --------
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

    def sample(self, n, mass_min=0.1, mass_max=10., steps=10000, seed=None):
        """
        Sample initial mass values between mass_min and mass_max,
        following the IMF distribution.

        ADW: Should this be `sample` or `simulate`?

        Parameters:
        -----------
        n : number of samples to draw
        mass_min : minimum mass to sample from
        mass_max : maximum mass to sample from
        steps    : number of steps for isochrone sampling
        seed     : random seed (passed to np.random.seed)

        Returns:
        --------
        mass     : array of randomly sampled mass values
        """
        if seed is not None: np.random.seed(seed)
        d_mass = (mass_max - mass_min) / float(steps)
        mass = np.linspace(mass_min, mass_max, steps)
        cdf = np.insert(np.cumsum(d_mass * self.pdf(mass[1:], log_mode=False)), 0, 0.)
        cdf = cdf / cdf[-1]
        f = scipy.interpolate.interp1d(cdf, mass)
        return f(np.random.uniform(size=n))

    @abstractmethod
    def pdf(cls): pass
        

class Chabrier2003(IMF):
    """ Initial mass function from Chabrier (2003):
    "Galactic Stellar and Substellar Initial Mass Function"
    Chabrier PASP 115:763-796 (2003)
    https://arxiv.org/abs/astro-ph/0304382    
    """

    @classmethod
    def pdf(cls, mass, log_mode=True, a=1.31357499301):
        """ PDF for the Chabrier IMF.
         
        The functional form and coefficients are described in Eq 17
        and Tab 1 of Chabrier (2003):

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

        # Constants from Chabrier 2003
        m_c = 0.079
        sigma = 0.69
        x = 1.3

        # This value is required so that the two components match at 1 Msun
        b = 0.279087531047 

        dn_dlogm = ((log_mass <= 0) * a * np.exp(-(log_mass - np.log10(m_c))**2 / (2 * (sigma**2)))) + ((log_mass  > 0) * a * b * mass**(-x))
            
        if log_mode:
            # Number per logarithmic mass range, i.e., dN/dlog(M)
            return dn_dlogm
        else:
            # Number per linear mass range, i.e., dN/dM
            return dn_dlogm / (mass * np.log(10))
        

class Kroupa2002(IMF): pass
class Salpeter2002(IMF): pass

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

kernelFactory = factory

    
############################################################

def chabrierIMF(mass, log_mode=True, a=1.31357499301):
    """ Backward compatible wrapper around Chabrier2003.pdf """
    return Chabrier2003.pdf(mass,log_mode=log_mode,a=a)

############################################################
