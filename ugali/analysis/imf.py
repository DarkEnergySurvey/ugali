"""
Classes to handle initial mass functions (IMFs).

https://github.com/keflavich/imf
"""
from abc import abstractmethod
import numpy as np
import scipy.interpolate

from ugali.utils.logger import logger

############################################################

class IMF(object):
    """
    Base class for initial mass functions (IMFs).
    """

    def __call__(self, mass, **kwargs):
        """ Call the pdf of the mass function """
        return self.pdf(mass,**kwargs)

    def integrate(self, mass_min, mass_max, log_mode=True, weight=False, steps=10000):
        """ Numerical Riemannn integral of the IMF (stupid simple).

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
        steps = int(steps)
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
        steps = int(steps)
        d_mass = (mass_max - mass_min) / float(steps)
        mass = np.linspace(mass_min, mass_max, steps)
        cdf = np.insert(np.cumsum(d_mass * self.pdf(mass[1:], log_mode=False)), 0, 0.)
        cdf = cdf / cdf[-1]
        f = scipy.interpolate.interp1d(cdf, mass)
        return f(np.random.uniform(size=n))

    @abstractmethod
    def pdf(cls, mass, **kwargs): pass
        

class Chabrier2003(IMF):
    """ Initial mass function from Chabrier (2003):
    "Galactic Stellar and Substellar Initial Mass Function"
    Chabrier PASP 115:763-796 (2003)
    https://arxiv.org/abs/astro-ph/0304382    
    """

    @classmethod
    def pdf(cls, mass, log_mode=True):
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

        The normalization is set so that the IMF integrates to 1 over
        the mass range from 0.1 Msun to 100 Msun

        Parameters:
        -----------
        mass: stellar mass (solar masses)
        log_mode[True]: return number per logarithmic mass range, i.e., dN/dlog(M)
         
        Returns:
        --------
        number per (linear or log) mass bin, i.e., dN/dM or dN/dlog(M) where mass unit is solar masses
        """
        log_mass = np.log10(mass)

        # Constants from Chabrier 2003
        m_c = 0.079
        sigma = 0.69
        x = 1.3 

        # This value is set to normalize integral from 0.1 to 100 Msun
        a=1.31357499301
        # This value is required so that the two components match at 1 Msun
        b = 0.279087531047 

        dn_dlogm = ((log_mass <= 0) * a * np.exp(-(log_mass - np.log10(m_c))**2 / (2 * (sigma**2)))) 
        dn_dlogm += ((log_mass  > 0) * a * b * mass**(-x))
        
        if log_mode:
            # Number per logarithmic mass range, i.e., dN/dlog(M)
            return dn_dlogm
        else:
            # Number per linear mass range, i.e., dN/dM
            return dn_dlogm / (mass * np.log(10))

class Kroupa2001(IMF):
    """ IMF from Kroupa (2001):

    "On the variation of the initial mass function"
    MNRAS 322:231 (2001)
    https://arxiv.org/abs/astro-ph/0009005
    """

    @classmethod
    def pdf(cls, mass, log_mode=True):
        """ PDF for the Kroupa IMF.

        Normalization is set over the mass range from 0.1 Msun to 100 Msun
        """
        log_mass = np.log10(mass)
        # From Eq 2
        mb = mbreak  = [0.08, 0.5] # Msun
        a = alpha = [0.3, 1.3, 2.3] # alpha

        # Normalization set from 0.1 -- 100 Msun
        norm = 0.27947743949440446

        b = 1./norm
        c = b * mbreak[0]**(alpha[1]-alpha[0])
        d = c * mbreak[1]**(alpha[2]-alpha[1])

        dn_dm  = b * (mass < 0.08) * mass**(-alpha[0])
        dn_dm += c * (0.08 <= mass) * (mass < 0.5) * mass**(-alpha[1])
        dn_dm += d * (0.5  <= mass) * mass**(-alpha[2])

        if log_mode:
            # Number per logarithmic mass range, i.e., dN/dlog(M)
            return dn_dm * (mass * np.log(10))
        else:
            # Number per linear mass range, i.e., dN/dM
            return dn_dm
    
class Salpeter1955(IMF): 
    """ IMF from Salpeter (1955):
    "The Luminosity Function and Stellar Evolution"
    ApJ 121, 161S (1955)
    http://adsabs.harvard.edu/abs/1955ApJ...121..161S
    """

    @classmethod
    def pdf(cls, mass, log_mode=True):
        """ PDF for the Salpeter IMF.

        Value of 'a' is set to normalize the IMF to 1 between 0.1 and 100 Msun
        """
        alpha = 2.35

        a = 0.060285569480482866
        dn_dm  = a * mass**(-alpha)

        if log_mode:
            # Number per logarithmic mass range, i.e., dN/dlog(M)
            return dn_dm * (mass * np.log(10))
        else:
            # Number per linear mass range, i.e., dN/dM
            return dn_dm
        
def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

imfFactory = factory

############################################################

def chabrierIMF(mass, log_mode=True):
    """ Backward compatible wrapper around Chabrier2003.pdf """
    return Chabrier2003.pdf(mass,log_mode=log_mode)

############################################################
