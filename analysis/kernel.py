"""
Documentation.
"""

import numpy
import numpy as np
import scipy.signal
import scipy.integrate
from abc import abstractmethod
from collections import OrderedDict as odict

import ugali.utils.projector

def kernelFactory(kernel_type,*args):
    """
    Factory for cerating spatial kernels. Arguments
    are passed directly to the constructor of the chosen
    kernel. Currently allowed kernel_types are:
    - Plummer
    - King
    - Exponential
    - Disk
    """
    kernel_type = kernel_type.lower()
    if kernel_type == "plummer":
        return PlummerKernel(*args)
    elif kernel_type == "king":  
        return KingKernel(*args)
    elif kernel_type == "exponential":
        return ExponentialKernel(*args)
    elif kernel_type == "disk":
        return DiskKernel(*args)
    else:
        raise Exception("Unrecognized kernel: %s"%kernel_type)
"""
# Start new kernel object base for elliptical kernels.
class Kernel(object):
    params = odict([])

    def __init__(self, lon, lat, **kwargs):
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        self.name = self.__class__.__name__
        params = dict(self.params)
        params.update(kwargs)
        for param,value in params.items():
            self.setp(param,value)

    def __getattr__(self,name):
        # Return 'value' of parameters
        # __getattr__ tries the usual places first.
        if name in self.params:
            return self.params[name]
        else:
            # Raises AttributeError
            return object.__getattribute__(self,name)

    def setp(self,param,value):
        self.params[param] = value
                        
    @property
    def extension(self):
        return self.params.values[0]

    @abstractmethod
    def pdf(self):
        pass

"""

class RadialKernel(object):
    """
    Base class for radially symmetric kernel profiles. Kernels are 
    explicitly normalized to integrate to unity. This means that all
    pdf's must be integrable...
    Each subclass must implement:
        pdf       : value of the pdf as a function of angular radius
        extension : parameter for controlling extension
        edge      : parameter for defining effective edge of the kernel
    """
    def __init__(self, lon, lat, *args):
        self.name = self.__class__.__name__
        self._setup(*args)
        self.setCenter(lon,lat)

    def __call__(self, radius):
        return self.pdf(radius)

    def _setup(self, *args):
        self.params = list(args)
        self.norm = 1.0
        self.norm /= self.integrate()

    def setCenter(self, lon, lat):
        self.lon = float(lon)
        self.lat = float(lat)
        self.projector = ugali.utils.projector.Projector(self.lon, self.lat)

    def setExtension(self, extension):
        if extension <= 0: 
            raise ValueError("Extension must be positive.")
        args = self.params
        args[0] = extension
        self._setup(*args)

    @abstractmethod
    def extension(self):
        pass

    @abstractmethod
    def edge(self):
        pass

    @abstractmethod
    def pdf(self, radius):
        """
        Evaluate the surface brightness profile, I(R), at a given radius (deg).
        The surface brightness profiles should be defined in the manner of
        second convention of Binney and Tremain, 2008 (Box 2.1)
        radius : radius to evaluate the local surface brightness
        return : surface brightness (1/deg^2)
        """
        pass

    # For back-compatibility
    def surfaceIntensity(self, radius): 
        return self.pdf(radius)

    def integrate(self, r_min=0, r_max=numpy.inf):
        """
        Calculate the 2D integral of the surface brightness profile (i.e, the flux) 
        between r_min and r_max. 
        r_min : minimum integration radius (deg)
        r_max : maximum integration radius (deg)
        return : Solid angle integral (deg^2)
        """
        if r_min < 0: raise Exception('r_min must be >= 0')
        #r_max = r_max if r_max < self.edge() else self.edge()
        integrand = lambda r: self.pdf(r) * 2*numpy.pi * r
        return scipy.integrate.quad(integrand, r_min, r_max, full_output=True, epsabs=0)[0]

    # For back-compatibility
    def integratePDF(self, r_min, r_max, steps=1e4, prenormalized=True): 
        return self.integrate(r_min, r_max)

    def sample_radius(self, n):
        """
        Sample the radial distribution (deg) from the 2D stellar density.
        """
        edge = self.edge() if self.edge()<20*self.extension() else 20*self.extension()
        #edge = 20*self.r_h
        radius = numpy.linspace(0, edge, 1.e5)
        rpdf = self.pdf(radius) * numpy.sin(numpy.radians(radius))
        rcdf = numpy.cumsum(rpdf)
        rcdf /= rcdf[-1]
        fn = scipy.interpolate.interp1d(rcdf, range(0, len(rcdf)), bounds_error=False, fill_value=-1)
        index = numpy.floor(fn(numpy.random.rand(n))).astype(int)
        return radius[index]

    def sample_lonlat(self, n):
        """
        Sample 2D distribution of points in lon, lat
        Careful, doesn't respect sky projections
        """
        radius = self.sample_radius(n)
        phi = 2. * numpy.pi * numpy.random.rand(n)
        x = radius * numpy.cos(phi)
        y = radius * numpy.sin(phi)
        lon, lat = self.projector.imageToSphere(x, y)
        return lon, lat

    simulate = sample_lonlat

class DiskKernel(RadialKernel):
    def _setup(self, *args):
        self.r_0 = args[0]
        super(DiskKernel,self)._setup(*args)

    ### Need to figure out a stellar mass conversion...
    def pdf(self, radius):
        """
        Disk stellar density distribution:
        f(r) = C  for r <= r_0
        f(r) = 0  for r > r_0
        """
        return numpy.where(radius<=self.r_0, self.norm, 0)

    def extension(self):
        return self.r_0

    def edge(self):
        return self.r_0
        
class PlummerKernel(RadialKernel):
    """
    Stellar density distribution for Plummer profile:
    f(r) = C * r_h**2 / (r_h**2 + r**2)**2
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 6)
    """

    def _setup(self, *args):
        self.r_h = args[0]
        self.u_t = args[1] if len(args)>1 else 5. #numpy.inf #5.
        super(PlummerKernel,self)._setup(*args)

    def pdf(self, radius):
        kernel = self.norm / (numpy.pi * self.r_h**2 * (1. + (radius / self.r_h)**2)**2)
        return numpy.where(radius<=self.edge(), kernel, 0 )

    def extension(self):
        return self.r_h

    def edge(self):
        return self.u_t * self.r_h

class KingKernel(RadialKernel):
    """
    Stellar density distribution for King profile:
    f(r) = C * [ 1/sqrt(1 + (r/r_c)**2) - 1/sqrt(1 + (r_t/r_c)**2) ]**2
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 4)
    """
    def _setup(self, *args):
        self.r_c = args[0]                 # Critical radius
        self.c   = args[1]                 # Concentration
        self.r_t = self.r_c * (10**self.c) # Tidal radius
        super(KingKernel,self)._setup(*args)

    def pdf(self, radius):
        kernel = self.norm * ((1. / numpy.sqrt(1. + (radius / self.r_c)**2)) - (1. / numpy.sqrt(1. + (self.r_t / self.r_c)**2)))**2
        return numpy.where( radius < self.edge(), kernel, 0 )

    def extension(self):
        return self.r_c

    def edge(self):
        return self.r_t

class ExponentialKernel(RadialKernel):
    """
    Stellar density distribution for Exponential profile:
    f(r) = C * exp(-r / r_e)
    where r_e = r_h/1.68
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 5)        
    """

    def _setup(self, *args):
        self.r_h = args[0]
        self.r_e = self.r_h/1.68
        super(ExponentialKernel,self)._setup(*args)

    def pdf(self, radius):
        kernel = self.norm * numpy.exp(-radius/self.r_e)
        return numpy.where(radius<=self.edge(), kernel, 0 )

    def extension(self):
        return self.r_h

    def edge(self):
        return 20. * self.r_h

class EllipticalKernel(object):
    """
    Stellar density for elliptical exponential profile:
    f(r_i) = C * exp(-r_i /r_e)
    where the elliptical radius is defined:
    r_i = sqrt{((x*cos(theta)-y*sin(theta))/(1-epsilon))**2+(x*sin(theta)+y*cos(theta))**2}
    with ellipticity epsilon = 1 - b/a (a/b represent the major/minor axes)
    and position angle theta (deg) from north to east.
    http://adsabs.harvard.edu/abs/2008ApJ...684.1075M
    """

    def __init__(self, lon, lat, *args):
        self.name = self.__class__.__name__
        self._setup(*args)
        self.setCenter(lon,lat)

    def __call__(self, lon, lat):
        return self.pdf(lon,lat)

    def _setup(self, *args):
        self.r_h = args[0]
        self.epsilon = args[1]
        self.theta = args[2]

        # Base class
        self.params = list(args)

    @property
    def norm(self):
        return 1./(2*np.pi*self.r_e**2*(1-self.epsilon))

    @property
    def r_e(self):
        return self.r_h/1.68

    def setCenter(self, lon, lat):
        self.lon = float(lon)
        self.lat = float(lat)
        self.projector = ugali.utils.projector.Projector(self.lon, self.lat)

    def setExtension(self, extension):
        if extension <= 0: 
            raise ValueError("Extension must be positive.")
        args = self.params
        args[0] = extension
        self._setup(*args)

    def extension(self):
        return self.r_h

    def edge(self):
        return 20. * self.r_h

    def pdf(self, x, y):
        costheta = np.cos(np.radians(self.theta))
        sintheta = np.sin(np.radians(self.theta))
        r_i = np.sqrt(((x*costheta-y*sintheta)/(1-self.epsilon))**2 + (x*sintheta+y*costheta)**2)
        kernel = self.norm * numpy.exp(-r_i/self.r_e)
        radius = np.sqrt(x**2+y**2)

        return numpy.where(radius<=self.edge(), kernel, 0 )
        
    def integrate(self, r_min=0, r_max=numpy.inf):
        """
        Calculate the 2D integral of the surface brightness profile (i.e, the flux) 
        between r_min and r_max.
        r_min : minimum integration radius
        r_max : maximum integration radius
        return : Solid angle integral 
        """
        if r_min < 0: raise Exception('r_min must be >= 0')
        #r_max = r_max if r_max < self.edge() else self.edge()
        integrand = lambda r: self.pdf(np.sqrt(r),np.sqrt(r)) * 2*numpy.pi * r
        return scipy.integrate.quad(integrand, r_min, r_max, full_output=True, epsabs=0)[0]
    
############################################################

def halfLightRadius(m_stellar):
    """
    Return the half-light radius (kpc) given stellar mass (M_sol).

    Three-dimensional to two-dimensional conversion needed?

    Based on plot from Andrey Kravtsov with fit by hand.
    http://adsabs.harvard.edu/abs/2013ApJ...764L..31K
    """
    A = -1.
    B = 0.194
    m_stellar_0 = 1.e3 # M_sol
    return 10**(A + B * numpy.log10(m_stellar / m_stellar_0))

############################################################
