#!/usr/bin/env python
"""
Spatial kernels for dwarf galaxies.
"""

import sys
import inspect
from abc import abstractmethod
from collections import OrderedDict as odict

import numpy
import numpy as np
import scipy.integrate

import ugali.utils.projector
from ugali.utils.projector import Projector, angsep
from ugali.utils.logger import logger

class Kernel(object):
    """
    Base class for kernels.
    """
    _params = odict([
        ('lon',0.0),
        ('lat',0.0),
        ('proj','car'),
    ])
    _mapping = odict([])
    
    def __init__(self, lon, lat, **kwargs):
        self.name = self.__class__.__name__
        params = dict()
        params.update(lon=lon,lat=lat,**kwargs)
        for param,value in params.items():
            # Raise AttributeError if attribute not found
            self.__getattr__(param) 
            # Set attribute
            self.__setattr__(param,value)
 
    def __call__(self, lon, lat):
        return self.pdf(lon, lat)

    def __getattr__(self,name):
        # Return 'value' of parameters
        # __getattr__ tries the usual places first.
        if name in self._mapping:
            return self.__getattr__(self._mapping[name])

        if name in self._params:
            return self._params[name]
        else:
            # Raises AttributeError
            return object.__getattribute__(self,name)

    def __setattr__(self, name, value):
        # Call 'set_value' on parameters
        # __setattr__ tries the usual places first.
        if name in self._mapping.keys():
            return self.__setattr__(self._mapping[name],value)
        
        if name in self._params:
            self.setp(name,value)
        else:
            return object.__setattr__(self, name, value)

    def setp(self,param,value):
        self._params[param] = value

    @abstractmethod
    def _kernel(self, r):
        # unnormalized, untruncated kernel
        pass

    @property
    def projector(self):
        #return Projector(self.lon, self.lat, proj_type='tan')
        if self.proj is None:
            return None
        else:
            return Projector(self.lon, self.lat, self.proj)

    def _pdf(self, radius):
        return np.where(radius<=self.edge,self._kernel(radius),0.)

    @abstractmethod
    def pdf(self, lon, lat):
        # normalized, truncated pdf
        pass

    @property
    def norm(self):
        # Numerically integrate the pdf
        return 1./self.integrate()

    def integrate(self, rmin=0, rmax=numpy.inf):
        """
        Calculate the 2D integral of the 1D surface brightness profile 
        (i.e, the flux) between rmin and rmax. 
        rmin : minimum integration radius (deg)
        rmax : maximum integration radius (deg)
        return : Solid angle integral (deg^2)
        """
        if rmin < 0: raise Exception('rmin must be >= 0')
        integrand = lambda r: self._pdf(r) * 2*numpy.pi * r
        return scipy.integrate.quad(integrand,rmin,rmax,full_output=True,epsabs=0)[0]

class EllipticalKernel(Kernel):
    """
    Base class for elliptical kernels.
    Ellipticity is defined as 1 - b/a where a,b are the semi-major,semi-minor
    axes respectively. The position angle is defined in degrees east of north.
    This definition follows from Martin et al. 2008:
    http://adsabs.harvard.edu/abs/2008ApJ...684.1075M
    """
    _params = odict(
        Kernel._params.items() + 
        [
            ('extension',0.5),
            ('ellipticity',0.0),     # Default 0 for RadialKernel
            ('position_angle',0.0),  # Default 0 for RadialKernel
        ])
    _mapping = odict([
        ('e','ellipticity'),
        ('theta','position_angle'),
    ])

    @property
    def norm(self):
        return super(EllipticalKernel,self).norm * 1./self.jacobian

    @property
    def jacobian(self):
        return 1. - self.e

    @property
    def a(self):
        return self.extension

    @property
    def b(self):
        return self.a*self.jacobian

    @property
    def edge(self):
        return 5.*self.extension

    def pdf(self,lon,lat):
        x,y = self.projector.sphereToImage(lon,lat)
        costh = np.cos(np.radians(self.theta))
        sinth = np.sin(np.radians(self.theta))
        radius = np.sqrt(((x*costh-y*sinth)/(1-self.e))**2 + (x*sinth+y*costh)**2)
        return self.norm*self._pdf(radius)

    def sample_radius(self, n):
        """
        Sample the radial distribution (deg) from the 2D stellar density.
        Output is elliptical radius in true projected coordinates.
        """
        edge = self.edge if self.edge<20*self.extension else 20*self.extension
        radius = np.linspace(0, edge, 1.e5)
        rpdf = self._pdf(radius) * np.sin(np.radians(radius))
        rcdf = np.cumsum(rpdf)
        rcdf /= rcdf[-1]
        fn = scipy.interpolate.interp1d(rcdf, range(0, len(rcdf)), bounds_error=False, fill_value=-1)
        index = numpy.floor(fn(numpy.random.rand(n))).astype(int)
        return radius[index]
 
    def sample_lonlat(self, n):
        """
        Sample 2D distribution of points in lon, lat
        """
        radius = self.sample_radius(n)
        a = radius; b = self.jacobian * radius
        phi = 2. * np.pi * numpy.random.rand(n)
        cosphi,sinphi = np.cos(phi),np.sin(phi)
        costheta,sintheta = np.cos(np.deg2rad(self.theta)),np.sin(np.deg2rad(self.theta))
        # From http://en.wikipedia.org/wiki/Ellipse#General_parametric_form
        # However, Martin et al. (2009) use PA theta "from north to east"
        # Maybe the definitions of phi and theta are both offset by pi/2???
        # In the end, everything is messed up because we use glon, glat
        x = a*cosphi*costheta - b*sinphi*sintheta
        y = a*cosphi*sintheta + b*sinphi*costheta
        lon, lat = self.projector.imageToSphere(x, y)
        return lon, lat
 
    simulate = sample_lonlat
    sample = sample_lonlat

class EllipticalDisk(EllipticalKernel):
    """
    Simple uniform disk kernel for testing.
    f(r) = 1  for r <= r_0
    f(r) = 0  for r > r_0
    """
    _params = EllipticalKernel._params
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_0','extension')
        ])
                     
    ### ADW: stellar mass conversion?
    def _kernel(self, radius):
        return np.where(radius<=self.r_0, 1.0, 0.0) 
 
    @property
    def norm(self):
        return 1./(np.pi*self.r_0**2 * self.jacobian)
 
class EllipticalGaussian(EllipticalKernel):
    """
    Simple Gaussian kernel for testing:
    f(r) = C * exp(-r / 2*sigma**2)
    """
    _params = EllipticalKernel._params
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('sigma','extension')
        ])
 
    ### ADW: stellar mass conversion?
    def _kernel(self, radius):
        return np.exp(-radius/(2*self.sigma**2))
 
    @property
    def norm(self):
        # Analytic integral from 0 to edge
        return 1./(2*np.pi*self.sigma**2*(1-np.exp(-self.edge/(2*self.sigma**2)))*self.jacobian)
 
class EllipticalExponential(EllipticalKernel):
    """
    Stellar density distribution for Exponential profile:
    f(r) = C * exp(-r / r_e)
    where r_e = r_h/1.68
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 5)        
    """
    _params = odict(EllipticalKernel._params)
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_h','extension'), # Half-light radius
        ])
    
    def _kernel(self,radius):
        return np.exp(-radius/self.r_e)        
 
    @property
    def norm(self):
        # Analytic integral
        xedge = self.edge/self.r_e
        return 1./(2*np.pi*self.r_e**2*(1 - (xedge+1)*np.exp(-xedge))*self.jacobian)
 
    @property
    def r_e(self):
        # Exponential scale radius
        return self.r_h/1.68
 
    @property
    def edge(self):
        return 20.*self.r_h
 
class EllipticalPlummer(EllipticalKernel):
    """
    Stellar density distribution for Plummer profile:
    f(r) = C * r_h**2 / (r_h**2 + r**2)**2
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 6)
    """
    _params = odict(
        EllipticalKernel._params.items() + 
        [
            ('truncate',3.0), # Truncation radius
        ])
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_h','extension'), # Half-light radius
            ('r_t','truncate'),  # Tidal radius
        ])
 
    def _kernel(self, radius):
        return 1./(numpy.pi*self.r_h**2 * (1.+(radius/self.r_h)**2)**2)
 
    @property
    def u_t(self):
        # Truncation factor
        return self.truncate/self.extension
 
    @property
    def edge(self):
        return self.r_t
 
class EllipticalKing(EllipticalKernel):
    """
    Stellar density distribution for King profile:
    f(r) = C * [ 1/sqrt(1 + (r/r_c)**2) - 1/sqrt(1 + (r_t/r_c)**2) ]**2
    http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 4)
    """
    _params = odict(
        EllipticalKernel._params.items() + 
        [
            ('truncate',3.0), # Truncation radius
        ])
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_c','extension'), # Core radius
            ('r_t','truncate'),  # Tidal radius
        ])
 
    def _kernel(self, radius):
        return ((1./np.sqrt(1.+(radius/self.r_c)**2))-(1./np.sqrt(1.+(self.r_t/self.r_c)**2)))**2
 
    @property
    def c(self):
        return np.log10(self.r_t/self.r_c)
 
    @property
    def edge(self):
        return self.r_t

#####################################################
### Radial Kernels
#####################################################
 
class RadialKernel(EllipticalKernel):
    """
    Radial kernel subclass fixing ellipticity and 
    position angle to zero.
    """
    _fixed_params = ['ellipticity','position_angle']

    def setp(self,param,value):
        if param in self._fixed_params:
            msg = "Fixed parameter cannot be updated: %s"%param
            raise AttributeError(msg)
        super(RadialKernel,self).setp(param,value)
 
    def pdf(self, lon, lat):
        if self.proj is None:
            radius = angsep(self.lon,self.lat,lon,lat)
            return self.norm*self._pdf(radius)
        else:
            return super(RadialKernel,self).pdf(lon,lat)
        

class RadialDisk(RadialKernel,EllipticalDisk):
    pass

class RadialExponential(RadialKernel,EllipticalExponential):
    pass

class RadialGaussian(RadialKernel,EllipticalGaussian):
    pass

class RadialPlummer(RadialKernel,EllipticalPlummer):
    pass

class RadialKing(RadialKernel,EllipticalKing):
    pass
 
# For fast access...
Disk = RadialDisk
Gaussian = RadialGaussian
Exponential = RadialExponential
Plummer = RadialPlummer
King = RadialKing

def kernelFactory(name, **kwargs):
    """
    Factory for cerating spatial kernels. Arguments
    are passed directly to the constructor of the chosen
    kernel.
    """
    fn = lambda member: inspect.isclass(member) and member.__module__==__name__
    kernels = odict(inspect.getmembers(sys.modules[__name__], fn))

    if name not in kernels.keys():
        msg = "%s not found in kernels:\n %s"%(name,kernels.keys())
        logger.error(msg)
        msg = "Unrecognized kernel: %s"%name
        raise Exception(msg)

    return kernels[name](**kwargs)

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
