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
import healpy

import ugali.utils.projector
from ugali.utils.projector import Projector, angsep
from ugali.analysis.model import Model, Parameter
from ugali.utils.healpix import ang2vec, ang2pix, query_disc

from ugali.utils.logger import logger

class Kernel(Model):
    """
    Base class for kernels.
    """
    _params = odict([
        ('lon',      Parameter(0.0, [0.0,  360.  ])),
        ('lat',      Parameter(0.0, [-90., 90.   ])),
    ])
    _mapping = odict([])
    
    def __init__(self, proj='ait',**kwargs):
        self.proj = proj
        super(Kernel,self).__init__(**kwargs)
 
    def __call__(self, lon, lat):
        return self.pdf(lon, lat)

    @abstractmethod
    def _kernel(self, r):
        # unnormalized, untruncated kernel
        pass

    def _pdf(self, radius):
        # unnormalized, truncated kernel
        return np.where(radius<=self.edge, self._kernel(radius), 0.)

    @abstractmethod
    def pdf(self, lon, lat):
        # normalized, truncated pdf
        pass

    @property
    def norm(self):
        # Numerically integrate the pdf
        return 1./self.integrate()

    @property
    def projector(self):
        if self.proj is None or self.proj.lower()=='none':
            return None
        else:
            return Projector(self.lon, self.lat, self.proj)

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

class ToyKernel(Kernel):
    """
    Simple toy kernel that selects healpix pixels within
    the given extension radius. Similar to 'RadialDisk'.
    """

    _params = odict(
        Kernel._params.items() + 
        [
            ('extension',     Parameter(0.5, [0.01,5.0]) ),
            ('nside',         Parameter(4096,[4096,4096])),
        ])

    def _cache(self, name):
        pixel_area = healpy.nside2pixarea(self.nside,degrees=True)
        vec = ang2vec(self.lon, self.lat)
        self.pix = query_disc(self.nside,vec,self.extension)
        self._norm = 1./(len(self.pix)*pixel_area)

    @property
    def norm(self):
        return self._norm

    def _pdf(self,pix):
        return  np.in1d(pix,self.pix)
        
    def pdf(self,lon,lat):
        pix = ang2pix(self.nside,lon,lat)
        return self.norm * self._pdf(pix)
       
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
            ('extension',     Parameter(0.5, [0.01,5.0]) ),
            ('ellipticity',   Parameter(0.0, [0.0, 1.0]) ),  # Default 0 for RadialKernel
            ('position_angle',Parameter(0.0, [0.0, 1.0]) ),  # Default 0 for RadialKernel
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
        # In the end, everything is messed up because we use glon, glat...
        x = a*cosphi*costheta - b*sinphi*sintheta
        y = a*cosphi*sintheta + b*sinphi*costheta
        
        if self.projector  is None:
            logger.debug("Creating AITOFF projector for sampling")
            projector = Projector(self.lon,self.lat,'ait')
        else:
            projector = self.projector
        lon, lat = projector.imageToSphere(x, y)
        return lon, lat
 
    simulate = sample_lonlat
    sample = sample_lonlat

    # Back-compatibility
    def setExtension(self,extension):
        self.extension = extension

    def setCenter(self,lon,lat):
        self.lon = lon
        self.lat = lat

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
    f(r) = C * exp(-r**2 / 2*sigma**2)
    """
    _params = EllipticalKernel._params
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('sigma','extension')
        ])
 
    ### ADW: stellar mass conversion?
    def _kernel(self, radius):
        return np.exp(-radius**2/(2*self.sigma**2))
 
    @property
    def norm(self):
        # Analytic integral from 0 to edge
        return 1./(2*np.pi*self.sigma**2*(1-np.exp(-self.edge**2/(2*self.sigma**2)))*self.jacobian)
 
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
            ('truncate', Parameter(3.0, [0.0, np.inf]) ), # Truncation radius
        ])
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_h','extension'), # Half-light radius
            ('r_t','truncate'),  # Tidal radius
        ])
 
    def _kernel(self, radius):
        return 1./(numpy.pi*self.r_h**2 * (1.+(radius/self.r_h)**2)**2)

    def _cache(self, name=None):
        if name in ['extension','ellipticity','truncate']:
            self._norm = 1./self.integrate()
        else:
            return

    @property
    def norm(self):
        return self._norm

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
            ('truncate', Parameter(3.0, [0.0, np.inf]) ), # Truncation radius
        ])
    _mapping = odict(
        EllipticalKernel._mapping.items() +
        [
            ('r_c','extension'), # Core radius
            ('r_t','truncate'),  # Tidal radius
        ])
 
    def _kernel(self, radius):
        return ((1./np.sqrt(1.+(radius/self.r_c)**2))-(1./np.sqrt(1.+(self.r_t/self.r_c)**2)))**2

    def _cache(self, name=None):
        if name in ['extension','ellipticity','truncate']:
            self._norm = 1./self.integrate()
        else:
            return

    @property
    def norm(self):
        return self._norm
 
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

    def __init__(self,**kwargs):
        self._params['ellipticity'].set(0.0, [0.0,0.0])
        self._params['position_angle'].set(0.0, [0.0,0.0])
        super(RadialKernel,self).__init__(**kwargs)
        
    def pdf(self, lon, lat):
        if self.projector is None:
            radius = angsep(self.lon,self.lat,lon,lat)
            return self.norm*self._pdf(radius)
        else:
            return super(RadialKernel,self).pdf(lon,lat)

    # Back-compatibility
    def surfaceIntensity(self,radius):
        return self.norm*self._pdf(radius)
        
# For fast access...
class RadialDisk(RadialKernel,EllipticalDisk): pass
class RadialExponential(RadialKernel,EllipticalExponential): pass
class RadialGaussian(RadialKernel,EllipticalGaussian): pass
class RadialPlummer(RadialKernel,EllipticalPlummer): pass
class RadialKing(RadialKernel,EllipticalKing): pass
 
Disk        = RadialDisk
Gaussian    = RadialGaussian
Exponential = RadialExponential
Plummer     = RadialPlummer
King        = RadialKing

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
