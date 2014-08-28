"""
Documentation.
"""

import numpy
import scipy.signal
import scipy.integrate
from abc import abstractmethod

import ugali.utils.plotting
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
        if extension < 0: 
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
        r_min : minimum integration radius
        r_max : maximum integration radius
        return : Solid angle integral 
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
    def _setup(self, *args):
        self.r_h = args[0]
        self.u_t = args[1] if len(args)>1 else 5. #numpy.inf #5.
        super(PlummerKernel,self)._setup(*args)

    def pdf(self, radius):
        """
        Stellar density distribution for Plummer profile:
        f(r) = C * r_h**2 / (r_h**2 + r**2)**2
        http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 6)
        """
        kernel = self.norm / (numpy.pi * self.r_h**2 * (1. + (radius / self.r_h)**2)**2)
        return numpy.where(radius<=self.edge(), kernel, 0 )

    def extension(self):
        return self.r_h

    def edge(self):
        return self.u_t * self.r_h

class KingKernel(RadialKernel):
    def _setup(self, *args):
        self.r_c = args[0] # Critical radius
        self.c   = args[1] # Concentration
        self.r_t = self.r_c * (10**self.c) # Tidal radius
        super(KingKernel,self)._setup(*args)

    def pdf(self, radius):
        """
        Stellar density distribution for King profile:
        f(r) = C * [ 1/sqrt(1 + (r/r_c)**2) - 1/sqrt(1 + (r_t/r_c)**2) ]**2
        http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 4)
        """
        kernel = self.norm * ((1. / numpy.sqrt(1. + (radius / self.r_c)**2)) - (1. / numpy.sqrt(1. + (self.r_t / self.r_c)**2)))**2
        return numpy.where( radius < self.edge(), kernel, 0 )

    def extension(self):
        return self.r_c

    def edge(self):
        return self.r_t

class ExponentialKernel(RadialKernel):
    def _setup(self, *args):
        self.r_h = args[0]
        self.r_e = self.r_h/1.68
        super(ExponentialKernel,self)._setup(*args)

    def pdf(self, radius):
        """
        Stellar density distribution for Exponential profile:
        f(r) = C * exp(-r / r_e)
        http://adsabs.harvard.edu//abs/2006MNRAS.365.1263M (Eq. 5)        
        """
        kernel = self.norm * numpy.exp(-radius/self.r_e)
        return numpy.where(radius<=self.edge(), kernel, 0 )

    def extension(self):
        return self.r_h

    def edge(self):
        return 20. * self.r_h


### ############################################################
###  
### class Kernel:
###     """
###     Normalized spatial kernel to describe candidate clusters.
###     """
###     
###     def __init__(self, config):
###  
###         self.config = config
###  
###         if self.config.params['kernel']['type'].lower() == 'disk':
###             self.kernel, self.bins_x, self.bins_y = disk(self.config.params['kernel']['params'],
###                                                          self.config.params['coords']['pixel_size'])
###         else:
###             print 'WARNING: kernel type %s not recognized'%(self.config.params['kernel']['type'])
###  
###     def plot(self):
###         ugali.utils.plotting.twoDimensionalHistogram('kernel', 'x (deg)', 'y (deg)',
###                                                      self.kernel, self.bins_x, self.bins_y,
###                                                      lim_x = [self.bins_x[0], self.bins_x[-1]],
###                                                      lim_y = [self.bins_y[0], self.bins_y[-1]])
###     
###     #def convolve(self, map):
###     #    map_convolve = scipy.signal.convolve(map, self.kernel, mode='same')
###     #    return map_convolve()
###         
###  
### ############################################################
###  
### def disk(params, pixel_size):
###  
###     disk_radius = params[0]
###  
###     disk_radius_pixels = numpy.ceil(disk_radius / pixel_size) # Disk radius in projected map pixels
###  
###     bins_x = numpy.linspace(-pixel_size * (disk_radius_pixels + 0.5),
###                             pixel_size * (disk_radius_pixels + 0.5),
###                             2 * (int(disk_radius_pixels) + 1))
###     bins_y = bins_x
###     centers_x = bins_x[0: -1] + (0.5 * pixel_size)
###     centers_y = centers_x
###     
###     mesh_x, mesh_y = numpy.meshgrid(centers_x, centers_y)
###  
###     r = numpy.sqrt(mesh_x**2 + mesh_y**2)
###  
###     kernel = r < disk_radius
###  
###     kernel = kernel.astype(float)
###  
###     kernel /= float(numpy.sum(kernel))
###  
###     return kernel, bins_x, bins_y
###  
### ############################################################


### ############################################################
###  
### # Should actually make child classes here for spatial profiles with azimuthal symmetry
###  
### class King:
###  
###     def __init__(self, lon, lat, r_c, c):
###         """
###         r_c is critical radius (deg). c is concentration.
###         """
###         self.setParams([lon, lat, r_c, c])
###  
###     def setParams(self, params):
###         self.lon = params[0]
###         self.lat = params[1]
###         self.r_c = params[2]
###         self.c = params[3]
###  
###         self.r_t = self.r_c * (10**self.c)
###         self.norm = 1. / self.integratePDF(0., self.r_t, prenormalized = False)
###  
###     def surfaceIntensity(self, r, prenormalized = True):
###         """
###         Evaluate surface intensity (deg^-2) at an arbitrary radius (deg).
###         """
###         if prenormalized:
###             norm = self.norm
###         else:
###             norm = 1.
###         r = numpy.array([r]).flatten()
###         result = numpy.zeros(len(r))
###         result += (r < self.r_t) * norm \
###                  * ((1. / numpy.sqrt(1. + (r / self.r_c)**2)) - (1. / numpy.sqrt(1. + (self.r_t / self.r_c)**2)))**2
###         return result
###  
###     def integratePDF(self, r_min, r_max, steps = 10000, prenormalized = True):
###         """
###         Integrate the King function PDF between r_min and r_max (deg).
###         """
###         r_edges = numpy.linspace(r_min, r_max, steps + 1)
###         d_r = (r_max - r_min) / float(steps)
###         r_centers = r_edges[1:] - (0.5 * d_r)
###         d_area = numpy.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
###         pdf = d_area * self.surfaceIntensity(r_centers, prenormalized = prenormalized)
###         return numpy.sum(pdf)
###  
###     def eval(self, r, r_max = 0., steps = 10000):
###         """
###         r is an array of angular separations (deg).
###         Retuns the PDF associated with each object and the area (deg^2) within the selected circular annuli.
###         """
###         if r_max <= numpy.max(r):
###             r_max = numpy.max(r) * (1. + (1./float(steps)))
###         r = numpy.array([r]).flatten()
###         r_edges = numpy.linspace(0., r_max, steps + 1)
###         d_r = r_max / float(steps)
###         r_centers = r_edges[1:] - (0.5 * d_r)
###         d_area = numpy.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
###         pdf = d_area * self.surfaceIntensity(r_centers)
###         r_digitized = numpy.digitize(r, r_edges) - 1
###         return pdf[r_digitized], d_area[r_digitized]
###  
###     def effectiveBackground(self, r_min, r_max, background_density, steps = 10000):
###         """
###         Compute the effective background (events) between r_min and r_max (deg).
###         """
###         r_edges = numpy.linspace(r_min, r_max, steps + 1)
###         d_r = (r_max - r_min) / float(steps)
###         r_centers = r_edges[1:] - (0.5 * d_r)
###         d_area = numpy.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
###         pdf = d_area * self.surfaceIntensity(r_centers, prenormalized = True)
###  
###         #print numpy.sum(pdf > 0.)
###  
###         #print numpy.sum(pdf * background_density * d_area > 0.)
###  
###         #print numpy.sum(background_density * d_area) / (numpy.pi * r_max**2)
###         
###         return numpy.sum(pdf * background_density * 2. * numpy.pi * r_centers**2)
###  
###         #return pdf, d_area
###  
###     def sample(self, n, steps = 10000):
###         """
###         Draw n samples from the King function PDF.
###         """
###         r_edges = numpy.linspace(0., self.r_t, steps + 1)
###         d_r = self.r_t / float(steps)
###         r_centers = r_edges[1:] - (0.5 * d_r)
###         d_area = numpy.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
###         pdf = d_area * self.surfaceIntensity(r_centers)
###  
###         f = scipy.interpolate.interp1d(numpy.insert(numpy.cumsum(pdf), 0, 0.),
###                                        range(0, len(r_centers) + 1))
###         r_indices = numpy.array(map(int, numpy.floor(f(numpy.random.rand(n)))))
###         return numpy.take(r_centers, r_indices)
###  
###  
### ############################################################
###  
### class Plummer:
###  
###     def __init__(self, lon, lat, r_h, u_t = 5.):
###         """
###         r_h is half-light radius (deg). u_t is truncation radius in units of the half-light radius.
###         """
###         self.setParams([lon, lat, r_h, u_t])
###  
###     def setParams(self, params):
###         self.lon = params[0]
###         self.lat = params[1]
###         self.r_h = params[2]
###         self.u_t = params[3]
###         
###         self.norm = 1. / self.integratePDF(0., self.r_h * self.u_t, prenormalized = False)
###  
###     def surfaceIntensity(self, r):
###         """
###         Evaluate surface intensity (deg^-2) at an arbitrary radius.
###         """
###         # ORIGINAL
###         #r = numpy.array([r]).flatten()
###         #u = r / self.r_h
###         #result = (u <= self.u_t) * norm * (1. + u**2)**(-2)
###         #return result
###         # ORIGINAL
###         
###         # Numerically fast version avoids negative exponents
###         return 1. / (numpy.pi * self.r_h**2 * (1. + (r / self.r_h)**2)**2)
###  
###     def integratePDF(self, r_min, r_max, prenormalized = True):
###         """
###         Integrate the Plummer function PDF between r_min and r_max (deg).
###         """
###         if prenormalized:
###             norm = self.norm
###         else:
###             norm = 1.
###         if r_max > (self.r_h * self.u_t):
###             r_max = self.r_h * self.u_t
###         return norm * numpy.pi * self.r_h**2 \
###                * (((r_min / self.r_h)**2 + 1.)**(-1) - ((r_max / self.r_h)**2 + 1.)**(-1))
###  
###     def simulate(self, n):
###         """
###         Return n simulated angular separations (deg) from center.
###         """
###         r = numpy.linspace(0, 20. * self.r_h, 1.e5)
###         pdf = self.surfaceIntensity(r) * numpy.sin(numpy.radians(r))
###         cdf = numpy.cumsum(pdf)
###         cdf /= cdf[-1]
###         f = scipy.interpolate.interp1d(cdf, range(0, len(cdf)), bounds_error=False, fill_value=-1)
###         index = numpy.floor(f(numpy.random.rand(n))).astype(int)
###         return r[index]
###  
###     def sample_lonlat(self, n):
###         """
###         Sample 2D distribution of points in lon, lat
###         Careful, doesn't respect sky projections
###         """
###         radius = self.simulate(n)
###         phi = 2. * numpy.pi * numpy.random.rand(n)
###         x = radius * numpy.cos(phi)
###         y = radius * numpy.sin(phi)
###         projector = ugali.utils.projector.Projector(self.lon, self.lat)
###         lon, lat = projector.imageToSphere(x, y)
###         return lon, lat

    
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
