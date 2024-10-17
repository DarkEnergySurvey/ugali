#!/usr/bin/env python
"""
Implementation of the (log) likelihood function.
"""
from collections import OrderedDict as odict
import time

import numpy as np

import healpy as hp
import fitsio

import ugali.utils.binning
import ugali.utils.parabola

from ugali.utils.projector import angsep
from ugali.utils.healpix import ang2pix,pix2ang,ang2disc
from ugali.utils.logger import logger

from ugali.utils.config import Config
from ugali.analysis.source import Source

class Observation(object):
    """
    Dummy class for storing catalog, roi, mask and other quantities
    associated with an "observation".
    """
    def __init__(self,**kwargs):
        self.__dict__.update(**kwargs)

class LogLikelihood(object):
    """
    Class for calculating the log-likelihood from a set of models.
    """

    def __init__(self, config, observation, source):
        # Currently assuming that input mask is ROI-specific
        self.config = Config(config)

        self.roi = observation.roi
        self.mask = observation.mask
        self.catalog_full = observation.catalog

        self.clip_catalog()

        # The source model (includes kernel and isochrone)
        self.source = source

        # Effective bin size in color-magnitude space
        self.delta_mag = self.config['likelihood']['delta_mag']

        self.spatial_only = self.config['likelihood'].get('spatial_only',False)
        self.color_only   = self.config['likelihood'].get('color_only',False)

        if self.spatial_only and self.color_only:
            msg = "Both 'spatial_only' and 'color_only' set"
            logger.error(msg)
            raise ValueError(msg)
        elif self.spatial_only: 
            logger.warning("Likelihood calculated from spatial information only!!!")
        elif self.color_only:
            logger.warning("Likelihood calculated from color information only!!!")

        self.calc_background()

    def __call__(self):
        # The signal probability for each object
        #self.p = (self.richness * self.u) / ((self.richness * self.u) + self.b)
        # The total model predicted counts
        #return -1. * np.sum(np.log(1.-self.p)) - (self.f * self.source.richness)
        return -1. * np.log(1.-self.p).sum() - (self.f * self.source.richness)
        
    def __str__(self):
        ret = "%s:\n"%self.__class__.__name__
        ret += str(self.source)
        return ret

    ############################################################################
    # Derived and protected properties
    ############################################################################

    # Various derived properties
    @property
    def kernel(self):
        #return self.models['spatial']
        return self.source.kernel

    @property
    def isochrone(self):
        #return self.models['color']
        return self.source.isochrone

    @property
    def pixel(self):
        nside = self.config.params['coords']['nside_pixel']
        pixel = ang2pix(nside,float(self.source.lon),float(self.source.lat))
        return pixel

    # Protect the basic elements of the likelihood
    @property
    def u(self):
        return self._u
    @property
    def b(self):
        return self._b
    @property
    def p(self):
        return self._p
    @property
    def f(self):
        return self._f
        
    def value(self,**kwargs):
        """
        Evaluate the log-likelihood at the given input parameter values
        """
        self.set_params(**kwargs)
        self.sync_params()
        return self()

    @property
    def nobs(self):
        """
        Number of observed stars.
        """
        return self.source.richness * self.f

    ############################################################################
    # Methods for setting model parameters
    ############################################################################

    def set_params(self,**kwargs):
        self.source.set_params(**kwargs)

        if self.pixel not in self.roi.pixels_interior:
            # ADW: Raising this exception is not strictly necessary, 
            # but at least a warning should be printed if target outside of region.
            raise ValueError("Coordinate outside interior ROI.")

    def sync_params(self):
        # The sync_params step updates internal quantities based on
        # newly set parameters. The goal is to only update required quantities
        # to keep computational time low.
 
        if self.source.get_sync('richness'):
            # No sync necessary for richness
            pass
        if self.source.get_sync('isochrone'):
            self.observable_fraction = self.calc_observable_fraction(self.source.distance_modulus)
            self.u_color = self.calc_signal_color(self.source.distance_modulus)
        if self.source.get_sync('kernel'):
            self.u_spatial = self.calc_signal_spatial()

        # Combined object-by-object signal probability
        self._u = self.u_spatial * self.u_color
         
        # Observable fraction requires update if isochrone changed
        # Fraction calculated over interior region
        self._f = self.roi.area_pixel * \
            (self.surface_intensity_sparse*self.observable_fraction).sum()

        ## ADW: Spatial information only (remember mag binning in calc_background)
        if self.spatial_only:
            self._u = self.u_spatial
            observable_fraction = (self.observable_fraction > 0)
            self._f = self.roi.area_pixel * \
                     (self.surface_intensity_sparse*observable_fraction).sum()

        # The signal probability for each object
        self._p = (self.source.richness * self.u)/((self.source.richness * self.u) + self.b)

        #logging.debug('b: %s'%np.unique(self.b)[:20])
        #logging.debug('u: %s'%np.unique(self.u)[:20])
        #logging.debug('f: %s'%np.unique(self.f)[:20])
        #logging.debug('p: %s'%np.unique(self.p)[:20])

        # Reset the sync toggle
        self.source.reset_sync()

    ############################################################################
    # Methods for calculating observation quantities
    ############################################################################

    def clip_catalog(self):
        # ROI-specific catalog
        logger.debug("Clipping full catalog...")
        cut_observable = self.mask.restrictCatalogToObservableSpace(self.catalog_full)

        # All objects within disk ROI
        logger.debug("Creating roi catalog...")
        self.catalog_roi = self.catalog_full.applyCut(cut_observable)
        self.catalog_roi.project(self.roi.projector)
        self.catalog_roi.spatialBin(self.roi)

        # All objects interior to the background annulus
        logger.debug("Creating interior catalog...")
        cut_interior = np.in1d(ang2pix(self.config['coords']['nside_pixel'], self.catalog_roi.lon, self.catalog_roi.lat), 
                                  self.roi.pixels_interior)
        #cut_interior = self.roi.inInterior(self.catalog_roi.lon,self.catalog_roi.lat)
        self.catalog_interior = self.catalog_roi.applyCut(cut_interior)
        self.catalog_interior.project(self.roi.projector)
        self.catalog_interior.spatialBin(self.roi)

        # Set the default catalog
        #logger.info("Using interior ROI for likelihood calculation")
        self.catalog = self.catalog_interior
        #self.pixel_roi_cut = self.roi.pixel_interior_cut

    def calc_backgroundCMD(self):
        #ADW: At some point we may want to make the background level a fit parameter.
        logger.info('Calculating background CMD ...')
        self.cmd_background = self.mask.backgroundCMD(self.catalog_roi)
        #self.cmd_background = self.mask.backgroundCMD(self.catalog_roi,mode='histogram')
        #self.cmd_background = self.mask.backgroundCMD(self.catalog_roi,mode='uniform')
        # Background density (deg^-2 mag^-2) and background probability for each object
        logger.info('Calculating background probabilities ...')
        b_density = ugali.utils.binning.take2D(self.cmd_background,
                                               self.catalog.color, self.catalog.mag,
                                               self.roi.bins_color, self.roi.bins_mag)

        # ADW: I don't think this 'area_pixel' or 'delta_mag' factors are necessary, 
        # so long as it is also removed from u_spatial and u_color
        #self._b = b_density * self.roi.area_pixel * self.delta_mag**2
        self._b = b_density

        if self.spatial_only:
            # ADW: This assumes a flat mask...
            #solid_angle_annulus = (self.mask.mask_1.mask_annulus_sparse > 0).sum()*self.roi.area_pixel
            solid_angle_annulus = ((self.mask.mask_1.mask_annulus_sparse > 0)*self.mask.frac_annulus_sparse).sum()*self.roi.area_pixel
            b_density = self.roi.inAnnulus(self.catalog_roi.lon,self.catalog_roi.lat).sum()/solid_angle_annulus
            self._b = np.array([b_density*self.roi.area_pixel])

    def calc_backgroundMMD(self):
        #ADW: At some point we may want to make the background level a fit parameter.
        logger.info('Calculating background MMD ...')
        self.mmd_background = self.mask.backgroundMMD(self.catalog_roi)
        #self.mmd_background = self.mask.backgroundMMD(self.catalog_roi,mode='histogram')
        #self.mmd_background = self.mask.backgroundMMD(self.catalog_roi,mode='uniform')
        # Background density (deg^-2 mag^-2) and background probability for each object
        logger.info('Calculating background probabilities ...')
        b_density = ugali.utils.binning.take2D(self.mmd_background,
                                               self.catalog.mag_2, self.catalog.mag_1,
                                               self.roi.bins_mag, self.roi.bins_mag)

        # ADW: I don't think this 'area_pixel' or 'delta_mag' factors are necessary, 
        # so long as it is also removed from u_spatial and u_color
        #self._b = b_density * self.roi.area_pixel * self.delta_mag**2
        self._b = b_density

        if self.spatial_only:
            # ADW: This assumes a flat mask...
            #solid_angle_annulus = (self.mask.mask_1.mask_annulus_sparse > 0).sum()*self.roi.area_pixel
            solid_angle_annulus = ((self.mask.mask_1.mask_annulus_sparse > 0)*self.mask.frac_annulus_sparse).sum()*self.roi.area_pixel
            b_density = self.roi.inAnnulus(self.catalog_roi.lon,self.catalog_roi.lat).sum()/solid_angle_annulus
            self._b = np.array([b_density*self.roi.area_pixel])

    # FIXME: Need to parallelize CMD and MMD formulation
    calc_background = calc_backgroundCMD

    def calc_observable_fraction(self,distance_modulus):
        """
        Calculated observable fraction within each pixel of the target region.
        """
        # This is the observable fraction after magnitude cuts in each 
        # pixel of the ROI.
        observable_fraction = self.isochrone.observableFraction(self.mask,distance_modulus)
        if not observable_fraction.sum() > 0:
            msg = "No observable fraction"
            msg += ("\n"+str(self.source.params))
            logger.error(msg)
            raise ValueError(msg)
        return observable_fraction

    def calc_signal_color1(self, distance_modulus, mass_steps=10000):
        """
        Compute signal color probability (u_color) for each catalog object on the fly.
        """
        mag_1, mag_2 = self.catalog.mag_1,self.catalog.mag_2
        mag_err_1, mag_err_2 = self.catalog.mag_err_1,self.catalog.mag_err_2
        u_density = self.isochrone.pdf(mag_1,mag_2,mag_err_1,mag_err_2,distance_modulus,self.delta_mag,mass_steps)

        #u_color = u_density * self.delta_mag**2
        u_color = u_density

        return u_color

    def calc_signal_color2(self, distance_modulus, mass_steps=1000):
        """
        Compute signal color probability (u_color) for each catalog object on the fly.
        """
        logger.info('Calculating signal color from MMD')

        mag_1, mag_2 = self.catalog.mag_1,self.catalog.mag_2
        lon, lat = self.catalog.lon,self.catalog.lat
        u_density = self.isochrone.pdf_mmd(lon,lat,mag_1,mag_2,distance_modulus,self.mask,self.delta_mag,mass_steps)

        #u_color = u_density * self.delta_mag**2
        u_color = u_density

        # ADW: Should calculate observable fraction here as well...

        return u_color

    # FIXME: Need to parallelize CMD and MMD formulation
    calc_signal_color = calc_signal_color1

    def calc_surface_intensity(self, factor=10):
        """Calculate the surface intensity for each pixel in the interior
        region of the ROI. Pixels are adaptively subsampled around the
        kernel centroid out to a radius of 'factor * max_pixrad'.

        Parameters:
        -----------
        factor : the radius of the oversample region in units of max_pixrad

        Returns:
        --------
        surface_intensity : the surface intensity at each pixel
        """
        # First we calculate the surface intensity at native resolution
        pixels = self.roi.pixels_interior
        nside_in = self.config['coords']['nside_pixel']
        surface_intensity = self.kernel.pdf(pixels.lon,pixels.lat)

        # Then we recalculate the surface intensity around the kernel
        # centroid at higher resolution
        for i in np.arange(1,5):
            # Select pixels within the region of interest
            nside_out = 2**i * nside_in
            radius = factor*np.degrees(hp.max_pixrad(nside_out))
            pix = ang2disc(nside_in,self.kernel.lon,self.kernel.lat,
                           radius,inclusive=True)

            # Select pix within the interior region of the ROI
            idx = ugali.utils.healpix.index_pix_in_pixels(pix,pixels)
            pix = pix[(idx >= 0)]; idx = idx[(idx >= 0)]

            # Reset the surface intensity for the subsampled pixels
            subpix = ugali.utils.healpix.ud_grade_ipix(pix,nside_in,nside_out)
            pix_lon,pix_lat = pix2ang(nside_out,subpix)
            surface_intensity[idx]=np.mean(self.kernel.pdf(pix_lon,pix_lat),axis=1)

        return surface_intensity

    def calc_signal_spatial(self):
        """
        Calculate the spatial signal probability for each catalog object.

        Parameters:
        -----------
        None

        Returns:
        --------
        u_spatial : array of spatial probabilities per object
        """
        # Calculate the surface intensity
        self.surface_intensity_sparse = self.calc_surface_intensity()

        # Calculate the probability per object-by-object level
        self.surface_intensity_object = self.kernel.pdf(self.catalog.lon,
                                                        self.catalog.lat)
        
        # Spatial component of signal probability
        u_spatial = self.surface_intensity_object
        return u_spatial

    ############################################################################
    # Methods for fitting and working with the likelihood
    ############################################################################

    def ts(self):
        return 2*self()

    def fit_richness(self, atol=1.e-3, maxiter=50):
        """
        Maximize the log-likelihood as a function of richness.

        ADW 2018-06-04: Does it make sense to set the richness to the mle?

        Parameters:
        -----------
        atol : absolute tolerence for conversion
        maxiter : maximum number of iterations

        Returns:
        --------
        loglike, richness, parabola : the maximum loglike, the mle, and the parabola
        """
        # Check whether the signal probability for all objects are zero
        # This can occur for finite kernels on the edge of the survey footprint
        if np.isnan(self.u).any():
            logger.warning("NaN signal probability found")
            return 0., 0., None
        
        if not np.any(self.u):
            logger.warning("Signal probability is zero for all objects")
            return 0., 0., None

        if self.f == 0:
            logger.warning("Observable fraction is zero")
            return 0., 0., None

        # Richness corresponding to 0, 1, and 10 observable stars
        richness = np.array([0., 1./self.f, 10./self.f])
        loglike = np.array([self.value(richness=r) for r in richness])

        found_maximum = False
        iteration = 0
        while not found_maximum:
            parabola = ugali.utils.parabola.Parabola(richness, 2.*loglike)
            if parabola.vertex_x < 0.:
                found_maximum = True
            else:
                richness = np.append(richness, parabola.vertex_x)
                loglike  = np.append(loglike, self.value(richness=richness[-1]))

                if np.fabs(loglike[-1] - np.max(loglike[0: -1])) < atol:
                    found_maximum = True
            iteration+=1
            if iteration > maxiter:
                logger.warning("Maximum number of iterations reached")
                break
            
        index = np.argmax(loglike)
        return loglike[index], richness[index], parabola

    def richness_interval(self, alpha=0.6827, n_pdf_points=100):
        loglike_max, richness_max, parabola = self.fit_richness()

        richness_range = parabola.profileUpperLimit(delta=25.) - richness_max
        richness = np.linspace(max(0., richness_max - richness_range),
                                  richness_max + richness_range,
                                  n_pdf_points)
        if richness[0] > 0.:
            richness = np.insert(richness, 0, 0.)
            n_pdf_points += 1
        
        log_likelihood = np.zeros(n_pdf_points)
        for kk in range(0, n_pdf_points):
            log_likelihood[kk] = self.value(richness=richness[kk])
        parabola = ugali.utils.parabola.Parabola(richness, 2.*log_likelihood)
        return parabola.confidenceInterval(alpha)

    # Writing membership files
    def write_membership(self,filename):
        """
        Write a catalog file of the likelihood region including
        membership properties.

        Parameters:
        -----------
        filename : output filename
        
        Returns:
        --------
        None
        """
        # Column names
        name_objid = self.config['catalog']['objid_field']
        name_mag_1 = self.config['catalog']['mag_1_field']
        name_mag_2 = self.config['catalog']['mag_2_field']
        name_mag_err_1 = self.config['catalog']['mag_err_1_field']
        name_mag_err_2 = self.config['catalog']['mag_err_2_field']

        # Coordinate conversion
        #ra,dec = gal2cel(self.catalog.lon,self.catalog.lat)
        glon,glat = self.catalog.glon_glat
        ra,dec    = self.catalog.ra_dec

        # Angular and isochrone separations
        sep = angsep(self.source.lon,self.source.lat,
                     self.catalog.lon,self.catalog.lat)
        isosep = self.isochrone.separation(self.catalog.mag_1,self.catalog.mag_2)

        # If size becomes an issue we can make everything float32
        data = odict()
        data[name_objid]     = self.catalog.objid
        data['GLON']         = glon
        data['GLAT']         = glat
        data['RA']           = ra
        data['DEC']          = dec
        data[name_mag_1]     = self.catalog.mag_1
        data[name_mag_err_1] = self.catalog.mag_err_1
        data[name_mag_2]     = self.catalog.mag_2
        data[name_mag_err_2] = self.catalog.mag_err_2
        data['COLOR']        = self.catalog.color
        data['ANGSEP']       = sep.astype(np.float32)
        data['ISOSEP']       = isosep.astype(np.float32)
        data['PROB']         = self.p.astype(np.float32)
     
        # HIERARCH allows header keywords longer than 8 characters
        header = []
        for param,value in self.source.params.items():
            card = dict(name='HIERARCH %s'%param.upper(),
                        value=value.value,
                        comment=param)
            header.append(card)
        card = dict(name='HIERARCH %s'%'TS',value=self.ts(),
                    comment='test statistic')
        header.append(card)
        card = dict(name='HIERARCH %s'%'TIMESTAMP',value=time.asctime(),
                    comment='creation time')
        header.append(card)
        fitsio.write(filename,data,header=header,clobber=True)

def write_membership(filename,config,srcfile,section=None):
    """
    Top level interface to write the membership from a config and source model.
    """
    source = Source()
    source.load(srcfile,section=section)
    loglike = createLoglike(config,source)
    loglike.write_membership(filename)

# This should probably be moved into ugali.analysis.source...
def createSource(config, section=None, **kwargs):
    config = Config(config)
    source = Source()

    if config.get(section) is not None:
        params = config.get(section).get('source')
    else:
        params = config.get('source')

    if params is not None:
        source.load(params)

    source.set_params(**kwargs)
    return source

### probably don't need these ###
def createKernel(config, **kwargs):
    return createSource(config,**kwargs).kernel

def createIsochrone(config, **kwargs):
    return createSource(config,**kwargs).isochrone

### probably move these to observation ###
def createObservation(config,lon,lat):
    roi = createROI(config,lon,lat)
    catalog = createCatalog(config,roi)
    mask = createMask(config,roi)
    return Observation(roi=roi,mask=mask,catalog=catalog)

def createROI(config,lon,lat):
    import ugali.observation.roi
    roi = ugali.observation.roi.ROI(config, lon, lat)        
    return roi

def createMask(config,roi=None,lon=None,lat=None):
    import ugali.observation.mask
    if roi is None: 
        if lon is None or lat is None: 
            msg = "Without `roi`, `lon` and `lat` must be specified"
            raise Exception(msg)
        roi = createROI(config,lon,lat)
    mask = ugali.observation.mask.Mask(config, roi)
    return mask

def createCatalog(config,roi=None,lon=None,lat=None):
    """
    Create a catalog object
    """
    import ugali.observation.catalog
    if roi is None: roi = createROI(config,lon,lat)
    catalog = ugali.observation.catalog.Catalog(config,roi=roi)
    return catalog

### move to simulate ###
def simulateCatalog(config,roi=None,lon=None,lat=None):
    """
    Simulate a catalog object.
    """
    import ugali.simulation.simulator
    if roi is None: roi = createROI(config,lon,lat)
    sim = ugali.simulation.simulator.Simulator(config,roi)
    return sim.catalog()

def createLoglike(config,source=None,lon=None,lat=None):

    if isinstance(source,str):
        srcfile = source
        source = ugali.analysis.source.Source()
        source.load(srcfile,section='source')
    if source is not None:
        lon,lat = source.lon,source.lat
    else:
        if lon is None or lat is None:
            msg = "Without `source`, `lon` and `lat` must be specified"
            raise Exception(msg)
        source = createSource(config,lon=lon,lat=lat)
        
    observation = createObservation(config,lon,lat)
    loglike = LogLikelihood(config,observation,source)
    loglike.sync_params()
    return loglike

if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args

