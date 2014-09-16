#!/usr/bin/env python
"""
Class to create and run an individual likelihood analysis.

Classes:
    Scan
    GridSearch

"""

import os
import sys
from collections import OrderedDict as odict

import numpy
import numpy as np
import pyfits
import healpy

import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.color_lut
import ugali.observation.catalog
import ugali.observation.mask
import ugali.utils.parabola
import ugali.utils.skymap
from ugali.analysis.loglike  import LogLikelihood 

from ugali.utils.config import Config
from ugali.utils.logger import logger
from ugali.utils.healpix import superpixel, subpixel, pix2ang, ang2pix


############################################################

class Scan(object):
    """
    The base of a likelihood analysis scan.
    """
    def __init__(self, config, coords):
        self.config = Config(config)
        # Should only be one coordinate
        if len(coords)!=1: raise Exception('Must specify one coordinate.')
        self.lon,self.lat,radius = coords[0]
        self._setup()

    def _setup(self):
        self.nside_catalog    = self.config['coords']['nside_catalog']
        self.nside_likelihood = self.config['coords']['nside_likelihood']
        self.nside_pixel      = self.config['coords']['nside_pixel']

        # All possible filenames
        self.filenames = self.config.getFilenames()
        # ADW: Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])

        #self.roi = ugali.observation.roi.ROI(self.config, self.lon, self.lat)
        self.roi = self.createROI(self.config,self.lon,self.lat)
        # All possible catalog pixels spanned by the ROI
        catalog_pixels = numpy.unique(superpixel(self.roi.pixels,self.nside_pixel,self.nside_catalog))
        # Only catalog pixels that exist in catalog files
        self.catalog_pixels = numpy.intersect1d(catalog_pixels, self.filenames['pix'].compressed())

        self.kernel = self.createKernel(self.config,self.lon,self.lat)
        self.isochrone = self.createIsochrone(self.config)
        self.catalog = self.createCatalog(self.config,self.roi)
        self.mask = self.createMask(self.config,self.roi)

        self.grid = GridSearch(self.config, self.roi, self.mask,self.catalog, 
                               self.isochrone, self.kernel)

    @property
    def loglike(self):
        return self.grid.loglike

    @staticmethod
    def createROI(config,lon,lat):
        roi = ugali.observation.roi.ROI(config, lon, lat)        
        return roi

    @staticmethod
    def createKernel(config,lon=0.0,lat=0.0):
        params = config['scan']['kernel']
        params.setdefault('lon',lon)
        params.setdefault('lat',lat)
        kernel = ugali.analysis.kernel.kernelFactory(**params)
        return kernel

    @staticmethod
    def createIsochrone(config):
        isochrones = []
        for ii, name in enumerate(config['isochrone']['infiles']):
            isochrones.append(ugali.analysis.isochrone.Isochrone(config, name))
        isochrone = ugali.analysis.isochrone.CompositeIsochrone(isochrones, config['isochrone']['weights'])
        return isochrone

    @staticmethod
    def createCatalog(config,roi=None,lon=None,lat=None):
        """
        Find the relevant catalog files for this scan.
        """
        if roi is None: roi = Scan.createROI(config,lon,lat)
        catalog = ugali.observation.catalog.Catalog(config,roi=roi)  
        return catalog

    @staticmethod
    def simulateCatalog(config,roi=None,lon=None,lat=None):
        """
        !!! PLACEHOLDER: Integrate the simulation structure more tightly with
        the analysis structure to avoid any nasty disconnects. !!!
        """
        pass

    @staticmethod
    def createMask(config,roi=None,lon=None,lat=None):
        if roi is None: roi = Scan.createROI(config,lon,lat)
        mask = ugali.observation.mask.Mask(config, roi)
        return mask

    def run(self, coords=None, debug=False):
        """
        Run the likelihood grid search
        """
        #self.grid.precompute()
        self.grid.search()
        return self.grid
        
    def write(self, outfile):
        self.grid.write(outfile)


############################################################

class GridSearch:

    def __init__(self, config, roi, mask, catalog, isochrone, kernel):
        """
        Object to efficiently search over a grid of ROI positions.

        ADW: This should probably be renamed GridSearch or something like that
        since we would like a pure access to the likelihood itself.
        """

        self.config = config
        self.roi = roi
        self.mask = mask # Currently assuming that input mask is ROI-specific

        logger.info("Creating log-likelihood...")
        self.loglike=LogLikelihood(config,roi,mask,catalog,isochrone,kernel)

        self.stellar_mass_conversion = self.loglike.stellar_mass()
        self.distance_modulus_array = self.config['scan']['distance_modulus_array']

    def precompute(self, distance_modulus_array=None):
        """
        Precompute u_background and u_color for each star in catalog.
        Precompute observable fraction in each ROI pixel.
        # Precompute still operates over the full ROI, not just the likelihood region
        """
        if distance_modulus_array is not None:
            self.distance_modulus_array = distance_modulus_array
        else:
            self.distance_modulus_array = self.config['scan']['distance_modulus_array']

        # Observable fraction for each pixel
        self.u_color_array = [[]] * len(self.distance_modulus_array)
        self.observable_fraction_sparse_array = [[]] * len(self.distance_modulus_array)

        logger.info('Looping over distance moduli in precompute ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            logger.info('  (%i/%i) Distance Modulus = %.2f ...'%(ii+1, len(self.distance_modulus_array), distance_modulus))

            self.u_color_array[ii] = False
            if self.config['scan']['color_lut_infile'] is not None:
                logger.info('  Precomputing signal color from %s'%(self.config['scan']['color_lut_infile']))
                self.u_color_array[ii] = ugali.analysis.color_lut.readColorLUT(self.config['scan']['color_lut_infile'],
                                                                               distance_modulus,
                                                                               self.loglike.catalog.mag_1,
                                                                               self.loglike.catalog.mag_2,
                                                                               self.loglike.catalog.mag_err_1,
                                                                               self.loglike.catalog.mag_err_2)
            if not numpy.any(self.u_color_array[ii]):
                logger.info('  Precomputing signal color on the fly...')
                self.u_color_array[ii] = self.loglike.calc_signal_color(distance_modulus) 
            
            # Calculate over all pixels in ROI
            self.observable_fraction_sparse_array[ii] = self.loglike.calc_observable_fraction(distance_modulus)
            
        self.u_color_array = numpy.array(self.u_color_array)

                
    def search(self, coords=None, distance_modulus_index=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        """
        nmoduli = len(self.distance_modulus_array)
        npixels    = len(self.roi.pixels_target)
        self.log_likelihood_sparse_array       = numpy.zeros([nmoduli, npixels])
        self.richness_sparse_array             = numpy.zeros([nmoduli, npixels])
        self.richness_lower_sparse_array       = numpy.zeros([nmoduli, npixels])
        self.richness_upper_sparse_array       = numpy.zeros([nmoduli, npixels])
        self.richness_upper_limit_sparse_array = numpy.zeros([nmoduli, npixels])
        self.stellar_mass_sparse_array         = numpy.zeros([nmoduli, npixels])
        self.fraction_observable_sparse_array  = numpy.zeros([nmoduli, npixels])

        # Specific pixel
        if coords is not None:
            pix_coords = ang2pix(coords)

        lon, lat = self.roi.pixels_target.lon, self.roi.pixels_target.lat
            
        logger.info('Looping over distance moduli in grid search ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):

            # Specific pixel
            if distance_modulus_index is not None:
                if ii != distance_modulus_index: continue

            logger.info('  (%-2i/%i) Distance Modulus=%.1f ...'%(ii+1,nmoduli,distance_modulus))

            # Set distance_modulus once to save time
            self.loglike.set_params(distance_modulus=distance_modulus)
            #self.loglike.sync_params()

            for jj in range(0, npixels):
                # Specific pixel
                if coords is not None:
                    if self.roi.pixels_target[jj] != pix_coords:
                        continue
                
                # Set kernel location
                self.loglike.set_params(lon=lon[jj],lat=lat[jj])
                # Doesn't re-sync distance_modulus each time
                self.loglike.sync_params()
                                         
                args = (jj+1, npixels, self.loglike.lon, self.loglike.lat)
                message = '    (%-3i/%i) Candidate at (%.2f, %.2f) ... '%(args)

                self.log_likelihood_sparse_array[ii][jj], self.richness_sparse_array[ii][jj], parabola = self.loglike.fit_richness()
                self.stellar_mass_sparse_array[ii][jj] = self.stellar_mass_conversion * self.richness_sparse_array[ii][jj]
                self.fraction_observable_sparse_array[ii][jj] = self.loglike.f
                if self.config['scan']['full_pdf'] \
                   or (coords is not None and distance_modulus_index is not None):

                    n_pdf_points = 100
                    richness_range = parabola.profileUpperLimit(delta=25.) - self.richness_sparse_array[ii][jj]
                    richness = numpy.linspace(max(0., self.richness_sparse_array[ii][jj] - richness_range),
                                              self.richness_sparse_array[ii][jj] + richness_range,
                                              n_pdf_points)
                    if richness[0] > 0.:
                        richness = numpy.insert(richness, 0, 0.)
                        n_pdf_points += 1
                    
                    log_likelihood = numpy.zeros(n_pdf_points)
                    for kk in range(0, n_pdf_points):
                        log_likelihood[kk] = self.loglike.value(richness=richness[kk])
                    parabola = ugali.utils.parabola.Parabola(richness, 2.*log_likelihood)
                    self.richness_lower_sparse_array[ii][jj], self.richness_upper_sparse_array[ii][jj] = parabola.confidenceInterval(0.6827)
                    self.richness_upper_limit_sparse_array[ii][jj] = parabola.bayesianUpperLimit(0.95)

                    args = (
                        2. * self.log_likelihood_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_lower_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_upper_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_upper_limit_sparse_array[ii][jj]
                    )
                    message += 'TS=%.1f, Stellar Mass=%.1f (%.1f -- %.1f @ 0.68 CL, < %.1f @ 0.95 CL)'%(args)
                else:
                    args = (
                        2. * self.log_likelihood_sparse_array[ii][jj], 
                        self.stellar_mass_conversion * self.richness_sparse_array[ii][jj],
                        self.fraction_observable_sparse_array[ii][jj]
                    )
                    message += 'TS=%.1f, Stellar Mass=%.1f, Fraction=%.2g'%(args)
                logger.debug( message )
                
                if coords is not None and distance_modulus_index is not None:
                    results = [self.richness_sparse_array[ii][jj],
                               self.log_likelihood_sparse_array[ii][jj],
                               self.richness_lower_sparse_array[ii][jj],
                               self.richness_upper_sparse_array[ii][jj],
                               self.richness_upper_limit_sparse_array[ii][jj],
                               richness, log_likelihood, self.loglike.p, self.loglike.f]
                    return results

            jj_max = self.log_likelihood_sparse_array[ii].argmax()
            args = (
                jj_max+1, npixels, lon[jj_max], lat[jj_max],
                2. * self.log_likelihood_sparse_array[ii][jj_max], 
                self.stellar_mass_conversion * self.richness_sparse_array[ii][jj_max]
            )
            message = '  (%-3i/%i) Maximum at (%.2f, %.2f) ... TS=%.1f, Stellar Mass=%.1f'%(args)
            logger.info( message )
 
    def mle(self):
        a = self.log_likelihood_sparse_array
        j,k = np.unravel_index(a.argmax(),a.shape)
        mle = odict()
        mle['richness'] = self.richness_sparse_array[j][k]
        mle['lon'] = self.roi.pixels_target.lon[k]
        mle['lat'] = self.roi.pixels_target.lat[k]
        mle['distance_modulus'] = self.distance_modulus_array[j]
        mle['extension'] = float(self.loglike.extension)
        mle['ellipticity'] = float(self.loglike.ellipticity)
        mle['position_angle'] = float(self.loglike.position_angle)
        return mle

    def write(self, outfile):
        """
        Save the likelihood fitting results as a sparse HEALPix map.
        """
        # Full data output (too large for survey)
        if self.config['scan']['full_pdf']:
            data_dict = {'LOG_LIKELIHOOD': self.log_likelihood_sparse_array.transpose(),
                         'RICHNESS':       self.richness_sparse_array.transpose(),
                         'RICHNESS_LOWER': self.richness_lower_sparse_array.transpose(),
                         'RICHNESS_UPPER': self.richness_upper_sparse_array.transpose(),
                         'RICHNESS_LIMIT': self.richness_upper_limit_sparse_array.transpose(),
                         #'STELLAR_MASS': self.stellar_mass_sparse_array.transpose(),
                         'FRACTION_OBSERVABLE': self.fraction_observable_sparse_array.transpose()}
        else:
            data_dict = {'LOG_LIKELIHOOD': self.log_likelihood_sparse_array.transpose(),
                         'RICHNESS': self.richness_sparse_array.transpose(),
                         'FRACTION_OBSERVABLE': self.fraction_observable_sparse_array.transpose()}

        # Stellar Mass can be calculated from STELLAR * RICHNESS
        header_dict = {
            'STELLAR' : round(self.stellar_mass_conversion,8),
            'LKDNSIDE': self.config['coords']['nside_likelihood'],
            'LKDPIX'  : ang2pix(self.config['coords']['nside_likelihood'],self.roi.lon,self.roi.lat),
            'NROI'    : self.roi.inROI(self.loglike.catalog_roi.lon,self.loglike.catalog_roi.lat).sum(), 
            'NANNULUS': self.roi.inAnnulus(self.loglike.catalog_roi.lon,self.loglike.catalog_roi.lat).sum(), 
            'NINSIDE' : self.roi.inInterior(self.loglike.catalog_roi.lon,self.loglike.catalog_roi.lat).sum(), 
            'NTARGET' : self.roi.inTarget(self.loglike.catalog_roi.lon,self.loglike.catalog_roi.lat).sum(), 
        }

        # In case there is only a single distance modulus
        if len(self.distance_modulus_array) == 1:
            for key in data_dict:
                data_dict[key] = data_dict[key].flatten()

        ugali.utils.skymap.writeSparseHealpixMap(self.roi.pixels_target,
                                                 data_dict,
                                                 self.config['coords']['nside_pixel'],
                                                 outfile,
                                                 distance_modulus_array=self.distance_modulus_array,
                                                 coordsys='NULL', ordering='NULL',
                                                 header_dict=header_dict)

############################################################
    
if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for executing the likelihood scan."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_argument('outfile',metavar='outfile.fits',help='Output fits file.')
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True,radius=False)
    opts = parser.parse_args()


    scan = Scan(opts.config,opts.coords)
    if not opts.debug:
        result = scan.run()
        scan.write(opts.outfile)
