"""
Likelihood evaluation.

Classes
    GridSearch

Functions
    someFunction
"""

import time
import numpy
import scipy.stats
import scipy.optimize
import pylab
import healpy

import ugali.utils.binning
import ugali.utils.parabola
import ugali.utils.skymap
import ugali.analysis.color_lut
import ugali.analysis.loglike

from ugali.utils.logger import logger
from ugali.utils.healpix import ang2pix
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
        self.loglike=ugali.analysis.loglike.LogLikelihood(config,roi,
                                                          mask,catalog,
                                                          isochrone,kernel)

        self.stellar_mass_conversion = self.loglike.stellar_mass()

    def precompute(self, distance_modulus_array=None):
        """
        Precompute u_background and u_color for each star in catalog.
        Precompute observable fraction in each ROI pixel.
        # Precompute still operates over the full ROI, not just the likelihood region
        """
        if distance_modulus_array is not None:
            self.distance_modulus_array = distance_modulus_array
        else:
            self.distance_modulus_array = self.config.params['likelihood']['distance_modulus_array']

        # Observable fraction for each pixel
        self.u_color_array = [[]] * len(self.distance_modulus_array)
        self.observable_fraction_sparse_array = [[]] * len(self.distance_modulus_array)

        logger.info('Loop over distance moduli in precompute step ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            logger.info('  (%i/%i) distance modulus = %.2f ...'%(ii+1, len(self.distance_modulus_array), distance_modulus))

            self.u_color_array[ii] = False
            if self.config.params['likelihood']['color_lut_infile'] is not None:
                logger.info('  Precomputing signal color from %s'%(self.config.params['likelihood']['color_lut_infile']))
                self.u_color_array[ii] = ugali.analysis.color_lut.readColorLUT(self.config.params['likelihood']['color_lut_infile'],
                                                                               distance_modulus,
                                                                               self.catalog.mag_1,
                                                                               self.catalog.mag_2,
                                                                               self.catalog.mag_err_1,
                                                                               self.catalog.mag_err_2)
            if not numpy.any(self.u_color_array[ii]):
                logger.info('  Precomputing signal color on the fly...')
                self.u_color_array[ii] = self.loglike.calc_signal_color(distance_modulus) 
            
            # Calculate over all pixels in ROI
            self.observable_fraction_sparse_array[ii] = self.loglike.calc_observable_fraction(distance_modulus)
            
            time_end = time.time()

        self.u_color_array = numpy.array(self.u_color_array)

                
    def search(self, coords=None, distance_modulus_index=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        """
        len_distance_modulus = len(self.distance_modulus_array)
        len_pixels_target    = len(self.roi.pixels_target)
        self.log_likelihood_sparse_array       = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.richness_sparse_array             = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.richness_lower_sparse_array       = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.richness_upper_sparse_array       = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.richness_upper_limit_sparse_array = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.stellar_mass_sparse_array         = numpy.zeros([len_distance_modulus, len_pixels_target])
        self.fraction_observable_sparse_array  = numpy.zeros([len_distance_modulus, len_pixels_target])

        # Specific pixel
        if coords is not None:
            pix_coords = ang2pix(coords)

        lon, lat = self.roi.pixels_target.lon, self.roi.pixels_target.lat
            
        logger.info('Begin loop over distance moduli in likelihood fitting ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):

            # Specific pixel
            if distance_modulus_index is not None:
                if ii != distance_modulus_index: continue
            
            logger.info('  (%i/%i) distance modulus = %.2f ...'%(ii+1, len_distance_modulus, distance_modulus))

            for jj in range(0, len_pixels_target):
                # Specific pixel
                if coords is not None:
                    if self.roi.pixels_target[jj] != pix_coords:
                        continue

                self.loglike.set_params(lon=lon[jj],lat=lat[jj],distance_modulus=distance_modulus)
                self.loglike.sync_params(u_color=self.u_color_array[ii],
                                         observable_fraction=self.observable_fraction_sparse_array[ii])
                                         
                args = (
                    jj+1, len_pixels_target, 
                    self.loglike.kernel.lon, self.loglike.kernel.lat
                )
                message = """    (%i/%i) Candidate at (%.3f, %.3f) ... """%(args)

                self.log_likelihood_sparse_array[ii][jj], self.richness_sparse_array[ii][jj], parabola = self.loglike.fit_richness()
                self.stellar_mass_sparse_array[ii][jj] = self.stellar_mass_conversion * self.richness_sparse_array[ii][jj]
                self.fraction_observable_sparse_array[ii][jj] = self.loglike.f
                if self.config.params['likelihood']['full_pdf'] \
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
                    message += 'TS = %.2f, Stellar Mass = %.1f (%.1f -- %.1f @ 0.68 CL, < %.1f @ 0.95 CL)'%(args)
                else:
                    args = (
                        2. * self.log_likelihood_sparse_array[ii][jj], 
                        self.stellar_mass_conversion * self.richness_sparse_array[ii][jj],
                        self.fraction_observable_sparse_array[ii][jj]
                    )
                    message += 'TS = %.2f, Stellar Mass = %.1f, Fraction = %.2g'%(args)
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
                jj_max+1, len_pixels_target, lon[jj_max], lat[jj_max],
                2. * self.log_likelihood_sparse_array[ii][jj_max], 
                self.stellar_mass_conversion * self.richness_sparse_array[ii][jj_max]
            )
            message = "  (%i/%i) Maximum at (%.3f, %.3f) ... TS = %.2f, Stellar Mass = %.1f"%(args)
            logger.info( message )
 

    def membership(self, index_distance_modulus=None, index_pixel_target=None):
        """
        Get membership probabilities for each catalog object based on fit from grid search
        """
        if index_distance_modulus is not None and index_pixel_target is None:
            index_pixel_target = numpy.argmax(likelihood.log_likelihood_sparse_array[index_distance_modulus])
        elif index_distance_modulus is None and index_pixel_target is not None:
            index_distance_modulus = numpy.argmax(numpy.take(likelihood.log_likelihood_sparse_array,
                                                             [index_pixel_target], axis=1))
        elif index_distance_modulus is None and index_pixel_target is None:
            index_distance_modulus, index_pixel_target = numpy.unravel_index(numpy.argmax(self.log_likelihood_sparse_array),
                                                                             self.log_likelihood_sparse_array.shape)
        else:
            pass

        # ADW: This needs to be updated with loglike calls
        distance_modulus = self.distance_modulus_array[index_distance_modulus]
        richness = self.richness_sparse_array[index_distance_modulus][index_pixel_target]

        self.kernel.lon = self.roi.pixels_target.lon[index_pixel_target]
        self.kernel.lat = self.roi.pixels_target.lat[index_pixel_target]

        self.angsep_sparse = self.roi.angsep[index_pixel_target] # deg
        self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi_index] # deg
            
        log_likelihood, p, f = self.logLikelihood(distance_modulus, richness, grid_search=True)
        return p

    def write(self, outfile):
        """
        Save the likelihood fitting results as a sparse HEALPix map.
        """
        # Full data output (too large for survey)
        if self.config.params['likelihood']['full_pdf']:
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
            'LKDNSIDE': self.config.params['coords']['nside_likelihood'],
            'LKDPIX'  : ang2pix(self.config.params['coords']['nside_likelihood'],self.roi.lon,self.roi.lat),
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
                                                 self.config.params['coords']['nside_pixel'],
                                                 outfile,
                                                 distance_modulus_array=self.distance_modulus_array,
                                                 coordsys='NULL', ordering='NULL',
                                                 header_dict=header_dict)

############################################################

