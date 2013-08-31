"""
Classes to evaluate the likelihood. 

Classes
    Likelihood

Functions
    someFunction
"""

import time
import numpy
import scipy.stats
import pylab
import healpy

import ugali.utils.binning
import ugali.utils.parabola
import ugali.utils.projector
import ugali.utils.skymap
import ugali.analysis.color_lut

############################################################

class Likelihood:

    def __init__(self, config, roi, mask, catalog_full, isochrone, kernel):
        """
        Object to efficiently search over a grid of ROI positions.
        """

        self.config = config
        self.roi = roi
        self.mask = mask # Currently assuming that input mask is ROI-specific

        # ROI-specific setup for catalog
        self.catalog_full = catalog_full
        cut_observable = self.mask.restrictCatalogToObservableSpace(self.catalog_full)
        self.catalog = self.catalog_full.applyCut(cut_observable)
        self.catalog.project(self.roi.projector)
        self.catalog.spatialBin(self.roi)
        
        self.isochrone = isochrone
        self.kernel = kernel

        self.delta_mag = 0.03 # 1.e-3 Effective bin size in color-magnitude space

        #self.precomputeGridSearch()

    def precomputeGridSearch(self, distance_modulus_array):
        """
        Precompute u_background and u_color for each star in catalog.
        Precompute observable fraction in each ROI pixel.
        """

        self.distance_modulus_array = distance_modulus_array

        print 'Precompute angular separations between %i target pixels and %i other ROI pixels ...'%(len(self.roi.pixels_target),
                                                                                                     len(self.roi.pixels))
        self.roi.precomputeAngsep()
        
        print 'Precompute field CMD ...'
        self.cmd_background = self.mask.backgroundCMD(self.catalog)

        # Background density (deg^-2 mag^-2)
        print 'Precompute background probability for each object ...'
        b_density = ugali.utils.binning.take2D(self.cmd_background,
                                               self.catalog.color, self.catalog.mag,
                                               self.roi.bins_color, self.roi.bins_mag)
        self.b = b_density * self.roi.area_pixel * self.delta_mag**2
        
        self.u_color_array = [[]] * len(self.distance_modulus_array)
        self.observable_fraction_sparse_array = [[]] * len(self.distance_modulus_array)

        print 'Begin loop over distance moduli ...'
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            print '  (%i/%i) distance modulus = %.2f ...'%(ii, len(self.distance_modulus_array), distance_modulus)#,
            time_start = time.time()

            self.u_color_array[ii] = False
            if self.config.params['likelihood']['color_lut_infile'] is not None:
                print '  Using look-up table %s'%(self.config.params['likelihood']['color_lut_infile'])
                self.u_color_array[ii] = ugali.analysis.color_lut.readColorLUT(self.config.params['likelihood']['color_lut_infile'],
                                                                               distance_modulus,
                                                                               self.catalog.mag_1,
                                                                               self.catalog.mag_2,
                                                                               self.catalog.mag_err_1,
                                                                               self.catalog.mag_err_2)
            if not numpy.any(self.u_color_array[ii]):
                self.u_color_array[ii] = self.signalColor(distance_modulus) # Compute on the fly instead
            
            self.observable_fraction_sparse_array[ii] = self.isochrone.observableFraction(self.mask,
                                                                                          distance_modulus)
            time_end = time.time()
            #print '%.2f s'%(time_end - time_start)

    def signalColor(self, distance_modulus, mass_steps=10000):
        """
        Compute signal color probability (u_color) for each catalog object on the fly.
        """
        
        # Isochrone will be binned in next step, so can sample many points efficiently
        isochrone_mass_init, isochrone_mass_pdf, isochrone_mass_act, isochrone_mag_1, isochrone_mag_2 = self.isochrone.sample(mass_steps=mass_steps)

        bins_mag_1 = numpy.arange(distance_modulus + numpy.min(isochrone_mag_1) - (0.5 * self.delta_mag),
                                  distance_modulus + numpy.max(isochrone_mag_1) + (0.5 * self.delta_mag),
                                  self.delta_mag)
        bins_mag_2 = numpy.arange(distance_modulus + numpy.min(isochrone_mag_2) - (0.5 * self.delta_mag),
                                  distance_modulus + numpy.max(isochrone_mag_2) + (0.5 * self.delta_mag),
                                  self.delta_mag)        
        histo_isochrone_pdf = numpy.histogram2d(distance_modulus + isochrone_mag_1,
                                                distance_modulus + isochrone_mag_2,
                                                bins=[bins_mag_1, bins_mag_2],
                                                weights=isochrone_mass_pdf)[0]

        # Keep only isochrone bins that are within the color-magnitude space of the ROI
        mag_1_mesh, mag_2_mesh = numpy.meshgrid(bins_mag_2[1:], bins_mag_1[1:])
        buffer = 1. # mag
        if self.config.params['catalog']['band_1_detection']:
            in_color_magnitude_space = (mag_1_mesh < (self.mask.mag_1_clip + buffer)) \
                                       * (mag_2_mesh > (self.roi.bins_mag[0] - buffer))
        else:
            in_color_magnitude_space = (mag_2_mesh < (self.mask.mag_2_clip + buffer)) \
                                       * (mag_2_mesh > (self.roi.bins_mag[0] - buffer))
        histo_isochrone_pdf *= in_color_magnitude_space
        index_mag_1, index_mag_2 = numpy.nonzero(histo_isochrone_pdf)
        isochrone_pdf = histo_isochrone_pdf[index_mag_1, index_mag_2]

        n_catalog = len(self.catalog.mag_1)
        n_isochrone_bins = len(index_mag_1)
        ones = numpy.ones([n_catalog, n_isochrone_bins])

        # Calculate distance between each catalog object and isochrone bin
        # Assume normally distributed photometry uncertainties
        delta_mag_1_hi = (self.catalog.mag_1.reshape([n_catalog, 1]) * ones) - (bins_mag_1[index_mag_1] * ones)
        arg_mag_1_hi = delta_mag_1_hi / (self.catalog.mag_err_1.reshape([n_catalog, 1]) * ones)
        delta_mag_1_lo = (self.catalog.mag_1.reshape([n_catalog, 1]) * ones) - (bins_mag_1[index_mag_1 + 1] * ones)
        arg_mag_1_lo = delta_mag_1_lo / (self.catalog.mag_err_1.reshape([n_catalog, 1]) * ones)
        #pdf_mag_1 = (scipy.stats.norm.cdf(arg_mag_1_hi) - scipy.stats.norm.cdf(arg_mag_1_lo))

        delta_mag_2_hi = (self.catalog.mag_2.reshape([n_catalog, 1]) * ones) - (bins_mag_2[index_mag_2] * ones)
        arg_mag_2_hi = delta_mag_2_hi / (self.catalog.mag_err_2.reshape([n_catalog, 1]) * ones)
        delta_mag_2_lo = (self.catalog.mag_2.reshape([n_catalog, 1]) * ones) - (bins_mag_2[index_mag_2 + 1] * ones)
        arg_mag_2_lo = delta_mag_2_lo / (self.catalog.mag_err_2.reshape([n_catalog, 1]) * ones)
        #pdf_mag_2 = scipy.stats.norm.cdf(arg_mag_2_hi) - scipy.stats.norm.cdf(arg_mag_2_lo)

        # PDF is only ~nonzero for object-bin pairs within 5 sigma in both magnitudes  
        index_nonzero_0, index_nonzero_1 = numpy.nonzero((arg_mag_1_hi > -5.) \
                                                         * (arg_mag_1_lo < 5.) \
                                                         * (arg_mag_2_hi > -5.) \
                                                         * (arg_mag_2_lo < 5.))
        pdf_mag_1 = numpy.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_2 = numpy.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_1[index_nonzero_0, index_nonzero_1] = scipy.stats.norm.cdf(arg_mag_1_hi[index_nonzero_0,
                                                                                        index_nonzero_1]) \
                                                      - scipy.stats.norm.cdf(arg_mag_1_lo[index_nonzero_0,
                                                                                          index_nonzero_1])
        pdf_mag_2[index_nonzero_0, index_nonzero_1] = scipy.stats.norm.cdf(arg_mag_2_hi[index_nonzero_0,
                                                                                        index_nonzero_1]) \
                                                      - scipy.stats.norm.cdf(arg_mag_2_lo[index_nonzero_0,
                                                                                          index_nonzero_1])

        # Signal color probability is product of PDFs for each object-bin pair summed over isochrone bins
        u_color = numpy.sum(pdf_mag_1 * pdf_mag_2 * (isochrone_pdf * ones), axis=1)
        return u_color

    def gridSearch2(self, coords=None, distance_modulus_index=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        Uses iterative procedure.
        """
        self.log_likelihood_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                  len(self.roi.pixels_target)])
        self.richness_error_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_upper_limit_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                              len(self.roi.pixels_target)])

        # Specific pixel
        if coords is not None and distance_modulus_index is not None:
            lon, lat = coords
            pix_coords = ugali.utils.projector.angToPix(self.config.params['coords']['nside_pixel'], lon, lat)
            
        print 'Begin loop over distance moduli ...'
        for ii, distance_modulus in enumerate(self.distance_modulus_array):

            # Specific pixel
            if coords is not None and distance_modulus_index is not None:
                if ii != distance_modulus_index:
                    continue
            
            print '  (%i/%i) distance modulus = %.2f ...'%(ii, len(self.distance_modulus_array), distance_modulus)
            self.u_color = self.u_color_array[ii]
            self.observable_fraction_sparse = self.observable_fraction_sparse_array[ii]
            
            for jj in range(0, len(self.roi.pixels_target)):

                # Specific pixel
                if coords is not None and distance_modulus_index is not None:
                    if self.roi.pixels_target[jj] != pix_coords:
                        continue

                self.kernel.lon = self.roi.centers_lon_target[jj]
                self.kernel.lat = self.roi.centers_lat_target[jj]

                print '    (%i/%i) Candidate at (%.3f, %.3f) ... '%(jj, len(self.roi.pixels_target),
                                                                    self.kernel.lon, self.kernel.lat),

                self.angsep_sparse = self.roi.angsep[jj] # deg
                self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi_index] # deg

                # Define a starting point for richness estimation
                f = numpy.sum(self.roi.area_pixel * self.kernel.surfaceIntensity(self.angsep_sparse) \
                              * self.observable_fraction_sparse)
                richness = numpy.array([1. / f])

                cmd_background = self.mask.backgroundCMD(self.catalog, weights=None)
                b_density = ugali.utils.binning.take2D(cmd_background,
                                                       self.catalog.color, self.catalog.mag,
                                                       self.roi.bins_color, self.roi.bins_mag) 
                self.b = b_density * self.roi.area_pixel * self.delta_mag**2
                
                log_likelihood_current, p = self.logLikelihood(distance_modulus, richness[-1], grid_search=True)[0: 2]
                log_likelihood = numpy.array([log_likelihood_current])

                # Begin iterative procedure using membership probabilities
                while True:
                    # Update the empirical field number density model (deg^-2 mag^-2)
                    #cmd_background = self.mask.backgroundCMD(self.catalog, weights=(1. - p))
                    #b_density = ugali.utils.binning.take2D(cmd_background,
                    #                                       self.catalog.color, self.catalog.mag,
                    #                                       self.roi.bins_color, self.roi.bins_mag) 
                    #self.b = b_density * self.roi.area_pixel * self.delta_mag**2

                    # Update the richness
                    richness = numpy.append(richness, numpy.sum(p) / f)
                    
                    log_likelihood_current, p, f = self.logLikelihood(distance_modulus, richness[-1], grid_search=True)
                    log_likelihood = numpy.append(log_likelihood, log_likelihood_current)

                    print richness[-1], log_likelihood[-1], len(richness)
                    raw_input('Continue')

                    # Convergence condition
                    if numpy.fabs(log_likelihood[-1] - numpy.max(log_likelihood[0: -1])) < tolerance:
                        break
                    
                # Store results
                self.richness_sparse_array[ii][jj] = richness[-1]
                self.richness_error_sparse_array[ii][jj] = numpy.sqrt(numpy.sum(p * (1. - p))) / f
                self.log_likelihood_sparse_array[ii][jj] = log_likelihood[-1]
                # Skip upper limits for the moment
                self.richness_upper_limit_sparse_array[ii][jj] = healpy.UNSEEN

                print 'TS = %.3f richness = %.3f +/- %.3f iterations = %i'%(2. * self.log_likelihood_sparse_array[ii][jj],
                                                                            self.richness_sparse_array[ii][jj],
                                                                            self.richness_error_sparse_array[ii][jj],
                                                                            len(richness))

                if coords is not None and distance_modulus_index is not None:
                    return self.richness_sparse_array[ii][jj], self.log_likelihood_sparse_array[ii][jj], self.richness_error_sparse_array[ii][jj], richness, log_likelihood, p, f

                
    def gridSearch(self, coords=None, distance_modulus_index=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        """

        self.log_likelihood_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                  len(self.roi.pixels_target)])
        self.richness_lower_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_upper_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_upper_limit_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                              len(self.roi.pixels_target)])

        # Specific pixel
        if coords is not None and distance_modulus_index is not None:
            lon, lat = coords
            theta = numpy.radians(90. - lat)
            phi = numpy.radians(lon)
            pix_coords = healpy.ang2pix(self.config.params['coords']['nside_pixel'], theta, phi)
            
        print 'Begin loop over distance moduli ...'
        for ii, distance_modulus in enumerate(self.distance_modulus_array):

            # Specific pixel
            if coords is not None and distance_modulus_index is not None:
                if ii != distance_modulus_index:
                    continue
            
            print '  (%i/%i) distance modulus = %.2f ...'%(ii, len(self.distance_modulus_array), distance_modulus)
            self.u_color = self.u_color_array[ii]
            self.observable_fraction_sparse = self.observable_fraction_sparse_array[ii]
            
            for jj in range(0, len(self.roi.pixels_target)):

                # Specific pixel
                if coords is not None and distance_modulus_index is not None:
                    if self.roi.pixels_target[jj] != pix_coords:
                        continue

                self.kernel.lon = self.roi.centers_lon_target[jj]
                self.kernel.lat = self.roi.centers_lat_target[jj]

                print '    (%i/%i) Candidate at (%.3f, %.3f) ... '%(jj, len(self.roi.pixels_target),
                                                                    self.kernel.lon, self.kernel.lat),

                self.angsep_sparse = self.roi.angsep[jj] # deg
                self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi_index] # deg

                #richness = numpy.array([0., 1., 1.e1, 1.e2, 1.e3])
                #log_likelihood = numpy.array([0.,
                #                              self.logLikelihood(distance_modulus, richness[1], grid_search=True)[0],
                #                              self.logLikelihood(distance_modulus, richness[2], grid_search=True)[0],
                #                              self.logLikelihood(distance_modulus, richness[3], grid_search=True)[0],
                #                              self.logLikelihood(distance_modulus, richness[4], grid_search=True)[0]])

                richness = numpy.array([0., 0., 0.])
                log_likelihood = numpy.array([0., 0., 0.])
                power = 0.
                while True:
                    richness[1] = 10.**power
                    log_likelihood[1] = self.logLikelihood(distance_modulus, richness[1], grid_search=True)[0]
                    power += 1.
                    if numpy.fabs(log_likelihood[1]) > 1.e-1:
                        break
                richness[2] = 10. * richness[1]
                log_likelihood[2] = self.logLikelihood(distance_modulus, richness[2], grid_search=True)[0]
                
                # First search for maximum likelihood richness
                found_maximum = False
                while not found_maximum:
                    parabola = ugali.utils.parabola.Parabola(richness, 2. * log_likelihood)
                    if parabola.vertex_x < 0.:
                            found_maximum = True
                    else:
                        richness = numpy.append(richness, parabola.vertex_x)
                        log_likelihood = numpy.append(log_likelihood,
                                                      self.logLikelihood(distance_modulus,
                                                                         richness[-1], grid_search=True)[0])    
                        if numpy.fabs(log_likelihood[-1] - numpy.max(log_likelihood[0: -1])) < tolerance:
                            found_maximum = True 

                # Use parabolic shape to begin filling in likelihoods at high richness values
                parabola = ugali.utils.parabola.Parabola(richness, 2. * log_likelihood)
                for delta in [1., 3., 10., 30.]:
                    richness = numpy.append(richness, parabola.profileUpperLimit(delta))
                    log_likelihood = numpy.append(log_likelihood,
                                                  self.logLikelihood(distance_modulus,
                                                                     richness[-1], grid_search=True)[0])
                
                # Continue to large enough richness values to ensure that richness upper limit can be computed
                found_upper_limit = False
                while not found_upper_limit:
                    if (numpy.min(log_likelihood) / 2.) > -30.:
                        richness = numpy.append(richness, 2. * numpy.max(richness))
                        log_likelihood = numpy.append(log_likelihood,
                                                      self.logLikelihood(distance_modulus,
                                                                         richness[-1], grid_search=True)[0])
                    else:
                        found_upper_limit = True

                # Argmax algorithm to find maximum
                #argmax = numpy.argmax(log_likelihood)
                #if argmax == 0:
                #    self.log_likelihood_sparse_array[ii][jj] = 0.
                #    self.richness_sparse_array[ii][jj] = 0.
                #else:
                #    self.log_likelihood_sparse_array[ii][jj] = log_likelihood[argmax]
                #    self.richness_sparse_array[ii][jj] = richness[argmax]

                parabola = ugali.utils.parabola.Parabola(richness, 2. * log_likelihood)

                # Vertex algorithm to find maximum
                self.richness_sparse_array[ii][jj] = parabola.vertex_x
                if self.richness_sparse_array[ii][jj] > 0.:
                    self.log_likelihood_sparse_array[ii][jj] = parabola.vertex_y / 2.
                else:
                    self.richness_sparse_array[ii][jj] = 0.
                    self.log_likelihood_sparse_array[ii][jj] = 0.

                #richness_lower = parabola.bayesianUpperLimit(0.159) # 1-sigma low
                #richness_upper = parabola.bayesianUpperLimit(0.841) # 1-sigma high
                #print richness_lower, richness_upper

                self.richness_lower_sparse_array[ii][jj], self.richness_upper_sparse_array[ii][jj] = parabola.confidenceInterval(0.6827)
                self.richness_upper_limit_sparse_array[ii][jj] = parabola.bayesianUpperLimit(0.95)
                
                print 'TS = %.2f richness = %.1f (%.1f -- %.1f, 0.68 CL) richness < %.1f (0.95 CL) iterations = %i'%(2. * self.log_likelihood_sparse_array[ii][jj],
                                                                                                                     self.richness_sparse_array[ii][jj],
                                                                                                                     self.richness_lower_sparse_array[ii][jj],
                                                                                                                     self.richness_upper_sparse_array[ii][jj],
                                                                                                                     self.richness_upper_limit_sparse_array[ii][jj],
                                                                                                                     len(richness))

                #if self.log_likelihood_sparse_array[ii][jj] == 0.:
                #    pylab.figure()
                #    pylab.scatter(richness, log_likelihood, c='b')
                #    raw_input('WAIT')

                if coords is not None and distance_modulus_index is not None:
                    p, f = self.logLikelihood(distance_modulus,
                                              self.richness_sparse_array[ii][jj],
                                              grid_search=True)[1:3]
                    return self.richness_sparse_array[ii][jj], self.log_likelihood_sparse_array[ii][jj], self.richness_lower_sparse_array[ii][jj], self.richness_upper_sparse_array[ii][jj], self.richness_upper_limit_sparse_array[ii][jj], richness, log_likelihood, p, f

    def logLikelihood(self, distance_modulus, richness, grid_search=False):
        """
        Return log(likelihood). If grid_search=True, take computational shortcuts.
        """

        # Option to rescale the kernel size??
        
        if grid_search:
            u_spatial = self.roi.area_pixel * self.kernel.surfaceIntensity(self.angsep_object)
            u = u_spatial * self.u_color
            f = numpy.sum(self.roi.area_pixel * self.kernel.surfaceIntensity(self.angsep_sparse) \
                          * self.observable_fraction_sparse)
        else:
            # Not implemented yet
            pass

        p = (richness * u) / ((richness * u) + self.b)
        log_likelihood = -1. * numpy.sum(numpy.log(1. - p)) - (f * richness)
        return log_likelihood, p, f

    def membershipGridSearch(self, index_distance_modulus = None, index_pixel_target = None):
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

        distance_modulus = self.distance_modulus_array[index_distance_modulus]
        richness = self.richness_sparse_array[index_distance_modulus][index_pixel_target]

        self.kernel.lon = self.roi.centers_lon_target[index_pixel_target]
        self.kernel.lat = self.roi.centers_lat_target[index_pixel_target]
        self.angsep_sparse = self.roi.angsep[index_pixel_target] # deg
        self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi_index] # deg
            
        log_likelihood, p, f = self.logLikelihood(distance_modulus, richness, grid_search=True)
        return p

    def write(self, outfile):
        """
        Save the likelihood fitting results as a sparse HEALPix map.
        """
        data_dict = {'LOG_LIKELIHOOD': self.log_likelihood_sparse_array.transpose(),
                     'RICHNESS': self.richness_sparse_array.transpose(),
                     'RICHNESS_LOWER': self.richness_lower_sparse_array.transpose(),
                     'RICHNESS_UPPER': self.richness_upper_sparse_array.transpose(),
                     'RICHNESS_LIMIT': self.richness_upper_limit_sparse_array.transpose()}

        ugali.utils.skymap.writeSparseHealpixMap(self.roi.pixels_target,
                                                 data_dict,
                                                 self.config.params['coords']['nside_pixel'],
                                                 outfile,
                                                 distance_modulus_array=self.distance_modulus_array,
                                                 coordsys='NULL', ordering='NULL')

############################################################
