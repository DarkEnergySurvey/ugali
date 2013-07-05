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

import ugali.utils.binning
import ugali.utils.parabola
import ugali.utils.skymap

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
            print '  (%i/%i) distance modulus = %.2f ...'%(ii, len(self.distance_modulus_array), distance_modulus),
            time_start = time.time()
            self.u_color_array[ii] = self.signalColor(distance_modulus)
            self.observable_fraction_sparse_array[ii] = self.isochrone.observableFraction(self.mask,
                                                                                          distance_modulus)
            time_end = time.time()
            print '%.2f s'%(time_end - time_start)

    def signalColor(self, distance_modulus, mass_steps=10000):
        """
        Compute color probability (u_color) for each catalog object.
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

    def signalColor2(self, distance_modulus, mass_steps=250, intrinsic_width=0.02):
        """
        Compute u_color for each object. Intrinsic width (mag).
        """
        isochrone_mass_init, isochrone_mass_pdf, isochrone_mass_act, isochrone_mag_1, isochrone_mag_2 = self.isochrone.sample(mass_steps=mass_steps)
        n_sample = len(isochrone_mass_init)
        n_catalog = len(self.catalog.lon)

        delta_mag_1 = self.catalog.mag_1 * numpy.ones([n_sample, n_catalog]) \
                      - (isochrone_mag_1 + distance_modulus).reshape([n_sample, 1]) * numpy.ones([n_sample, n_catalog])
        
        delta_mag_2 = self.catalog.mag_2 * numpy.ones([n_sample, n_catalog]) \
                      - (isochrone_mag_2 + distance_modulus).reshape([n_sample, 1]) * numpy.ones([n_sample, n_catalog])
        
        mag_err_1 = self.catalog.mag_err_1 * numpy.ones([n_sample, n_catalog])
        mag_err_2 = self.catalog.mag_err_2 * numpy.ones([n_sample, n_catalog])
        # Add in quadrature some minimum magnitude uncertainty??
        mag_err_1 = numpy.sqrt(mag_err_1**2 + intrinsic_width**2)
        mag_err_2 = numpy.sqrt(mag_err_2**2 + intrinsic_width**2)
        
        arg_mag_1_hi = (delta_mag_1 + (0.5 * self.delta_mag)) / mag_err_1
        arg_mag_1_lo = (delta_mag_1 - (0.5 * self.delta_mag)) / mag_err_1

        arg_mag_2_hi = (delta_mag_2 + (0.5 * self.delta_mag)) / mag_err_2
        arg_mag_2_lo = (delta_mag_2 - (0.5 * self.delta_mag)) / mag_err_2

        arg_mag_1_hi = arg_mag_1_hi.flatten()
        arg_mag_1_lo = arg_mag_1_lo.flatten()
        arg_mag_2_hi = arg_mag_2_hi.flatten()
        arg_mag_2_lo = arg_mag_2_lo.flatten()

        index_nonzero = numpy.nonzero(numpy.logical_or(numpy.fabs(arg_mag_1_hi) < 5.,
                                                       numpy.fabs(arg_mag_1_lo) < 5.) \
                                      * numpy.logical_or(numpy.fabs(arg_mag_2_hi) < 5.,
                                                         numpy.fabs(arg_mag_2_lo) < 5.))[0]

        u_mag_1 = numpy.zeros(n_sample * n_catalog)
        u_mag_2 = numpy.zeros(n_sample * n_catalog)

        u_mag_1[index_nonzero] = scipy.stats.norm.cdf(arg_mag_1_hi[index_nonzero]) \
                                 - scipy.stats.norm.cdf(arg_mag_1_lo[index_nonzero])
        u_mag_2[index_nonzero] = scipy.stats.norm.cdf(arg_mag_2_hi[index_nonzero]) \
                                 - scipy.stats.norm.cdf(arg_mag_2_lo[index_nonzero])
        
        u_mag_1 = u_mag_1.reshape([n_sample, n_catalog])
        u_mag_2 = u_mag_2.reshape([n_sample, n_catalog])

        isochrone_mass_pdf_matrix = isochrone_mass_pdf.reshape([n_sample, 1]) \
                                    * numpy.ones([n_sample, n_catalog])

        u_color = numpy.sum(isochrone_mass_pdf_matrix * u_mag_1 * u_mag_2, axis = 0)

        return u_color

    def gridSearch(self):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        """

        self.log_likelihood_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                        len(self.roi.pixels_target)])
        self.richness_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                  len(self.roi.pixels_target)])
        self.richness_upper_limit_sparse_array = numpy.zeros([len(self.distance_modulus_array),
                                                              len(self.roi.pixels_target)])

        print 'Begin loop over distance moduli ...'
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            
            print '  (%i/%i) distance modulus = %.2f ...'%(ii, len(self.distance_modulus_array), distance_modulus)
            self.u_color = self.u_color_array[ii]
            self.observable_fraction_sparse = self.observable_fraction_sparse_array[ii]
            
            for jj in range(0, len(self.roi.pixels_target)):
                self.kernel.lon = self.roi.centers_lon_target[jj]
                self.kernel.lat = self.roi.centers_lat_target[jj]

                print '    (%i/%i) Candidate at (%.3f, %.3f) ... '%(jj, len(self.roi.pixels_target),
                                                                    self.kernel.lon, self.kernel.lat),

                self.angsep_sparse = self.roi.angsep[jj] # deg
                self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi] # deg

                richness = numpy.array([0., 1., 1.e1, 1.e2, 1.e3])
                log_likelihood = numpy.array([0.,
                                              self.logLikelihood(distance_modulus, richness[1], grid_search=True)[0],
                                              self.logLikelihood(distance_modulus, richness[2], grid_search=True)[0],
                                              self.logLikelihood(distance_modulus, richness[3], grid_search=True)[0],
                                              self.logLikelihood(distance_modulus, richness[4], grid_search=True)[0]])

                """
                found_maximum = False
                found_upper_limit = False
                while not (found_maximum and found_upper_limit):
                    parabola = ugali.utils.parabola.Parabola(richness, 2. * log_likelihood)
                    
                    if not found_maximum:
                        if parabola.vertex_x < 0.:
                            found_maximum = True
                        else:
                            richness = numpy.append(richness, parabola.vertex_x)
                            log_likelihood = numpy.append(log_likelihood,
                                                          self.logLikelihood(distance_modulus,
                                                                             richness[-1], grid_search=True)[0])

                            if numpy.fabs(log_likelihood[-1] - log_likelihood[-2]) < 1.e-2:
                                found_maximum = True
                    else:
                        if numpy.min(log_likelihood) > -100.:
                            richness = numpy.append(richness, 2. * numpy.max(richness))
                            log_likelihood = numpy.append(log_likelihood,
                                                          self.logLikelihood(distance_modulus,
                                                                             richness[-1], grid_search=True)[0])
                        else:
                            found_upper_limit = True
                """

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
                        if numpy.fabs(log_likelihood[-1] - numpy.max(log_likelihood[0: -1])) < 1.e-2:
                            found_maximum = True

                # Continue far enough to compute richness upper limit
                found_upper_limit = False
                while not found_upper_limit:
                    if numpy.min(log_likelihood) > -100.:
                        richness = numpy.append(richness, 2. * numpy.max(richness))
                        log_likelihood = numpy.append(log_likelihood,
                                                      self.logLikelihood(distance_modulus,
                                                                         richness[-1], grid_search=True)[0])
                    else:
                        found_upper_limit = True
                            
                argmax = numpy.argmax(log_likelihood)
                if argmax == 0:
                    self.log_likelihood_sparse_array[ii][jj] = 0.
                    self.richness_sparse_array[ii][jj] = 0.
                else:
                    self.log_likelihood_sparse_array[ii][jj] = log_likelihood[argmax]
                    self.richness_sparse_array[ii][jj] = richness[argmax]

                self.richness_upper_limit_sparse_array[ii][jj] = parabola.bayesianUpperLimit(0.95)
                
                print 'TS = %.3f richness = %.3f richness < %.3f (0.95 CL) interations = %i'%(2. * self.log_likelihood_sparse_array[ii][jj],
                                                                                              self.richness_sparse_array[ii][jj],
                                                                                              self.richness_upper_limit_sparse_array[ii][jj],
                                                                                              len(richness))

                #if self.log_likelihood_sparse_array[ii][jj] == 0.:
                #    pylab.figure()
                #    pylab.scatter(richness, log_likelihood, c='b')
                #    raw_input('WAIT')
                #return richness, log_likelihood

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
        self.angsep_object = self.angsep_sparse[self.catalog.pixel_roi] # deg
            
        log_likelihood, p, f = self.logLikelihood(distance_modulus, richness, grid_search=True)
        return p

    def write(self, outfile):
        """

        """

        data_dict = {'LOG_LIKELIHOOD': self.log_likelihood_sparse_array.transpose(),
                     'RICHNESS': self.richness_sparse_array.transpose(),
                     'RICHNESS_LIM': self.richness_upper_limit_sparse_array.transpose()}

        ugali.utils.skymap.writeSparseHealpixMap(self.roi.pixels_target,
                                                 data_dict,
                                                 self.config.params['coords']['nside_pixel'],
                                                 outfile,
                                                 distance_modulus_array=self.distance_modulus_array,
                                                 coordsys='NULL', ordering='NULL')

############################################################
