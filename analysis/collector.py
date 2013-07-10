"""
Classes to organize high level results from grid search.

Classes
    Collector

Functions
    someFunction
"""

import numpy
import pyfits
import pylab
import healpy

import ugali.analysis.farm
import ugali.utils.parse_config
import ugali.utils.plotting
import ugali.utils.projector
import ugali.utils.skymap

pylab.ion()

############################################################

class Collector:

    def __init__(self, infile,
                 config=None, stellar_mass=None,
                 distance_modulus_extension='DISTANCE_MODULUS', distance_modulus_field='DISTANCE_MODULUS',
                 pix_data_extension='PIX_DATA'):
        """
        Infile is the output of likelihood grid search. 
        """

        self.infile = infile

        print 'Distance Modulus'
        reader = pyfits.open(self.infile)
        self.nside = reader[pix_data_extension].header['NSIDE']
        self.area_pixel = healpy.nside2pixarea(self.nside, degrees=True)
        self.distance_modulus_array = reader[distance_modulus_extension].data.field(distance_modulus_field)
        reader.close()

        print 'Pixels'
        self.pixels = ugali.utils.skymap.readSparseHealpixMap(self.infile, 'LOG_LIKELIHOOD', construct_map=False)[0]
        print 'Log-likelihood'
        self.log_likelihood_sparse = ugali.utils.skymap.readSparseHealpixMap(self.infile, 'LOG_LIKELIHOOD',
                                                                             construct_map=False)[1]
        print 'Richness'
        self.richness_sparse = ugali.utils.skymap.readSparseHealpixMap(self.infile, 'RICHNESS',
                                                                       construct_map=False)[1]
        print 'Richness Limit'
        self.richness_lim_sparse = ugali.utils.skymap.readSparseHealpixMap(self.infile, 'RICHNESS_LIM',
                                                                           construct_map=False)[1]

        print 'Data covers %.2f deg^2'%(len(self.pixels) * self.area_pixel)

        if config is not None:
            self.config = ugali.utils.parse_config.Config(config)
            # Determine richness to stellar mass conversion from isochrone
            isochrones = []
            for ii, name in enumerate(self.config.params['isochrone']['infiles']):
                isochrones.append(ugali.analysis.isochrone.Isochrone(self.config, name))
            isochrone = ugali.analysis.isochrone.CompositeIsochrone(isochrones, self.config.params['isochrone']['weights'])
            self.stellar_mass = isochrone.stellarMass()
        elif stellar_mass is not None:
            self.stellar_mass = stellar_mass
        else:
            print 'WARNING: could not determine stellar mass conversion factor, please supply config file.'
            self.stellar_mass = 1.

    def map(self, mode='ts', distance_modulus_index=None, **kwargs):
        """
        Documentation.
        """
        theta, phi =  healpy.pix2ang(self.nside, self.pixels)
        lon = numpy.degrees(phi)
        lat = 90. - numpy.degrees(theta)

        lon_median = numpy.median(lon)
        lat_median = numpy.median(lat)

        radius = 1.1 * numpy.max(ugali.utils.projector.angsep(lon_median, lat_median, lon, lat))

        for ii in range(0, len(self.distance_modulus_array)):

            if distance_modulus_index is not None and ii != distance_modulus_index:
                continue

            distance = ugali.utils.projector.distanceModulusToDistance(self.distance_modulus_array[ii])
        
            if mode == 'ts':
                title = 'Test Statistic (%.1f kpc)'%(distance)
                map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
                map[self.pixels] = self.log_likelihood_sparse[ii]
                index = map != healpy.UNSEEN
                map[index] = 2. * map[index]
            if mode == 'lim':
                title = 'Upper Limit (%.1f kpc)'%(distance)
                map = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
                map[self.pixels] = self.richness_lim_sparse[ii]
                        
            ugali.utils.plotting.zoomedHealpixMap(title, map, lon_median, lat_median, radius, **kwargs)

    def testStatistic(self):
        """
        Documentation.
        """
        #fig = pylab.figure()
        #ax = fig.add_subplot(1,1,1)
        #ax.hist(2. * self.log_likelihood_sparse.flatten(), bins=40, log=True, color='b')
        #ax.set_xlabel('Test Statistic')
        #ax.set_title('Significance Distribution')
        #ax.set_yscale('log')
        
        pylab.figure()
        pylab.hist(2. * self.log_likelihood_sparse.flatten(), bins=40, log=True, color='b')
        pylab.xlabel('Test Statistic')
        pylab.title('Significance Distribution')

    def sensitivity(self, steps=1000):
        """
        stellar_mass is the average stellar mass (M_Sol) of the isochrone, i.e., the average mass per star.
        """
        
        richness_lim_array = self.stellar_mass * numpy.linspace(numpy.min(self.richness_lim_sparse), numpy.max(self.richness_lim_sparse), steps)

        print len(richness_lim_array)

        sensitivity_curve_array = []

        for ii in range(0, len(self.distance_modulus_array)):
            sensitivity_curve_array.append([])
            for richness_lim in richness_lim_array:
                sensitivity_curve_array[ii].append(numpy.sum(self.stellar_mass * self.richness_lim_sparse[ii] < richness_lim))

            sensitivity_curve_array[ii] = self.area_pixel * numpy.array(sensitivity_curve_array[ii])
        
        pylab.figure()

        for ii in range(0, len(sensitivity_curve_array)):
            distance = ugali.utils.projector.distanceModulusToDistance(self.distance_modulus_array[ii])
            pylab.plot(richness_lim_array, sensitivity_curve_array[ii],
                       c=pylab.cm.jet(numpy.linspace(0, 1, len(sensitivity_curve_array))[ii]),
                       label='%.1f kpc'%(distance))

        pylab.xlabel(r'Stellar Mass ($M_{\odot}$)')
        pylab.ylabel(r'Area (deg$^2$)')
        pylab.title('Upper Limits (0.95 CL)')
        #pylab.xscale('log')
        #pylab.yscale('log')
        pylab.legend(loc='lower right')

    def inspect(self, coords, distance_modulus_index):
        """
        A useful tool would be to recreate the likelihood object for a given set of coordinates.
        """
        farm = ugali.analysis.farm.Farm(self.config)
        likelihood = farm.farmLikelihoodFromCatalog(local=True, coords=coords)

        likelihood.precomputeGridSearch([self.distance_modulus_array[distance_modulus_index]])

        #richness, log_likelihood = likelihood.gridSearch(coords=coords, distance_modulus_index=0)
        richness, log_likelihood, richness_upper_limit, richness_raw, log_likelihood_raw, p, f = likelihood.gridSearch(coords=coords, distance_modulus_index=0)

        #pylab.figure()
        #pylab.scatter(richness, log_likelihood, c='b')

        #return likelihood, richness, log_likelihood
        return likelihood, richness, log_likelihood, richness_upper_limit, richness_raw, log_likelihood_raw, p, f

    def findPeaks(self, ts_threshold):
        """
        Return a list of peaks in the TS maps.
        """
        peaks = []
        for distance_modulus_index in range(0, len(self.distance_modulus_array)):
            peaks.append([])

            pix_index = numpy.nonzero((2. * self.log_likelihood_sparse[distance_modulus_index]) > ts_threshold)[0]
            if len(pix_index) == 0:
                continue
            
            pix = self.pixels[pix_index]

            peaks[distance_modulus_index].append([pix[0]])

            peak_index = 0
            pix_assigned = numpy.concatenate(peaks[distance_modulus_index])

            while len(pix_assigned) < len(pix):

                pix_neighbors = numpy.unique(healpy.pixelfunc.get_all_neighbours(self.nside,
                                                                                 peaks[distance_modulus_index][peak_index]))
                pix_neighbors_valid = numpy.intersect1d(pix_neighbors, pix)
                pix_neighbors_valid_new = numpy.setdiff1d(pix_neighbors_valid, peaks[distance_modulus_index][peak_index])

                if len(pix_neighbors_valid_new) == 0:
                    peaks[distance_modulus_index].append([numpy.min(numpy.setdiff1d(pix, pix_assigned))])
                else:
                    peaks[distance_modulus_index][peak_index] = numpy.concatenate([peaks[distance_modulus_index][peak_index],
                                                                                   pix_neighbors_valid_new])

                peak_index = len(peaks[distance_modulus_index]) - 1
                pix_assigned = numpy.concatenate(peaks[distance_modulus_index])

        for distance_modulus_index in range(0, len(self.distance_modulus_array)):
            print '  distance modulus = %.2f'%(self.distance_modulus_array[distance_modulus_index])

            map_log_likelihood = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
            map_log_likelihood[self.pixels] = self.log_likelihood_sparse[distance_modulus_index]

            for peak_index in range(0, len(peaks[distance_modulus_index])):
                log_likelihood_peak = map_log_likelihood[peaks[distance_modulus_index][peak_index]]
                lon, lat = ugali.utils.projector.pixToAng(self.nside,
                                                          peaks[distance_modulus_index][peak_index][numpy.argmax(log_likelihood_peak)])
                print '    TS_peak = %.2f, (lon, lat) = (%.3f, %.3f), size = %.4f deg^2 (%i pix)'%(numpy.max(2. * log_likelihood_peak),
                                                                                                   lon, lat,
                                                                                                   self.area_pixel * len(peaks[distance_modulus_index][peak_index]),
                                                                                                   len(peaks[distance_modulus_index][peak_index]))

        # Merge peaks on distance modulus slices??

        #for distance_modulus_index in range(0, len(self.distance_modulus_array)):
        #    for peak_index in range(0, len(peaks[distance_modulus_index])):       
                
        return peaks

############################################################
