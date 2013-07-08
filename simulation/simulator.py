"""
Documentation.
"""

import numpy
import scipy.interpolate
import pylab

import ugali.observation.catalog
import ugali.observation.mask
import ugali.observation.roi
import ugali.utils.parse_config
import ugali.utils.projector

pylab.ion()

############################################################

class Simulator:

    def __init__(self, config, lon, lat):

        self.config = ugali.utils.parse_config.Config(config)
        self.catalog = ugali.observation.catalog.Catalog(self.config)

        self.roi = ugali.observation.roi.ROI(self.config, lon, lat)

        mask_1 = ugali.observation.mask.MaskBand(self.config.params['mask']['infile_1'], self.roi)
        mask_2 = ugali.observation.mask.MaskBand(self.config.params['mask']['infile_2'], self.roi)
        self.mask = ugali.observation.mask.Mask(self.config, mask_1, mask_2)

        cut = self.mask.restrictCatalogToObservableSpace(self.catalog)
        self.catalog = self.catalog.applyCut(cut)

        self.cmd_background = self.mask.backgroundCMD(self.catalog)

        self._photometricErrors()

    def _photometricErrors(self, n_per_bin=100, plot=False):
        """
        Realistic photometric errors estimated from catalog objects and mask.
        """

        self.catalog.spatialBin(self.roi)

        # Band 1
        
        mag_1_thresh = self.mask.mask_1.mask_roi[self.roi.pixels][self.catalog.pixel_roi] - self.catalog.mag_1
        sorting_indices = numpy.argsort(mag_1_thresh)
        mag_1_thresh_sort = mag_1_thresh[sorting_indices]
        mag_err_1_sort = self.catalog.mag_err_1[sorting_indices]

        mag_1_thresh_medians = []
        mag_err_1_medians = []
        for ii in range(0, int(len(mag_1_thresh) / float(n_per_bin))):
            mag_1_thresh_medians.append(numpy.median(mag_1_thresh_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))
            mag_err_1_medians.append(numpy.median(mag_err_1_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))

        if mag_1_thresh_medians[0] > 0.:
            mag_1_thresh_medians = numpy.insert(mag_1_thresh_medians, 0, 0.)
            mag_err_1_medians = numpy.insert(mag_err_1_medians, 0, mag_err_1_medians[0])
        
        self.photo_err_1 = scipy.interpolate.interp1d(mag_1_thresh_medians, mag_err_1_medians,
                                                      bounds_error=False, fill_value=mag_err_1_medians[-1])

        # Band 2

        mag_2_thresh = self.mask.mask_2.mask_roi[self.roi.pixels][self.catalog.pixel_roi] - self.catalog.mag_2
        sorting_indices = numpy.argsort(mag_2_thresh)
        mag_2_thresh_sort = mag_2_thresh[sorting_indices]
        mag_err_2_sort = self.catalog.mag_err_2[sorting_indices]

        mag_2_thresh_medians = []
        mag_err_2_medians = []
        for ii in range(0, int(len(mag_2_thresh) / float(n_per_bin))):
            mag_2_thresh_medians.append(numpy.median(mag_2_thresh_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))
            mag_err_2_medians.append(numpy.median(mag_err_2_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))

        if mag_2_thresh_medians[0] > 0.:
            mag_2_thresh_medians = numpy.insert(mag_2_thresh_medians, 0, 0.)
            mag_err_2_medians = numpy.insert(mag_err_2_medians, 0, mag_err_2_medians[0])
        
        self.photo_err_2 = scipy.interpolate.interp1d(mag_2_thresh_medians, mag_err_2_medians,
                                                      bounds_error=False, fill_value=mag_err_2_medians[-1])

        if plot:
            pylab.figure()
            pylab.scatter(mag_1_thresh, self.catalog.mag_err_1, c='blue')
            x = numpy.linspace(0., 10., 1.e4)        
            #pylab.scatter(mag_1_thresh_medians, mag_err_1_medians, c='red')
            pylab.plot(x, self.photo_err_1(x), c='red')

    def satellite(self, isochrone, kernel, stellar_mass, distance_modulus, mc_src_id=1):
        """

        """
        mag_1, mag_2, lon, lat = satellite(isochrone, kernel, stellar_mass, distance_modulus)
        pix = ugali.utils.projector.angToPix(self.config.params['coords']['nside_pixel'], lon, lat)
        mag_1_lim = self.mask.mask_1.mask_roi[pix]
        mag_2_lim = self.mask.mask_2.mask_roi[pix]
        mag_1_obs = mag_1 + (numpy.random.normal(size=len(mag_1)) * self.photo_err_1(mag_1_lim - mag_1))
        mag_2_obs = mag_2 + (numpy.random.normal(size=len(mag_2)) * self.photo_err_2(mag_2_lim - mag_2))

        cut = numpy.logical_and(mag_1_obs < mag_1_lim, mag_2_obs < mag_2_lim)

        # What about objects below threshold??

        return mag_1_obs[cut], mag_2_obs[cut], lon[cut], lat[cut]

    def write(self, outfile):
        """

        """
        pass

############################################################

def satellite(isochrone, kernel, stellar_mass, distance_modulus):
    """

    """
    projector = ugali.utils.projector.Projector(kernel.lon, kernel.lat)
    mag_1, mag_2 = isochrone.simulate(stellar_mass, distance_modulus)
    r = kernel.simulate(len(mag_1))
    phi = 2. * numpy.pi * numpy.random.rand(len(r))
    x = r * numpy.cos(phi)
    y = r * numpy.sin(phi)
    lon, lat = projector.imageToSphere(x, y)
    return mag_1, mag_2, lon, lat

############################################################
