"""
Documentation.
"""

import numpy
import scipy.interpolate
import pyfits
import healpy
import pylab

import ugali.observation.catalog
import ugali.observation.mask
import ugali.observation.roi
import ugali.utils.config
import ugali.utils.projector

pylab.ion()

############################################################

class Simulator:

    def __init__(self, config, roi):

        self.config = ugali.utils.config.Config(config)
        
        self.roi = roi

        self.catalog_full = ugali.observation.catalog.Catalog(self.config,roi=self.roi)
        self.mask = ugali.observation.mask.Mask(self.config, roi=self.roi)
        
        cut = self.mask.restrictCatalogToObservableSpace(self.catalog_full)
        self.catalog = self.catalog_full.applyCut(cut)

        self.cmd_background = self.mask.backgroundCMD(self.catalog)

        self._photometricErrors()

    def _photometricErrors(self, n_per_bin=100, plot=False):
        """
        Realistic photometric errors estimated from catalog objects and mask.
        Extend below the magnitude threshold with a flat extrapolation.
        """

        self.catalog.spatialBin(self.roi)

        # Band 1
        
        mag_1_thresh = self.mask.mask_1.mask_roi_sparse[self.catalog.pixel_roi_index] - self.catalog.mag_1
        sorting_indices = numpy.argsort(mag_1_thresh)
        mag_1_thresh_sort = mag_1_thresh[sorting_indices]
        mag_err_1_sort = self.catalog.mag_err_1[sorting_indices]

        mag_1_thresh_medians = []
        mag_err_1_medians = []
        for ii in range(0, int(len(mag_1_thresh) / float(n_per_bin))):
            mag_1_thresh_medians.append(numpy.median(mag_1_thresh_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))
            mag_err_1_medians.append(numpy.median(mag_err_1_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))
        
        if mag_1_thresh_medians[0] > 0.:
            mag_1_thresh_medians = numpy.insert(mag_1_thresh_medians, 0, -99.)
            mag_err_1_medians = numpy.insert(mag_err_1_medians, 0, mag_err_1_medians[0])
        
        self.photo_err_1 = scipy.interpolate.interp1d(mag_1_thresh_medians, mag_err_1_medians,
                                                      bounds_error=False, fill_value=mag_err_1_medians[-1])

        # Band 2

        mag_2_thresh = self.mask.mask_2.mask_roi_sparse[self.catalog.pixel_roi_index] - self.catalog.mag_2
        sorting_indices = numpy.argsort(mag_2_thresh)
        mag_2_thresh_sort = mag_2_thresh[sorting_indices]
        mag_err_2_sort = self.catalog.mag_err_2[sorting_indices]

        mag_2_thresh_medians = []
        mag_err_2_medians = []
        for ii in range(0, int(len(mag_2_thresh) / float(n_per_bin))):
            mag_2_thresh_medians.append(numpy.median(mag_2_thresh_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))
            mag_err_2_medians.append(numpy.median(mag_err_2_sort[n_per_bin * ii: n_per_bin * (ii + 1)]))

        if mag_2_thresh_medians[0] > 0.:
            mag_2_thresh_medians = numpy.insert(mag_2_thresh_medians, 0, -99.)
            mag_err_2_medians = numpy.insert(mag_err_2_medians, 0, mag_err_2_medians[0])
        
        self.photo_err_2 = scipy.interpolate.interp1d(mag_2_thresh_medians, mag_err_2_medians,
                                                      bounds_error=False, fill_value=mag_err_2_medians[-1])

        if plot:
            pylab.figure()
            pylab.scatter(mag_1_thresh, self.catalog.mag_err_1, c='blue')
            x = numpy.linspace(0., 10., 1.e4)        
            #pylab.scatter(mag_1_thresh_medians, mag_err_1_medians, c='red')
            pylab.plot(x, self.photo_err_1(x), c='red')

    def satellite(self, isochrone, kernel, stellar_mass, distance_modulus, mc_source_id=1):
        """
        Create a simulated satellite. Returns a catalog object.
        """
        mag_1, mag_2, lon, lat = satellite(isochrone, kernel, stellar_mass, distance_modulus)
        pix = ugali.utils.projector.angToPix(self.config.params['coords']['nside_pixel'], lon, lat)

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * numpy.ones(healpy.nside2npix(self.config.params['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * numpy.ones(healpy.nside2npix(self.config.params['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]

        mag_err_1 = self.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.photo_err_2(mag_lim_2 - mag_2)
        mag_obs_1 = mag_1 + (numpy.random.normal(size=len(mag_1)) * mag_err_1)
        mag_obs_2 = mag_2 + (numpy.random.normal(size=len(mag_2)) * mag_err_2)

        cut = numpy.logical_and(mag_obs_1 < mag_lim_1, mag_obs_2 < mag_lim_2)
        #return mag_1_obs[cut], mag_2_obs[cut], lon[cut], lat[cut]
        mc_source_id = mc_source_id * numpy.ones(len(mag_1))
        
        if self.config.params['catalog']['coordsys'].lower() == 'cel' \
           and self.config.params['coords']['coordsys'].lower() == 'gal':
            lon, lat = ugali.utils.projector.galToCel(lon, lat)
        elif self.config.params['catalog']['coordsys'].lower() == 'gal' \
           and self.config.params['coords']['coordsys'].lower() == 'cel':
            lon, lat = ugali.utils.projector.celToGal(lon, lat)

        #hdu = self.makeHDU(mag_1[cut], mag_err_1[cut], mag_2[cut], mag_err_2[cut], lon[cut], lat[cut], mc_source_id[cut])
        hdu = self.makeHDU(mag_obs_1[cut], mag_err_1[cut], mag_obs_2[cut], mag_err_2[cut], lon[cut], lat[cut], mc_source_id[cut])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog

    def background(self,mc_source_id=1):
        """
        Create a simulation of the background stellar population.
        Because some stars have been clipped to generate the CMD,
        this function tends to slightly underestimate (~1%) the 
        background as compared to the true catalog.

        """
        # self.cmd_background = Number of objects / deg^2 / mag^2


        ### # Number of objects in the background annulus
        ### ntotal = (self.cmd_background * self.mask.solid_angle_cmd * self.roi.delta_color * self.roi.delta_mag).sum()
        ###  
        ### # Number of objects per square degree
        ### num_per_deg = self.cmd_background  * self.roi.delta_color * self.roi.delta_mag
        ###  
        ### # Number of objects per pixel per bin
        ### num_per_pix = self.cmd_background * self.roi.area_pixel
        ###  
        ### # To simulate at each pixel
        ### #npix = 1e4
        ### #numpy.random.poisson(lam=self.cmd_background, size=[npix]+list(self.cmd_background.shape))

        # Simulate over full ROI
        #area_roi = len(self.roi.pixels) * self.roi.area_pixel
        roi_radius = self.config.params['coords']['roi_radius']
        nside_pixel = self.config.params['coords']['nside_pixel']
        area_roi = numpy.pi*roi_radius**2
        lambda_per_bin = self.cmd_background * area_roi * self.roi.delta_color * self.roi.delta_mag
        nstar_per_bin = numpy.round(lambda_per_bin).astype(int) 
        #nstar_per_bin = numpy.random.poisson(lam=lambda_per_bin)
        nstar = nstar_per_bin.sum()
        print "nstar", nstar

        xx,yy = numpy.meshgrid(self.roi.centers_color,self.roi.centers_mag)
        color = numpy.repeat(xx.flatten(),repeats=nstar_per_bin.flatten())
        color += numpy.random.uniform(-self.roi.delta_color/2.,self.roi.delta_color/2.,size=len(color))
        mag_1 = numpy.repeat(yy.flatten(),repeats=nstar_per_bin.flatten())
        mag_1 += numpy.random.uniform(-self.roi.delta_mag/2.,self.roi.delta_mag/2.,size=len(color))
        mag_2 = mag_1 - color

        # Simulate random positions
        # Careful, may not respect sky projections
        radius = roi_radius*numpy.sqrt(numpy.random.uniform(size=nstar))
        theta = 2*numpy.pi*numpy.random.uniform(size=nstar)
        x = numpy.sqrt(radius)*numpy.cos(theta)
        y = numpy.sqrt(radius)*numpy.sin(theta)
        lon,lat = self.roi.projector.imageToSphere(x, y)
        pix = ugali.utils.projector.angToPix(nside_pixel, lon, lat)

        ### # Simulate position by drawing randomly from subpixels
        ### nside_pixel = self.config.params['coords']['nside_pixel']
        ### nside_subpix = 2**18
        ### print "Generating subpix"
        ### subpix = ugali.utils.projector.query_disc(nside_subpix, self.roi.vec, self.roi.config.params['coords']['roi_radius']+np.degrees(healpy.max_pixrad(nside_pixel)))
        ###  
        ### print "Calculating superpix"
        ### superpix = ugali.utils.skymap.superpixel(subpix,nside_subpix,nside_pixel)
        ### subpix = subpix[numpy.in1d(superpix,self.roi.pixels)]
        ### print "Random subpix"
        ### pixel = subpix[numpy.random.randint(0,len(subpix),nstar)]
        ### lon,lat = ugali.utils.projector.pixToAng(nside_subpix,pixel)
        ### pix = ugali.utils.projector.angToPix(nside_pixel, lon, lat)
        ### print len(pix)

        #return mag_1,mag_2,lon,lat,pix

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * numpy.ones(healpy.nside2npix(nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * numpy.ones(healpy.nside2npix(nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]

        mag_err_1 = self.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.photo_err_2(mag_lim_2 - mag_2)

        cut = numpy.logical_and(mag_1 < mag_lim_1, mag_2 < mag_lim_2)
        mc_source_id = mc_source_id * numpy.ones(len(mag_1))
        
        if self.config.params['catalog']['coordsys'].lower() == 'cel' \
           and self.config.params['coords']['coordsys'].lower() == 'gal':
            lon, lat = ugali.utils.projector.galToCel(lon, lat)
        elif self.config.params['catalog']['coordsys'].lower() == 'gal' \
           and self.config.params['coords']['coordsys'].lower() == 'cel':
            lon, lat = ugali.utils.projector.celToGal(lon, lat)

        hdu = self.makeHDU(mag_1[cut], mag_err_1[cut], mag_2[cut], mag_err_2[cut], lon[cut], lat[cut], mc_source_id[cut])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog

    def makeHDU(self, mag_1, mag_err_1, mag_2, mag_err_2, lon, lat, mc_source_id):
        """
        Create a pyfits header object based on input data.
        """
        columns_array = []
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['lon_field'],
                                           format = 'D',
                                           array = lon))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['lat_field'],
                                           format = 'D',
                                           array = lat))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['mag_1_field'],
                                           format = 'E',
                                           array = mag_1))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['mag_err_1_field'],
                                           format = 'E',
                                           array = mag_err_1))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['mag_2_field'],
                                           format = 'E',
                                           array = mag_2))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['mag_err_2_field'],
                                           format = 'E',
                                           array = mag_err_2))
        columns_array.append(pyfits.Column(name = self.config.params['catalog']['mc_source_id_field'],
                                           format = 'I',
                                           array = mc_source_id))
        hdu = pyfits.new_table(columns_array)
        return hdu

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
    # ADW: Updated kernel
    lon, lat = kernel.simulate(len(mag_1))

    return mag_1, mag_2, lon, lat

############################################################
