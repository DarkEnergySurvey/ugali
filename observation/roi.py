"""
Define a region of interest (ROI) in color, magnitude, and direction space.
"""

import numpy
import healpy

import ugali.utils.binning
import ugali.utils.projector
import ugali.utils.plotting
import ugali.utils.skymap

############################################################

class ROI():

    def __init__(self, config, lon, lat):

        self.config = config
        self.lon = lon
        self.lat = lat

        self.projector = ugali.utils.projector.Projector(self.lon, self.lat)

        vec = healpy.ang2vec(numpy.radians(90. - self.lat), numpy.radians(self.lon))
        try:
            self.pixels = healpy.query_disc(self.config.params['coords']['nside_pixel'],
                                            vec,
                                            self.config.params['coords']['roi_radius'],
                                            deg=True)
            self.pixels_annulus = numpy.setdiff1d(self.pixels, healpy.query_disc(self.config.params['coords']['nside_pixel'],
                                                                                 vec,
                                                                                 self.config.params['coords']['roi_radius_annulus'],
                                                                                 deg=True))
        except:
            self.pixels = healpy.query_disc(self.config.params['coords']['nside_pixel'],
                                            vec,
                                            numpy.radians(self.config.params['coords']['roi_radius']))
            self.pixels_annulus = numpy.setdiff1d(self.pixels, healpy.query_disc(self.config.params['coords']['nside_pixel'],
                                                                                 vec,
                                                                                 numpy.radians(self.config.params['coords']['roi_radius_annulus'])))
                                                  
        theta, phi = healpy.pix2ang(self.config.params['coords']['nside_pixel'], self.pixels)
        self.centers_lon, self.centers_lat = numpy.degrees(phi), 90. - numpy.degrees(theta)

        self.pixels_target = ugali.utils.skymap.subpixel(healpy.ang2pix(self.config.params['coords']['nside_likelihood_segmentation'],
                                                                        numpy.radians(90. - self.lat),
                                                                        numpy.radians(self.lon)),
                                                         self.config.params['coords']['nside_likelihood_segmentation'],
                                                         self.config.params['coords']['nside_pixel'])
        theta, phi = healpy.pix2ang(self.config.params['coords']['nside_pixel'], self.pixels_target)
        self.centers_lon_target, self.centers_lat_target = numpy.degrees(phi), 90. - numpy.degrees(theta)

        self.area_pixel = healpy.nside2pixarea(self.config.params['coords']['nside_pixel'], degrees=True) # deg^2
                                     
        """
        self.centers_x = self._centers(self.bins_x)
        self.centers_y = self._centers(self.bins_y)

        self.delta_x = self.config.params['coords']['pixel_size']
        self.delta_y = self.config.params['coords']['pixel_size']
        
        # Should actually try to take projection effects into account for better accuracy
        # MC integration perhaps?
        # Throw points in a cone around full ROI and see what fraction fall in
        self.area_pixel = self.config.params['coords']['pixel_size']**2
        
        self.centers_lon, self.centers_lat = self.projector.imageToSphere(self.centers_x, self.centers_y)
        """
        
        self.bins_mag = numpy.linspace(self.config.params['mag']['min'],
                                       self.config.params['mag']['max'],
                                       self.config.params['mag']['n_bins'] + 1)
        
        self.bins_color = numpy.linspace(self.config.params['color']['min'],
                                         self.config.params['color']['max'],
                                         self.config.params['color']['n_bins'] + 1)

        self.centers_mag = ugali.utils.binning.centers(self.bins_mag)
        self.centers_color = ugali.utils.binning.centers(self.bins_color)

        self.delta_mag = self.bins_mag[1] - self.bins_mag[0]
        self.delta_color = self.bins_color[1] - self.bins_color[0]

        # Axis labels
        self.label_x = 'x (deg)'
        self.label_y = 'y (deg)'
        
        if self.config.params['catalog']['band_1_detection']:
            self.label_mag = '%s (mag)'%(self.config.params['catalog']['mag_1_field'])
        else:
            self.label_mag = '%s (mag)'%(self.config.params['catalog']['mag_2_field'])
        self.label_color = '%s - %s (mag)'%(self.config.params['catalog']['mag_1_field'],
                                            self.config.params['catalog']['mag_2_field'])

        #self.precomputeAngsep()

    def plot(self, value=None, pixel=None):
        """
        Plot the ROI
        """
        map_roi = numpy.array(healpy.UNSEEN \
                              * numpy.ones(healpy.nside2npix(self.config.params['coords']['nside_pixel'])))
        
        if value is None:
            #map_roi[self.pixels] = ugali.utils.projector.angsep(self.lon, self.lat, self.centers_lon, self.centers_lat)
            map_roi[self.pixels] = 1
            map_roi[self.pixels_annulus] = 0
            map_roi[self.pixels_target] = 2
        elif value is not None and pixel is None:
            map_roi[self.pixels] = value
        elif value is not None and pixel is not None:
            map_roi[pixel] = value
        else:
            print 'ERROR: count not parse input'
        
        ugali.utils.plotting.zoomedHealpixMap('Region of Interest',
                                              map_roi,
                                              self.lon, self.lat,
                                              self.config.params['coords']['roi_radius'])
        
    def precomputeAngsep(self):
        """
        Precompute the angular separations to each pixel in ROI for each target pixel
        """
        self.angsep = []
        for ii in range(0, len(self.pixels_target)):
            self.angsep.append(ugali.utils.projector.angsep(self.centers_lon_target[ii],
                                                            self.centers_lat_target[ii],
                                                            self.centers_lon, 
                                                            self.centers_lat))

############################################################

