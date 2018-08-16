"""
Define a region of interest (ROI) in color, magnitude, and direction space.

The ROI is divided into 3 regions:
1) The 'target' region: The region occuping the likelihood-scale healpix
   pixel over which the likelihood is evaluated. (Size controlled by
   'nside_likelihood')
2) The 'interior' region: The region where objects are included into the
   likelihood fit.
3) The 'annulus' region: The region where the background is fit.

"""

import numpy as np
import healpy as hp

import ugali.utils.binning
import ugali.utils.projector
import ugali.utils.skymap

from ugali.utils.config import Config
from ugali.utils.healpix import query_disc, ang2pix, pix2ang, ang2vec

############################################################

# ADW: Should really write some "PixelSet" object that contains the pixels for each region...

class PixelRegion(np.ndarray):
    # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

    def __new__(cls, nside, pixels):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(pixels).view(cls)
        # add the new attribute to the created instance
        obj._nside = nside
        obj._pix = pixels
        obj._lon,obj._lat = pix2ang(nside,pixels)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return

        # NOTE: the _nside, _pix etc. attributes don't get set on
        # slicing because we are worried it will be too slow

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def nside(self):
        return self._nside

    @property
    def pix(self):
        return self._pix

class ROI(object):

    def __init__(self, config, lon, lat):

        self.config = Config(config)
        self.lon = lon
        self.lat = lat

        self.projector = ugali.utils.projector.Projector(self.lon, self.lat)

        self.vec = vec = ang2vec(self.lon, self.lat)
        self.pix = ang2pix(self.config['coords']['nside_likelihood'],self.lon,self.lat)

        # Pixels from the entire ROI disk
        pix = query_disc(self.config['coords']['nside_pixel'], vec, 
                         self.config['coords']['roi_radius'])
        self.pixels = PixelRegion(self.config['coords']['nside_pixel'],pix)

        # Pixels in the interior region
        pix = query_disc(self.config['coords']['nside_pixel'], vec, 
                         self.config['coords']['roi_radius_interior'])
        self.pixels_interior = PixelRegion(self.config['coords']['nside_pixel'],pix)

        # Pixels in the outer annulus
        pix = query_disc(self.config['coords']['nside_pixel'], vec, 
                         self.config['coords']['roi_radius_annulus'])
        pix = np.setdiff1d(self.pixels, pix)
        self.pixels_annulus = PixelRegion(self.config['coords']['nside_pixel'],pix)

        # Pixels within target healpix region
        pix = ugali.utils.skymap.subpixel(self.pix,self.config['coords']['nside_likelihood'],
                                          self.config['coords']['nside_pixel'])
        self.pixels_target = PixelRegion(self.config['coords']['nside_pixel'],pix)

        # Boolean arrays for selecting given pixels 
        # (Careful, this works because pixels are pre-sorted by query_disc before in1d)
        self.pixel_interior_cut = np.in1d(self.pixels, self.pixels_interior)
        self.pixel_annulus_cut  = np.in1d(self.pixels, self.pixels_annulus)

        # Some pixel properties
        self.area_pixel = hp.nside2pixarea(self.config.params['coords']['nside_pixel'],degrees=True) # deg^2
        self.max_pixrad = np.degrees(hp.max_pixrad(self.config['coords']['nside_pixel'])) # deg
                                     
        # ADW: These are really bin edges, should be careful and consistent
        # It would be cleaner to separate the CMD from ROI
        self.bins_mag = np.linspace(self.config.params['mag']['min'],
                                       self.config.params['mag']['max'],
                                       self.config.params['mag']['n_bins'] + 1)
        
        self.bins_color = np.linspace(self.config.params['color']['min'],
                                         self.config.params['color']['max'],
                                         self.config.params['color']['n_bins'] + 1)

        self.centers_mag = ugali.utils.binning.centers(self.bins_mag)
        self.centers_color = ugali.utils.binning.centers(self.bins_color)

        self.delta_mag = self.bins_mag[1] - self.bins_mag[0]
        self.delta_color = self.bins_color[1] - self.bins_color[0]

        ## Axis labels (DEPRECATED)
        #self.label_x = 'x (deg)'
        #self.label_y = 'y (deg)'
        # 
        #if self.config.params['catalog']['band_1_detection']:
        #    self.label_mag = '%s (mag)'%(self.config.params['catalog']['mag_1_field'])
        #else:
        #    self.label_mag = '%s (mag)'%(self.config.params['catalog']['mag_2_field'])
        #self.label_color = '%s - %s (mag)'%(self.config.params['catalog']['mag_1_field'],
        #                                    self.config.params['catalog']['mag_2_field'])

    def plot(self, value=None, pixel=None):
        """
        Plot the ROI
        """
        # DEPRECATED
        import ugali.utils.plotting

        map_roi = np.array(hp.UNSEEN \
                              * np.ones(hp.nside2npix(self.config.params['coords']['nside_pixel'])))
        
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
            logger.error("Can't parse input")
        
        ugali.utils.plotting.zoomedHealpixMap('Region of Interest',
                                              map_roi,
                                              self.lon, self.lat,
                                              self.config.params['coords']['roi_radius'])

    # ADW: Maybe these should be associated with the PixelRegion objects
    def inPixels(self,lon,lat,pixels):
        """ Function for testing if coordintes in set of ROI pixels. """
        nside = self.config.params['coords']['nside_pixel']
        return ugali.utils.healpix.in_pixels(lon,lat,pixels,nside)
        
    def inROI(self,lon,lat):
        return self.inPixels(lon,lat,self.pixels)

    def inAnnulus(self,lon,lat):
        return self.inPixels(lon,lat,self.pixels_annulus)

    def inInterior(self,lon,lat):
        return self.inPixels(lon,lat,self.pixels_interior)

    def inTarget(self,lon,lat):
        return self.inPixels(lon,lat,self.pixels_target)

    def indexPixels(self,lon,lat,pixels):
        nside = self.config.params['coords']['nside_pixel']
        return ugali.utils.healpix.index_pixels(lon,lat,pixels,nside)

    def indexROI(self,lon,lat):
        return self.indexPixels(lon,lat,self.pixels)

    def indexAnnulus(self,lon,lat):
        return self.indexPixels(lon,lat,self.pixels_annulus)

    def indexInterior(self,lon,lat):
        return self.indexPixels(lon,lat,self.pixels_interior)

    def indexTarget(self,lon,lat):
        return self.indexPixels(lon,lat,self.pixels_target)
        
    def getCatalogPixels(self):
        """
        Return the catalog pixels spanned by this ROI.
        """
        filenames = self.config.getFilenames()

        nside_catalog = self.config.params['coords']['nside_catalog']
        nside_pixel = self.config.params['coords']['nside_pixel']
        # All possible catalog pixels spanned by the ROI
        superpix = ugali.utils.skymap.superpixel(self.pixels,nside_pixel,nside_catalog)
        superpix = np.unique(superpix)
        # Only catalog pixels that exist in catalog files
        pixels = np.intersect1d(superpix, filenames['pix'].compressed())
        return pixels
        
############################################################

