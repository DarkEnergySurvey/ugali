"""
Classes which manage object catalogs live here.
"""

import numpy
import pyfits
import pylab
import healpy

import ugali.utils.projector
import ugali.utils.plotting

############################################################

class Catalog:

    def __init__(self, config, data = None):
        """
        Class to store information about detected objects.

        INPUTS:
            config: Config object
            data[None]: pyfits table data (fitsrec) object.
        """
        #self.params = config.merge(config_merge) # Maybe you would want to update parameters??
        self.config = config

        if data is None:
            self._parse()
        else:
            self.data = data 
    
        self._defineVariables()

    def applyCut(self, cut):
        """
        Return a new catalog which is a subset of objects selected using the input cut array.
        """
        return Catalog(self.config, self.data[cut])

    def project(self, projector = None):
        """
        Project coordinates on sphere to image plane using Projector class.
        """
        if projector is None:
            try:
                self.projector = ugali.utils.projector.Projector(self.config.params['coords']['reference'][0],
                                                                 self.config.params['coords']['reference'][1])
            except KeyError:
                print 'WARNING: projection reference point is median (lon, lat) of catalog objects'
                self.projector = ugali.utils.projector.Projector(numpy.median(self.lon), numpy.median(self.lat))
        else:
            self.projector = projector

        self.x, self.y = self.projector.sphereToImage(self.lon, self.lat)

    def spatialBin(self, roi):
        """
        Return indices of ROI pixels corresponding to object locations.
        """
        theta = numpy.radians(90. - self.lat)
        phi = numpy.radians(self.lon)
        pix = healpy.ang2pix(self.config.params['coords']['nside_pixel'], theta, phi)

        # There must be a better way than using this loop??
        #self.pixel_roi = -1. * numpy.ones(len(self.lon))
        #for ii in range(0, len(roi.pixels)):
        #    self.pixel_roi[numpy.nonzero(pix == roi.pixels[ii])[0]] = ii

        # No for loop, but overhead of creating a full map
        map_roi = numpy.array(-1. * numpy.ones(healpy.nside2npix(self.config.params['coords']['nside_pixel'])))
        map_roi[roi.pixels] = numpy.linspace(0, len(roi.pixels) - 1, len(roi.pixels))
        self.pixel_roi = map_roi[pix]

        self.pixel_roi = self.pixel_roi.astype(int)

        if numpy.any(self.pixel_roi < 0):
            print "WARNING: objects found that are not contained within ROI"

    def write(self, outfile):
        """
        Write the current object catalog to fits file.
        """
        hdu = pyfits.BinTableHDU(self.data)
        hdu.writeto(outfile, clobber=True)

    def plotCMD(self, mode='scatter'):
        """
        Show the color-magnitude diagram for catalog objects as scatter plot or two-dimensional histogram.
        """
        if mode == 'scatter':
            ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
                                                       self.color, self.mag)
            y_min, y_max = pylab.axis()[2], pylab.axis()[3]
            pylab.ylim(y_max, y_min)
        elif mode == 'histogram':
            # ROI object needed here
            pass
        else:
            print 'WARNING: did not recognize plotting mode %s'%(mode)

    def plotMap(self, mode='scatter'):
        """
        Show map of catalog objects in image (projected) coordinates.
        """
        if mode == 'scatter':
            ugali.utils.plotting.twoDimensionalScatter('test', r'$\Delta$x', '$\Delta$y',
                                                       self.x, self.y, color=self.color)
                                                       #lim_x = lim_x
                                                       #lim_y = lim_y)
        else:
            print 'WARNING: did not recognize plotting mode %s'%(mode)


    def plotMag(self):
        """

        """
        pass

    def _parse(self):
        """
        Helper function to parse a catalog file and return a pyfits table.

        CSV format not yet validated.
        """
        file_type = self.config.params['catalog']['infile'].split('.')[-1].strip().lower()

        if file_type == 'csv':
            self.data = numpy.recfromcsv(self.config.params['catalog']['infile'], delimiter = ',')
        elif file_type in ['fit', 'fits']:
            self.data = pyfits.open(self.config.params['catalog']['infile'])[1].data
        else:
            print 'WARNING: did not recognize catalog file extension %s'%(file_type)
            
        print 'Found %i objects'%(len(self.data))

    def _defineVariables(self):
        """
        Helper funtion to define pertinent variables from catatalog data.
        """
        self.lon = self.data.field(self.config.params['catalog']['lon_field'])
        self.lat = self.data.field(self.config.params['catalog']['lat_field'])

        if self.config.params['catalog']['coordsys'].lower() == 'cel' \
           and self.config.params['catalog']['coordsys'].lower() == 'gal':
            self.lon, self.lat = ugali.utils.projector.celToGal(self.lon, self.lat)
        elif self.config.params['catalog']['coordsys'].lower() == 'gal' \
           and self.config.params['catalog']['coordsys'].lower() == 'cel':
            self.lon, self.lat = ugali.utils.projector.galToCel(self.lon, self.lat)

        self.mag_1 = self.data.field(self.config.params['catalog']['mag_1_field'])
        self.mag_err_1 = self.data.field(self.config.params['catalog']['mag_err_1_field'])
        self.mag_2 = self.data.field(self.config.params['catalog']['mag_2_field'])
        self.mag_err_2 = self.data.field(self.config.params['catalog']['mag_err_2_field'])

        if self.config.params['catalog']['mc_source_id_field'] is not None:
            self.mc_source_id = self.data.field(self.config.params['catalog']['mc_source_id_field'])
            print 'Found %i MC source objects'%(numpy.sum(self.mc_source_id == 1))

        if self.config.params['catalog']['band_1_detection']:
            self.mag = self.mag_1
            self.mag_err = self.mag_err_1
        else:
            self.mag = self.mag_2
            self.mag_err = self.mag_err_2
            
        self.color = self.mag_1 - self.mag_2
        self.color_err = numpy.sqrt(self.mag_err_1**2 + self.mag_err_2**2)

############################################################
