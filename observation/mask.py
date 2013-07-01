"""
Classes and functions that handle masks (i.e., exposure depth). 

Classes
    Mask

Functions
    someFunction
"""

import os
import numpy
import scipy.signal
import healpy

import ugali.observation.roi
import ugali.utils.plotting
import ugali.utils.binning

############################################################

class Mask:
    """
    Contains maps of completeness depth in magnitudes for multiple observing bands, and associated products.
    """
    def __init__(self, config, mask_1, mask_2):

        self.config = config
        self.mask_1 = mask_1
        self.mask_2 = mask_2

        self.roi = self.mask_1.roi

        self.minimum_solid_angle = self.config.params['mask']['minimum_solid_angle'] # deg^2

        self._solidAngleCMD()
        self._pruneCMD(self.minimum_solid_angle)

    def _solidAngleCMD(self):
        """
        Compute solid angle within the mask (deg^2) as a function of color and magnitude.
        """

        self.solid_angle_cmd = numpy.zeros([len(self.roi.centers_mag),
                                            len(self.roi.centers_color)])

        for index_mag in range(0, len(self.roi.centers_mag)):
            for index_color in range(0, len(self.roi.centers_color)):

                mag = self.roi.centers_mag[index_mag]
                color = self.roi.centers_color[index_color]

                if self.config.params['catalog']['band_1_detection']:
                    # Evaluating at the center of the color-magnitude bin, be consistent!
                    #mag_1 = self.roi.centers_mag[index_mag]
                    #color = self.roi.centers_color[index_color]
                    #mag_2 = mag_1 - color
                    # Evaluating at corner of the color-magnitude bin, be consistent!
                    mag_1 = mag + (0.5 * self.roi.delta_mag)
                    mag_2 = mag - color + (0.5 * self.roi.delta_color)
                else:
                    # Evaluating at the center of the color-magnitude bin, be consistent!
                    #mag_2 = self.roi.centers_mag[index_mag]
                    #color = self.roi.centers_color[index_color]
                    #mag_1 = mag_2 + color
                    # Evaluating at corner of the color-magnitude bin, be consistent!
                    mag_1 = mag + color + (0.5 * self.roi.delta_color)
                    mag_2 = mag + (0.5 * self.roi.delta_mag)

                #self.solid_angle_cmd[index_mag, index_color] = self.roi.area_pixel * numpy.sum((self.mask_1.mask > mag_1) * (self.mask_2.mask > mag_2))
                n_unmasked_pixels = numpy.sum((self.mask_1.mask[self.roi.pixels] > mag_1) \
                                              * (self.mask_2.mask[self.roi.pixels] > mag_2))
                self.solid_angle_cmd[index_mag, index_color] = self.roi.area_pixel * n_unmasked_pixels

    def _pruneCMD(self, minimum_solid_angle):
        """
        Remove regions of color-magnitude space where the unmasked solid angle is
        statistically insufficient to estimate the background.

        INPUTS:
            solid_angle[1]: minimum solid angle (deg^2)
        """

        print 'Prunning CMD based on minimum solid angle of %.2f deg^2'%(minimum_solid_angle)
        
        self.solid_angle_cmd *= self.solid_angle_cmd > minimum_solid_angle

        # Compute which magnitudes the clipping correspond to
        index_mag, index_color = numpy.nonzero(self.solid_angle_cmd)
        mag = self.roi.centers_mag[index_mag]
        color = self.roi.centers_color[index_color]
        if self.config.params['catalog']['band_1_detection']:
            mag_1 = mag
            mag_2 = mag_1 - color
            self.mag_1_clip = numpy.max(mag_1) + (0.5 * self.roi.delta_mag)
            self.mag_2_clip = numpy.max(mag_2) + (0.5 * self.roi.delta_color)
        else:
            mag_2 = mag
            mag_1 = color + mag_2
            self.mag_1_clip = numpy.max(mag_1) + (0.5 * self.roi.delta_color)
            self.mag_2_clip = numpy.max(mag_2) + (0.5 * self.roi.delta_mag)

        print 'Clipping mask 1 at %.2f mag'%(self.mag_1_clip)
        print 'Clipping mask 2 at %.2f mag'%(self.mag_2_clip)
        self.mask_1.mask_roi = numpy.clip(self.mask_1.mask_roi, 0., self.mag_1_clip)
        self.mask_2.mask_roi = numpy.clip(self.mask_2.mask_roi, 0., self.mag_2_clip)
        
    def plotSolidAngleCMD(self):
        """
        Solid angle within the mask as a function of color and magnitude.
        """
        ugali.utils.plotting.twoDimensionalHistogram('mask', 'color', 'magnitude',
                                                     self.solid_angle_cmd,
                                                     self.roi.bins_color,
                                                     self.roi.bins_mag,
                                                     lim_x = [self.roi.bins_color[0],
                                                              self.roi.bins_color[-1]],
                                                     lim_y = [self.roi.bins_mag[-1],
                                                              self.roi.bins_mag[0]])

    def backgroundCMD(self, catalog, mode='cloud-in-cells', plot=False):
        """
        Generate an empirical background model in color-magnitude space.
        
        INPUTS:
            catalog: Catalog object
        OUTPUTS:
            background
        """

        if mode == 'cloud-in-cells':

            # Weight each object before binning
            # Divide by solid angle and bin size in magnitudes
            # Units are (deg^-2 mag^-2)
            solid_angle = ugali.utils.binning.take2D(self.solid_angle_cmd,
                                                     catalog.color, catalog.mag,
                                                     self.roi.bins_color, self.roi.bins_mag)
            weights = (solid_angle * self.roi.delta_color * self.roi.delta_mag)**(-1)
            
            # Apply cloud-in-cells algorithm
            cmd_background = ugali.utils.binning.cloudInCells(catalog.color,
                                                              catalog.mag,
                                                              [self.roi.bins_color,
                                                               self.roi.bins_mag],
                                                              weights)[0]

            # Account for the events that spill out of observable space
            for index_color in range(0, len(self.roi.centers_color)):
                for index_mag in range(0, len(self.roi.centers_mag)):
                    if self.solid_angle_cmd[index_mag][index_color] < self.minimum_solid_angle:
                        cmd_background[index_mag - 1][index_color] += cmd_background[index_mag][index_color]
                        cmd_background[index_mag][index_color] = 0.
                        break

            # Divide by solid angle and bin size in magnitudes
            # Units are (deg^-2 mag^-2)
            # For numerical stability, avoid dividing by zero
            #epsilon = 1.e-10
            #cmd_background /= (self.solid_angle_cmd + epsilon) * self.roi.delta_color * self.roi.delta_mag
            #cmd_background *= self.solid_angle_cmd > epsilon
                  
        elif mode == 'bootstrap':
            #
            mag_1_array = catalog.mag_1
            mag_2_array = catalog.mag_2

            catalog.mag_1 + (catalog.mag_1_err * numpy.random.normal(0, 1., len(catalog.mag_1)))
            catalog.mag_2 + (catalog.mag_2_err * numpy.random.normal(0, 1., len(catalog.mag_2)))

        if plot:
            ugali.utils.plotting.twoDimensionalHistogram(r'CMD Background (deg$^{-2}$ mag$^{-2}$)',
                                                         'color (mag)', 'magnitude (mag)',
                                                         cmd_background,
                                                         self.roi.bins_color,
                                                         self.roi.bins_mag,
                                                         lim_x = [self.roi.bins_color[0],
                                                                  self.roi.bins_color[-1]],
                                                         lim_y = [self.roi.bins_mag[-1],
                                                                  self.roi.bins_mag[0]])

        return cmd_background 
        
    def restrictCatalogToObservableSpace(self, catalog):
        """
        Retain only the catalog objects which fall within the observable (i.e., unmasked) space.

        INPUTS:
            catalog: a Catalog object
        OUTPUTS:
            boolean cut array where True means the object would be observable (i.e., unmasked).
        """

        # Check that the objects fall in the color-magnitude space of the ROI
        cut_mag = numpy.logical_and(catalog.mag > self.roi.bins_mag[0],
                                    catalog.mag < self.roi.bins_mag[-1])
        cut_color = numpy.logical_and(catalog.color > self.roi.bins_color[0],
                                      catalog.color < self.roi.bins_color[-1])

        # and are observable in the ROI-specific mask for both bands
        theta = numpy.radians(90. - catalog.lat)
        phi = numpy.radians(catalog.lon)
        pix = healpy.ang2pix(self.config.params['coords']['nside_pixel'], theta, phi)
        cut_mag_1 = catalog.mag_1 < self.mask_1.mask_roi[pix]
        cut_mag_2 = catalog.mag_2 < self.mask_2.mask_roi[pix]

        # and are located in the region of color-magnitude space where background can be estimated
        cut_cmd = ugali.utils.binning.take2D(self.solid_angle_cmd,
                                             catalog.color, catalog.mag,
                                             self.roi.bins_color, self.roi.bins_mag) > 0.

        cut = numpy.all([cut_mag,
                         cut_color,
                         cut_mag_1,
                         cut_mag_2,
                         cut_cmd], axis=0)
        
        return cut
        

############################################################

class MaskBand:
    """
    Map of completeness depth in magnitudes for a single observing band.
    """

    def __init__(self, infile, roi):
        """
        Infile is a HEALPix map.
        """
        self.roi = roi
        self.mask = ugali.utils.skymap.readSparseHealpixMap(infile)
        
        self.mask_roi = numpy.zeros(len(self.mask))
        self.mask_roi[self.roi.pixels] =  self.mask[self.roi.pixels] # ROI specific

    def depth(self, x, y):
        """
        Return completeness depth in magnitudes at given image coordinates
        """
        pass

    def plot(self, roi = False):
        """
        Plot the magnitude depth.
        """

        if roi:
            mask = self.mask_roi
        else:
            mask = self.mask
        
        ugali.utils.plotting.zoomedHealpixMap('Completeness Depth',
                                              mask,
                                              self.roi.lon, self.roi.lat,
                                              self.roi.config.params['coords']['roi_radius'])

############################################################

def simpleMask(config):

    #params = ugali.utils.(config, kwargs)

    roi = ugali.observation.roi.ROI(config)

    # De-project the bin centers to get magnitude depths

    mesh_x, mesh_y = numpy.meshgrid(roi.centers_x, roi.centers_y)
    r = numpy.sqrt(mesh_x**2 + mesh_y**2) # Think about x, y conventions here

    #z = (0. * (r > 1.)) + (21. * (r < 1.))
    #z = 21. - r
    #z = (21. - r) * (mesh_x > 0.) * (mesh_y < 0.)
    z = (21. - r) * numpy.logical_or(mesh_x > 0., mesh_y > 0.)

    return MaskBand(z, roi)
    
############################################################

def readMangleFile(infile, lon, lat, index = None):
    """
    Mangle must be set up on your system.
    The index argument is a temporary file naming convention to avoid file conflicts.
    Coordinates must be given in the native coordinate system of the Mangle file.
    """

    if index is None:
        index = numpy.random.randint(0, 1.e10)
    
    coordinates_file = 'temp_coordinates_%010i.dat'%(index)
    maglims_file = 'temp_maglims_%010i.dat'%(index)

    writer = open(coordinates_file, 'w')
    for ii in range(0, len(lon)):
        writer.write('%12.5f%12.5f\n'%(lon[ii], lat[ii]))
    writer.close()

    os.system('polyid -W %s %s %s || exit'%(infile,
                                            coordinates_file,
                                            maglims_file))

    reader = open(maglims_file)
    lines = reader.readlines()
    reader.close()

    os.remove(maglims_file)
    os.remove(coordinates_file)

    maglims = []
    for ii in range(1, len(lines)):
        if len(lines[ii].split()) == 3:
            maglims.append(float(lines[ii].split()[2]))
        elif len(lines[ii].split()) == 2:
            maglims.append(0.) # Coordinates outside of the MANGLE ploygon
        elif len(lines[ii].split()) > 3:
            #print 'WARNING: coordinate inside multiple polygons, using weight from first polygon'
            #maglims.append(float(lines[ii].split()[2])) # Mask out the pixels inside multiple polygons
            print 'WARNING: coordinate inside multiple polygons, masking that coordinate'
            maglims.append(0.)
        else:
            print 'WARNING: cannot parse maglims file, unexpected number of columns, stop reading now'
            break
            
    maglims = numpy.array(maglims)
    return maglims

############################################################

def farmMaskFromCatalog(catalog, infile_mangle, nside_pix, nside_subpix, writedir):
    """
    Given an object catalog, farm out the task of creating a mask.
    """

    if not os.path.exists(writedir):
        os.mkdir(writedir)
    
    if self.config.params['coords']['coordsys'].lower() == 'cel' \
           and self.config.params['mask']['coordsys'].lower() == 'gal':
        lon_catalog, lat_catalog = ugali.utils.projector.celToGal(catalog.lon, catalog.lat)
    elif self.config.params['coords']['coordsys'].lower() == 'gal' \
             and self.config.params['mask']['coordsys'].lower() == 'cel':
        lon_catalog, lat_catalog = ugali.utils.projector.galToCel(catalog.lon, catalog.lat)
    else:
        lon_catalog, lat_catalog = catalog.lon, catalog.lat

    pix, subpix = ugali.utils.skymap.surveyPixel(lon_catalog, lat_catalog, nside_pix, nside_subpix)

    for ii in range(0, len(pix)):
        theta, phi =  healpy.pix2ang(nside_subpix, subpix[ii])
        lon, lat = numpy.degrees(phi), 90. - numpy.degrees(theta)
        outfile = '%s/mask_%010i.fits'%(writedir, pix[ii])
        print '(%i/%i) %s %i (%.3f, %.3f)'%(ii, len(pix), outfile, len(lon), lon[0], lat[0])
        maglims = readMangleFile(infile_mangle, lon, lat, index = pix[ii])
        ugali.utils.skymap.writeSparseHealpixMap(subpix[ii], maglims, nside_subpix, outfile)

############################################################
