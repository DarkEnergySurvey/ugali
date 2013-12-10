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
import ugali.utils.projector
import ugali.utils.skymap
from ugali.utils.logger import logger

############################################################

class Mask:
    """
    Contains maps of completeness depth in magnitudes for multiple observing bands, and associated products.
    """
    def __init__(self, config, roi):

        self.config = config
        self.roi = roi
        filenames = self.config.getFilenames()
        catalog_pixels = self.roi.getCatalogPixels()
        self.mask_1 = MaskBand(filenames['mask_1'][catalog_pixels],self.roi)
        self.mask_2 = MaskBand(filenames['mask_2'][catalog_pixels],self.roi)

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
                n_unmasked_pixels = numpy.sum((self.mask_1.mask_annulus_sparse > mag_1) \
                                              * (self.mask_2.mask_annulus_sparse > mag_2))
                self.solid_angle_cmd[index_mag, index_color] = self.roi.area_pixel * n_unmasked_pixels

    def _pruneCMD(self, minimum_solid_angle):
        """
        Remove regions of color-magnitude space where the unmasked solid angle is
        statistically insufficient to estimate the background.

        INPUTS:
            solid_angle[1]: minimum solid angle (deg^2)
        """

        logger.info('Prunning CMD based on minimum solid angle of %.2f deg^2'%(minimum_solid_angle))
        
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

        logger.info('Clipping mask 1 at %.2f mag'%(self.mag_1_clip) )
        logger.info('Clipping mask 2 at %.2f mag'%(self.mag_2_clip) )
        self.mask_1.mask_roi_sparse = numpy.clip(self.mask_1.mask_roi_sparse, 0., self.mag_1_clip)
        self.mask_2.mask_roi_sparse = numpy.clip(self.mask_2.mask_roi_sparse, 0., self.mag_2_clip)
        self.mask_1.mask_annulus_sparse = numpy.clip(self.mask_1.mask_annulus_sparse, 0., self.mag_1_clip)
        self.mask_2.mask_annulus_sparse = numpy.clip(self.mask_2.mask_annulus_sparse, 0., self.mag_2_clip)
        
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

    def backgroundCMD(self, catalog, mode='cloud-in-cells', weights=None, plot=False):
        """
        Generate an empirical background model in color-magnitude space.
        
        INPUTS:
            catalog: Catalog object
        OUTPUTS:
            background
        """

        if mode == 'cloud-in-cells':
            # Select objects in annulus
            cut_annulus = numpy.in1d(ugali.utils.projector.angToPix(self.config.params['coords']['nside_pixel'],
                                                                    catalog.lon, catalog.lat),
                                     self.roi.pixels_annulus)
            color = catalog.color[cut_annulus]
            mag = catalog.mag[cut_annulus]

            # Weight each object before binning
            # Divide by solid angle and bin size in magnitudes to get number density
            # Units are (deg^-2 mag^-2)
            solid_angle = ugali.utils.binning.take2D(self.solid_angle_cmd,
                                                     color, mag,
                                                     self.roi.bins_color, self.roi.bins_mag)

            # Optionally weight each catalog object
            if weights is None:
                number_density = (solid_angle * self.roi.delta_color * self.roi.delta_mag)**(-1)
            else:
                number_density = weights * (solid_angle * self.roi.delta_color * self.roi.delta_mag)**(-1)
            
            # Apply cloud-in-cells algorithm
            cmd_background = ugali.utils.binning.cloudInCells(color,
                                                              mag,
                                                              [self.roi.bins_color,
                                                               self.roi.bins_mag],
                                                              weights=number_density)[0]

            # Account for the objects that spill out of the observable space
            # But what about the objects that spill out to red colors??
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

            # Avoid dividing by zero
            cmd_background[numpy.logical_and(self.solid_angle_cmd > 0.,
                                             cmd_background == 0.)] = numpy.min(cmd_background[cmd_background > 0.])
                  
        elif mode == 'bootstrap':
            # Not yet implemented
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
        #if not hasattr(catalog, 'pixel_roi_index'): # TODO: An attempt to save computations, but not robust
        #    catalog.spatialBin(self.roi)
        catalog.spatialBin(self.roi)
        cut_roi = (catalog.pixel_roi_index >= 0) # Objects outside ROI have pixel_roi_index of -1
        # ADW: This creates a slope in color-magnitude space near the magnitude limit
        # i.e., if color=g-r then you can't have an object with g-r=1 and mag_r > mask_r-1
        cut_mag_1 = catalog.mag_1 < self.mask_1.mask_roi_sparse[catalog.pixel_roi_index]
        cut_mag_2 = catalog.mag_2 < self.mask_2.mask_roi_sparse[catalog.pixel_roi_index]

        # and are located in the region of color-magnitude space where background can be estimated
        cut_cmd = ugali.utils.binning.take2D(self.solid_angle_cmd,
                                             catalog.color, catalog.mag,
                                             self.roi.bins_color, self.roi.bins_mag) > 0.

        cut = numpy.all([cut_mag,
                         cut_color,
                         cut_roi,
                         cut_mag_1,
                         cut_mag_2,
                         cut_cmd], axis=0)
        
        return cut
        

############################################################

class MaskBand:
    """
    Map of completeness depth in magnitudes for a single observing band.
    """

    def __init__(self, infiles, roi):
        """
        Infile is a sparse HEALPix map fits file.
        """
        self.roi = roi
        mask = ugali.utils.skymap.readSparseHealpixMaps(infiles, field='MAGLIM')
        self.mask_roi_sparse = mask[self.roi.pixels] # Sparse map for pixels in ROI
        self.mask_annulus_sparse = mask[self.roi.pixels_annulus] # Sparse map for pixels in annulus part of ROI
        self.nside = healpy.npix2nside(len(mask))

    def depth(self, x, y):
        """
        Return completeness depth in magnitudes at given image coordinates.
        """
        pass

    def plot(self):
        """
        Plot the magnitude depth.
        """
        mask = healpy.UNSEEN * numpy.ones(healpy.nside2npix(self.nside))
        mask[self.roi.pixels] = self.mask_roi_sparse
        mask[mask == 0.] = healpy.UNSEEN
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
    
    coordinate_file = 'temp_coordinate_%010i.dat'%(index)
    maglim_file = 'temp_maglim_%010i.dat'%(index)

    writer = open(coordinate_file, 'w')
    for ii in range(0, len(lon)):
        writer.write('%12.5f%12.5f\n'%(lon[ii], lat[ii]))
    writer.close()

    os.system('polyid -W %s %s %s || exit'%(infile,
                                            coordinate_file,
                                            maglim_file))

    reader = open(maglim_file)
    lines = reader.readlines()
    reader.close()

    os.remove(maglim_file)
    os.remove(coordinate_file)

    maglim = []
    for ii in range(1, len(lines)):
        if len(lines[ii].split()) == 3:
            maglim.append(float(lines[ii].split()[2]))
        elif len(lines[ii].split()) == 2:
            maglim.append(0.) # Coordinates outside of the MANGLE ploygon
        elif len(lines[ii].split()) > 3:
            #print 'WARNING: coordinate inside multiple polygons, using weight from first polygon'
            #maglim.append(float(lines[ii].split()[2])) # Mask out the pixels inside multiple polygons
            logger.warning('Coordinate inside multiple polygons, masking that coordinate.')
            maglim.append(0.)
        else:
            logger.warning('Cannot parse maglim file, unexpected number of columns, stop reading now.')
            break
            
    maglim = numpy.array(maglim)
    return maglim

############################################################

def allSkyMask(infile, nside):
    lon, lat = ugali.utils.skymap.allSkyCoordinates(nside)
    maglim = readMangleFile(infile, lon, lat, index = None)
    return maglim

############################################################

def scale(mask, mag_scale, outfile=None):
        """
        Scale the completeness depth of a mask such that mag_new = mag + mag_scale.
        Input is a full HEALPix map.
        Optionally write out the scaled mask as an sparse HEALPix map.
        """
        mask_new = healpy.UNSEEN * numpy.ones(len(mask))
        mask_new[mask == 0.] = 0.
        mask_new[mask > 0.] = mask[mask > 0.] + mag_scale

        if outfile is not None:
            pix = numpy.nonzero(mask_new > 0.)[0]
            data_dict = {'MAGLIM': mask_new[pix]}
            nside = healpy.npix2nside(len(mask_new))
            ugali.utils.skymap.writeSparseHealpixMap(pix, data_dict, nside, outfile)

        return mask_new

############################################################

