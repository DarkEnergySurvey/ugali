#!/usr/bin/env python

"""
Class to farm out analysis tasks.

Classes
    Mask

Functions
    someFunction
"""

import os
import numpy
import healpy

import ugali.utils.parse_config
import ugali.utils.skymap
import ugali.observation.catalog
import ugali.observation.mask

############################################################

class Farm:
    """
    The Farm class is the master analysis coordinator.
    """

    def __init__(self, config):
        
        self.config = ugali.utils.parse_config.Config(config)
        self.catalog = ugali.observation.catalog.Catalog(self.config)

    def farmMaskFromCatalog(self, local=True):
        """
        Given an object catalog, farm out the task of creating a mask.
        """
        
        pix, subpix = ugali.utils.skymap.surveyPixel(self.catalog.lon, self.catalog.lat,
                                                     self.config.params['coords']['nside_mask_segmentation'],
                                                     self.config.params['coords']['nside_pixel'])

        print '=== Mask From Catalog ==='

        for infile in [self.config.params['mangle']['infile_1'],
                       self.config.params['mangle']['infile_2']]:

            print 'Mangle infile = %s'%(infile)

            if infile == self.config.params['mangle']['infile_1']:
                savedir = self.config.params['output']['savedir_mag_1_mask']
            elif infile == self.config.params['mangle']['infile_2']:
                savedir = self.config.params['output']['savedir_mag_2_mask']
            else:
                print 'WARNING: did not recognize the Mangle file %s.'%(infile)

            if not os.path.exists(savedir):
                os.mkdir(savedir)
            
            print 'Savedir = %s'%(savedir)
            
            for ii in range(0, len(pix)):

                #if ii >= 2:
                #    break # Just for testing

                theta, phi =  healpy.pix2ang(self.config.params['coords']['nside_mask_segmentation'], pix[ii])
                lon, lat = numpy.degrees(phi), 90. - numpy.degrees(theta)
                
                print '  (%i/%i) pixel %i nside %i; %i query points; %s (lon, lat) = (%.3f, %.3f)'%(ii, len(pix), pix[ii],
                                                                                                    self.config.params['coords']['nside_mask_segmentation'],
                                                                                                    len(subpix[ii]),
                                                                                                    self.config.params['coords']['coordsys'],
                                                                                                    lon, lat)

                # Should actually check to see if outfile exists
                outfile = '%s/mask_%010i_nside_pix_%i_nside_subpix_%i_%s.fits'%(savedir,
                                                                                pix[ii],
                                                                                self.config.params['coords']['nside_mask_segmentation'],
                                                                                self.config.params['coords']['nside_pixel'],
                                                                                self.config.params['coords']['coordsys'].lower())

                if os.path.exists(outfile):
                    print '  %s already exists. Skipping ...'%(outfile)
                    continue
                
                if local:
                    self.farmMaskFromCatalogNow(pix[ii], infile, outfile)
                else:
                    # Submit to queue
                    pass                

    def farmMaskFromCatalogNow(self, pix, infile, outfile):
        """
        
        """

        subpix = ugali.utils.skymap.subpixel(pix,
                                             self.config.params['coords']['nside_mask_segmentation'],
                                             self.config.params['coords']['nside_pixel'])
        
        theta, phi =  healpy.pix2ang(self.config.params['coords']['nside_pixel'], subpix)
        lon, lat = numpy.degrees(phi), 90. - numpy.degrees(theta)

        # Conversion between coordinate systems of object catalog and Mangle mask
        if self.config.params['coords']['coordsys'].lower() == 'cel' \
               and self.config.params['mangle']['coordsys'].lower() == 'gal':
            lon, lat = ugali.utils.projector.celToGal(lon, lat)
        elif self.config.params['coords']['coordsys'].lower() == 'gal' \
                 and self.config.params['mangle']['coordsys'].lower() == 'cel':
            lon, lat = ugali.utils.projector.galToCel(lon, lat)
        else:
            pass
            
        maglim = ugali.observation.mask.readMangleFile(infile, lon, lat, index = pix)
        data_dict = {'MAGLIM': maglim}
        ugali.utils.skymap.writeSparseHealpixMap(subpix, data_dict, self.config.params['coords']['nside_pixel'],
                                                 outfile, coordsys=self.config.params['coords']['coordsys'])
        
    def farmLikelihoodFromCatalog(self):
        """
        Given an object catalog, farm out the task of evaluating the likelihood.
        """
        pass
    
############################################################

def main():
    """
    Placeholder for when users want to call this script as an executable
    """
    from optparse import OptionParser
    
if __name__ == "__main__":
    main()
