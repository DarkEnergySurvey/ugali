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

import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.likelihood
import ugali.observation.catalog
import ugali.observation.mask
import ugali.utils.parse_config
import ugali.utils.skymap

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
        
    def farmLikelihoodFromCatalog(self, local=True):
        """
        Given an object catalog, farm out the task of evaluating the likelihood.
        """
        pix = ugali.utils.skymap.surveyPixel(self.catalog.lon, self.catalog.lat,
                                             self.config.params['coords']['nside_likelihood_segmentation'])

        if not os.path.exists(self.config.params['output']['savedir_likelihood']):
            os.mkdir(self.config.params['output']['savedir_likelihood'])

        print '=== Likelihood From Catalog ==='
        for ii in range(0, len(pix)):

            # Just for testing
            #if ii >= 2:
            #    continue
            
            theta, phi =  healpy.pix2ang(self.config.params['coords']['nside_likelihood_segmentation'], pix[ii])
            lon, lat = numpy.degrees(phi), 90. - numpy.degrees(theta)

            n_query_points = healpy.nside2npix(self.config.params['coords']['nside_pixel']) \
                             / healpy.nside2npix(self.config.params['coords']['nside_likelihood_segmentation'])

            print '  (%i/%i) pixel %i nside %i; %i query points; %s (lon, lat) = (%.3f, %.3f)'%(ii, len(pix), pix[ii],
                                                                                                self.config.params['coords']['nside_likelihood_segmentation'],
                                                                                                n_query_points,
                                                                                                self.config.params['coords']['coordsys'],
                                                                                                lon, lat)

            # Should actually check to see if outfile exists
            outfile = '%s/likelihood_%010i_nside_pix_%i_nside_subpix_%i_%s.fits'%(self.config.params['output']['savedir_likelihood'],
                                                                                  pix[ii],
                                                                                  self.config.params['coords']['nside_likelihood_segmentation'],
                                                                                  self.config.params['coords']['nside_pixel'],
                                                                                  self.config.params['coords']['coordsys'].lower())
            
            if os.path.exists(outfile):
                print '  %s already exists. Skipping ...'%(outfile)
                continue
                
            if local:
                likelihood = self.farmLikelihoodFromCatalogNow(pix[ii], outfile)
                #pass
            else:
                # Submit to queue
                pass

            #return likelihood

    def farmLikelihoodFromCatalogNow(self, pix, outfile):
        """

        """

        theta, phi =  healpy.pix2ang(self.config.params['coords']['nside_likelihood_segmentation'], pix)
        lon, lat = numpy.degrees(phi), 90. - numpy.degrees(theta)

        # Make these config parameters
        isochronedir = '/home/s1/bechtol/des10.a/projects/mw_substructure/stellar_evolution/isochrones/des/'
        isochrone_names = ['isota1010z0.1.dat',
                           'isota1010z0.2.dat',
                           'isota1015z0.1.dat',
                           'isota1015z0.2.dat']
        
        roi = ugali.observation.roi.ROI(self.config, lon, lat)

        mask_1 = ugali.observation.mask.MaskBand(self.config.params['mask']['infile_1'], roi)
        mask_2 = ugali.observation.mask.MaskBand(self.config.params['mask']['infile_2'], roi)
        mask = ugali.observation.mask.Mask(self.config, mask_1, mask_2)

        cut = mask.restrictCatalogToObservableSpace(self.catalog)
        catalog = self.catalog.applyCut(cut)

        isochrones = []
        for ii, name in enumerate(isochrone_names):
            isochrones.append(ugali.analysis.isochrone.Isochrone(self.config, '%s/%s'%(isochronedir, name)))

        isochrone = ugali.analysis.isochrone.CompositeIsochrone(isochrones, [0.25, 0.25, 0.25, 0.25])

        kernel = ugali.analysis.kernel.Plummer(lon, lat, 0.1)

        likelihood = ugali.analysis.likelihood.Likelihood(self.config, roi, mask, self.catalog, isochrone, kernel)
        
        likelihood.precomputeGridSearch(self.config.params['likelihood']['distance_modulus_array'])

        likelihood.gridSearch()

        likelihood.write(outfile)

        #return likelihood
    
############################################################

def main():
    """
    Placeholder for when users want to call this script as an executable
    """
    from optparse import OptionParser
    
if __name__ == "__main__":
    main()
