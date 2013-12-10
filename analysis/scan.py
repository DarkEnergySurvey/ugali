#!/usr/bin/env python

"""
Class to create and run likelihood analysis.

Classes
    Class

Functions
    someFunction
"""

import os
import sys
import numpy
import subprocess
import time
import getpass

import pyfits
import healpy

from os.path import join

import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.likelihood
import ugali.observation.catalog
import ugali.observation.mask
import ugali.simulation.simulator
import ugali.utils.parse_config
import ugali.utils.skymap

from ugali.utils.logger import logger
from ugali.utils.skymap import superpixel, subpixel


############################################################

class Scan:
    """
    The base of a likelihood analysis scan.
    """
    def __init__(self, config, pix):
        self.config = ugali.utils.parse_config.Config(config)
        self.pix = pix
        self._setup()

    def _setup(self):
        self.nside_catalog    = self.config.params['coords']['nside_catalog']
        self.nside_likelihood = self.config.params['coords']['nside_likelihood']
        self.nside_pixel      = self.config.params['coords']['nside_pixel']

        # All possible filenames
        self.filenames = self.config.getFilenames()
        # ADW: Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])

        theta, phi =  healpy.pix2ang(self.nside_likelihood, self.pix)
        self.lon, self.lat = numpy.degrees(phi), 90. - numpy.degrees(theta)

        self.roi = ugali.observation.roi.ROI(self.config, self.lon, self.lat)
        # All possible catalog pixels spanned by the ROI
        catalog_pixels = numpy.unique(superpixel(self.roi.pixels,self.nside_pixel,self.nside_catalog))
        # Only catalog pixels that exist in catalog files
        self.catalog_pixels = numpy.intersect1d(catalog_pixels, self.filenames['pix'].compressed())

        self.kernel = self.createKernel()
        self.isochrone = self.createIsochrone()
        self.catalog = self.createCatalog()
        self.mask = self.createMask()

        self.likelihood = ugali.analysis.likelihood.Likelihood(self.config, self.roi, 
                                                               self.mask, self.catalog, 
                                                               self.isochrone, self.kernel)

    def createKernel(self):
        name = self.config.params['kernel']['type'].lower()
        params = self.config.params['kernel']['params']
        kernel = ugali.analysis.kernel.kernelFactory(name,self.lon,self.lat,*params)
        return kernel

    def createIsochrone(self):
        isochrones = []
        for ii, name in enumerate(self.config.params['isochrone']['infiles']):
            isochrones.append(ugali.analysis.isochrone.Isochrone(self.config, name))
        isochrone = ugali.analysis.isochrone.CompositeIsochrone(isochrones, self.config.params['isochrone']['weights'])
        return isochrone

    def createCatalog(self):
        """
        Find the relevant catalog files for this scan.
        """
        catalog = ugali.observation.catalog.Catalog(self.config,roi=self.roi)  
        return catalog

    def simulateCatalog(self):
        """
        !!! PLACEHOLDER: Integrate the simulation structure more tightly with
        the analysis structure to avoid any nasty disconnects. !!!
        """
        pass

    def createMask(self):
        mask = ugali.observation.mask.Mask(self.config, self.roi)
        return mask

    def run(self, coords=None, debug=False):
        """
        Run the likelihood analysis
        """
        self.likelihood.precomputeGridSearch(self.config.params['likelihood']['distance_modulus_array'])
        self.likelihood.gridSearch(coords=coords)
        return self.likelihood
        
    def write(self, outfile):
        self.likelihood.write(outfile)
    
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] config pix outfile"
    description = "Script housing the setup and execution of the likelihood scan."
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-d','--debug',action='store_true',
                      help="Setup, but don't run")
    parser.add_option('-v','--verbose',action='store_true')
    (opts, args) = parser.parse_args()
    if opts.verbose: logger.setLevel(logger.DEBUG)
    else:            logger.setLevel(logger.INFO)
    config = args[0]
    pix = int(args[1])
    outfile = args[2]
    scan = Scan(config,pix)
    if not opts.debug:
        scan.run()
        scan.write(outfile)
