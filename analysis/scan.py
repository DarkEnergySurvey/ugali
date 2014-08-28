#!/usr/bin/env python

"""
Class to create and run an individual likelihood analysis.

Classes:
    Scan

Functions:
    someFunction
"""

import os
import sys

import numpy
import numpy as np
import pyfits

import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.grid
import ugali.observation.catalog
import ugali.observation.mask
import ugali.simulation.simulator
import ugali.utils.config

from ugali.utils.logger import logger
from ugali.utils.healpix import superpixel, subpixel, pix2ang

############################################################

class Scan(object):
    """
    The base of a likelihood analysis scan.
    """
    def __init__(self, config, coords):
        self.config = ugali.utils.config.Config(config)
        self.lon,self.lat = coords
        self._setup()

    def _setup(self):
        self.nside_catalog    = self.config.params['coords']['nside_catalog']
        self.nside_likelihood = self.config.params['coords']['nside_likelihood']
        self.nside_pixel      = self.config.params['coords']['nside_pixel']

        # All possible filenames
        self.filenames = self.config.getFilenames()
        # ADW: Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])

        self.roi = ugali.observation.roi.ROI(self.config, self.lon, self.lat)
        # All possible catalog pixels spanned by the ROI
        catalog_pixels = numpy.unique(superpixel(self.roi.pixels,self.nside_pixel,self.nside_catalog))
        # Only catalog pixels that exist in catalog files
        self.catalog_pixels = numpy.intersect1d(catalog_pixels, self.filenames['pix'].compressed())

        self.kernel = self.createKernel()
        self.isochrone = self.createIsochrone()
        self.catalog = self.createCatalog()
        self.mask = self.createMask()

        self.grid = ugali.analysis.grid.GridSearch(self.config, self.roi, 
                                                   self.mask,self.catalog, 
                                                   self.isochrone, self.kernel)
        self.loglike = self.grid.loglike

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
        Run the likelihood grid search
        """
        self.grid.precompute()
        self.grid.search()
        return self.grid
        
    def write(self, outfile):
        self.grid.write(outfile)
    
if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for executing the likelihood scan."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_argument('outfile',metavar='outfile.fits',help='Output fits file.')
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True)
    opts = parser.parse_args()


    scan = Scan(opts.config,opts.coords)
    if not opts.debug:
        result = scan.run()
        scan.write(opts.outfile)
