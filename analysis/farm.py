#!/usr/bin/env python

"""
Class to farm out analysis tasks.

Classes
    Mask

Functions
    someFunction
"""

import os
import sys
import numpy
import healpy
import subprocess
import time
import getpass
import glob

from os.path import join, exists

import numpy.ma.mrecords

import ugali.analysis.isochrone
import ugali.analysis.kernel
import ugali.analysis.likelihood
import ugali.analysis.scan
import ugali.observation.catalog
import ugali.observation.mask
import ugali.utils.config
import ugali.utils.skymap
import ugali.utils.batch

from ugali.utils.projector import gal2cel,cel2gal
from ugali.utils.healpix import subpixel,superpixel,query_disc
from ugali.utils.healpix import pix2ang,ang2pix,ang2vec

from ugali.utils.logger import logger

import ugali.utils.shell
from ugali.utils.shell import mkdir

class Farm:
    def __init__(self, configfile):
        self.configfile = configfile
        self.config = ugali.utils.config.Config(configfile)
        self._setup()

    def _setup(self):
        self.nside_catalog    = self.config.params['coords']['nside_catalog']
        self.nside_likelihood = self.config.params['coords']['nside_likelihood']
        self.nside_pixel      = self.config.params['coords']['nside_pixel']

        self.filenames = self.config.getFilenames()
        # Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])
        self.catalog_pixels = self.filenames['pix'].compressed()

    # ADW: This should be moved to a "target" or utility module
    @staticmethod
    def loadTargetCoordinates(filename):
        """
        Load a text file with target coordinates. Returns
        an array of target locations in Galactic coordinates.
        File description:
        [NAME] [LON] [LAT] [RADIUS] [COORD]
        
        The values of LON and LAT will depend on COORD:
        COORD = [GAL  | CEL | HPX  ],
        LON   = [GLON | RA  | NSIDE]
        LAT   = [GLAT | DEC | PIX  ]

        """
        data = numpy.loadtxt(filename,unpack=True,usecols=range(5),dtype=object)
        # Deal with one-line input files
        if data.ndim == 1: data = numpy.array([data]).T
        names = data[0]
        out   = data[1:4].astype(float)
        lon,lat,radius = out

        coord = numpy.array([s.lower() for s in data[4]])
        gal = (coord=='gal')
        cel = (coord=='cel')
        hpx = (coord=='hpx')

        if cel.any():
            glon,glat = cel2gal(lon[cel],lat[cel])
            out[0][cel] = glon
            out[1][cel] = glat
        if hpx.any():
            glon,glat = pix2ang(lat[hpx],lon[hpx])
            out[0][hpx] = glon
            out[1][hpx] = glat

        return names,out.T
               
    def command(self, outfile, configfile, pix):
        """
        Placeholder for a function to generate the command for running
        the likelihood scan.
        """
        cmd = '%s %s %s --hpx %i %i'%(self.config['queue']['script'], configfile, outfile, self.nside_likelihood, pix)
        return cmd

    def queue(self):
        """
        Placeholder for a function that will interface with the various
        batch systems (i.e., lsf, condor, etc.)
        """
        pass

    # ADW: Should probably be in a utility
    def footprint(self, nside=None):
        """
        UNTESTED.
        Should return a boolean array representing the pixels in the footprint.
        """
        if nside is None:
            nside = self.nside_pixel
        elif nside > self.nside_catalog: 
            raise Exception('Requested nside=%i is greater than catalog_nside'%nside)
        elif nside < self.nside_pixel:
            raise Exception('Requested nside=%i is less than pixel_nside'%nside)
        pix = numpy.arange( healpy.nside2npix(nside), dtype=int )
        map = self.inFootprint(pix)
        return map 

    # ADW: Should probably be in a utility
    def inFootprint(self, pixels, nside=None):
        """
        Open each valid filename for the set of pixels and determine the set 
        of subpixels with valid data.
        """
        if numpy.isscalar(pixels): pixels = numpy.array([pixels])
        if nside is None: nside = self.nside_likelihood

        inside = numpy.zeros( len(pixels), dtype='bool')
        if not self.nside_catalog:
            catalog_pix = [0]
        else:
            catalog_pix = superpixel(pixels,nside,self.nside_catalog)
            catalog_pix = numpy.intersect1d(catalog_pix,self.catalog_pixels)

        for filenames in self.filenames[catalog_pix]:
            logger.debug("Loading %s"%filenames['mask_1'])
            subpix_1,val_1 = ugali.utils.skymap.readSparseHealpixMap(filenames['mask_1'],'MAGLIM',construct_map=False)
            logger.debug("Loading %s"%filenames['mask_2'])
            subpix_2,val_2 = ugali.utils.skymap.readSparseHealpixMap(filenames['mask_2'],'MAGLIM',construct_map=False)
            subpix = numpy.intersect1d(subpix_1,subpix_2)
            superpix = numpy.unique(ugali.utils.skymap.superpixel(subpix,self.nside_pixel,nside))
            inside |= numpy.in1d(pixels, superpix)
            
        return inside
        
    def submit_all(self, coords=None, local=False, debug=False):
        """
        Submit likelihood analyses on a set of coordinates. If
        coords == None, submit all coordinates in the footprint.

        Inputs:
        coords : Array of target locations in Galactic coordinates. 
                 [Can optionally contain search radii about the specified coordinates.]
                 [ (GLON, GLAT, [RADIUS]) ]
        local  : Run locally
        debug  : Don't run.
        """
        if coords is None:
            pixels = numpy.arange( healpy.nside2npix(self.nside_likelihood) )
        else:
            coords = numpy.asarray(coords)
            if coords.ndim == 1:
                coords = numpy.array([coords])
            if coords.shape[1] == 2:
                glon,glat = coords.T
                radius    = numpy.zeros(len(glon))
            elif coords.shape[1] == 3:
                glon,glat,radius = coords.T
            else:
                raise Exception("Unrecognized coords shape:"+str(coords.shape))
            vec = ang2vec(glon,glat)
            pixels = numpy.zeros( 0, dtype=int)
            for v,r in zip(vec,radius):
                pix = query_disc(self.nside_likelihood,v,r,inclusive=True,fact=32)
                pixels = numpy.hstack([pixels, pix])
            #pixels = numpy.unique(pixels)

        inside = self.inFootprint(pixels)
        if inside.sum() != len(pixels):
            logger.warning("Ignoring pixels outside survey footprint:\n"+str(pixels[~inside]))
        if inside.sum() == 0:
            logger.warning("No pixels inside footprint.")
            return

        # Only write the configfile once
        outdir = mkdir(self.config.params['output']['savedir_likelihood'])
        configfile = '%s/config_queue.py'%(outdir)
        self.config.write(configfile)

        pixels = pixels[inside]
        self.submit(pixels,local=local,debug=debug,configfile=configfile)

    def submit(self, pixels, local=True, debug=False, configfile=None):
        """
        Submit the likelihood job for the given pixel(s).
        """
        if numpy.isscalar(pixels): pixels = numpy.array([pixels])

        outdir = mkdir(self.config['output']['savedir_likelihood'])
        logdir = mkdir(join(outdir,'log'))
        subdir = mkdir(join(outdir,'sub'))

        # Need to develop some way to take command line arguments...
        if local or self.config['queue']['cluster']=='local':
            batch = ugali.utils.batch.Local()
        elif self.config['queue']['cluster']=='midway':
            batch = ugali.utils.batch.Midway()
        elif self.config['queue']['cluster']=='slac':
            # Need to add an option for which slac queue [short/long/kipac-ibq]
            batch = ugali.utils.batch.LSF()
        elif self.config['queue']['cluster']=='fnal':
            # Need to learn how to use condor first...
            batch = ugali.utils.batch.Condor()
            raise Exception("FNAL cluster not implemented")

        # Save the current configuation settings; avoid writing 
        # file multiple times if configfile passed as argument.
        if configfile is None:
            if local:
                configfile = self.configfile
            else:
                configfile = '%s/config_queue.py'%(outdir)
                self.config.write(configfile)
                
        lon,lat = pix2ang(self.nside_likelihood,pixels)
        commands = []
        chunk = self.config['queue']['chunk']
        istart = 0
        logger.info('=== Submit Likelihood ===')
        for ii,pix in enumerate(pixels):
            logger.info('  (%i/%i) pixel=%i nside=%i; (lon, lat) = (%.3f, %.3f)'%(ii+1, len(pixels), pix, self.nside_likelihood, lon[ii], lat[ii] ))

            # Create outfile name
            outbase = 'likelihood_%08i_%s.fits'%(pix,self.config['coords']['coordsys'].lower())
            outfile = join(outdir, outbase)
            jobname = self.config['queue']['jobname']

            # Check if outfile exists
            if os.path.exists(outfile) and not local:
                logger.info('  %s already exists. Skipping ...'%(outfile))
                continue

            # Submission command
            # This should be able to submit multiple pixels...
            cmd = self.command(outfile,configfile,pix)

            commands.append([ii,cmd])
            
            if local or chunk == 0:
                # Not chunking
                command = cmd
                logfile = join(logdir,os.path.splitext(outbase)[0]+'.log')
            elif (len(commands)%chunk==0) or (len(pixels)==ii+1):
                # End of chunk, create submit script
                istart, iend = commands[0][0], commands[-1][0]
                info = 'echo -e "\n### %s ###";\n'
                subfile = join(subdir,'submit_%08i_%08i.sh'%(istart,iend))
                logfile = join(logdir,'submit_%08i_%08i.log'%(istart,iend))
                script = open(subfile,'w')
                script.write(info%('Pixels %i to %i'%(istart,ii)))
                for i,c in commands: 
                    script.write(info%('Pixel %i (%.3f, %.3f)'%(i,lon[i],lat[i])))
                    script.write('%s;\n'%c)
                script.close()
                commands = []; istart = ii+1
                command = "sh %s"%subfile
            else:
                # Not end of chunk
                continue

            while True:
                njobs = batch.njobs()
                if njobs < self.config.params['queue']['max_jobs']:
                    break
                else:
                    logger.info('%i jobs already in queue, waiting ...'%(njobs))
                    time.sleep(15)

            opts = self.config['queue'].get('opts',{})
            job = batch.submit(command,jobname,logfile,**opts)
            logger.info("  "+job)
            time.sleep(0.5)


if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for dispatching the likelihood scan to the queue."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_queue()
    parser.add_verbose()
    parser.add_coords(required=True)
    parser.add_argument('--radius',default=0,type=float,
                      help="Radius surrounding specified coordinates")
    parser.add_argument('-t','--targets',default=None,
                      help="List of target coordinates")
    opts = parser.parse_args()

    if opts.targets:
        names,coords = farm.loadTargetCoordinates(opts.targets)
    else:
        coords = [ (opts.coords[0],opts.coords[1],opts.radius) ]

    local = (opts.queue=='local')

    farm = Farm(opts.config)
    x = farm.submit_all(coords=coords,local=local,debug=opts.debug)
