#!/usr/bin/env python

"""
Class to farm out analysis tasks.

@author: Keith Bechtol      <bechtol@kicp.uchicago.edu>
@author: Alex Drlica-Wagner <kadrlica@fnal.gov>
"""

import os
from os.path import join, exists
import sys
import subprocess
import time
import glob

import numpy
import numpy as np
import healpy

import ugali.utils.config
import ugali.utils.skymap
import ugali.utils.batch

from ugali.utils.projector import gal2cel,cel2gal
from ugali.utils.healpix import subpixel,superpixel,query_disc
from ugali.utils.healpix import pix2ang,ang2vec
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

class Farm:

    def __init__(self, configfile, verbose=False):
        self.configfile = configfile
        self.config = ugali.utils.config.Config(configfile)
        self._setup()
        self.verbose = verbose

    def _setup(self):
        self.nside_catalog    = self.config['coords']['nside_catalog']
        self.nside_likelihood = self.config['coords']['nside_likelihood']
        self.nside_pixel      = self.config['coords']['nside_pixel']

        self.filenames = self.config.getFilenames()
        self.skip = "Outfile already exists. Skipping..."

        # Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])
        self.catalog_pixels = self.filenames['pix'].compressed()

    def command(self, outfile, configfile, pix):
        """
        Generate the command for running the likelihood scan.
        """
        params = dict(script=self.config['scan']['script'],
                      config=configfile, outfile=outfile, 
                      nside=self.nside_likelihood, pix=pix,
                      verbose='-v' if self.verbose else '')
        cmd = '%(script)s %(config)s %(outfile)s --hpx %(nside)i %(pix)i %(verbose)s'%params
        return cmd

    # ADW: Should probably be in a utility
    def footprint(self, nside=None):
        """
        UNTESTED.
        Should return a boolean array representing the pixels in the footprint.
        """
        if nside is None:
            nside = self.nside_pixel
        elif nside < self.nside_catalog: 
            raise Exception('Requested nside=%i is less than catalog_nside'%nside)
        elif nside > self.nside_pixel:
            raise Exception('Requested nside=%i is greater than pixel_nside'%nside)
        pix = numpy.arange(healpy.nside2npix(nside), dtype=int)
        map = self.inFootprint(pix,nside)
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
            #logger.debug("Loading %s"%filenames['mask_1'])
            subpix_1,val_1 = ugali.utils.skymap.readSparseHealpixMap(filenames['mask_1'],'MAGLIM',construct_map=False)
            #logger.debug("Loading %s"%filenames['mask_2'])
            subpix_2,val_2 = ugali.utils.skymap.readSparseHealpixMap(filenames['mask_2'],'MAGLIM',construct_map=False)
            subpix = numpy.intersect1d(subpix_1,subpix_2)
            superpix = numpy.unique(ugali.utils.skymap.superpixel(subpix,self.nside_pixel,nside))
            inside |= numpy.in1d(pixels, superpix)
            
        return inside
        
    def submit_all(self, coords=None, queue=None, debug=False):
        """
        Submit likelihood analyses on a set of coordinates. If
        coords is `None`, submit all coordinates in the footprint.

        Inputs:
        coords : Array of target locations in Galactic coordinates. 
        queue  : Overwrite submit queue.
        debug  : Don't run.
        """
        if coords is None:
            pixels = numpy.arange(healpy.nside2npix(self.nside_likelihood))
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
            pixels = numpy.zeros(0, dtype=int)
            for v,r in zip(vec,radius):
                pix = query_disc(self.nside_likelihood,v,r,inclusive=True,fact=32)
                pixels = numpy.hstack([pixels, pix])
            #pixels = numpy.unique(pixels)

        inside = ugali.utils.skymap.inFootprint(self.config,pixels)
        if inside.sum() != len(pixels):
            logger.warning("Ignoring pixels outside survey footprint:\n"+str(pixels[~inside]))
        if inside.sum() == 0:
            logger.warning("No pixels inside footprint.")
            return

        # Only write the configfile once
        outdir = mkdir(self.config['output']['likedir'])
        configfile = '%s/config_queue.py'%(outdir)
        self.config.write(configfile)

        pixels = pixels[inside]
        self.submit(pixels,queue=queue,debug=debug,configfile=configfile)

    def submit(self, pixels, queue=None, debug=False, configfile=None):
        """
        Submit the likelihood job for the given pixel(s).
        """
        queue = self.config['batch']['cluster'] if queue is None else queue
        local = (queue == 'local')

        # Need to develop some way to take command line arguments...
        self.batch = ugali.utils.batch.batchFactory(queue,**self.config['batch']['opts'])

        if numpy.isscalar(pixels): pixels = numpy.array([pixels])

        outdir = mkdir(self.config['output']['likedir'])
        logdir = mkdir(join(outdir,'log'))
        subdir = mkdir(join(outdir,'sub'))

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
        chunk = self.config['batch']['chunk']
        istart = 0
        logger.info('=== Submit Likelihood ===')
        for ii,pix in enumerate(pixels):
            logger.info('  (%i/%i) pixel=%i nside=%i; (lon, lat) = (%.2f, %.2f)'%(ii+1,len(pixels),pix, self.nside_likelihood,lon[ii],lat[ii]))

            # Create outfile name
            outfile = self.config.likefile%(pix,self.config['coords']['coordsys'].lower())
            outbase = os.path.basename(outfile)
            jobname = self.config['batch']['jobname']

            # Submission command
            sub = not os.path.exists(outfile)
            cmd = self.command(outfile,configfile,pix)
            commands.append([ii,cmd,lon[ii],lat[ii],sub])
            
            if local or chunk == 0:
                # Not chunking
                command = cmd
                submit = sub
                logfile = join(logdir,os.path.splitext(outbase)[0]+'.log')
            elif (len(commands)%chunk==0) or (ii+1 == len(pixels)):
                # End of chunk, create submission script
                commands = np.array(commands,dtype=object)
                istart, iend = commands[0][0], commands[-1][0]
                subfile = join(subdir,'submit_%08i_%08i.sh'%(istart,iend))
                logfile = join(logdir,'submit_%08i_%08i.log'%(istart,iend))
                command = "sh %s"%subfile

                submit = np.any(commands[:,-1])
                if submit: self.write_script(subfile,commands)
            else:
                # Not end of chunk
                continue
            commands=[]

            # Actual job submission
            if not submit:
                logger.info(self.skip)
                continue
            else:
                while True:
                    njobs = self.batch.njobs()
                    if njobs < self.config['batch']['max_jobs']:
                        break
                    else:
                        logger.info('%i jobs already in queue, waiting...'%(njobs))
                        time.sleep(5*chunk)

                job = self.batch.submit(command,jobname,logfile)
                logger.info("  "+job)
                time.sleep(0.5)

    def write_script(self, filename, commands):
        info = 'echo "{0:=^60}";\n'
        hline = info.format("")
        newline = 'echo;\n'

        istart, iend = commands[0][0], commands[-1][0]
        script = open(filename,'w')
        script.write(hline)
        script.write(info.format('Submit Pixels %i to %i'%(istart,iend)))
        script.write(hline)
        script.write(newline)
        script.write('status=0;\n')
        for i,cmd,lon,lat,sub in commands: 
            script.write(info.format('Pixel %i: (%.2f, %.2f)'%(i,lon,lat)))
            if sub: script.write('%s; [ $? -ne 0 ] && status=1;\n'%cmd)
            else:   script.write('echo "%s";\n'%self.skip)
            script.write(hline)
            script.write(newline)
        script.write('exit $status;\n')
        script.close()

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for dispatching the likelihood scan to the queue."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_queue()
    parser.add_verbose()
    parser.add_coords(required=True,radius=True,targets=True)
    opts = parser.parse_args()

    farm = Farm(opts.config,verbose=opts.verbose)
    x = farm.submit_all(coords=opts.coords,queue=opts.queue,debug=opts.debug)
