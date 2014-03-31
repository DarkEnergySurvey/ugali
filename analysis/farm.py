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
import ugali.utils.parse_config
import ugali.utils.skymap

from ugali.utils.projector import galToCel,celToGal,pixToAng,angToPix,angToVec
from ugali.utils.skymap import subpixel, superpixel


from ugali.utils.logger import logger

import ugali.utils.shell


class Farm:
    def __init__(self, configfile):
        self.configfile = configfile
        self.configfile_queue = configfile
        self.config = ugali.utils.parse_config.Config(self.configfile)
        self._setup()

    def _setup(self):
        self.nside_catalog    = self.config.params['coords']['nside_catalog']
        self.nside_likelihood = self.config.params['coords']['nside_likelihood']
        self.nside_pixel      = self.config.params['coords']['nside_pixel']

        self.filenames = self.config.getFilenames()
        # Might consider storing only the good filenames
        # self.filenames = self.filenames.compress(~self.filenames.mask['pix'])
        self.catalog_pixels = self.filenames['pix'].compressed()

    # ADW: This should be moved to a "target" module
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
            glon,glat = celToGal(lon[cel],lat[cel])
            out[0][cel] = glon
            out[1][cel] = glat
        if hpx.any():
            glon,glat = pixToAng(lat[hpx],lon[hpx])
            out[0][hpx] = glon
            out[1][hpx] = glat

        return names,out.T
               
    def command(self):
        """
        Placeholder for a function to generate the command for running
        the likelihood scan.
        """
        pass

    def queue(self):
        """
        Placeholder for a function that will interface with the various
        batch systems (i.e., lsf, condor, etc.)
        """
        pass

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
            vec = angToVec(glon,glat)
            pixels = numpy.zeros( 0, dtype=int)
            for v,r in zip(vec,radius):
                pix = ugali.utils.projector.query_disc(self.nside_likelihood,v,r,inclusive=True,fact=32)
                pixels = numpy.hstack([pixels, pix])
            #pixels = numpy.unique(pixels)

        inside = self.inFootprint(pixels)
        if inside.sum() != len(pixels):
            logger.warning("Ignoring pixels outside survey footprint:\n"+str(pixels[~inside]))
        if inside.sum() == 0:
            logger.warning("No pixels inside footprint.")
            return

        # Only write the configfile once
        outdir = ugali.utils.shell.mkdir(self.config.params['output']['savedir_likelihood'])
        configfile = '%s/config_queue.py'%(outdir)
        self.config.writeConfig(configfile)

        pixels = pixels[inside]
        lon,lat = pixToAng(self.nside_likelihood,pixels)
        for ii,pix in enumerate(pixels):
            logger.info('=== Submit Likelihood ===')
            logger.info('  (%i/%i) pixel=%i nside=%i; (glon, glat) = (%.3f, %.3f)'%(ii+1, len(pixels), pix, self.nside_likelihood, lon[ii], lat[ii] ))

            self.submit(pix,local=local,debug=debug,configfile=configfile)

    def submit(self, pixels, local=True, debug=False, configfile=None):
        """
        Submit the likelihood job for the given pixel(s).
        """
        if numpy.isscalar(pixels): pixels = numpy.array([pixels])

        outdir = ugali.utils.shell.mkdir(self.config.params['output']['savedir_likelihood'])

        # Save the current configuation settings; avoid writing 
        # file multiple times if configfile passed as argument.
        if configfile is None:
            if local:
                configfile = self.configfile
            else:
                configfile = '%s/config_queue.py'%(outdir)
                self.config.writeConfig(configfile)
                
        lon,lat = pixToAng(self.nside_likelihood,pixels)
        n_query_points = healpy.nside2npix(self.nside_pixel)/healpy.nside2npix(self.nside_likelihood)

        for ii,pix in enumerate(pixels):
            #logger.info('  (%i/%i) pixel %i nside %i; %i query points; %s (lon, lat) = (%.3f, %.3f)'%(ii+1, len(pixels), pix, self.nside_likelihood, n_query_points, self.config.params['coords']['coordsys'],lon[ii], lat[ii]))

            # Create outfile name
            outfile = '%s/likelihood_%010i_nside_pix_%i_nside_subpix_%i_%s.fits'%(outdir,pix,self.nside_likelihood,self.nside_pixel,self.config.params['coords']['coordsys'].lower())

            # Submission command
            # This should be able to submit multiple pixels...
            command = '%s %s %i %s'%(self.config.params['queue']['script'], configfile, pix, outfile)
            # Check if outfile exists
            if os.path.exists(outfile) and not local:
                logger.info('  %s already exists. Skipping ...'%(outfile))
                continue
                
            if local or self.config.params['queue']['cluster'] == 'local':
                #scan = ugali.analysis.scan.Scan(self.configfile,pix[ii])
                #likelihood = scan.run(outfile,debug=debug)
                #return likelihood
                logger.info(command)
                subprocess.call(command,shell=True)

            else:
                # Submit to queue
                logfile = '%s/%s_%i.log'%(self.config.params['output']['logdir_likelihood'], self.config.params['queue']['jobname'], pix)
                logdir = ugali.utils.shell.mkdir(os.path.dirname(logfile))
                username = getpass.getuser()
 
                # Midway cluster
                if self.config.params['queue']['cluster'] == 'midway':
                    #batch = 'sbatch --account=kicp --partition=kicp-ht --output=%s --job-name=%s --mem=10000 '%(logfile, self.config.params['queue']['jobname'])
                    batch = """sbatch --account=kicp --partition=kicp-ht --output=%(logfile)s --job-name=%(jobname)s --mem=10000 """
                    check_jobs = 'squeue -u %s | wc\n'%username
                # SLAC cluster
                elif self.config.params['queue']['cluster'] == 'slac':
                    # Need to add an option for which slac queue [short/long/kipac-ibq]
                    batch = """bsub -q %(queue)s -R \"%(require)s\" -oo %(logfile)s -J %(jobname)s -C 0 """
                    check_jobs = 'bjobs -u %s | wc\n'%username
                # FNAL cluster
                elif self.config.params['queue']['cluster'] == 'fnal':
                    # Need to learn how to use condor first...
                    raise Exception("FNAL cluster not implemented")
 
                batch = batch % dict(self.config.params['queue'],logfile=logfile)
                sleep = "; sleep 0.5"
                command_queue = batch + command + sleep
                logger.info(command_queue)
                    
                if not os.path.exists(self.config.params['output']['logdir_likelihood']):
                    logdir = os.mkdir(self.config.params['output']['logdir_likelihood'])
 
                while True:
                    n_submitted = int(subprocess.Popen(check_jobs, shell=True, 
                                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate()[0].split()[0]) - 1
                    if n_submitted < self.config.params['queue']['max_jobs']:
                        break
                    else:
                        logger.info('%i jobs already in queue, waiting ...'%(n_submitted))
                        time.sleep(15)
 
                subprocess.call(command_queue,shell=True)
                #break

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] config"
    description = "Script housing the setup and execution of the likelihood scan."
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-l','--glon',default=None,type='float',
                      help='Galactic longitude of target')
    parser.add_option('-b','--glat',default=None,type='float',
                      help='Galactic latitude of target')
    parser.add_option('-r','--ra',default=None,type='float',
                      help="RA of target")
    parser.add_option('-d','--dec',default=None,type='float',
                      help="DEC of target")
    parser.add_option('-p','--pix',default=None,type='int',
                      help="HEALPix pixel of target (Galactic coordinates)")
    parser.add_option('-n','--nside',default=None,type='int',
                      help="HEALPix nside of target pixel")
    parser.add_option('--radius',default=0,type='float',
                      help="Radius surrounding specified coordinates")
    parser.add_option('-t','--targets',default=None,type='str',
                      help="List of target coordinates")
    parser.add_option('--local',action='store_true',
                      help="Run locally")
    parser.add_option('--debug',action='store_true',
                      help="Setup, but don't run")
    (opts, args) = parser.parse_args()

    if opts.glon is not None and (opts.ra is not None or opts.pix is not None):
        logger.error("Only one coordinate type allowed")
        parser.print_help()
        sys.exit(1)

    config = args[0]
    farm = Farm(config)

    coords = None
    if opts.glon is not None and opts.glat is not None:
        glon = opts.glon; glat = opts.glat
        coords = [ (glon,glat,radius) ]
    elif opts.ra is not None and opts.dec is not None:
        glon,glat = celToGal(opts.ra,opts.dec)
        coords = [ (glon,glat,radius) ]
    elif opts.pix is not None and opts.nside is not None:
        glon,glat = pixToAng(opts.nside,opts.pix)
        coords = [ (glon,glat,radius) ]
    elif opts.targets:
        names,coords = farm.loadTargetCoordinates(opts.targets)

    x = farm.submit_all(coords=coords,local=opts.local,debug=opts.debug)
