#!/usr/bin/env python
"""Run the likelihood search."""
import glob
import os
from os.path import join, exists

from ugali.analysis.farm import Farm
from ugali.analysis.pipeline import Pipeline

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
import ugali.utils.skymap
from ugali.utils import healpix

components = ['scan','merge','tar','plot','check']
defaults = ['scan','merge']

def run(self):
    if 'scan' in self.opts.run:
        logger.info("Running 'scan'...")
        farm = Farm(self.config,verbose=self.opts.verbose)
        farm.submit_all(coords=self.opts.coords,queue=self.opts.queue,debug=self.opts.debug)

    if 'merge' in self.opts.run:
        import numpy as np

        logger.info("Running 'merge'...")
        mergefile = self.config.mergefile
        roifile = self.config.roifile
        filenames = self.config.likefile.split('_%')[0]+'_*.fits'
        infiles = np.array(sorted(glob.glob(filenames)))

        if 'mergedir' in self.config['output']:
            mkdir(self.config['output']['mergedir'])

        pixels = np.char.rpartition(np.char.rpartition(infiles,'_')[:,0],'_')[:,-1]
        pixels = pixels.astype(int)
        superpixel = healpix.superpixel(pixels,
                                        self.config['coords']['nside_likelihood'],
                                        self.config['coords']['nside_merge'])
                         
        for pix in np.unique(superpixel):
            outfile = mergefile%pix
            if exists(outfile) and not self.opts.force:
                logger.warning("  Found %s; skipping..."%outfile)
            else:
                healpix.merge_partial_maps(infiles[superpixel == pix],
                                           outfile,multiproc=8)

        if exists(roifile) and not self.opts.force:
            logger.warning("  Found %s; skipping..."%roifile)
        else:
            logger.info("  Merging likelihood headers...")
            healpix.merge_likelihood_headers(infiles,roifile)
            
    if 'tar' in self.opts.run:
        logger.info("Running 'tar'...")
        outdir = mkdir(self.config['output']['likedir'])
        logdir = mkdir(join(outdir,'log'))

        scanfile = self.config.likefile.split('_%')[0]+'_[0-9]*.fits'
        tarfile = join(self.config.likefile.split('_%')[0]+'_pixels.tar.gz')
        jobname = 'tar'
        logfile = os.path.join(logdir,'scan_tar.log')
        cmd = 'tar --remove-files -cvzf %s %s'%(tarfile,scanfile)
        if exists(tarfile) and not self.opts.force:
            logger.warning("  Found %s; skipping..."%tarfile)
        else:
            logger.info("  Tarring likelihood files...")
            logger.info(cmd)
            self.batch.submit(cmd,jobname,logfile)

    if 'plot' in self.opts.run:
        # WARNING: Loading the full 3D healpix map is memory intensive.
        logger.info("Running 'plot'...")
        import numpy as np
        # Should do this in environment variable
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        import ugali.utils.plotting as plotting
        from skymap import DESSkymap

        filenames = self.config.mergefile.split('_%')[0]+'_*.fits'
        infiles = np.array(sorted(glob.glob(filenames)))
        hpxmaps = ugali.utils.healpix.read_partial_map(infiles,'LOG_LIKELIHOOD',fullsky=True)[2]

        outdir = mkdir(self.config['output']['plotdir'])
        basename = os.path.basename(self.config.mergefile.split('_%')[0])
        for i,hpxmap in enumerate(hpxmaps):
            #plotting.plotSkymap(hpxmap)
            smap = DESSkymap()
            smap.draw_hpxmap(hpxmap, cmap='gray_r', vmax=3.5)
            smap.draw_inset_colorbar()
            outfile = os.path.join(outdir,basename+'_%02d.png'%i)
            print("Writing %s..."%outfile)
            plt.savefig(outfile)
        # Make the movie
        import subprocess
        cmd = "convert -delay 30 -loop 0 %s/*.png %s.gif"%(outdir,basename)
        print(cmd)
        subprocess.check_call(cmd,shell=True)

    if 'check' in self.opts.run:
        # Check the completion fraction
        logger.info("Running 'check'...")

        import fitsio
        import numpy as np
        import healpy as hp

        from ugali.utils.skymap import inFootprint

        # Load the ROI file
        roi = fitsio.read(self.config.roifile)
        done = roi['PIXEL']
        
        # Get all target pixels
        nside = self.config['coords']['nside_likelihood']
        pixels = np.arange(hp.nside2npix(nside))
        pixarea = hp.nside2pixarea(nside,degrees=True)
        foot = pixels[inFootprint(self.config,pixels)]

        # And find the pixels that haven't been processed
        undone = ~np.in1d(foot,done)
        hpxmap = np.zeros(len(pixels))
        hpxmap[foot[undone]] = True

        logger.info("Found %i incomplete pixels with an area of %.1f deg^2."%(hpxmap.sum(), hpxmap.sum()*pixarea))
        hp.write_map('check.fits.gz',hpxmap)
        
Pipeline.run = run
Pipeline.defaults = defaults
pipeline = Pipeline(__doc__,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
