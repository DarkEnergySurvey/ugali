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
import ugali.utils.healpix

components = ['scan','merge','tar','plot']
defaults = ['scan','merge']

def run(self):
    if 'scan' in self.opts.run:
        logger.info("Running 'scan'...")
        farm = Farm(self.config,verbose=self.opts.verbose)
        farm.submit_all(coords=self.opts.coords,queue=self.opts.queue,debug=self.opts.debug)

    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")
        mergefile = self.config.mergefile
        roifile = self.config.roifile
        filenames = self.config.likefile.split('_%')[0]+'_*.fits'
        infiles = sorted(glob.glob(filenames))

        if exists(mergefile) and not self.opts.force:
            logger.warn("  Found %s; skipping..."%mergefile)
        else:
            logger.info("  Merging likelihood files...")
            ugali.utils.healpix.merge_partial_maps(infiles,mergefile)

        if exists(roifile) and not self.opts.force:
            logger.warn("  Found %s; skipping..."%roifile)
        else:
            logger.info("  Merging likelihood headers...")
            ugali.utils.healpix.merge_likelihood_headers(infiles,roifile)

            #ugali.utils.skymap.mergeLikelihoodFiles(infiles,mergefile,roifile)
            
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
            logger.warn("  Found %s; skipping..."%tarfile)
        else:
            logger.info("  Tarring likelihood files...")
            logger.info(cmd)
            self.batch.submit(cmd,jobname,logfile)

    if 'plot' in self.opts.run:
        # WARNING: Loading the full 3D healpix map is memory intensive.
        logger.info("Running 'plot'...")
        # Should do this in environment variable
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        import ugali.utils.plotting as plotting
        skymap = ugali.utils.skymap.readSparseHealpixMap(self.config.mergefile,'LOG_LIKELIHOOD')[1]
        plotting.plotSkymap(skymap)
        outdir = mkdir(self.config['output']['plotdir'])
        basename = os.path.basename(self.config.mergefile.replace('.fits','.png'))
        outfile = os.path.join(outdir,basename)
        plt.savefig(outfile)

Pipeline.run = run
Pipeline.defaults = defaults
pipeline = Pipeline(__doc__,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
