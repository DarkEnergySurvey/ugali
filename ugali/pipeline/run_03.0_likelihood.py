#!/usr/bin/env python
import glob
import os
from os.path import join, exists

from ugali.analysis.farm import Farm
from ugali.analysis.pipeline import Pipeline

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
import ugali.utils.skymap
import ugali.utils.healpix

description="Run the likelihood search."
components = ['scan','merge','tar','plot']

def run(self):
    if 'scan' in self.opts.run:
        logger.info("Running 'scan'...")
        farm = Farm(self.config,verbose=self.opts.verbose)
        farm.submit_all(coords=self.opts.coords,queue=self.opts.queue,debug=self.opts.debug)

    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")
        mergefile = self.config.mergefile
        roifile = self.config.roifile
        filenames = self.config.likefile.split('_%')[0]+'_*'
        infiles = sorted(glob.glob(filenames))

        if exists(mergefile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%mergefile)
        else:
            ugali.utils.healpix.merge_partial_maps(infiles,mergefile)

        if exists(roifile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%roifile)
        else:
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
        print(cmd)
        self.batch.submit(cmd,jobname,logfile)

    if 'plot' in self.opts.run:
        # WARNING: Loading the full 3D healpix map is memory intensive.
        logger.info("Running 'plot'...")
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
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
