#!/usr/bin/env python
import glob
import os
from os.path import join, exists

import ugali.analysis.farm
from ugali.analysis.pipeline import Pipeline

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

description="Run the likelihood search."
components = ['scan','merge','plot']

def run(self):
    if 'scan' in self.opts.run:
        logger.info("Running 'scan'...")
        farm = ugali.analysis.farm.Farm(self.config)
        farm.submit_all(coords=self.opts.coords,queue=self.opts.queue,debug=self.opts.debug)
    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")
        mergefile = self.config.mergefile
        roifile = self.config.roifile
        if (exists(mergefile) or exists(roifile)) and not self.opts.force:
            logger.info("  Found %s; skipping..."%mergefile)
            logger.info("  Found %s; skipping..."%roifile)
        else:
            filenames = self.config.likefile.split('_%')[0]+'_*'
            infiles = sorted(glob.glob(filenames))
            ugali.utils.skymap.mergeLikelihoodFiles(infiles,mergefile,roifile)
    if 'plot' in self.opts.run:
        # WARNING: Loading the full 3D healpix map is memory intensive.
        logger.info("Running 'plot'...")
        import pylab as plt
        import ugali.utils.plotting
        skymap = ugali.utils.skymap.readSparseHealpixMap(self.config.mergefile,'LOG_LIKELIHOOD')[1]
        ugali.utils.plotting.plotSkymap(skymap)
        outdir = mkdir(self.config['output']['plotdir'])
        basename = os.path.basename(self.config.mergefile.replace('.fits','.png'))
        outfile = os.path.join(outdir,basename)
        plt.savefig(outfile)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
