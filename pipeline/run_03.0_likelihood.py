#!/usr/bin/env python
import glob
import os
from os.path import join, exists

import ugali.analysis.farm
from ugali.analysis.pipeline import Pipeline

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

description="Run the likelihood search."
components = ['likelihood','merge','targets']

def run(self):
    if 'likelihood' in self.opts.run:
        logger.info("Running 'likelihood'...")
        farm = ugali.analysis.farm.Farm(self.config)
        farm.submit_all(coords=self.opts.coords,queue=self.opts.queue,debug=self.opts.debug)
    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")
        filenames = "%s/likelihood*.fits"%self.config['output']['savedir_likelihood']
        infiles = sorted(glob.glob(filenames))
        outdir = mkdir(self.config['output']['savedir_results'])
        mergefile = join(outdir,self.config['output']['mergefile'])
        roifile = join(outdir,self.config['output']['roifile'])
        if (exists(mergefile) or exists(roifile)) and not self.opts.force:
            logger.info("  Found %s; skipping..."%mergefile)
            logger.info("  Found %s; skipping..."%roifile)
        else:
            ugali.utils.skymap.mergeLikelihoodFiles(infiles,mergefile,roifile)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
