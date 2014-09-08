#!/usr/bin/env python
import glob
import os
from os.path import join, exists

import ugali.analysis.farm
from ugali.analysis.pipeline import Pipeline

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

description="Run the likelihood search."
components = ['scan','merge']

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
            outdir = mkdir(self.config['output']['searchdir'])

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
