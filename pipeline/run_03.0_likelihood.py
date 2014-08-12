#!/usr/bin/env python
import glob
import os
from os.path import join

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
        farm.submit_all(coords=None,local=False,debug=False)
    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")
        filenames = "%s/likelihood*.fits"%self.config.params['output']['savedir_likelihood']
        infiles = sorted(glob.glob(filenames))
        outdir = mkdir(self.config.params['output']['savedir_results'])
        mergefile = join(outdir,self.config.params['output']['mergefile'])
        roifile = join(outdir,self.config.params['output']['roifile'])
        if os.path.exists(mergefile) or os.path.exists(roifile):
            raise Exception("Output file already exists; skipping...")
        #ugali.utils.skymap.mergeSparseHealpixMaps(infiles,outfile)
        ugali.utils.skymap.mergeLikelihoodFiles(infiles,mergefile,roifile)
    if 'targets' in self.opts.run and len(args) > 1:
        logger.info("Running 'targets'...")
        farm = ugali.analysis.farm.Farm(self.config)
        names,coords = farm.loadTargetCoordinates(args[1])
        farm.submit_all(coords=coords,local=False,debug=False)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()


"""
if __name__ == "__main__":
    import argparse
    description = "Pipeline script for running the likelihood analysis"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config',help='Configuration file.')
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Output verbosity')
    parser.add_argument('-r','--run', default=[],
                        action='append',choices=COMPONENTS,
                        help="Choose analysis component to run")
    opts = parser.parse_args()

    config = Config(opts.config)
    if not opts.run: opts.run = COMPONENTS
    if opts.verbose: logger.setLevel(logger.DEBUG)

    if 'likelihood' in opts.run:
        logger.info("Running 'likelihood'...")
        farm = ugali.analysis.farm.Farm(config)
        farm.submit_all(coords=None,local=False,debug=False)
    if 'merge' in opts.run:
        logger.info("Running 'merge'...")
        filenames = "%s/likelihood*.fits"%config.params['output']['savedir_likelihood']
        infiles = sorted(glob.glob(filenames))
        outdir = mkdir(config.params['output']['savedir_results'])
        mergefile = join(outdir,config.params['output']['mergefile'])
        roifile = join(outdir,config.params['output']['roifile'])
        if os.path.exists(mergefile) or os.path.exists(roifile):
            raise Exception("Output file already exists; skipping...")
        #ugali.utils.skymap.mergeSparseHealpixMaps(infiles,outfile)
        ugali.utils.skymap.mergeLikelihoodFiles(infiles,mergefile,roifile)
    if 'targets' in opts.run and len(args) > 1:
        logger.info("Running 'targets'...")
        farm = ugali.analysis.farm.Farm(config)
        names,coords = farm.loadTargetCoordinates(args[1])
        farm.submit_all(coords=coords,local=False,debug=False)
"""
