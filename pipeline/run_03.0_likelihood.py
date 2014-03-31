#!/usr/bin/env python
import glob
import os
from os.path import join

import ugali.analysis.farm
from ugali.utils.parse_config import Config
from ugali.utils.shell import mkdir
from ugali.utils.logger import logger

COMPONENTS = ['likelihood','merge','targets']
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] config.py"
    description = "..."
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-v','--verbose', action='store_true')
    parser.add_option('-r','--run', default=[],
                      action='append',choices=COMPONENTS,
                      help="Choose analysis component to run")
    (opts, args) = parser.parse_args()
    if not opts.run: opts.run = COMPONENTS
    if opts.verbose: logger.setLevel(logger.DEBUG)

    config = Config(args[0])

    if 'likelihood' in opts.run:
        logger.info("Running 'likelihood'...")
        farm = ugali.analysis.farm.Farm(config)
        farm.submit_all(coords=None,local=False,debug=False)
    if 'merge' in opts.run:
        logger.info("Running 'merge'...")
        filenames = "%s/likelihood*.fits"%config.params['output']['savedir_likelihood']
        infiles = sorted(glob.glob(filenames))
        outdir = mkdir(config.params['output']['savedir_results'])
        outfile = join(outdir,config.params['output']['mergefile'])
        if os.path.exists(outfile):
            raise Exception("File already exists; skipping...")
        ugali.utils.skymap.mergeSparseHealpixMaps(infiles,outfile)
    if 'targets' in opts.run and len(args) > 1:
        logger.info("Running 'targets'...")
        farm = ugali.analysis.farm.Farm(config)
        names,coords = farm.loadTargetCoordinates(args[1])
        farm.submit_all(coords=coords,local=False,debug=False)
        
        
