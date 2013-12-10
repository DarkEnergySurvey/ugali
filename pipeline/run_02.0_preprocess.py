#!/usr/bin/env python
import os
import glob

import ugali.data.pixelize
import ugali.data.maglims
from ugali.utils.parse_config import Config

from ugali.utils.logger import logger

COMPONENTS = ['pixelize','density','maglims']
if __name__ == "__main__":
    from optparse import OptionParser
    usage = "Usage: %prog  [options] input"
    description = "python script"
    parser = OptionParser(usage=usage,description=description)
    parser.add_option('-v','--verbose', action='store_true')
    parser.add_option('-r','--run', default=[],
                      action='append',choices=COMPONENTS,
                      help="Choose analysis component to run")
    (opts, args) = parser.parse_args()
    if not opts.run: opts.run = COMPONENTS
    if opts.verbose: logger.setLevel(logger.DEBUG)

    configfile = args[0]
    config = Config(configfile)

    if 'pixelize' in opts.run:
        # Pixelize the raw catalog data
        logger.info("Running 'pixelize'...")
        rawdir = config.params['data']['dirname']
        rawfiles = sorted(glob.glob(os.path.join(rawdir,'*.fits')))
        x = ugali.data.pixelize.pixelizeCatalog(rawfiles,config)
    if 'density' in opts.run:
        # Calculate magnitude limits
        logger.info("Running 'density'...")
        x = ugali.data.pixelize.pixelizeDensity(config,nside=2**9)
    if 'maglims' in opts.run:
        # Calculate magnitude limits
        logger.info("Running 'maglims'...")
        maglims = ugali.data.maglims.Maglims(config)
        x = maglims.run()
        
