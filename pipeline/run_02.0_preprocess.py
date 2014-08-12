#!/usr/bin/env python
import os
import glob

import ugali.preprocess.pixelize
import ugali.preprocess.maglims

from ugali.utils.logger import logger

components = ['pixelize','density','maglims','simple']
description="Pipeline script for data pre-processing."

def run(self):
    if 'pixelize' in self.opts.run:
        # Pixelize the raw catalog data
        logger.info("Running 'pixelize'...")
        rawdir = config.params['data']['dirname']
        rawfiles = sorted(glob.glob(os.path.join(rawdir,'*.fits')))
        x = ugali.preprocess.pixelize.pixelizeCatalog(rawfiles,config)
    if 'density' in self.opts.run:
        # Calculate magnitude limits
        logger.info("Running 'density'...")
        x = ugali.preprocess.pixelize.pixelizeDensity(config,nside=2**9)
    if 'maglims' in self.opts.run:
        # Calculate magnitude limits
        logger.info("Running 'maglims'...")
        maglims = ugali.preprocess.maglims.Maglims(config)
        x = maglims.run()
    if 'simple' in self.opts.run:
        # Calculate simple magnitude limits
        logger.info("Running 'simple'...")
        ugali.preprocess.maglims.simple_maglims(config)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()


"""
COMPONENTS = ['pixelize','density','maglims','simple']
if __name__ == "__main__":
    import argparse
    description = "Pipeline script for data pre-processing."
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

    if 'pixelize' in opts.run:
        # Pixelize the raw catalog data
        logger.info("Running 'pixelize'...")
        rawdir = config.params['data']['dirname']
        rawfiles = sorted(glob.glob(os.path.join(rawdir,'*.fits')))
        x = ugali.preprocess.pixelize.pixelizeCatalog(rawfiles,config)
    if 'density' in opts.run:
        # Calculate magnitude limits
        logger.info("Running 'density'...")
        x = ugali.preprocess.pixelize.pixelizeDensity(config,nside=2**9)
    if 'maglims' in opts.run:
        # Calculate magnitude limits
        logger.info("Running 'maglims'...")
        maglims = ugali.preprocess.maglims.Maglims(config)
        x = maglims.run()
    if 'simple' in opts.run:
        # Calculate simple magnitude limits
        logger.info("Running 'simple'...")
        ugali.preprocess.maglims.simple_maglims(config)
"""
