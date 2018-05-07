#!/usr/bin/env python
"""Pipeline script for data pre-processing."""
import os
import glob

import numpy as np

from ugali.analysis.pipeline import Pipeline
import ugali.preprocess.pixelize
import ugali.preprocess.maglims

from ugali.utils.logger import logger

components = ['pixelize','density','maglims','simple','split']
defaults = ['pixelize','density','simple']

def run(self):
    # The three mask options are (semi-)mutually exclusive
    if np.in1d(['maglims','simple','split'],self.opts.run).sum() > 1:
        raise Exception("Too many 'mask' run options.")

    if 'pixelize' in self.opts.run:
        # Pixelize the raw catalog data
        logger.info("Running 'pixelize'...")
        rawdir = self.config['data']['dirname']
        rawfiles = sorted(glob.glob(os.path.join(rawdir,'*.fits')))
        x = ugali.preprocess.pixelize.pixelizeCatalog(rawfiles,self.config)
    if 'density' in self.opts.run:
        # Calculate magnitude limits
        logger.info("Running 'density'...")
        x = ugali.preprocess.pixelize.pixelizeDensity(self.config,nside=512,force=self.opts.force)
    if 'maglims' in self.opts.run:
        # Calculate magnitude limits
        logger.info("Running 'maglims'...")
        maglims = ugali.preprocess.maglims.Maglims(self.config)
        x = maglims.run(force=self.opts.force)
    if 'simple' in self.opts.run:
        # Calculate simple magnitude limits
        logger.info("Running 'simple'...")
        #ugali.preprocess.maglims.simple_maglims(self.config,force=self.opts.force)
        maglims = ugali.preprocess.maglims.Maglims(self.config)
        x = maglims.run(simple=True,force=self.opts.force)
    if 'split' in self.opts.run:
        # Split up a pre-existing maglim map
        logger.info("Running 'split'...")
        ugali.preprocess.maglims.split(self.config,'split',force=self.opts.force)


Pipeline.run = run
Pipeline.defaults = defaults
pipeline = Pipeline(__doc__,components)
pipeline.parse_args()
pipeline.execute()
