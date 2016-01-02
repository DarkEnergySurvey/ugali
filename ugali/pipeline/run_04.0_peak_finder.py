#!/usr/bin/env python
import os
from os.path import exists, join
import time

from ugali.analysis.pipeline import Pipeline
from ugali.analysis.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

description="Perform object finding and association."
components = ['label','objects','associate','candidate','plot']

def run(self):
    search = CandidateSearch(self.config)
    self.search = search

    if 'label' in self.opts.run:
        logger.info("Running 'label'...")
        if exists(search.labelfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.labelfile)
        else:
            #search.createLabels3D()
            search.createLabels2D()
            search.writeLabels()
    if 'objects' in self.opts.run:
        logger.info("Running 'objects'...")
        if exists(search.objectfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.labelfile)
        else:
            search.loadLabels()
            search.createObjects()
            search.writeObjects()
    if 'associate' in self.opts.run:
        logger.info("Running 'associate'...")
        if exists(search.assocfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.assocfile)
        else:
            search.loadObjects()
            search.createAssociations()
            search.writeAssociations()
    if 'candidate' in self.opts.run:
        logger.info("Running 'candidate'...")
        if exists(search.candfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.candfile)
        else:
            search.loadAssociations()
            search.writeCandidates()
    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        import pyfits

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])
        logdir = mkdir(os.path.join(outdir,'log'))

        # Eventually move this into 'plotting' module
        candidates = pyfits.open(self.config.candfile)[1].data
        candidates = candidates[candidates['TS'] >= threshold]

        for candidate in candidates:
            logger.info("Plotting %s (%.2f,%.2f)..."%(candidate['name'],candidate['glon'],candidate['glat']))
            params = (self.opts.config,outdir,candidate['name'],candidate['ra'],
                      candidate['dec'],candidate['modulus'])
            cmd = 'ugali/scratch/PlotCandidate.py %s %s -n="%s" --cel %f %f -m %.2f'
            cmd = cmd%params

            jobname = candidate['name'].lower().replace(' ','_')
            logfile = os.path.join(logdir,jobname+'.log')
            self.batch.submit(cmd,jobname,logfile)
            time.sleep(5)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()
