#!/usr/bin/env python
import os
from os.path import exists, join

from ugali.analysis.pipeline import Pipeline
from ugali.candidate.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger

description="Perform object finding and association."
components = ['label','objects','associate','candidates']

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
    if 'candidates' in self.opts.run:
        logger.info("Running 'candidates'...")
        if exists(search.candfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.candfile)
        else:
            search.loadAssociations()
            search.writeCandidates()
        
            
Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()
