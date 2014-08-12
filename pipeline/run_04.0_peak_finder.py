#!/usr/bin/env python

from ugali.analysis.pipeline import Pipeline

from ugali.candidate.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger

description="Perform object association."
components = ['label','objects','associate']

def run(self):
    search = CandidateSearch(self.config)

    if 'label' in self.opts.run:
        logger.info("Running 'label'...")
        search.createLabels()
        search.writeLabels()
    if 'objects' in self.opts.run:
        logger.info("Running 'objects'...")
        search.loadLabels()
        search.createObjects()
        search.writeObjects()
    if 'associate' in self.opts.run:
        logger.info("Running 'associate'...")
        ugali.candidate.associate.associate_sources(config)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()

"""    
if __name__ == "__main__":
    import argparse
    description = "Pipeline script for object association."
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

    search = CandidateSearch(config)

    if 'label' in opts.run:
        logger.info("Running 'label'...")
        search.createLabels()
        search.writeLabels()
    if 'objects' in opts.run:
        logger.info("Running 'objects'...")
        search.loadLabels()
        search.createObjects()
        search.writeObjects()
    if 'associate' in opts.run:
        logger.info("Running 'associate'...")
        x,o = ugali.candidate.associate.associate_sources(config)
"""
