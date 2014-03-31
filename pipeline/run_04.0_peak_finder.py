#!/usr/bin/env python

from ugali.candidate.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.parse_config import Config
from ugali.utils.logger import logger

COMPONENTS = ['label','objects','associate']
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

    configfile = args[0]
    config = Config(configfile)

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
        
        
