#!/usr/bin/env python
"""
Base functionality for pipeline script instances
"""
import argparse

from ugali.utils.logger import logger
from ugali.utils.config import Config

class Pipeline(object):
    """
    A pipeline script owns:
    - A set of command line arguments
    - A set of runtime components
    """

    def __init__(self, description, components=[]):
        self.description = description
        self.components = components
        self._setup_parser()

    def _setup_parser(self):
        self.parser = argparse.ArgumentParser(description=self.description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('config',metavar='config.py',help='Configuration file.')
        self.parser.add_argument('-d','--dryrun',action='store_true',
                            help="NOT IMPLEMENTED.")
        self.parser.add_argument('-q','--queue',
                            help="NOT IMPLEMENTED.")
        self.parser.add_argument('-r','--run', default=[],
                            action='append',choices=self.components,
                            help="Analysis component(s) to run.")
        self.parser.add_argument('-v','--verbose',action='store_true',
                            help='Output verbosity.')

    def parse_args(self):
        self.opts = self.parser.parse_args()

        self.config = Config(self.opts.config)        
        if not self.opts.run: 
            self.opts.run = self.components
        if self.opts.verbose: 
            logger.setLevel(logger.DEBUG)

    def run(self):
        logger.warning("Doing nothing...")
        return

    def execute(self):
        ret = self.run()
        logger.info("Done.")
        return ret

if __name__ == "__main__":
    description = "Pipeline test"
    components = ['test']
     
    def run(self):
        logger.info("Testing pipeline...")
        if 'test' in self.opts.run:
            logger.info("  This should run.")
        if 'foo' in self.opts.run:
            logger.error("  This should NOT run")
            raise Exception
    Pipeline.run = run
        
    pipeline = Pipeline(description,components)
    pipeline.parser.print_help()
    pipeline.parse_args()
    pipeline.execute()

