#!/usr/bin/env python
"""
Base functionality for pipeline scripts
"""

import ugali.utils.batch
#from ugali.utils.batch import factory as batchFactory

from ugali.utils.parser import Parser
from ugali.utils.logger import logger
from ugali.utils.config import Config

class Pipeline(object):
    """
    A pipeline script owns:
    - A set of command line arguments
    - A set of runtime components
    """

    def __init__(self, description=__doc__, components=[]):
        self.description = description
        self.components = components
        self._setup_parser()

    def _setup_parser(self):
        self.parser = Parser(description=self.description)
        self.parser.add_config()
        self.parser.add_debug()
        self.parser.add_force()
        self.parser.add_queue()
        self.parser.add_run(choices=self.components) 
        self.parser.add_verbose()
        self.parser.add_version()

    def parse_args(self):
        self.opts = self.parser.parse_args()
        if not self.opts.run: 
            self.opts.run = self.components

        self.config = Config(self.opts.config)        
        self.batch = ugali.utils.batch.batchFactory(self.opts.queue)

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

