#!/usr/bin/env python
from ugali.analysis.pipeline import Pipeline
from ugali.preprocess.database import databaseFactory

description="Download data from database."
components = []

def run(self):
    db = databaseFactory(self.config)
    db.run()

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()

"""
COMPONENTS = []
if __name__ == "__main__":
    import argparse
    description = "Pipeline script for database querying."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config',help='Configuration file.')
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Output verbosity')
    parser.add_argument('-r','--run', default=[],
                        action='append',choices=COMPONENTS,
                        help="Choose analysis component to run")
    opts = parser.parse_args()

    config = Config(opts.config)
    db = databaseFactory(config)
    db.run()
"""
