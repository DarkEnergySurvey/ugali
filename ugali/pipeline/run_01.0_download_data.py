#!/usr/bin/env python
"""Download data from database."""
from os.path import join

from ugali.analysis.pipeline import Pipeline
from ugali.preprocess.database import databaseFactory
from ugali.utils.shell import mkdir

#components = ['data','isochrone']
components = ['data']

def run(self):

    #db = databaseFactory(self.config)
    #db.run(outdir=self.config['data']['dirname'])

    if 'data' in self.opts.run:
        outdir=self.config['data']['dirname']
        logdir=join(outdir,'log')
        mkdir(logdir)
        jobname='download'
        logfile=join(logdir,jobname+'.log')
        script = self.config['data']['script']
         
        cmd='%s %s %s'%(script,self.opts.config,self.opts.pixfile)
        self.batch.submit(cmd,jobname,logfile)
    if 'isochrone' in self.opts.run:
        jobname='isochrone'
        script = self.config['data']['script'].replace('database.py','padova.py')
         
        cmd='%s %s'%(script,self.opts.config)
        self.batch.submit(cmd,jobname)
        

Pipeline.run = run
pipeline = Pipeline(__doc__,components)
pipeline.parser.add_argument('pixfile',nargs='?',metavar='pixels.dat',default='')
pipeline.parse_args()
pipeline.execute()
