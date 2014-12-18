#!/usr/bin/env python
from os.path import join

from ugali.analysis.pipeline import Pipeline
from ugali.preprocess.database import databaseFactory
from ugali.utils.shell import mkdir

description="Download data from database."
components = []

def run(self):

    #db = databaseFactory(self.config)
    #db.run(outdir=self.config['data']['dirname'])


    outdir=self.config['data']['dirname']
    logdir=join(outdir,'log')
    mkdir(logdir)
    jobname='download'
    logfile=join(logdir,jobname+'.log')
    script = self.config['data']['script']

    cmd='%s %s'%(script,self.opts.config)
    self.batch.submit(cmd,jobname,logfile)


Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()
