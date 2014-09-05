#!/usr/bin/env python
import os
from os.path import join
import glob
import numpy
import copy
import subprocess

from ugali.analysis.scan import Scan
from ugali.analysis.mcmc import MCMC
from ugali.analysis.pipeline import Pipeline

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

description="Perform targeted followup."
components = ['mcmc']

def run(self):
    if self.opts.coords:
        coords = self.opts.coords
    else:
        dirname = self.config['output']['savedir_results']
        filename = os.path.join(dirname,self.config['output']['candfile'])
        names,coords = self.parser.parse_targets(filename)
    labels=[n.lower().replace(' ','_').replace('(','').replace(')','') for n in names]

    if 'mcmc' in self.opts.run:
        logger.info("Running 'mcmc'...")
        outdir=mkdir(self.config['output2']['mcmcdir'])
        logdir=mkdir(join(outdir,'log'))
        for name,label,coord in zip(names,labels,coords):
            glon,glat,radius = coord
            print name,'(%.4f,%.4f)'%(glon,glat)
            logfile=join(logdir,'%s_mcmc.log'%label)
            outfile=join(outdir,'%s_mcmc.npy'%label)
            jobname=label
            cmd='ugali/analysis/mcmc.py %s --gal %.4f %.4f %s'%(self.opts.config,glon,glat,outfile)
            nthreads = self.config['mcmc']['nthreads']
            self.batch.submit(cmd,jobname,logfile,n=nthreads)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
