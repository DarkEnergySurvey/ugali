#!/usr/bin/env python
"""Perform object finding and association."""
import os, glob
from os.path import exists, join
import time

import fitsio
import numpy as np

from ugali.analysis.pipeline import Pipeline
from ugali.analysis.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

components = ['label','objects','associate','candidate','plot','www']

def load_candidates(filename,threshold=0):
    """ Load candidates for plotting """
    candidates = fitsio.read(filename,lower=True,trim_strings=True)
    candidates = candidates[candidates['ts'] >= threshold]

    return candidates

def run(self):
    if 'label' in self.opts.run:
        logger.info("Running 'label'...")
        if not hasattr(self,'search'): 
            self.search = CandidateSearch(self.config)
        if exists(self.search.labelfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%self.search.labelfile)
        else:
            #self.search.createLabels3D()
            #self.search.loadLikelhood()
            #self.search.loadROI()
            self.search.createLabels2D()
            self.search.writeLabels()
    if 'objects' in self.opts.run:
        logger.info("Running 'objects'...")
        if not hasattr(self,'search'): 
            self.search = CandidateSearch(self.config)
        if exists(self.search.objectfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%self.search.labelfile)
        else:
            self.search.loadLabels()
            self.search.createObjects()
            self.search.writeObjects()
    if 'associate' in self.opts.run:
        logger.info("Running 'associate'...")
        if not hasattr(self,'search'): 
            self.search = CandidateSearch(self.config)
        if exists(self.search.assocfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%self.search.assocfile)
        else:
            self.search.loadObjects()
            self.search.createAssociations()
            self.search.writeAssociations()
    if 'candidate' in self.opts.run:
        logger.info("Running 'candidate'...")
        if exists(self.search.candfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%self.search.candfile)
        else:
            self.search.loadAssociations()
            self.search.writeCandidates()
    if 'plot' in self.opts.run:
        self.opts.run.append('www')
        logger.info("Running 'plot'...")

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])
        logdir = mkdir(join(outdir,'log'))

        # Eventually move this into 'plotting' module
        candidates = load_candidates(self.config.candfile,threshold)

        for i,c in enumerate(candidates):
            name = c['name'].replace('(','').replace(')','')
            msg = "(%i/%i) Plotting %s (%.2f,%.2f)..."%(i,len(candidates),name,c['ra'],c['dec'])
            logger.info(msg)
            params = (self.opts.config,outdir,name,c['ra'],
                      c['dec'],0.5,c['modulus'])
            cmd = 'ugali/scratch/PlotCandidate.py %s %s -n="%s" --cel %f %f --radius %s -m %.2f'
            cmd = cmd%params
            jobname = name.lower().replace(' ','_')
            logfile = join(logdir,jobname+'.log')
            batch = self.config['search'].get('batch',self.config['batch'])
            out = [join(outdir,jobname+'.png'),
                   join(outdir,jobname+'_dist.png'),
                   join(outdir,jobname+'_scat.png')]
            if all([exists(o) for o in out]) and not self.opts.force:
                logger.info("  Found plots for %s; skipping..."%name)
            else:
                logger.info(cmd)
                self.batch.submit(cmd,jobname,logfile,**batch.get(self.opts.queue,{}))
                time.sleep(3)

    if 'www' in self.opts.run:
        logger.info("Running 'www'...")

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])

        # Eventually move this into 'plotting' module
        candidates = load_candidates(self.config.candfile,threshold)

        from ugali.utils.www import create_index_html
        filename = os.path.join(outdir,'index.html')
        create_index_html(filename,candidates)

        
Pipeline.run = run
pipeline = Pipeline(__doc__,components)
pipeline.parse_args()
pipeline.execute()
