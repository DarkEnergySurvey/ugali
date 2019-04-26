#!/usr/bin/env python
"""Perform object finding and association."""
import os, glob
from os.path import exists, join
import time

from ugali.analysis.pipeline import Pipeline
from ugali.analysis.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

components = ['label','objects','associate','candidate','plot','www']

def run(self):
    search = CandidateSearch(self.config)
    self.search = search

    if 'label' in self.opts.run:
        logger.info("Running 'label'...")
        if exists(search.labelfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.labelfile)
        else:
            #search.createLabels3D()
            #search.loadLikelhood()
            #search.loadROI()
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
    if 'candidate' in self.opts.run:
        logger.info("Running 'candidate'...")
        if exists(search.candfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.candfile)
        else:
            search.loadAssociations()
            search.writeCandidates()
    if 'plot' in self.opts.run:
        self.opts.run.append('www')
        logger.info("Running 'plot'...")
        import fitsio

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])
        logdir = mkdir(join(outdir,'log'))

        # Eventually move this into 'plotting' module
        candidates = fitsio.read(self.config.candfile,lower=True,trim_strings=True)
        candidates = candidates[candidates['ts'] >= threshold]
        
        for i,c in enumerate(candidates):
            name = c['name'].replace('(','').replace(')','')
            msg = "(%i/%i) Plotting %s (%.2f,%.2f)..."%(i,len(candidates),name,c['ra'],c['dec'])
            logger.info(msg)
            params = (self.opts.config,outdir,name,c['ra'],
                      c['dec'],0.5,c['modulus'])
            cmd = 'ugali/scratch/PlotCandidate.py %s %s -n="%s" --cel %f %f --radius %s -m %.2f'
            cmd = cmd%params
            logger.info(cmd)
            jobname = name.lower().replace(' ','_')
            logfile = join(logdir,jobname+'.log')
            batch = self.config['search'].get('batch',self.config['batch'])
            if len(glob.glob(join(outdir,jobname+'*.png')))==3 and not self.opts.force:
                logger.info("  Found plots for %s; skipping..."%name)
            else:
                self.batch.submit(cmd,jobname,logfile,**batch.get(self.opts.queue,{}))
                time.sleep(3)

    if 'www' in self.opts.run:
        logger.info("Running 'www'...")
        import fitsio

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])

        # Eventually move this into 'plotting' module
        candidates = fitsio.read(self.config.candfile,lower=True,trim_strings=True)
        candidates = candidates[candidates['ts'] >= threshold]

        from ugali.utils.www import create_index_html
        filename = os.path.join(outdir,'index.html')
        create_index_html(filename,candidates)

        
Pipeline.run = run
pipeline = Pipeline(__doc__,components)
pipeline.parse_args()
pipeline.execute()
