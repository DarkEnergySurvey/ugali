#!/usr/bin/env python
import os
from os.path import exists, join

from ugali.analysis.pipeline import Pipeline
from ugali.analysis.search import CandidateSearch
import ugali.candidate.associate

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

description="Perform object finding and association."
components = ['label','objects','associate','candidate','plot']

def run(self):
    search = CandidateSearch(self.config)
    self.search = search

    if 'label' in self.opts.run:
        logger.info("Running 'label'...")
        if exists(search.labelfile) and not self.opts.force:
            logger.info("  Found %s; skipping..."%search.labelfile)
        else:
            #search.createLabels3D()
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
        logger.info("Running 'plot'...")
        import pyfits

        threshold = self.config['search']['cand_threshold']
        outdir = mkdir(self.config['output']['plotdir'])
        logdir = mkdir(os.path.join(outdir,'log'))

        # Eventually move this into 'plotting' module
        candidates = pyfits.open(self.config.candfile)[1].data
        candidates = candidates[candidates['TS'] >= threshold]

        for candidate in candidates:
            logger.info("Plotting %s (%.2f,%.2f)..."%(candidate['name'],candidate['glon'],candidate['glat']))
            params = (self.opts.config,outdir,candidate['name'],candidate['ra'],
                      candidate['dec'],candidate['modulus'])
            cmd = 'ugali/scratch/PlotCandidate.py %s %s -n="%s" --cel %f %f -m %.2f'
            cmd = cmd%params

            jobname = candidate['name'].lower().replace(' ','_')
            logfile = os.path.join(logdir,jobname+'.log')
            self.batch.submit(cmd,jobname,logfile)

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parse_args()
pipeline.execute()


        #maglims = ugali.utils.skymap.readSparseHealpixMaps(pipeline.config.filenames['mask_1'].compressed(),'MAGLIM')
        #ugali.utils.plotting.plotSkymap(maglims,coord='G')
        #ugali.utils.plotting.projScatter(candidates['glon'],candidates['glat'],c='r',coord='G')
        #basename = 'candidates_gal.png'
        #outfile = os.path.join(outdir,basename)
        #plt.savefig(outfile)
        # 
        #ugali.utils.plotting.plotSkymap(maglims,coord='GC')
        #ugali.utils.plotting.projScatter(candidates['glon'],candidates['glat'],c='r',coord='GC')
        #basename = 'candidates_equ.png'
        #outfile = os.path.join(outdir,basename)
        #plt.savefig(outfile)

        #import numpy as np
        #from ugali.utils.projector import cel2gal
        # 
        #data = [
        #    #['Grus 1'   , 344.17, -50.1633, 19.6, 8], 
        #    #['Indus 1'  , 317.20, -51.1656, 20.0, 8], 
        #    #['Phoenix 2', 354.99, -54.4060, 20.4, 8], 
        #    ['J0443.8-5017',70.9, -50.3, 20, 8],
        #    ]
        #print data
        #data = [ d + list(cel2gal(d[1],d[2])) for d in data]
        #print data
        #names=['NAME','RA','DEC','DISTANCE_MODULUS','ZIDX_MAX','GLON','GLAT']
        #rec = np.rec.fromrecords(data,names=names)
        # 
        #candidates = pyfits.new_table(rec).data


            #logger.info("Plotting %s (%.2f,%.2f)..."%(candidate['name'],candidate['glon'],candidate['glat']))
            #label = candidate['name'].lower().replace(' ','_')
            #plotter = ugali.utils.plotting.ObjectPlotter(candidate,self.config)
            # 
            #fig,ax = plotter.plot4()
            #basename = '%s.png'%label
            #outfile = os.path.join(outdir,basename)
            #plt.savefig(outfile,bbox_inches='tight')
            # 
            #fig,ax = plotter.plotDistance()
            #basename = '%s_dist.png'%label
            #outfile = os.path.join(outdir,basename)
            #plt.savefig(outfile,bbox_inches='tight')
            # 
            #fig,axes = plt.subplots(1,2)
            #plotter.drawSpatial(axes[0])
            #plotter.drawCMD(axes[1],radius=0.2)
            #basename = '%s_scat.png'%label
            #outfile = os.path.join(outdir,basename)
            #plt.savefig(outfile,bbox_inches='tight')
