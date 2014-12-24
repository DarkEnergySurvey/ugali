#!/usr/bin/env python
"""
Perform targeted followup.

"""

import os
from os.path import join, exists,basename,splitext
import numpy
import numpy as np

from ugali.analysis.mcmc import MCMC
from ugali.analysis.pipeline import Pipeline
from ugali.analysis.scan import Scan

from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

description=__doc__
components = ['mcmc','membership','plot']

def run(self):
    if self.opts.coords is not None:
        coords = self.opts.coords
        names = vars(self.opts).get('names',len(coords)*[''])
    else:
        #dirname = self.config['output2']['searchdir']
        #filename = os.path.join(dirname,self.config['output2']['candfile'])
        names,coords = self.parser.parse_targets(self.config.candfile)
    labels=[n.lower().replace(' ','_').replace('(','').replace(')','') for n in names]

    outdir=mkdir(self.config['output']['mcmcdir'])
    logdir=mkdir(join(outdir,'log'))

    if 'mcmc' in self.opts.run:
        logger.info("Running 'mcmc'...")
        for name,label,coord in zip(names,labels,coords):
            glon,glat,radius = coord
            print name,'(%.4f,%.4f)'%(glon,glat)
            outfile=join(outdir,self.config['output']['mcmcfile']%label)
            base = splitext(basename(outfile))[0]
            logfile=join(logdir,base+'.log')
            jobname=base
            script = self.config['mcmc']['script']
            cmd='%s %s --gal %.4f %.4f %s'%(script,self.opts.config,glon,glat,outfile)
            nthreads = self.config['mcmc']['nthreads']
            self.batch.submit(cmd,jobname,logfile,n=nthreads)
    if 'membership' in self.opts.run:
        logger.info("Running 'membership'...")
        for name,label,coord in zip(names,labels,coords):
            glon,glat,radius = coord
            print name,'(%.4f,%.4f)'%(glon,glat)
            scan = Scan(self.config, [coord])
            #scan.run()
            #params = scan.grid.mle()
            #params['ellipticity'] = None
            #params['position_angle'] = None
            params = dict(distance_modulus=19.,extension=0.16,lat=69.63,lon=358.09,richness=51514.)
            scan.loglike.set_params(**params)
            scan.loglike.sync_params()
            outfile=join(outdir,self.config['output']['mcmcfile']%label).replace('.npy','.fits')
            scan.loglike.write_membership(outfile)

    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        import ugali.utils.plotting
        import matplotlib; matplotlib.use('Agg')
        import triangle

        for name,label,coord in zip(names,labels,coords):
            infile = join(outdir,self.config['output']['mcmcfile']%label)
            if not exists(infile): continue
            outfile = infile.replace('.npy','.png')
            nburn = self.config['mcmc']['nburn']
            nwalkers = self.config['mcmc']['nwalkers']
            params = self.config['mcmc']['params']
            samples = np.load(infile)[nburn*nwalkers:]
            if len(params) != len(samples.dtype):
                raise Exception("Samples shape does not match number of params")
            fig = triangle.corner(samples.view((float,len(params))), labels=params)
            fig.suptitle(name)
            logger.info("  Writing %s..."%outfile)
            fig.savefig(outfile)
            
Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
