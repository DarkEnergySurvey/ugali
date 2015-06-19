#!/usr/bin/env python
"""
Perform targeted followup.

"""

import os
from os.path import join, exists,basename,splitext
import numpy
import numpy as np
import yaml
import pyfits

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
    if 'results' is self.opts.run:
        logger.info("Running 'results'...")
        # Not implemented yet because no way to get the fixed parameter values...
    if 'membership' in self.opts.run:
        logger.info("Running 'membership'...")
        from ugali.analysis.kernel import kernelFactory
        for name,label,coord in zip(names,labels,coords):
            glon,glat,radius = coord
            print name,'(%.4f,%.4f)'%(glon,glat)
            scan = Scan(self.config, [coord])
            #scan.run()
            #params = scan.grid.mle()
            #params['ellipticity'] = None
            #params['position_angle'] = None
            #params = dict(distance_modulus=19.,extension=0.16,lat=69.63,lon=358.09,richness=51514.)
            datfile = join(outdir,self.config['output']['mcmcfile']%label)
            datfile = datfile.replace('.npy','.dat')
            params = yaml.load(open(datfile))['params']

            kernel = kernelFactory(**self.config['mcmc']['kernel'])
            scan.loglike.set_model('spatial',kernel)
            scan.loglike.set_params(**params)
            scan.loglike.sync_params()
            outfile=join(outdir,self.config['output']['mcmcfile']%label).replace('.npy','.fits')
            scan.loglike.write_membership(outfile)

    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        import ugali.utils.plotting
        import matplotlib; matplotlib.use('Agg')
        import pylab as plt
        import triangle

        for name,label,coord in zip(names,labels,coords):
            infile = join(outdir,self.config['output']['mcmcfile']%label)
            if not exists(infile): 
                logger.warning("Couldn't find %s; skipping..."%infile)
                continue
            outfile = infile.replace('.npy','.png')
            
            samples = ugali.analysis.mcmc.Samples(infile)

            nburn = self.config['mcmc']['nburn']
            nwalkers = self.config['mcmc']['nwalkers']
            burn = nburn * nwalkers
            clip = 10.

            params = samples.names
            datfile = join(outdir,self.config['output']['mcmcfile']%label)
            datfile = datfile.replace('.npy','.dat')
            if False: #os.path.exists(datfile):
                results = yaml.load(open(datfile))['results']
                truths = [results[param][0] for param in params]
            else:
                truths=[samples.kde_peak(p,burn=burn,clip=clip) for p in samples.names]
                #truths = None

            chain = samples.ndarray[nburn*nwalkers:]
            chain = chain[~np.all(chain==0,axis=1)]

            extents = None
            #extents = [[0,15e3],[323.6,323.8],[-59.8,-59.7],[0,0.1],[19.5,20.5]]
            fig = triangle.corner(chain, labels=params, truths=truths,extents=extents)
            fig.suptitle(name)
            logger.info("  Writing %s..."%outfile)
            fig.savefig(outfile)

            ### Plot membership probabilities
            resfile = infile.replace('.npy','.dat')
            params = yaml.load(open(resfile))['params']
            data = pyfits.open(infile.replace('.npy','.fits'))[1].data
            
            iso = MCMC.createIsochrone(self.config)
            kernel = MCMC.createKernel(self.config)

            for k,v in params.items():
                try: 
                    setattr(iso,k,v)
                    setattr(kernel,k,v)
                except:
                    pass

            print iso
            print kernel

            plt.figure()
            ugali.utils.plotting.plotMembership(self.config,data,kernel,iso)
            outfile = infile.replace('.npy','_mem.png')
            logger.info("  Writing %s..."%outfile)
            plt.savefig(outfile,bbox_inches='tight')

Pipeline.run = run
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
