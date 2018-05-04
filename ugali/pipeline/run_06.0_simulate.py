#!/usr/bin/env python
"""
Simulate the likelihood search.
"""
import os
from os.path import join, splitext, exists
import time
import glob

import numpy as np
import numpy.lib.recfunctions as recfuncs
import fitsio

from ugali.analysis.pipeline import Pipeline
import ugali.analysis.loglike
import ugali.simulation.simulator
import ugali.utils.skymap

from ugali.utils.shell import mkdir
from ugali.utils.logger import logger
from ugali.utils.healpix import pix2ang

components = ['simulate','analyze','merge','plot']

def run(self):
    outdir=mkdir(self.config['output']['simdir'])
    logdir=mkdir(join(outdir,'log'))

    if 'simulate' in self.opts.run:
        logger.info("Running 'simulate'...")

        if self.opts.num is None: self.opts.num = self.config['simulator']['njobs']
        for i in range(self.opts.num):
            outfile=join(outdir,self.config['output']['simfile']%i)
            base = splitext(os.path.basename(outfile))[0]
            logfile=join(logdir,base+'.log')
            jobname=base
            script = self.config['simulator']['script']
            cmd='%s %s %s --seed %i'%(script,self.opts.config,outfile,i)
            #cmd='%s %s %s'%(script,self.opts.config,outfile)
            self.batch.submit(cmd,jobname,logfile)
            time.sleep(0.1)

    if 'analyze' in self.opts.run:
        logger.info("Running 'analyze'...")
        dirname = self.config['simulate']['dirname']
        catfiles = sorted(glob.glob(join(dirname,self.config['simulate']['catfile'])))
        popfile = join(dirname,self.config['simulate']['popfile'])
        batch = self.config['simulate']['batch']

        for i,catfile in enumerate(catfiles):
            basename = os.path.basename(catfile)
            outfile = join(outdir,basename)

            if exists(outfile) and not self.opts.force:
                logger.info("  Found %s; skipping..."%outfile)
                continue

            base = splitext(os.path.basename(outfile))[0]
            logfile=join(logdir,base+'.log')
            jobname=base
            script = self.config['simulate']['script']
            cmd='%s %s -p %s -c %s -o %s'%(script,self.opts.config,popfile,catfile,outfile)
            opts = batch.get(self.opts.queue,dict())
            self.batch.submit(cmd,jobname,logfile,**opts)
            time.sleep(0.1)
        
    if 'sensitivity' in self.opts.run:
        logger.info("Running 'sensitivity'...")

    if 'merge' in self.opts.run:
        logger.info("Running 'merge'...")

        filenames=join(outdir,self.config['output']['simfile']).split('_%')[0]+'_*'
        infiles=sorted(glob.glob(filenames))

        f = fitsio.read(infiles[0])
        table = np.empty(0,dtype=data.dtype)
        for filename in infiles:
            logger.debug("Reading %s..."%filename)
            d = fitsio.read(filename)
            t = d[~np.isnan(d['ts'])]
            table = recfuncs.stack_arrays([table,t],usemask=False,asrecarray=True)

        logger.info("Found %i simulations."%len(table))
        outfile = join(outdir,"merged_sims.fits")
        logger.info("Writing %s..."%outfile)
        fitsio.write(outfile,table,clobber=True)
        
    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        import ugali.utils.plotting
        import pylab as plt

        plotdir = mkdir(self.config['output']['plotdir'])

        data = fitsio.read(join(outdir,"merged_sims.fits"))
        data = data[~np.isnan(data['ts'])]
        
        bigfig,bigax = plt.subplots()
        
        for dist in np.unique(data['fit_distance']):
            logger.info('  Plotting distance: %s'%dist)
            ts = data['ts'][data['fit_distance'] == dist]
            ugali.utils.plotting.drawChernoff(bigax,ts,bands='none',color='gray')
            
            fig,ax = plt.subplots(1,2,figsize=(10,5))
            ugali.utils.plotting.drawChernoff(ax[0],ts,bands='none',pdf=True)
            ugali.utils.plotting.drawChernoff(ax[1],ts)
            fig.suptitle(r'Chernoff ($\mu = %g$)'%dist)
            ax[0].annotate(r"$N=%i$"%len(ts), xy=(0.15,0.85), xycoords='axes fraction', 
                           bbox={'boxstyle':"round",'fc':'1'})
            basename = 'chernoff_u%g.png'%dist
            outfile = os.path.join(plotdir,basename)
            plt.savefig(outfile)
        bigfig.suptitle('Chernoff!')
        basename = 'chernoff_all.png'
        outfile = os.path.join(plotdir,basename)
        plt.savefig(outfile)

        #idx=np.random.randint(len(data['ts'])-1,size=400)
        #idx=slice(400)
        #ugali.utils.plotting.plotChernoff(data['ts'][idx])
        #ugali.utils.plotting.plotChernoff(data['fit_ts'])
        plt.ion()
        """
        try:
            fig = plt.figure()
            x = range(len(data))
            y = data['fit_mass']/data['stellar_mass']
            yclip,lo,hi = scipy.stats.sigmaclip(y)
            yerr = data['fit_mass_err']/data['stellar_mass']
             
            plt.errorbar(x,y,yerr=yerr,fmt='o',c='k')
            plt.axhline(1,ls='--',c='gray',lw=2)
            plt.axhline(np.mean(yclip),ls='--',c='r',lw=2)
            plt.ylim(lo,hi)
            plt.ylabel("Best-Fit Mass Residual")
            plt.xlabel("Simulation Number")
        except:
            pass
        """

Pipeline.run = run
pipeline = Pipeline(__doc__,components)
pipeline.parser.add_argument('-n','--num',default=None,type=int)
pipeline.parse_args()
pipeline.execute()

import pylab as plt
