#!/usr/bin/env python
"""Perform MCMC follow-up fitting."""

import os
from os.path import join,exists,basename,splitext
import shutil
from collections import OrderedDict as odict
from multiprocessing import Pool

import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')

import numpy
import numpy as np
import yaml
import fitsio

from ugali.analysis.pipeline import Pipeline
from ugali.analysis.scan import Scan
import ugali.analysis.source
import ugali.analysis.loglike
import ugali.analysis.results

import ugali.utils.config
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

components = ['mcmc','membership','results','plot','collect','scan']

def make_filenames(config,label):
    config = ugali.utils.config.Config(config)
    outdir=config['output']['mcmcdir']
    samfile=join(outdir,config['output']['mcmcfile']%label)
    srcfile=samfile.replace('.npy','.yaml')
    memfile=samfile.replace('.npy','.fits')
    ret = dict(outfile=samfile,samfile=samfile,srcfile=srcfile,memfile=memfile)
    return ret

def do_results(args):
    """ Write the results output file """
    config,name,label,coord = args

    filenames = make_filenames(config,label)
    srcfile = filenames['srcfile']
    samples = filenames['samfile']

    if not exists(srcfile):
        logger.warning("Couldn't find %s; skipping..."%srcfile)
        return
    if not exists(samples):
        logger.warning("Couldn't find %s; skipping..."%samples)
        return

    logger.info("Writing %s..."%srcfile)
    from ugali.analysis.results import write_results
    write_results(srcfile,config,srcfile,samples)

def do_membership(args):
    """ Write the membership output file """
    config,name,label,coord = args

    filenames = make_filenames(config,label)
    srcfile = filenames['srcfile']
    memfile = filenames['memfile']

    logger.info("Writing %s..."%memfile)
    from ugali.analysis.loglike import write_membership
    write_membership(memfile,config,srcfile,section='source')
    
def do_plot(args):
    """ Create plots of mcmc output """
    import ugali.utils.plotting
    import pylab as plt

    config,name,label,coord = args
    filenames = make_filenames(config,label)
    srcfile = filenames['srcfile']
    samfile = filenames['samfile']
    memfile = filenames['memfile']

    if not exists(srcfile):
        logger.warning("Couldn't find %s; skipping..."%srcfile)
        return
    if not exists(samfile):
        logger.warning("Couldn't find %s; skipping..."%samfile)
        return

    config = ugali.utils.config.Config(config)
    burn = config['mcmc']['nburn']*config['mcmc']['nwalkers']

    source = ugali.analysis.source.Source()
    source.load(srcfile,section='source')

    outfile = samfile.replace('.npy','.png')
    ugali.utils.plotting.plotTriangle(srcfile,samfile,burn=burn)
    logger.info("  Writing %s..."%outfile)
    plt.savefig(outfile,bbox_inches='tight',dpi=60)
    plt.close()

    plotter = ugali.utils.plotting.SourcePlotter(source,config,radius=0.5)

    data = fitsio.read(memfile,trim_strings=True) if exists(memfile) else None
    if data is not None:
        plt.figure()
        kernel,isochrone = source.kernel,source.isochrone
        ugali.utils.plotting.plotMembership(config,data,kernel,isochrone)
        outfile = samfile.replace('.npy','_mem.png')
        logger.info("  Writing %s..."%outfile)
        plt.savefig(outfile,bbox_inches='tight',dpi=60)
        plt.close()
            
        plotter.plot6(data)

        outfile = samfile.replace('.npy','_6panel.png')
        logger.info("  Writing %s..."%outfile)
        plt.savefig(outfile,bbox_inches='tight',dpi=60)

        outfile = samfile.replace('.npy','_6panel.pdf')
        logger.info("  Writing %s..."%outfile)
        plt.savefig(outfile,bbox_inches='tight',dpi=60)

        plt.close()

    try:
        title = name
        plotter.plot4()
        outfile = samfile.replace('.npy','_4panel.png')
        logger.info("  Writing %s..."%outfile)
        plt.suptitle(title)
        plt.savefig(outfile,bbox_inches='tight',dpi=60)
        plt.close()
    except:
        logger.warning("  Failed to create plotter.plot4()")
    
def run(self):
    if self.opts.coords is not None:
        coords = self.opts.coords
        names = vars(self.opts).get('names',len(coords)*[''])
    else:
        names,coords = self.parser.parse_targets(self.config.candfile)
    labels=[n.lower().replace(' ','_').replace('(','').replace(')','') for n in names]

    self.outdir=mkdir(self.config['output']['mcmcdir'])
    self.logdir=mkdir(join(self.outdir,'log'))

    args = list(zip(len(names)*[self.opts.config],names,labels,coords))

    if 'mcmc' in self.opts.run:
        logger.info("Running 'mcmc'...")
        try:      shutil.copy(self.opts.config,self.outdir)
        except Exception as e: logger.warn(e.message)

        for config,name,label,coord in args:
            glon,glat,radius = coord
            outfile = make_filenames(self.config,label)['samfile']
            base = splitext(basename(outfile))[0]
            logfile=join(self.logdir,base+'.log')
            jobname=base
            script = self.config['mcmc']['script']
            nthreads = self.config['mcmc']['nthreads']
            srcmdl = self.config['mcmc'].get('srcmdl')

            if srcmdl is not None:
                try:      shutil.copy(srcmdl,self.outdir)
                except Exception as e: logger.warn(e.message)
                logger.info('%s (%s)'%(name,srcmdl))
                cmd='%s %s --name %s --srcmdl %s %s' % (
                    script,self.opts.config,name,srcmdl,outfile)
            else:
                logger.info('%s (%.4f,%.4f)'%(name,glon,glat))
                cmd='%s %s --name %s --gal %.4f %.4f --grid %s'% (
                    script,self.opts.config,name,glon,glat,outfile)
            logger.info(cmd)
            self.batch.submit(cmd,jobname,logfile,n=nthreads,a='mpirun')

    if 'results' in self.opts.run:
        logger.info("Running 'results'...")
        if len(args) > 1:
            pool = Pool(maxtasksperchild=1)
            pool.map(do_results,args)
        else:
            do_results(*args)

    if 'membership' in self.opts.run:
        logger.info("Running 'membership'...")
        if len(args) > 1:
            pool = Pool(maxtasksperchild=1)
            pool.map(do_membership,args)
        else:
            do_membership(*args)

    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        if len(args) > 1:
            pool = Pool(maxtasksperchild=1)
            pool.map(do_plot,args)
            #map(do_plot,args)
        else:
            do_plot(*args)

    if 'collect' in self.opts.run:
        logger.info("Running 'collect'...")
        results = odict()
        srcmdl = odict()
        params = odict()
        for config,name,label,coord in args:
            srcfile = make_filenames(self.config,name)['srcfile']
            results[name] = yaml.load(open(srcfile))['results']
            srcmdl[name] = yaml.load(open(srcfile))['source']
            params[name] = yaml.load(open(srcfile))['params']

        for base,output in [('results.yaml',results),('srcmdl.yaml',srcmdl),('params.yaml',params)]:
            outfile = join(self.outdir,base)
            out = open(outfile,'w')
            out.write(yaml.dump(output))
            out.close()

    if 'scan' in self.opts.run:
        logger.info("Running 'scan'...")
        for config,name,label,coord in args:
            logdir = mkdir('plots/log')
            logfile=join(logdir,'%s_lnlscan.log')

            cmd = 'python lnlscan.py %s  --name %s --xpar %s --xbins 45 --ypar %s --ybins 45'%(self.opts.config,name,'age','metallicity')
            self.batch.submit(cmd,logfile=logfile)

            cmd = 'python lnlscan.py %s  --name %s --xpar %s --xbins 45 --ypar %s --ybins 45'%(self.opts.config,name,'metallicity','distance_modulus')
            self.batch.submit(cmd,logfile=logfile)

            cmd = 'python lnlscan.py %s  --name %s --xpar %s --xbins 45 --ypar %s --ybins 45'%(self.opts.config,name,'age','distance_modulus')
            self.batch.submit(cmd,logfile=logfile)


Pipeline.run = run
pipeline = Pipeline(__doc__,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parser.add_ncores()
pipeline.parse_args()
pipeline.execute()
