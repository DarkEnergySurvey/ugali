#!/usr/bin/env python
"""
Perform targeted followup.

"""

import os
from os.path import join,exists,basename,splitext
import shutil
from collections import OrderedDict as odict

import matplotlib
try:             os.environ['DISPLAY']
except KeyError: matplotlib.use('Agg')

import numpy
import numpy as np
import yaml
import pyfits

from ugali.analysis.mcmc import MCMC
from ugali.analysis.pipeline import Pipeline
from ugali.analysis.scan import Scan
import ugali.analysis.source
import ugali.analysis.loglike
import ugali.analysis.mcmc

import ugali.utils.config
from ugali.utils.logger import logger
from ugali.utils.shell import mkdir

from multiprocessing import Pool

description=__doc__
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
    ugali.analysis.mcmc.write_results(config,srcfile,samples,srcfile)

def do_membership(args):
    """ Write the membership output file """
    config,name,label,coord = args

    filenames = make_filenames(config,label)
    srcfile = filenames['srcfile']
    memfile = filenames['memfile']

    source = ugali.analysis.source.Source()
    source.load(srcfile,'source')

    loglike = ugali.analysis.loglike.createLoglike(config,source)
    logger.info("Writing %s..."%memfile)
    loglike.write_membership(memfile)

def do_plot(args):
    import ugali.utils.plotting
    import pylab as plt
    import triangle

    config,name,label,coord = args
    print args

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

    data = pyfits.open(memfile)[1].data if exists(memfile) else None
    if data is not None:
        plt.figure()
        kernel,isochrone = source.kernel,source.isochrone
        ugali.utils.plotting.plotMembership(config,data,kernel,isochrone)
        outfile = samfile.replace('.npy','_mem.png')
        logger.info("  Writing %s..."%outfile)
        plt.savefig(outfile,bbox_inches='tight',dpi=60)
        plt.close()

    plotter = ugali.utils.plotting.SourcePlotter(source,config,radius=0.5)
    plotter.plot6(data)
    outfile = samfile.replace('.npy','_6panel.png')
    logger.info("  Writing %s..."%outfile)
    plt.savefig(outfile,bbox_inches='tight',dpi=60)
    plt.close()

    plotter.plot4()
    outfile = samfile.replace('.npy','_4panel.png')
    logger.info("  Writing %s..."%outfile)
    plt.savefig(outfile,bbox_inches='tight',dpi=60)
    plt.close()
    
def run(self):
    if self.opts.coords is not None:
        coords = self.opts.coords
        names = vars(self.opts).get('names',len(coords)*[''])
    else:
        names,coords = self.parser.parse_targets(self.config.candfile)
    labels=[n.lower().replace(' ','_').replace('(','').replace(')','') for n in names]

    self.outdir=mkdir(self.config['output']['mcmcdir'])
    self.logdir=mkdir(join(self.outdir,'log'))

    args = zip(len(names)*[self.opts.config],names,labels,coords)

    if 'mcmc' in self.opts.run:
        logger.info("Running 'mcmc'...")
        try:
            shutil.copy(self.opts.config,self.outdir)
        except m:
            print m

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
                print name, srcmdl
                cmd='%s %s --name %s --srcmdl %s %s'%(script,self.opts.config,name,srcmdl,outfile)
            else:
                print name,'(%.4f,%.4f)'%(glon,glat)
                cmd='%s %s --name %s --gal %.4f %.4f --grid %s'%(script,self.opts.config,name,glon,glat,outfile)
            print cmd
            self.batch.submit(cmd,jobname,logfile,n=nthreads)

    if 'results' in self.opts.run:
        logger.info("Running 'results'...")
        pool = Pool(maxtasksperchild=1)
        pool.map(do_results,args)

    if 'membership' in self.opts.run:
        logger.info("Running 'membership'...")
        pool = Pool(maxtasksperchild=1)
        pool.map(do_membership,args)

    if 'plot' in self.opts.run:
        logger.info("Running 'plot'...")
        #pool = Pool(maxtasksperchild=1)
        #pool.map(do_plot,args)
        map(do_plot,args)

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
pipeline = Pipeline(description,components)
pipeline.parser.add_coords(radius=True,targets=True)
pipeline.parse_args()
pipeline.execute()
