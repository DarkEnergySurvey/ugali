#!/usr/bin/env python
"""
Run MCMC follow-up on target parameters.

The default parellel execution by emcee uses multiprocess and pickeling
the function that is being evaluated. Things become a bit complicated 
because you can't pickle instance methods.
Steven Bethard has a solution registering a method with pickle:
http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods#edit2155350

Another simple solution is to create a plain method linked to a static method.
http://stackoverflow.com/questions/21111106/cant-pickle-static-method-multiprocessing-python
"""

import os,sys
from collections import OrderedDict as odict

import numpy
import numpy as np
import numpy.lib.recfunctions as recfuncs
import scipy.stats

import pyfits
import yaml

import ugali.utils.stats
from ugali.utils.stats import Samples
import ugali.analysis.scan
from ugali.utils.config import Config
from ugali.analysis.kernel import kernelFactory
from ugali.analysis.loglike import createSource, createLoglike
from ugali.analysis.prior import UniformPrior, InversePrior

from ugali.utils.logger import logger
from ugali.utils.projector import dist2mod,mod2dist,gal2cel,gal2cel_angle

import emcee

# Write a class to store priors

class MCMC(object):
    """
    This object creates the loglike object from a source model along
    with a log prior and passes them to emcee with some flow control
    parameters.
    """

    def __init__(self, config, loglike):
        self.config = Config(config)
        self.nsamples = self.config['mcmc'].get('nsamples',100)
        self.nthreads = self.config['mcmc'].get('nthreads',16)
        self.nchunk = self.config['mcmc'].get('nchunk',25)
        self.nwalkers = self.config['mcmc'].get('nwalkers',50)
        self.nburn = self.config['mcmc'].get('nburn',10)
        self.alpha = self.config['mcmc'].get('alpha',0.10)

        self.loglike = loglike
        self.source = self.loglike.source
        self.params = self.source.get_free_params().keys()
        self.samples = None

        self.priors = odict(zip(self.params,len(self.params)*[UniformPrior()]))
        self.priors['extension'] = InversePrior()

    def __str__(self):
        ret = "%s:\n"%self.__class__.__name__
        ret += str(self.loglike)
        return ret
 
    def get_mle(self):
        """
        Get the values of the source parameters.
        """
        
        #return self.grid.mle()
        mle = self.source.get_params()
        # FIXME: For composite isochrones
        if 'age' not in mle:
            mle['age'] = np.average(self.source.isochrone.age)
        if 'metallicity' not in mle:
            mle['metallicity'] = np.average(self.source.isochrone.metallicity)
            
        return mle
 
    def get_std(self):
        """
        Somewhat arbitrarily assign standard deviation to seed points.
        """
        mle = self.get_mle()
 
        std = odict([
            ('richness',0.1*mle['richness']), # delta_r (10% of max)
            ('lon',0.01),                     # delta_l (deg)
            ('lat',0.01),                     # delta_b (deg)
            ('distance_modulus',0.1),         # delta_mu                
            ('extension',0.01),               # delta_ext (deg)
            ('ellipticity',0.1),              # delta_e 
            ('position_angle',15.0),          # delta_pa (deg)
            ('age',0.5),                      # delta_age (Gyr)
            ('metallicity',0.0001),           # delta_z
        ])
        return std
 
    def get_ball(self, params, size=1):
        mle = self.get_mle() 
        std = self.get_std()
                                             
        p0 = np.array([mle[k] for k in params])
        s0 = np.array([std[k] for k in params])
 
        return emcee.utils.sample_ball(p0,s0,size)
 
    def lnlike(self, theta):
        """ Logarithm of the likelihood """
        kwargs = dict(zip(self.params,theta))
        try:
            lnlike = self.loglike.value(**kwargs)
        except ValueError,AssertionError:
            lnlike = -np.inf
        return lnlike
 
    def lnprior(self,theta):
        """ Logarithm of the prior """
        kwargs = dict(zip(self.params,theta))
        err = np.seterr(invalid='raise')
        try:
            lnprior = np.sum(np.log([self.priors[k](v) for k,v in kwargs.items()]))
        except FloatingPointError,ValueError:
            lnprior = -np.inf
        np.seterr(**err)
        return lnprior
 
    def run(self, params=None, outfile=None):
        # Initailize the likelihood to maximal value
        mle =self.get_mle()
        msg = "Setting inital values..."
        for k,v in mle.items():
            msg+='\n  %s : %s'%(k,v)
        logger.info(msg)
        logger.info(str(self.loglike))
 
        self.loglike.set_params(**mle)
        self.loglike.sync_params()
 
        if params is not None: self.params = params
 
        nwalkers = self.nwalkers
        nsamples = self.nsamples
        nthreads = self.nthreads
        ndim = len(params)
        
        logger.info("Running MCMC chain...")
        p0 = self.get_ball(params,nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads=nthreads)
        #self.sampler.run_mcmc(p0,nsamples)
 
        # Chain is shape (nwalkers,nsteps,nparams)
        # Samples is shape (nwalkers*nsteps,nparams):
        for i,result in enumerate(self.sampler.sample(p0,iterations=nsamples)):
            steps = i+1
            if steps%10 == 0: logger.info("%i steps ..."%steps)
            self.chain = self.sampler.chain
            if (i==0) or (steps%self.nchunk==0):
                samples = self.chain.reshape(-1,len(self.params),order='F')
                self.samples = Samples(samples.T,names=self.params)
                if outfile is not None: 
                    logger.info("Writing %i steps to %s..."%(steps,outfile))
                    self.write_samples(outfile)
 
        samples = self.chain.reshape(-1,len(self.params),order='F')
        self.samples = Samples(samples.T,names=self.params)
        if outfile is not None: self.write_samples(outfile)
 
    def write_samples(self,filename):
        np.save(filename,self.samples)
 
    def load_samples(self,filename):
        self.samples = Samples(filename)
 
    def load_srcmdl(self,filename,section='source'):
        self.source.load(filename,section)

    def write_srcmdl(self,filename,section='source'):
        source = dict()
        source[section] = self.source.todict()
        
        out = open(filename,'w')
        out.write(yaml.dump(source))
        out.close()

# Move to factory?
def createMCMC(config,srcfile,section='source',samples=None):
    """ Create an MCMC instance """

    source = ugali.analysis.source.Source()
    source.load(srcfile,section=section)
    loglike = ugali.analysis.loglike.createLoglike(config,source)

    mcmc = MCMC(config,loglike)
    if samples is not None:
        mcmc.load_samples(samples)

    return mcmc

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for running MCMC followup"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_verbose()
    #parser.add_coords(required=True)
    parser.add_coords()
    parser.add_name()
    parser.add_argument('--srcmdl',help='Source model file')
    parser.add_argument('--grid',action='store_true',help='Grid search for intial parameters')
    parser.add_argument('outfile',default=None,help="Output file name")
                        
    opts = parser.parse_args()

    if opts.coords is not None and len(opts.coords) != 1: 
        raise Exception('Must specify exactly one coordinate.')

    config = Config(opts.config)

    outfile = opts.outfile
    samfile = outfile                          # samples
    srcfile = outfile.replace('.npy','.yaml')  # srcmdl and results
    memfile = outfile.replace('.npy','.fits')  # membership
    resfile = srcfile                          # results file
    
    source = ugali.analysis.loglike.createSource(config,section='mcmc')
    source.name = opts.name

    if opts.srcmdl is not None:
        source.load(opts.srcmdl,section=opts.name)
    if opts.coords:
        lon,lat,radius = opts.coords[0]
        source.set_params(lon=lon,lat=lat)
    if config['mcmc'].get('params'):
        params = config['mcmc'].get('params')
        source.set_free_params(params)

    loglike = ugali.analysis.loglike.createLoglike(config,source)

    if opts.grid:
        grid = ugali.analysis.scan.GridSearch(config,loglike)
        grid.search()
        source.set_params(**grid.mle())

    mcmc = MCMC(config,loglike)

    #resfile = opts.outfile.replace('.npy','.dat')

    logger.info("Writing %s..."%srcfile)
    mcmc.write_srcmdl(srcfile)

    niter = 0
    def lnprob(theta):
        global niter
        # Avoid extra likelihood calls with bad priors
        lnprior = mcmc.lnprior(theta)
        if not np.isfinite(lnprior):
            lnprior = -np.inf
            lnlike = -np.inf
        else:
            lnlike = mcmc.lnlike(theta)
        lnprob = lnprior + lnlike

        if (niter%100==0):
            msg = "%i function calls ...\n"%niter
            msg+= ', '.join('%s: %.3f'%(k,v) for k,v in zip(mcmc.params,theta))
            msg+= '\nlog(like): %.3f, log(prior): %.3f'%(lnprior,lnlike)
            logger.debug(msg)
        niter+=1
        return lnprob
    mcmc.lnprob = lnprob

    #params = mcmc.config['mcmc']['params']
    params = source.get_free_params().keys()
    mcmc.run(params,samfile)

    logger.info("Writing %s..."%srcfile)
    from ugali.analysis.results import write_results
    write_results(resfile,config,srcfile,samfile)

    logger.info("Writing %s..."%memfile)
    from ugali.analysis.loglike import write_membership
    write_membership(memfile,config,srcfile,section='source')
