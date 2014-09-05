#!/usr/bin/env python

"""
Run MCMC follow-up on target parameters.
"""

import os
import sys
from collections import OrderedDict as odict

import numpy
import numpy as np
import scipy.stats

import pyfits
import healpy
import emcee
import yaml

import ugali.analysis.scan
import ugali.utils.config

from ugali.utils.logger import logger
from ugali.utils.skymap import superpixel, subpixel


"""
The default parellel execution by emcee uses multiprocess and pickeling
the function that is being evaluated. Things become a bit complicated 
because you can't pickle instance methods.
Steven Bethard has a solution registering a method with pickle:
http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods#edit2155350

Another simple solution is to create a plain method linked to a static method.
http://stackoverflow.com/questions/21111106/cant-pickle-static-method-multiprocessing-python
"""
#import copy_reg
#import types
# 
#def _pickle_method(method):
#    """
#    Author: Steven Bethard (author of argparse)
#    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
#    """
#    func_name = method.im_func.__name__
#    obj = method.im_self
#    cls = method.im_class
#    cls_name = ''
#    if func_name.startswith('__') and not func_name.endswith('__'):
#        cls_name = cls.__name__.lstrip('_')
#    if cls_name:
#        func_name = '_' + cls_name + func_name
#    return _unpickle_method, (func_name, obj, cls)
# 
# 
#def _unpickle_method(func_name, obj, cls):
#    """
#    Author: Steven Bethard
#    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
#    """
#    for cls in cls.mro():
#        try:
#            func = cls.__dict__[func_name]
#        except KeyError:
#            pass
#        else:
#            break
#    return func.__get__(obj, cls)
# 
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
 

class MCMC(object):
    """
    This object needs to:
    - Create the log-likelihood function
    - Create the log-probability in a way that emcee can handle
    - Run emcee
    """
    def __init__(self, config, coords):
        self.scan = ugali.analysis.scan.Scan(config, coords)
        self.config = self.scan.config
        self.scan.run()
        self.grid = self.scan.grid
        self.loglike = self.grid.loglike
        self.roi = self.loglike.roi

        self.nwalkers = self.config['mcmc']['nwalkers']
        self.nsamples = self.config['mcmc']['nsamples'] 
        self.nthreads = self.config['mcmc']['nthreads'] 

    def get_mle(self):
        return self.grid.mle()

    def get_std(self):
        mle = self.get_mle()

        std = odict([
            ('richness',0.1*mle['richness']), # delta_r = 10% of max richness 
            ('lon',0.01),                     # delta_l = 0.01 deg            
            ('lat',0.01),                     # delta_b = 0.01 deg            
            ('distance_modulus',0.1),         # delta_mu = 0.1                
            ('extension',0.01),               # delta_ext = 0.01 deg
        ])
        return std

    def get_ball(self, params, size=1):
        mle = self.get_mle() 
        std = self.get_std()
                                             
        p0 = np.array([mle[k] for k in params])
        s0 = np.array([std[k] for k in params])

        return emcee.utils.sample_ball(p0,s0,size)

    def lnlike(self, theta):
        kwargs = dict(zip(self.params,theta))
        try:
            return self.loglike.value(**kwargs)
        except ValueError,AssertionError:
            return -np.inf
        
    def lnprior(self,theta):
        # Nothing to see here...
        # but don't forget that the probability is the lkhd x prior
        kwargs = dict(zip(self.params,theta))
        return 0

    def run(self, params):
        logger.info("Setting inital values...")

        # Initailize the likelihood to maximal value
        mle =self.grid.mle()
        self.loglike.set_params(**mle)
        self.loglike.sync_params()

        self.params = params
 
        nwalkers = self.nwalkers
        nsamples = self.nsamples
        nthreads = self.nthreads
        ndim = len(params)
        
        logger.info("Running MCMC chain...")
        p0 = self.get_ball(params,nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads=nthreads)
        self.sampler.run_mcmc(p0,nsamples)

        # Chain is shape (nwalkers,nsteps,nparams)
        self.chain = mcmc.sampler.chain
        # Samples is shape (nwalkers*nsteps,nparams):
        # chain[walker][step] == samples[walker+nwalkers*step]
        samples = self.chain.reshape(-1,len(params),order='F')
        ## Non-optimal conversion...
        #self.samples = np.core.records.fromrecords(samples,names=self.params)
        # Faster...
        self.samples = np.core.records.fromarrays(samples.T,names=self.params)

    def estimate(self,param,burn=None,sigma=5.0):
        if param not in self.loglike.params.keys():
            raise Exception('Unrecognized parameter: %s'%param)

        if param not in self.params: 
            mle = self.get_mle()
            return [mle[param],0]

        if burn is None: burn = self.config['mcmc']['nburn']
        burn *= self.nwalkers
        clip = scipy.stats.sigmaclip(self.samples[param][burn:],sigma,sigma)
        mu,sigma = scipy.stats.norm.fit(clip[0])
        return [float(mu), float(sigma)]

    def estimate_params(self,burn=None,sigma=5.0):
        mle = self.get_mle()
        out = odict()
        for param in mle.keys():
            out[param] = self.estimate(param,burn,sigma)
        return out

    def save_samples(self,filename):
        np.save(filename,self.samples)

    def load_samples(self,filename):
        self.samples = np.load(filename)

    def save_results(self,filename):
        output = self.estimate_params()
        out = open(filename,'w')
        out.write(yaml.dump(dict(output)))
        out.close()

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for running MCMC followup"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True)
    parser.add_argument('outfile',default=None,help="Output file name")
                        
    opts = parser.parse_args()

    mcmc = MCMC(opts.config,opts.coords)

    niter = 0
    def lnprob(theta):
        global niter
        if (niter%100==0):
            logger.info("%i function calls ..."%niter)
        niter+=1
        lp = mcmc.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + mcmc.lnlike(theta)

    mcmc.lnprob = lnprob

    params = mcmc.config['mcmc']['params']
    mcmc.run(params)

    mcmc.save_samples(opts.outfile)
    resfile = opts.outfile.replace('.npy','.dat')
    mcmc.save_results(resfile)

    memfile = opts.outfile.replace('.npy','.fits')
    est = mcmc.estimate_params()
    kwargs = {k:v[0] for k,v in est.items()}
    mcmc.loglike.set_params(**kwargs)
    mcmc.loglike.sync_params()
    mcmc.loglike.write_membership(memfile)

    #import matplotlib
    #matplotlib.use('Agg')
    #
    #import triangle
    #fig = triangle.corner(samples[50*mcmc.nwalkers:], labels=params)
    #fig.savefig('mcmc_l%.1f_b%.1f.png'%(mcmc.roi.lon,mcmc.roi.lat))
