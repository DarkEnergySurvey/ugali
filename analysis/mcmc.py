#!/usr/bin/env python

"""
Run MCMC follow up on target parameters.
"""

import os
import sys

import numpy
import numpy as np
import matplotlib
matplotlib.use('Agg')

import pyfits
import healpy
import emcee

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


    def get_mle(self, params):
        # Maximum likelihood from scan
        a = self.grid.log_likelihood_sparse_array
        j,k = np.unravel_index(a.argmax(),a.shape)

        ret = []
        for i,p in enumerate(params):
            if p == 'richness':
                ret += [self.grid.richness_sparse_array[j][k]]
            if p == 'lon':
                ret += [self.roi.pixels_target.lon[k]]
            if p == 'lat':
                ret += [self.roi.pixels_target.lat[k]]
            if p == 'distance_modulus':
                ret += [self.grid.distance_modulus_array[j]]
            if p == 'extension':
                ret += [self.loglike.kernel.extension()]
                        
        return np.array(ret)

    def get_ball(self, params, size=1):
        mle = self.get_mle(params)

        std = []
        for i,p in enumerate(params):
            if p == 'richness':
                std += [0.1*mle[i]] # delta_r = 10% of max richness
            if p == 'lon':
                std += [0.01]       # delta_l = 0.01 deg
            if p == 'lat':
                std += [0.01]       # delta_b = 0.01 deg
            if p == 'distance_modulus':
                std += [0.1]        # delta_mu = 0.1 
            if p == 'extension':
                std += [0.01]       # delta_ext = 0.01 deg

        return emcee.utils.sample_ball(mle,std,size)
        
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
        self.params = params
 
        ball = self.get_ball(params,size=self.nwalkers)
        nwalkers = self.nwalkers
        nsamples = self.nsamples
        nthreads = self.nthreads

        ndim = len(params)
        
        p0 = self.get_ball(params,nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads=nthreads)
        self.sampler.run_mcmc(p0,nsamples)

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for running MCMC followup"
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True)
    opts = parser.parse_args()

    #test = Test(opts.config,opts.coords)
    #test.run()

    mcmc = MCMC(opts.config,opts.coords)

    niter = 0
    def lnprob(theta):
        global niter
        if (logger.level==logger.DEBUG) and (niter%100==0):
            logger.debug("%i function calls ..."%niter)
        niter+=1
        lp = mcmc.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + mcmc.lnlike(theta)

    mcmc.lnprob = lnprob
    #params = ['lon','lat','extension']
    params = ['richness','lon','lat','extension']
    mcmc.run(params)

    chain = mcmc.sampler.chain
    samples = chain.reshape((-1, len(params)))
    np.save('mcmc_l%.1f_b%.1f.npy'%(mcmc.roi.lon,mcmc.roi.lat),samples)

    import triangle
    fig = triangle.corner(samples[50*mcmc.nwalkers:], labels=params)
    fig.savefig('mcmc_l%.1f_b%.1f.png'%(mcmc.roi.lon,mcmc.roi.lat))

