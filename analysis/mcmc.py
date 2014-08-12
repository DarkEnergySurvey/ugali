#!/usr/bin/env python

"""
Run MCMC follow up on target parameters.
"""

import os
import sys

import numpy
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import pyfits
import healpy
import emcee

import ugali.analysis.scan
import ugali.utils.config

from ugali.utils.logger import logger
from ugali.utils.skymap import superpixel, subpixel


class MCMC:
    """
    """
    def __init__(self, config, pix):
        self.scan = ugali.analysis.scan.Scan(config,pix)
        self.scan.run()
        self.grid = self.scan.grid
        self.loglike = self.grid.loglike
        self.roi = self.loglike.roi
        self.niter = 0

    
    def lnlike(self, theta):
        richness,lon,lat,distance_modulus = theta
        try:
            self.loglike.set_params(richness=richness,
                               lon=lon,lat=lat,
                               distance_modulus=distance_modulus)
        except ValueError,AssertionError:
            return -np.inf
        self.loglike.sync_params()
        return self.loglike()

    def lnprior(self, theta):
        richness,lon,lat,distance_modulus = theta 
        if richness < 0 or not (15 < distance_modulus < 25):
            return -np.inf
        else:
            return 0

    def lnprob(self, theta):
        if (logger.level==logger.DEBUG) and (self.niter%10==0):
            logger.debug("%i function calls ..."%self.niter)
        self.niter+=1
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.lnlike(theta)

if __name__ == "__main__":
    import argparse
    description = "Script for running MCMC followup"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config',metavar='config.yaml',help="Configuration file.")
    parser.add_argument('pix',help="HEALPix pixel.",type=int)
    parser.add_argument('-v','--verbose',action='store_true')
    opts = parser.parse_args()
    if opts.verbose: logger.setLevel(logger.DEBUG)
    else:            logger.setLevel(logger.INFO)

    mcmc = MCMC(opts.config,opts.pix)

    grid = mcmc.grid
    roi = mcmc.roi

    a = grid.log_likelihood_sparse_array
    i,j = np.unravel_index(a.argmax(),a.shape)

    distance_modulus = grid.distance_modulus_array[i]
    lon = roi.centers_lon_target[j]
    lat = roi.centers_lat_target[j]
    richness = grid.richness_sparse_array[i][j]

    results = np.array([richness,lon,lat,distance_modulus])

    # Should come from config
    ndim,nwalkers,nsamples = 4,10,100

    # Need to constrain to the interior region...
    pos = emcee.utils.sample_ball(results, [10,0.01,0.01,0.1], nwalkers)
    #pos = [results + 1e-4*np.random.rand(ndim) for i in range(nwalkers)]

    print "Setting up sampler..."
    sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc.lnprob, threads=1)
    print "Sampling..."
    sampler.run_mcmc(pos,nsamples)

    print "Making triangle plot..."
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    import triangle
    fig = triangle.corner(samples, labels=["$r$", "$l$", "$b$", "$\mu$"])
    fig.savefig('mcmc_segue_1.png')
