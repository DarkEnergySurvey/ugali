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
import emcee
import yaml

import ugali.analysis.scan
import ugali.utils.config
from ugali.analysis.kernel import kernelFactory
from ugali.analysis.loglike import createIsochrone

from ugali.utils.logger import logger
from ugali.utils.projector import mod2dist, gal2cel


"""
The default parellel execution by emcee uses multiprocess and pickeling
the function that is being evaluated. Things become a bit complicated 
because you can't pickle instance methods.
Steven Bethard has a solution registering a method with pickle:
http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods#edit2155350

Another simple solution is to create a plain method linked to a static method.
http://stackoverflow.com/questions/21111106/cant-pickle-static-method-multiprocessing-python
"""

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
        self.nwalkers = self.config['mcmc']['nwalkers']
        self.nsamples = self.config['mcmc']['nsamples'] 
        self.nthreads = self.config['mcmc']['nthreads'] 
        self.nburn = self.config['mcmc']['nburn']
        
        self.scan.run()
        self.grid = self.scan.grid
        self.loglike = self.grid.loglike

        if self.config['mcmc'].get('kernel') is not None:
            kernel = kernelFactory(**self.config['mcmc']['kernel'])
        else:
            kernel = kernelFactory(**self.config['kernel'])
        print kernel
        self.loglike.set_model('spatial',kernel)
            

        if self.config['mcmc'].get('isochrone') is not None:
            config = dict(self.config)
            config.update(isochrone=self.config['mcmc']['isochrone'])
            iso = createIsochrone(config)
        else:
            iso = createIsochrone(config)
        print iso
        self.loglike.set_model('color',iso)

        self.roi = self.loglike.roi

    def get_mle(self):
        return self.grid.mle()

    def get_std(self):
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
        kwargs = dict(zip(self.params,theta))
        try:
            #print kwargs,
            #print self.loglike.value(**kwargs)
            return self.loglike.value(**kwargs)
        except ValueError,AssertionError:
            return -np.inf
        
    def lnprior(self,theta):
        # Nothing to see here...
        # but don't forget that the probability is the lkhd x prior
        kwargs = dict(zip(self.params,theta))
        return 0

    def run(self, params):
        # Initailize the likelihood to maximal value
        mle =self.grid.mle()
        msg = "Setting inital values..."
        for k,v in mle.items():
            msg+='\n  %s : %s'%(k,v)
        logger.info(msg)

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
        self.samples = Samples(samples.T,names=self.params)

    def estimate(self,param,burn=None,clip=10.0):
        if param not in self.loglike.params.keys():
            raise Exception('Unrecognized parameter: %s'%param)

        if param not in self.params: 
            mle = self.get_mle()
            return [float(mle[param]),[0,0]]

        return self.samples.peak_interval(param)

    def estimate_params(self,burn=None,clip=10.0):
        mle = self.get_mle()
        out = odict()
        for param in mle.keys():
            out[param] = self.estimate(param,burn,clip)
        return out

    def get_results(self,**kwargs):
        estimate = self.estimate_params()
        params = {k:v[0] for k,v in estimate.items()}
        results = dict(estimate)
        
        ts = 2*self.loglike.value(**params)
        results['ts'] = [float(ts),0]

        lon,lat = estimate['lon'][0],estimate['lat'][0]

        results.update(gal=[float(lon),float(lat)])
        ra,dec = gal2cel(lon,lat)
        results.update(equ=[float(ra),float(dec)])

        mod,mod_err = estimate['distance_modulus']
        dist = mod2dist(mod)
        dist_lo,dist_hi = [mod2dist(mod_err[0]),mod2dist(mod_err[1])]
        results['distance'] = Samples._interval(dist,dist_lo,dist_hi)
        
        rich,rich_err = estimate['richness']

        # Careful, depends on the isochrone...
        stellar_mass = self.loglike.stellar_mass()
        stellar_luminosity = self.loglike.stellar_luminosity()

        mass = rich*stellar_mass
        mass_lo,mass_hi = rich_err[0]*stellar_mass,rich_err[1]*stellar_mass
        results['mass'] = Samples._interval(mass,mass_lo,mass_hi)

        lum = rich*stellar_luminosity
        lum_lo,lum_hi = rich_err[0]*stellar_luminosity,rich_err[1]*stellar_luminosity
        results['luminosity'] = Samples._interval(lum,lum_lo,lum_hi)

        output = dict()
        output['params'] = params
        output['results'] = results
        return output
        
    def write_samples(self,filename):
        np.save(filename,self.samples)

    def load_samples(self,filename):
        self.samples = Samples(filename)

    def write_results(self,filename):
        results = dict(self.get_results())
        params  = dict(params=results.pop('params'))

        out = open(filename,'w')
        out.write(yaml.dump(params,default_flow_style=False))
        out.write(yaml.dump(results))
        out.close()


#class Samples(np.ndarray):
class Samples(np.recarray):
    """
    Wrapper class for recarray to deal with MCMC samples.
    
    A nice summary of various bayesian credible intervals can be found here:
    http://www.sumsar.net/blog/2014/10/probable-points-and-credible-intervals-part-one/
    """
    _alpha = 0.32

    def __new__(cls, input, names=None):
        # Load the array from file
        if not isinstance(input,np.ndarray):
            obj = np.load(input).view(cls)
        else:
            obj = np.asarray(input).view(cls)
            
        # (re)set the column names
        if names is not None:
            if obj.dtype.names is None:
                obj = np.rec.fromarrays(obj,names=names).view(cls)
            else:
                obj.dtype.names = names

        return obj

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self,out_arr,context)

    @property
    def names(self):
        return self.dtype.names

    def data(self, name, burn=None, clip=None):
        burn = slice(burn,None)
        data = self[name][burn]
        if clip is not None:
            data,low,high = scipy.stats.sigmaclip(data,clip,clip)   
        return data

    @classmethod
    def _interval(cls,best,lo,hi):
        """
        Pythonized interval for easy output to yaml
        """
        
        return [float(best),[float(lo),float(hi)]]

    def mean(self, name, **kwargs):
        """
        Mean of the distribution.
        """
        return np.mean(self.data(name,**kwargs))

    def mean_interval(self, name, alpha=_alpha, **kwargs):
        """
        Inerval assuming gaussin posterior.
        """
        data = self.data(name,**kwargs)
        mean =np.mean(data)
        sigma = np.std(data)
        return self._interval(mean,mean-sigma,mean+sigma)

    def median(self, name, **kwargs):
        """
        Median of the distribution.
        """
        data = self.data(name,**kwargs)
        q = [50]
        return np.percentile(data,q)

    def median_interval(self,name,alpha=_alpha, **kwargs):
        """
        Median including bayesian credible interval.
        """
        data = self.data(name,**kwargs)
        q = [100*alpha/2., 50, 100*(1-alpha/2.)]
        lo,med,hi = numpy.percentile(data,q)
        return self._interval(med,lo,hi)
        
    def peak(self, name, **kwargs):
        data = self.data(name,**kwargs)
        num,edges = np.histogram(data,bins=100)
        centers = (edges[1:]+edges[:-1])/2.
        return centers[np.argmax(num)]

    def peak_interval(self, name, alpha=_alpha, **kwargs):
        data = self.data(name, **kwargs)
        peak = self.peak(name, **kwargs)
        x = np.sort(data); n = len(x)
        # The number of entries in the interval
        window = int(np.rint((1.0-alpha)*n))
        # The start, stop, and width of all possible intervals
        starts = x[:n-window]; ends = x[window:]
        widths = ends - starts
        # Just the intervals containing the peak
        select = (peak >= starts) & (peak <= ends)
        widths = widths[select]
        if len(widths) == 0:
            raise ValueError('Too few elements for interval calculation')
        min_idx = np.argmin(widths)
        lo = x[min_idx]
        hi = x[min_idx+window]
        return self._interval(peak,lo,hi)

    def min_interval(self,name, alpha=_alpha, **kwargs):
        data = self.data(name, **kwargs)
        x = np.sort(data); n = len(x)
        # The number of entries in the interval
        window = int(np.rint((1.0-alpha)*n))
        # The start, stop, and width of all possible intervals
        starts = x[:n-window]; ends = x[window:]
        widths = ends - starts
        if len(widths) == 0:
            raise ValueError('Too few elements for interval calculation')
        min_idx = np.argmin(widths)
        lo = x[min_idx]
        hi = x[min_idx+window]
        mean = (hi+lo)/2.
        return self._interval(mean,lo,hi)

    def results(self, names=None, alpha=_alpha, mode='peak', **kwargs):
        if names is None: names = self.names
        ret = odict()
        for n in names:
            ret[n] = getattr(self,'%s_interval'%mode)(n, **kwargs)
        return ret

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
    mcmc.write_samples(opts.outfile)

    estimate = mcmc.estimate_params()
    kwargs = {k:v[0] for k,v in estimate.items()}

    mcmc.loglike.set_params(**kwargs)
    mcmc.loglike.sync_params()

    membfile = opts.outfile.replace('.npy','.fits')
    mcmc.loglike.write_membership(membfile)

    resfile = opts.outfile.replace('.npy','.dat')
    mcmc.write_results(resfile)
