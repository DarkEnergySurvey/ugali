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

We choose the second option here.
"""

import os
import sys
from collections import OrderedDict as odict

import numpy
import numpy as np
import numpy.lib.recfunctions as recfuncs
import scipy.stats

import pyfits
import emcee
import yaml

import ugali.utils.stats
import ugali.analysis.scan
from ugali.utils.config import Config
from ugali.analysis.kernel import kernelFactory
from ugali.analysis.loglike import createSource, createLoglike
from ugali.analysis.prior import UniformPrior, InversePrior

from ugali.utils.logger import logger
from ugali.utils.projector import dist2mod,mod2dist,gal2cel,gal2cel_angle

# Write a class to store priors

class MCMC(object):
    """
    This object needs to:
    - Create the log-likelihood function
    - Create the log-probability in a way that emcee can handle
    - Run emcee
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
        #return self.grid.mle()
        mle = self.source.get_params()
        # FIXME: For composite isochrones
        if 'age' not in mle:
            mle['age'] = np.average(self.source.isochrone.age)
        if 'metallicity' not in mle:
            mle['metallicity'] = np.average(self.source.isochrone.metallicity)
            
        return mle

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

    def estimate(self,param,burn=None,clip=10.0,alpha=0.32):
        # FIXME: Need to add age and metallicity to composite isochrone
        if param not in self.source.params.keys() + ['age','metallicity']:
            raise Exception('Unrecognized parameter: %s'%param)

        mle = self.get_mle()
        errors = [np.nan,np.nan] 

        if param in self.source.params:
            err = self.source.params[param].errors
            if err is not None: errors = err

        # For age an metallicity
        if param not in self.params: 
            return [float(mle[param]),errors]

        if param not in self.samples.names: 
            return [float(mle[param]),errors]

        if param == 'position_angle':
            return self.estimate_position_angle(burn=burn,clip=clip,alpha=alpha)

        #return self.samples.mean_interval(param,burn=burn,clip=clip,alpha=alpha)
        return self.samples.peak_interval(param,burn=burn,clip=clip,alpha=alpha)

    def estimate_params(self,burn=None,clip=10.0,alpha=0.32):
        mle = self.get_mle()
        out = odict()
        for param in mle.keys():
            out[param] = self.estimate(param,burn=burn,clip=clip,alpha=alpha)
        return out

    def estimate_position_angle(self,burn=None,clip=10.0,alpha=0.32):
        # Transform so peak in the middle of the distribution
        pa = self.samples.get('position_angle',burn=burn,clip=clip)
        peak = ugali.utils.stats.kde_peak(pa,samples=1000)
        pa -= 180.*((pa+90-peak)>180)
        ret = ugali.utils.stats.peak_interval(pa,alpha,samples=1000)
        if ret[0] < 0: 
            ret[0] += 180.; ret[1][0] += 180.; ret[1][1] += 180.;
        return ret

    def bayes_factor(self,param,burn=None,clip=10.0,bins=50):
        # CAREFUL: Assumes flat prior...
        try: 
            data = self.samples.get(param,burn=burn,clip=clip)
        except ValueError,msg:
            logger.warning(msg)
            return ugali.utils.stats.interval(np.nan)

        bmin,bmax = self.source.params[param].bounds
        bins = np.linspace(bmin,bmax,bins)
        n,b = np.histogram(data,bins=bins,normed=True)
        prior = 1.0/(bmax-bmin)
        posterior = n[0]
        # Excluding the null hypothesis
        bf = prior/posterior
        return ugali.utils.stats.interval(bf)

    def get_results(self,**kwargs):
        import astropy.coordinates
        kwargs = dict(alpha=self.alpha,burn=self.nburn*self.nwalkers)
        estimate = self.estimate_params(**kwargs)
        params = {k:v[0] for k,v in estimate.items()}
        results = dict(estimate)
        
        ts = 2*self.loglike.value(**params)
        results['ts'] = ugali.utils.stats.interval(ts,np.nan,np.nan)

        lon,lat = estimate['lon'][0],estimate['lat'][0]
        #coord = astropy.coordinates.SkyCoord(lon,lat,frame='galactic',unit='deg')

        results.update(gal=[float(lon),float(lat)])
        ra,dec = gal2cel(lon,lat)
        results.update(cel=[float(ra),float(dec)])
        results['ra'] = ugali.utils.stats.interval(ra,np.nan,np.nan)
        results['dec'] = ugali.utils.stats.interval(dec,np.nan,np.nan)

        # Celestial position angle
        # Break ambiguity in direction with '% 180.'
        pa,pa_err = results['position_angle']
        pa_cel = gal2cel_angle(lon,lat,pa) % 180.
        pa_cel_err = np.array(pa_err) - pa + pa_cel
        results['position_angle_cel'] = ugali.utils.stats.interval(pa_cel,pa_cel_err[0],pa_cel_err[1])
        
        mod,mod_err = estimate['distance_modulus']
        dist = mod2dist(mod)
        dist_lo,dist_hi = [mod2dist(mod_err[0]),mod2dist(mod_err[1])]
        results['distance'] = ugali.utils.stats.interval(dist,dist_lo,dist_hi)
        dist,dist_err = results['distance']

        ext,ext_err = estimate['extension']
        results['extension_arcmin'] = ugali.utils.stats.interval(60*ext,60*ext_err[0],60*ext_err[1])

        # Radially symmetric extension (correct for ellipticity).
        ell,ell_err = estimate['ellipticity']
        rext,rext_err = ext*np.sqrt(1-ell),np.array(ext_err)*np.sqrt(1-ell)
        rext_sigma = np.nan_to_num(np.array(rext_err) - rext)
        results['extension_radial'] = ugali.utils.stats.interval(rext,rext_err[0],rext_err[1])
        results['extension_radial_arcmin'] = ugali.utils.stats.interval(60*rext,60*rext_err[0],60*rext_err[1])

        # Bayes factor for ellipticity
        results['ellipticity_bayes_factor'] = self.bayes_factor('ellipticity',burn=kwargs['burn'])

        # Physical Size (should do this with the posteriors)
        # Radially symmetric
        size = np.arctan(np.radians(rext)) * dist
        dist_sigma = np.nan_to_num(np.array(dist_err) - dist)
        size_sigma = size * np.sqrt((rext_sigma/rext)**2 + (dist_sigma/dist)**2)
        size_err = [size-size_sigma[0],size+size_sigma[1]]
        results['physical_size'] = ugali.utils.stats.interval(size,size_err[0],size_err[1])

        # Richness
        rich,rich_err = estimate['richness']

        # Number of observed stars (sum of p-values)
        nobs = self.loglike.p.sum()
        nobs_lo,nobs_hi = nobs + np.sqrt(nobs)*np.array([-1,1])
        results['nobs'] = ugali.utils.stats.interval(nobs,nobs_lo,nobs_hi)

        # Number of predicted stars (pixelization effects?)
        npred = self.loglike.f*rich
        npred_lo,npred_hi = rich_err[0]*self.loglike.f,rich_err[1]*self.loglike.f
        results['npred'] = ugali.utils.stats.interval(npred,npred_lo,npred_hi)
        
        # Careful, depends on the isochrone...
        stellar_mass = self.source.stellar_mass()
        mass = rich*stellar_mass
        mass_lo,mass_hi = rich_err[0]*stellar_mass,rich_err[1]*stellar_mass
        results['mass'] = ugali.utils.stats.interval(mass,mass_lo,mass_hi)

        stellar_luminosity = self.source.stellar_luminosity()
        lum = rich*stellar_luminosity
        lum_lo,lum_hi = rich_err[0]*stellar_luminosity,rich_err[1]*stellar_luminosity
        results['luminosity'] = ugali.utils.stats.interval(lum,lum_lo,lum_hi)

        Mv = self.source.absolute_magnitude(rich)
        Mv_lo = self.source.absolute_magnitude(rich_err[0])
        Mv_hi = self.source.absolute_magnitude(rich_err[1])
        results['Mv'] = ugali.utils.stats.interval(Mv,Mv_lo,Mv_hi)

        # ADW: WARNING this is very fragile.
        # Also, this is not quite right, should cut on the CMD available space
        kwargs = dict(richness=rich,mag_bright=16., mag_faint=23.,
                      n_trials=5000,alpha=self.alpha, seed=0)
        if False:
            Mv_martin = self.source.isochrone.absolute_magnitude_martin(**kwargs)
            results['Mv_martin'] = Mv_martin
        else:
            logger.warning("Skipping Martin magnitude")
            results['Mv_martin'] = np.nan


        # Surface Brightness
        def surfaceBrightness(abs_mag, r_physical, distance):
            r_angle = np.degrees(np.arctan(r_physical / distance))
            # corrected to use radius
            c_v = 19.06 # mag/arcsec^2
            #c_v = 10.17 # mag/arcmin^2
            #c_v = 1.28  # mag/deg^2
            return abs_mag + dist2mod(distance) + c_v + 2.5 * np.log10(r_angle**2)
        mu = surfaceBrightness(Mv, size, dist)
        results['surface_brightness'] = ugali.utils.stats.interval(mu,np.nan,np.nan)

        try: 
            results['constellation'] = ugali.utils.projector.ang2const(lon,lat)[1]
        except:
            pass
        results['iau'] = ugali.utils.projector.ang2iau(lon,lat)

        output = dict()
        output['params'] = params
        output['results'] = results
        return output
        
    def write_samples(self,filename):
        np.save(filename,self.samples)

    def load_samples(self,filename):
        self.samples = Samples(filename)

    def write_results(self,filename):
        if self.samples is not None:
            results = dict(self.get_results())
            params  = dict(params=results.pop('params'))
        else:
            results = dict(results=dict())
            params = dict(params=dict())
        source = dict(source=self.source.todict())

        out = open(filename,'w')
        out.write(yaml.dump(params,default_flow_style=False))
        out.write(yaml.dump(results))
        out.write(yaml.dump(source))
        out.close()

    def load_srcmdl(self,filename,section='source'):
        self.source.load(filename,section)


#class Samples(np.ndarray):
class Samples(np.recarray):
    """
    Wrapper class for recarray to deal with MCMC samples.
    
    A nice summary of various bayesian credible intervals can be found here:
    http://www.sumsar.net/blog/2014/10/probable-points-and-credible-intervals-part-one/
    """
    _alpha = 0.10
    _nbins = 300

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

    @property
    def ndarray(self):
        # atleast_2d is for 
        if len(self.dtype) == 1:
            return np.expand_dims(self.view((float,len(self.dtype))),1)
        else:
            return self.view((float,len(self.dtype)))

    # ADW: Depricated for Samples.get
    #def data(self, name, burn=None, clip=None):
    #    # Remove zero entries
    #    zsel = ~np.all(self.ndarray==0,axis=1)
    #    # Remove burn entries
    #    bsel = np.zeros(len(self),dtype=bool)
    #    bsel[slice(burn,None)] = 1
    #    data = self[name][bsel&zsel]
    #    if clip is not None:
    #        data,low,high = scipy.stats.sigmaclip(data,clip,clip)   
    #    return data

    def get(self, names=None, burn=None, clip=None):
        if names is None: names = list(self.dtype.names)
        names = np.array(names,ndmin=1)

        missing = names[~np.in1d(names,self.dtype.names)]
        if len(missing):
            msg = "field(s) named %s not found"%(missing)
            print msg
            raise ValueError(msg)
        idx = np.where(np.in1d(self.dtype.names,names))[0]

        # Remove zero entries
        zsel = ~np.all(self.ndarray==0,axis=1)
        # Remove burn entries
        bsel = np.zeros(len(self),dtype=bool)
        bsel[slice(burn,None)] = 1

        data = self.ndarray[:,idx][bsel&zsel]
        if clip is not None:
            from astropy.stats import sigma_clip
            mask = sigma_clip(data,sig=clip,copy=False,axis=0).mask
            data = data[np.where(~mask.any(axis=1))]

        return data


    @classmethod
    def _interval(cls,best,lo,hi):
        """
        Pythonized interval for easy output to yaml
        """
        return ugali.utils.stats.interval(best,lo,hi)

    def mean(self, name, **kwargs):
        """
        Mean of the distribution.
        """
        return np.mean(self.get(name,**kwargs))

    def mean_interval(self, name, alpha=_alpha, **kwargs):
        """
        Interval assuming gaussian posterior.
        """
        data = self.get(name,**kwargs)
        return ugali.utils.stats.mean_interval(data,alpha)

    def median(self, name, **kwargs):
        """
        Median of the distribution.
        """
        data = self.get(name,**kwargs)
        return np.percentile(data,[50])

    def median_interval(self,name,alpha=_alpha, **kwargs):
        """
        Median including bayesian credible interval.
        """
        data = self.get(name,**kwargs)
        return ugali.utils.stats.median_interval(data,alpha)
        
    def peak(self, name, **kwargs):
        data = self.get(name,**kwargs)
        return ugali.utils.stats.peak(data,bins=self._nbins)

    def kde_peak(self, name, **kwargs):
        data = self.get(name,**kwargs)
        return ugali.utils.stats.kde_peak(data,samples=250)

    def kde(self, name, **kwargs):
        data = self.get(name,**kwargs)
        return ugali.utils.stats.kde(data,samples=250)

    def peak_interval(self, name, alpha=_alpha, **kwargs):
        data = self.get(name, **kwargs)
        return ugali.utils.stats.peak_interval(data,alpha,samples=250)

    def min_interval(self,name, alpha=_alpha, **kwargs):
        data = self.get(name, **kwargs)
        return ugali.utils.min_interval(data,alpha)

    def results(self, names=None, alpha=_alpha, mode='peak', **kwargs):
        if names is None: names = self.names
        ret = odict()
        for n in names:
            ret[n] = getattr(self,'%s_interval'%mode)(n, **kwargs)
        return ret

def createMCMC(config,srcfile,section='source',samples=None):
    """ Create an MCMC instance """
    source = ugali.analysis.source.Source()
    source.load(srcfile,section=section)
    loglike = ugali.analysis.loglike.createLoglike(config,source)

    mcmc = MCMC(config,loglike)
    if samples is not None:
        mcmc.load_samples(samples)

    return mcmc

def write_membership(config,srcfile,outfile):
    mcmc = createMCMC(config,srcfile)
    mcmc.loglike.write_membership(outfile)

def write_results(config,srcfile,samples,outfile):
    mcmc = createMCMC(config,srcfile,samples=samples)
    mcmc.write_results(outfile)

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
    srcfile = outfile.replace('.npy','.yaml')
    memfile = outfile.replace('.npy','.fits')
    
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
    mcmc.write_results(srcfile)

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
            msg += ', '.join('%s: %.3f'%(k,v) for k,v in zip(mcmc.params,theta))
            msg += '\nlog(like): %.3f, log(prior): %.3f'%(lnprior,lnlike)
            logger.debug(msg)
        niter+=1
        return lnprob
    mcmc.lnprob = lnprob

    #params = mcmc.config['mcmc']['params']
    params = source.get_free_params().keys()
    mcmc.run(params,opts.outfile)

    logger.info("Writing %s..."%srcfile)
    write_results(config,srcfile,outfile,srcfile)

    logger.info("Writing %s..."%memfile)
    write_membership(config,srcfile,memfile)

