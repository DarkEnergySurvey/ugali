#!/usr/bin/env python
"""
Calculate output results dictionary.
"""

from collections import OrderedDict as odict

import numpy as np
import yaml
import numpy.lib.recfunctions as recfuncs

import astropy.coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u

import ugali.analysis.source
import ugali.analysis.loglike

import ugali.utils.stats
from ugali.utils.stats import Samples
from ugali.utils.projector import dist2mod,mod2dist,gal2cel,gal2cel_angle
from ugali.utils.projector import ang2const, ang2iau
from ugali.utils.config import Config
from ugali.utils.logger import logger

class Results(object):
    """
    Calculate results from a MCMC chain.
    """
    def __init__(self, config, loglike, samples=None):
        self.config = Config(config)
        self.alpha = self.config['results'].get('alpha',0.10)
        self.nwalkers = self.config['mcmc'].get('nwalkers',100)
        self.nburn = self.config['results'].get('nburn',10)
        self.coordsys = self.config['coords']['coordsys'].lower()
        
        self.loglike = loglike
        self.source = self.loglike.source
        self.params = list(self.source.get_free_params().keys())
        self.samples = samples

    def load_samples(self,filename):
        samples = Samples(filename)
        self.samples = samples.supplement(coordsys=self.coordsys)


    def get_mle(self):
        mle = self.source.get_params()
        # FIXME: For composite isochrones
        if 'age' not in mle:
            mle['age'] = np.average(self.source.isochrone.age)
        if 'metallicity' not in mle:
            mle['metallicity'] = np.average(self.source.isochrone.metallicity)
            
        return mle

    def estimate(self,param,burn=None,clip=10.0,alpha=0.32):
        """ Estimate parameter value and uncertainties """
        # FIXME: Need to add age and metallicity to composite isochrone params (currently properties)
        if param not in list(self.samples.names) + list(self.source.params) + ['age','metallicity']:
            msg = 'Unrecognized parameter: %s'%param
            raise KeyError(msg)

        # If the parameter is in the samples
        if param in self.samples.names:
            if param.startswith('position_angle'):
                return self.estimate_position_angle(param,burn=burn,
                                                    clip=clip,alpha=alpha)

            return self.samples.peak_interval(param,burn=burn,clip=clip,alpha=alpha)
 
        mle = self.get_mle()
        errors = [np.nan,np.nan] 

        # Set default value to the MLE value
        if param in self.source.params:
            err = self.source.params[param].errors
            if err is not None: errors = err

        # For age and metallicity from composite isochrone
        return [float(mle[param]),errors]

        ### if (param not in self.params) or (param not in : 
        ###     return [float(mle[param]),errors]
        ###  
        ### if param not in self.samples.names: 
        ###     return [float(mle[param]),errors]
        ###  
        ### msg = "Unrecognized parameter: %s"%param
        ### raise KeyError(msg)
 
    def estimate_params(self,burn=None,clip=10.0,alpha=0.32):
        """ Estimate all source parameters """
        mle = self.get_mle()
        out = odict()
        for param in mle.keys():
            out[param] = self.estimate(param,burn=burn,clip=clip,alpha=alpha)
        return out
 
    def estimate_position_angle(self,param='position_angle',burn=None,clip=10.0,alpha=0.32):
        """ Estimate the position angle from the posterior dealing
        with periodicity.
        """
        # Transform so peak in the middle of the distribution
        pa = self.samples.get(param,burn=burn,clip=clip)
        peak = ugali.utils.stats.kde_peak(pa)
        shift = 180.*((pa+90-peak)>180)
        pa -= shift
        # Get the kde interval
        ret = ugali.utils.stats.peak_interval(pa,alpha)
        if ret[0] < 0: 
            ret[0] += 180.; ret[1][0] += 180.; ret[1][1] += 180.;
        return ret
 
    def bayes_factor(self,param,burn=None,clip=10.0,bins=50):
        # CAREFUL: Assumes a flat prior...
        try: 
            data = self.samples.get(param,burn=burn,clip=clip)
        except ValueError as msg:
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
        kwargs.setdefault('alpha',self.alpha)
        kwargs.setdefault('burn',self.nburn*self.nwalkers)

        # Calculate best-fit parameters from MCMC chain
        logger.debug('Estimating parameters...')
        estimate = self.estimate_params(**kwargs)
        params = {k:v[0] for k,v in estimate.items()}
        results = dict(estimate)

        # Extra parameters from the MCMC chain
        logger.debug('Estimating auxiliary parameters...')
        try: 
            results['ra']  = self.estimate('ra',**kwargs)
            results['dec'] = self.estimate('dec',**kwargs)
        except KeyError:
            logger.warn("Didn't find 'ra' or 'dec'")
            ra,dec = gal2cel(results['lon'][0],results['lat'][0])
            results['ra'] = ugali.utils.stats.interval(ra)
            results['dec'] = ugali.utils.stats.interval(dec)

        ra,dec = results['ra'][0],results['dec'][0]
        glon,glat = lon,lat = results['lon'][0],results['lat'][0]
        results.update(gal=[float(glon),float(glat)])
        results.update(cel=[float(ra),float(dec)])

        try:
            results['position_angle_cel']  = self.estimate('position_angle_cel',**kwargs)
        except KeyError:
            results['position_angle_cel'] = ugali.utils.stats.interval(np.nan)

        # Update the loglike to the best-fit parameters from the chain
        logger.debug('Calculating TS...')
        ts = 2*self.loglike.value(**params)
        results['ts'] = ugali.utils.stats.interval(ts,np.nan,np.nan)
 
        #lon,lat = estimate['lon'][0],estimate['lat'][0]
        # 
        #results.update(gal=[float(lon),float(lat)])
        #ra,dec = gal2cel(lon,lat)
        #results.update(cel=[float(ra),float(dec)])
        #results['ra'] = ugali.utils.stats.interval(ra,np.nan,np.nan)
        #results['dec'] = ugali.utils.stats.interval(dec,np.nan,np.nan)
 
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
        ext_sigma = np.nan_to_num(np.array(ext_err) - ext)
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
        dist_sigma = np.nan_to_num(np.array(dist_err) - dist)

        size = np.arctan(np.radians(ext)) * dist
        size_sigma = size * np.sqrt((ext_sigma/ext)**2 + (dist_sigma/dist)**2)
        size_err = [size-size_sigma[0],size+size_sigma[1]]
        results['physical_size'] = ugali.utils.stats.interval(size,size_err[0],size_err[1])

        rsize = np.arctan(np.radians(rext)) * dist
        rsize_sigma = rsize * np.sqrt((rext_sigma/rext)**2 + (dist_sigma/dist)**2)
        rsize_err = [rsize-rsize_sigma[0],rsize+rsize_sigma[1]]
        results['physical_size_radial'] = ugali.utils.stats.interval(rsize,rsize_err[0],rsize_err[1])
 
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

        # Absolute magnitude only calculated for DES isochrones with g,r 
        try:
            Mv = self.source.absolute_magnitude(rich)
            Mv_lo = self.source.absolute_magnitude(rich_err[0])
            Mv_hi = self.source.absolute_magnitude(rich_err[1])
            results['Mv'] = ugali.utils.stats.interval(Mv,Mv_lo,Mv_hi)
        except ValueError as e:
            logger.warning("Skipping absolute magnitude")
            logger.warn(str(e))
            Mv = np.nan
            results['Mv'] = Mv

        # ADW: WARNING this is very fragile.
        # Also, this is not quite right, should cut on the CMD available space
        kwargs = dict(richness=rich,mag_bright=16., mag_faint=23.,
                      n_trials=5000,alpha=self.alpha, seed=0)
        martin = self.config['results'].get('martin')
        if martin:
            logger.info("Calculating Martin magnitude...")
            if martin > 1: kwargs['n_trials'] = martin
            Mv_martin = self.source.isochrone.absolute_magnitude_martin(**kwargs)
            results['Mv_martin'] = Mv_martin
        else:
            logger.warning("Skipping Martin magnitude")
            results['Mv_martin'] = np.nan
        
        mu = surfaceBrightness(Mv, size, dist)
        results['surface_brightness'] = ugali.utils.stats.interval(mu,np.nan,np.nan)
 
        try: 
            results['constellation'] = ang2const(lon,lat,self.coordsys)[1]
        except:
            pass
        results['iau'] = ugali.utils.projector.ang2iau(lon,lat)
 
        coord = SkyCoord(ra*u.deg,dec*u.deg,distance=dist*u.kpc)
        results['ra_sex'] = str(coord.ra.to_string())
        results['dec_sex'] = str(coord.dec.to_string())
 
        # Calculate some separations from GC, LMC, SMC
        #NED coordinates with de Grisj distance
        LMC = SkyCoord(80.8939*u.deg,-69.7561*u.deg,distance=49.89*u.kpc)
        #NED coordinates with de Grisj distance
        SMC = SkyCoord(13.1866*u.deg,-72.8286*u.deg,distance=61.94*u.kpc)
        # GC from astropy?
        GC = SkyCoord(266.4168262*u.deg,-29.0077969*u.deg,distance=8.0*u.kpc)
         
        results['d_gc'] = coord.separation_3d(GC).value
        results['d_lmc'] = coord.separation_3d(LMC).value
        results['d_smc'] = coord.separation_3d(SMC).value

        try:
            results['feh'] = float(self.source.isochrone.feh)
        except:
            results['feh'] = np.nan
        
        output = dict()
        output['params'] = params
        output['results'] = results
        return output

    def write(self,filename):
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

def surfaceBrightness(abs_mag, r_physical, distance):
    """
    Compute the average surface brightness [mag arcsec^-2] within the half-light radius

    abs_mag = absolute magnitude [mag]
    r_physical = half-light radius [kpc] 
    distance = [kpc]

    The factor 2 in the c_v equation below account for half the luminosity 
    within the half-light radius. The 3600.**2 is conversion from deg^2 to arcsec^2

    c_v = 2.5 * np.log10(2.) + 2.5 * np.log10(np.pi * 3600.**2) = 19.78
    """
    r_angle = np.degrees(np.arctan(r_physical / distance))
    c_v = 19.78 # mag/arcsec^2
    return abs_mag + dist2mod(distance) + c_v + 2.5 * np.log10(r_angle**2)

def createResults(config,srcfile,section='source',samples=None):
    """ Create an MCMC instance """
    source = ugali.analysis.source.Source()
    source.load(srcfile,section=section)
    loglike = ugali.analysis.loglike.createLoglike(config,source)

    results = Results(config,loglike,samples)

    if samples is not None:
        results.load_samples(samples)

    return results

def write_results(filename,config,srcfile,samples):
    """ Package everything nicely """ 
    results = createResults(config,srcfile,samples=samples)
    results.write(filename)


if __name__ == "__main__":
    import ugali.utils.parser
    parser = ugali.utils.parser.Parser(description=__doc__)
    parser.add_config()
    parser.add_verbose()
    parser.add_argument('--srcmdl',required=True,
                        help='Source model file')
    parser.add_argument('--section',default='source',
                        help='Section of source file')
    parser.add_argument('--samples',required=True,
                        help='Posterior samples file')
    parser.add_argument('outfile',default=None,
                        help="Output file name")
                        
    args = parser.parse_args()

    #write_results(args.outfile,args.config,args.srcmdl,args.samples)
    results = createResults(args.config,args.srcmdl,samples=args.samples)
    results.write(args.outfile)

