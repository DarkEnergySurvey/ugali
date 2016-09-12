"""
Object for isochrone storage and basic calculations.

NOTE: only absolute magnitudes are used in the Isochrone class

ADW: There are some issues involved here. As we are generally using a
forward-folding likelihood technique, what we would like to do is to
convolve the isochrone model with the survey response functions to
derive a model of the observed distribution of objects given a
specific true isochrone. This convolution involves two distinct parts:
1) the object completeness as a function of delta-magnitude
(difference between magnitude and local limiting magnitude), 2) the
magnitude dispersion (magnitude uncertainty) as a function of
delta-magnitude.

Since the survey response (i.e., depth) changes on the pixel scale,
this would means deriving the convolved isochrone for every pixel in
the interior of the ROI. Assuming a magnitude binning of 70x70, and a
set of 3000 interior pixels, this is a 70x70x3000 matrix.  However,
the issue is that to generate this array you need to sample the
isochrone at roughly 1000 points and sum these points.  Needless to
say it is fairly intensive to calculate and store this matrix. Things
become much more reasonable if you only calculate this matrix once for
each unique magnitude limit, but again this becomes difficult because
you need each unique limit in both magnitudes.

"""

import sys
import os
from abc import abstractmethod
import collections
from collections import OrderedDict as odict
import inspect
import glob
from functools import wraps

import numpy
import numpy as np
import scipy.interpolate
import scipy.stats
import scipy.spatial
import scipy.ndimage as ndimage

import ugali.analysis.imf
from ugali.analysis.model import Model, Parameter
from ugali.utils.stats import norm_cdf
from ugali.utils.shell import get_ugali_dir
from ugali.utils.projector import mod2dist

from ugali.utils.config import Config
from ugali.utils.logger import logger

############################################################

# ADW: Depricated (2016-06-25)
###if __name__.endswith('isochrone2'):
###    import warnings
###    warnings.simplefilter('module', DeprecationWarning)
###    msg = "isochrone2 is deprecated"
###    warnings.warn(msg,DeprecationWarning)
###    warnings.simplefilter('default', DeprecationWarning)

def get_iso_dir():
    isodir = os.path.join(get_ugali_dir(),'isochrones')
    if not os.path.exists(isodir):
        msg = "Isochrone directory not found:\n%s"%isodir
        logger.warning(msg)
    return isodir

class Isochrone(Model):
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
        ('age',              Parameter(10.0, [0.1, 15.0]) ),  # Gyr
        ('metallicity',      Parameter(0.0002, [0.0,0.02]) ),
    ])

    _mapping = odict([
        ('mod','distance_modulus'),
        ('a','age'),                 
        ('z','metallicity'),
    ])

    # ADW: Careful, there are weird things going on with adding
    # defaults to subclasses...  When converted to a dict, the
    # last duplicate entry is filled.
    defaults = (
        ('survey','des','Name of survey filter system'),
        ('band_1','g','Field name for magnitude one'),
        ('band_2','r','Field name for magnitude two'),
        ('band_1_detection',True,'Band one is detection band'),
        ('imf_type','chabrier','Initial mass function'),
        ('hb_stage',None,'Horizontal branch stage name'),
        ('hb_spread',0.0,'Intrinisic spread added to horizontal branch'),
        )
    
    def __init__(self, **kwargs):
        self._setup(**kwargs)
        super(Isochrone,self).__init__(**kwargs)

    def _setup(self, **kwargs):
        defaults = odict([(d[0],d[1]) for d in self.defaults])
        [defaults.update([i]) for i in kwargs.items() if i[0] in defaults]

        for k,v in defaults.items():
            setattr(self,k,v)

        self.imf = ugali.analysis.imf.IMF(defaults['imf_type'])
        self.index = None

    def _parse(self,filename):
        msg = "Not implemented for base class"
        raise Exception(msg)

    def todict(self):
        ret = super(Isochrone,self).todict()
        defaults = odict([(d[0],d[1]) for d in self.defaults])
        for k,v in defaults.items():
            if getattr(self,k) != v: ret[k] = getattr(self,k)
        return ret

    @property
    def feh(self):
        #### For use with Marigo et al. (2008) and earlier
        ###metallicity_solar = 0.019 # Anders & Grevesse 1989
        # For use with Bressan et al. (2012) and later
        metallicity_solar = 0.0152 
        return np.log10(self.metallicity / metallicity_solar)

    @property
    def distance(self):
        """ Convert to physical distance (kpc) """
        return mod2dist(self.distance_modulus)

    def sample(self, mode='data', mass_steps=1000, mass_min=0.1, full_data_range=False):
        """Sample the isochrone in steps of mass interpolating between the
        originally defined isochrone points.
        """

        if full_data_range:
            # ADW: Might be depricated 02/10/2015
            # Generate points over full isochrone data range
            select = slice(None)
        else:
            # Not generating points for the post-AGB stars,
            # but still count those stars towards the normalization
            select = slice(self.index)
            
        mass_init = self.mass_init[select]
        mass_act = self.mass_act[select]
        mag_1 = self.mag_1[select]
        mag_2 = self.mag_2[select]
        
        # ADW: Assume that the isochrones are pre-sorted by mass_init
        # This avoids some numerical instability from points that have the same
        # mass_init value (discontinuities in the isochrone).
        # ADW: Might consider using numpy.interp for speed
        mass_act_interpolation = scipy.interpolate.interp1d(mass_init, mass_act,assume_sorted=True)
        mag_1_interpolation = scipy.interpolate.interp1d(mass_init, mag_1,assume_sorted=True)
        mag_2_interpolation = scipy.interpolate.interp1d(mass_init, mag_2,assume_sorted=True)

        # ADW: Any other modes possible?
        if mode=='data':
            # Mass interpolation with uniform coverage between data points from isochrone file 
            mass_interpolation = scipy.interpolate.interp1d(range(0, len(mass_init)), mass_init)
            mass_array = mass_interpolation(np.linspace(0, len(mass_init) - 1, mass_steps + 1))
            d_mass = mass_array[1:] - mass_array[0:-1]
            mass_init_array = np.sqrt(mass_array[1:] * mass_array[0:-1])
            mass_pdf_array = d_mass * self.imf.pdf(mass_init_array, log_mode = False)
            mass_act_array = mass_act_interpolation(mass_init_array)
            mag_1_array = mag_1_interpolation(mass_init_array)
            mag_2_array = mag_2_interpolation(mass_init_array)
            
        # Horizontal branch dispersion
        if self.hb_spread and (self.stage==self.hb_stage).any():
            logger.debug("Performing dispersion of horizontal branch...")
            mass_init_min = self.mass_init[self.stage==self.hb_stage].min()
            mass_init_max = self.mass_init[self.stage==self.hb_stage].max()
            cut = (mass_init_array>mass_init_min)&(mass_init_array<mass_init_max)
            if isinstance(self.hb_spread,collections.Iterable):
                # Explicit dispersion spacing
                dispersion_array = self.hb_spread
                n = len(dispersion_array)
            else:
                # Default dispersion spacing
                dispersion = self.hb_spread
                spacing = 0.025
                n = int(round(2.0*self.hb_spread/spacing))
                if n % 2 != 1: n += 1
                dispersion_array = np.linspace(-dispersion, dispersion, n)

            # Reset original values
            mass_pdf_array[cut] = mass_pdf_array[cut] / float(n)

            # Isochrone values for points on the HB
            mass_init_hb = mass_init_array[cut]
            mass_pdf_hb = mass_pdf_array[cut]
            mass_act_hb = mass_act_array[cut]
            mag_1_hb = mag_1_array[cut]
            mag_2_hb = mag_2_array[cut]

            # Add dispersed values
            for dispersion in dispersion_array:
                if dispersion == 0.: continue
                msg = 'Dispersion=%-.4g, HB Points=%i, Iso Points=%i'%(dispersion,cut.sum(),len(mass_init_array))
                logger.debug(msg)

                mass_init_array = np.append(mass_init_array, mass_init_hb) 
                mass_pdf_array = np.append(mass_pdf_array, mass_pdf_hb)
                mass_act_array = np.append(mass_act_array, mass_act_hb) 
                mag_1_array = np.append(mag_1_array, mag_1_hb + dispersion)
                mag_2_array = np.append(mag_2_array, mag_2_hb + dispersion)

        # Note that the mass_pdf_array is not generally normalized to unity
        # since the isochrone data range typically covers a different range
        # of initial masses
        #mass_pdf_array /= np.sum(mass_pdf_array) # ORIGINAL
        # Normalize to the number of stars in the satellite with mass > mass_min
        mass_pdf_array /= self.imf.integrate(mass_min, self.mass_init_upper_bound)
        out = np.vstack([mass_init_array,mass_pdf_array,mass_act_array,mag_1_array,mag_2_array])
        return out

    def stellar_mass(self, mass_min=0.1, steps=10000):
        """
        Compute the stellar mass (M_Sol; average per star). PDF comes from IMF, but weight by actual stellar mass.
        """
        mass_max = self.mass_init_upper_bound
            
        d_log_mass = (np.log10(mass_max) - np.log10(mass_min)) / float(steps)
        log_mass = np.linspace(np.log10(mass_min), np.log10(mass_max), steps)
        mass = 10.**log_mass

        if mass_min < np.min(self.mass_init):
            mass_act_interpolation = scipy.interpolate.interp1d(np.insert(self.mass_init, 0, mass_min),
                                                                np.insert(self.mass_act, 0, mass_min))
        else:
           mass_act_interpolation = scipy.interpolate.interp1d(self.mass_init, self.mass_act) 

        mass_act = mass_act_interpolation(mass)
        return np.sum(mass_act * d_log_mass * self.imf.pdf(mass, log_mode=True))

    def stellar_luminosity(self, steps=10000):
        """
        Compute the stellar luminosity (L_Sol; average per star). PDF comes from IMF.
        The range of integration only covers the input isochrone data (no extrapolation used),
        but this seems like a sub-percent effect if the isochrone goes to 0.15 stellar masses for the
        old and metal-poor stellar populations of interest.

        Note that the stellar luminosity is very sensitive to the post-AGB population.
        """
        mass_min = np.min(self.mass_init)
        mass_max = self.mass_init_upper_bound
        
        d_log_mass = (np.log10(mass_max) - np.log10(mass_min)) / float(steps)
        log_mass = np.linspace(np.log10(mass_min), np.log10(mass_max), steps)
        mass = 10.**log_mass
        
        luminosity_interpolation = scipy.interpolate.interp1d(self.mass_init, self.luminosity,fill_value=0,bounds_error=False)
        luminosity = luminosity_interpolation(mass)

        return np.sum(luminosity * d_log_mass * self.imf.pdf(mass, log_mode=True))

    def stellar_luminosity2(self, steps=10000):
        """
        Compute the stellar luminosity (L_Sol; average per star). 
        Usese "sample" to generate mass samplint and pdf.
        The range of integration only covers the input isochrone data (no extrapolation used),
        but this seems like a sub-percent effect if the isochrone goes to 0.15 stellar masses for the
        old and metal-poor stellar populations of interest.

        Note that the stellar luminosity is very sensitive to the post-AGB population.
        """
        mass_init, mass_pdf, mass_act, mag_1, mag_2 = self.sample(mass_steps=steps)
        luminosity_interpolation = scipy.interpolate.interp1d(self.mass_init, self.luminosity,fill_value=0,bounds_error=False)
        luminosity = luminosity_interpolation(mass_init)
        return np.sum(luminosity * mass_pdf)

    # ADW: For temporary backward compatibility
    stellarMass = stellar_mass
    stellarLuminosity = stellar_luminosity

    def absolute_magnitude(self, richness=1, steps=1e4):
        # Using the SDSS g,r -> V from Jester 2005 [arXiv:0506022]
        # for stars with R-I < 1.15
        # V = g_sdss - 0.59(g_sdss-r_sdss) - 0.01
        # g_des = g_sdss - 0.104(g_sdss - r_sdss) + 0.01
        # r_des = r_sdss - 0.102(g_sdss - r_sdss) + 0.02
        if self.survey.lower() != 'des':
            raise Exception('Only valid for DES')
        if 'g' not in [self.band_1,self.band_2]:
            msg = "Need g-band for absolute magnitude"
            raise Exception(msg)    
        if 'r' not in [self.band_1,self.band_2]:
            msg = "Need r-band for absolute magnitude"
            raise Exception(msg)    

        mass_init,mass_pdf,mass_act,mag_1,mag_2=self.sample(mass_steps = steps)
        g,r = (mag_1,mag_2) if self.band_1 == 'g' else (mag_2,mag_1)
        
        V = g - 0.487*(g - r) - 0.0249
        flux = np.sum(mass_pdf*10**(-V/2.5))
        Mv = -2.5*np.log10(richness*flux)
        return Mv

    def absolute_magnitude_martin(self, richness=1, steps=1e4, n_trials=1000, mag_bright=16., mag_faint=23., alpha=0.32, seed=None):
        # ADW: This function is not quite right. You should be restricting
        # the catalog to the obsevable space (using the function named as such)
        # Also, this needs to be applied in each pixel individually
        
        # Using the SDSS g,r -> V from Jester 2005 [arXiv:0506022]
        # for stars with R-I < 1.15
        # V = g_sdss - 0.59(g_sdss-r_sdss) - 0.01
        # g_des = g_sdss - 0.104(g_sdss - r_sdss) + 0.01
        # r_des = r_sdss - 0.102(g_sdss - r_sdss) + 0.02
        np.random.seed(seed)

        if self.survey.lower() != 'des':
            raise Exception('Only valid for DES')
        if 'g' not in [self.band_1,self.band_2]:
            msg = "Need g-band for absolute magnitude"
            raise Exception(msg)    
        if 'r' not in [self.band_1,self.band_2]:
            msg = "Need r-band for absolute magnitude"
            raise Exception(msg)    
        
        def visual(g, r, pdf=None):
            v = g - 0.487 * (g - r) - 0.0249
            if pdf is None:
                flux = np.sum(10**(-v / 2.5))
            else:
                flux = np.sum(pdf * 10**(-v / 2.5))
            abs_mag_v = -2.5 * np.log10(flux)
            return abs_mag_v

        def sumMag(mag_1, mag_2):
            flux_1 = 10**(-mag_1 / 2.5)
            flux_2 = 10**(-mag_2 / 2.5)
            return -2.5 * np.log10(flux_1 + flux_2)

        # Analytic part
        mass_init, mass_pdf, mass_act, mag_1, mag_2 = self.sample(mass_steps = steps)
        g,r = (mag_1,mag_2) if self.band_1 == 'g' else (mag_2,mag_1)
        #cut = numpy.logical_not((g > mag_bright) & (g < mag_faint) & (r > mag_bright) & (r < mag_faint))
        cut = ((g + self.distance_modulus) > mag_faint) if self.band_1 == 'g' else ((r + self.distance_modulus) > mag_faint)
        mag_unobs = visual(g[cut], r[cut], richness * mass_pdf[cut])

        # Stochastic part
        abs_mag_obs_array = numpy.zeros(n_trials)
        for ii in range(0, n_trials):
            if ii%100==0: logger.debug('%i absolute magnitude trials'%ii)
            g, r = self.simulate(richness * self.stellar_mass())
            #cut = (g > 16.) & (g < 23.) & (r > 16.) & (r < 23.)
            cut = (g < mag_faint) if self.band_1 == 'g' else (r < mag_faint)
            mag_obs = visual(g[cut] - self.distance_modulus, r[cut] - self.distance_modulus)
            abs_mag_obs_array[ii] = sumMag(mag_obs, mag_unobs)

        # ADW: This shouldn't be necessary
        #abs_mag_obs_array = numpy.sort(abs_mag_obs_array)[::-1]

        # ADW: Careful, fainter abs mag is larger (less negative) number
        q = [100*alpha/2., 50, 100*(1-alpha/2.)]
        hi,med,lo = numpy.percentile(abs_mag_obs_array,q)
        return ugali.utils.stats.interval(med,lo,hi)

    def simulate(self, stellar_mass, distance_modulus=None, **kwargs):
        """
        Simulate observed magnitudes for satellite of given mass and distance.
        """
        if distance_modulus is None: distance_modulus = self.distance_modulus
        # Total number of stars in system
        # ADW: is this the predicted number or the observed number?
        n = int(stellar_mass/self.stellar_mass()) 
        mass_init, mass_pdf, mass_act, mag_1, mag_2 = self.sample(**kwargs)
        
        ## ADW: This assumes that everything is sorted by increasing mass
        #mag_1, mag_2 = mag_1[::-1],mag_2[::-1]
        #mass_pdf[::-1]

        cdf = np.cumsum(mass_pdf[::-1])
        cdf = np.insert(cdf, 0, 0.)

        #mode='data', mass_steps=1000, mass_min=0.1, full_data_range=False
        #ADW: CDF is *not* normalized (because of minimum mass)
        f = scipy.interpolate.interp1d(cdf, range(0, len(cdf)), bounds_error=False, fill_value=-1)
        index = np.floor(f(np.random.uniform(size=n))).astype(int)
        #print "WARNING: non-random isochrone simulation"
        #index = np.floor(f(np.linspace(0,1,n))).astype(int)
        index = index[index >= 0]
        return mag_1[::-1][index]+distance_modulus, mag_2[::-1][index]+distance_modulus

    def observableFractionCMDX(self, mask, distance_modulus, mass_min=0.1):
        """
        Compute observable fraction of stars with masses greater than mass_min in each 
        pixel in the interior region of the mask.

        ADW: Careful, this function is fragile! The selection here should
             be the same as mask.restrictCatalogToObservable space. However,
             for technical reasons it is faster to do the calculation with
             broadcasting here.
        ADW: Could this function be even faster / more readable?
        ADW: Should this include magnitude error leakage?
        """
        mass_init_array,mass_pdf_array,mass_act_array,mag_1_array,mag_2_array = self.sample(mass_min=mass_min,full_data_range=False)
        mag = mag_1_array if self.band_1_detection else mag_2_array
        color = mag_1_array - mag_2_array

        # ADW: Only calculate observable fraction over interior pixels...
        pixels = mask.roi.pixels_interior
        mag_1_mask = mask.mask_1.mask_roi_sparse[mask.roi.pixel_interior_cut]
        mag_2_mask = mask.mask_2.mask_roi_sparse[mask.roi.pixel_interior_cut]

        # ADW: Restrict mag and color to range of mask with sufficient solid angle
        cmd_cut = ugali.utils.binning.take2D(mask.solid_angle_cmd,color,mag+distance_modulus,
                                             mask.roi.bins_color, mask.roi.bins_mag) > 0
        # Pre-apply these cuts to the 1D mass_pdf_array to save time
        mass_pdf_cut = mass_pdf_array*cmd_cut

        # Create 2D arrays of cuts for each pixel
        mask_1_cut = (mag_1_array+distance_modulus)[:,np.newaxis] < mag_1_mask
        mask_2_cut = (mag_2_array+distance_modulus)[:,np.newaxis] < mag_2_mask
        mask_cut_repeat = mask_1_cut & mask_2_cut

        observable_fraction = (mass_pdf_cut[:,np.newaxis]*mask_cut_repeat).sum(axis=0)
        return observable_fraction

    def observableFractionCMD(self, mask, distance_modulus, mass_min=0.1):
        """
        Compute observable fraction of stars with masses greater than mass_min in each 
        pixel in the interior region of the mask.

        ADW: Careful, this function is fragile! The selection here should
             be the same as mask.restrictCatalogToObservable space. However,
             for technical reasons it is faster to do the calculation with
             broadcasting here.
        ADW: Could this function be even faster / more readable?
        ADW: Should this include magnitude error leakage?
        """
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = self.sample(mass_min=mass_min,full_data_range=False)

        mag = mag_1 if self.band_1_detection else mag_2
        color = mag_1 - mag_2

        # ADW: Only calculate observable fraction for unique mask values
        mag_1_mask,mag_2_mask = mask.mask_roi_unique.T

        # ADW: Restrict mag and color to range of mask with sufficient solid angle
        cmd_cut = ugali.utils.binning.take2D(mask.solid_angle_cmd,color,mag+distance_modulus,
                                             mask.roi.bins_color, mask.roi.bins_mag) > 0
        # Pre-apply these cuts to the 1D mass_pdf_array to save time
        mass_pdf_cut = mass_pdf*cmd_cut

        # Create 2D arrays of cuts for each pixel
        mask_1_cut = (mag_1+distance_modulus)[:,np.newaxis] < mag_1_mask
        mask_2_cut = (mag_2+distance_modulus)[:,np.newaxis] < mag_2_mask
        mask_cut_repeat = mask_1_cut & mask_2_cut

        observable_fraction = (mass_pdf_cut[:,np.newaxis]*mask_cut_repeat).sum(axis=0)
        return observable_fraction[mask.mask_roi_digi[mask.roi.pixel_interior_cut]]


    def observableFractionCDF(self, mask, distance_modulus, mass_min=0.1):
        """
        Compute observable fraction of stars with masses greater than mass_min in each 
        pixel in the interior region of the mask. Incorporates simplistic
        photometric errors.

        ADW: Careful, this function is fragile! The selection here should
             be the same as mask.restrictCatalogToObservable space. However,
             for technical reasons it is faster to do the calculation with
             broadcasting here.
        ADW: This function is currently a rate-limiting step in the likelihood 
             calculation. Could it be faster?
        """
        method = 'step'

        mass_init,mass_pdf,mass_act,mag_1,mag_2 = self.sample(mass_min=mass_min,full_data_range=False)
         
        mag_1 = mag_1+distance_modulus
        mag_2 = mag_2+distance_modulus
         
        mask_1,mask_2 = mask.mask_roi_unique.T
         
        mag_err_1 = mask.photo_err_1(mask_1[:,np.newaxis]-mag_1)
        mag_err_2 = mask.photo_err_2(mask_2[:,np.newaxis]-mag_2)
         
        # "upper" bound set by maglim
        delta_hi_1 = (mask_1[:,np.newaxis]-mag_1)/mag_err_1
        delta_hi_2 = (mask_2[:,np.newaxis]-mag_2)/mag_err_2
         
        # "lower" bound set by bins_mag (maglim shouldn't be 0)
        delta_lo_1 = (mask.roi.bins_mag[0]-mag_1)/mag_err_1
        delta_lo_2 = (mask.roi.bins_mag[0]-mag_2)/mag_err_2
         
        cdf_1 = norm_cdf(delta_hi_1) - norm_cdf(delta_lo_1)
        cdf_2 = norm_cdf(delta_hi_2) - norm_cdf(delta_lo_2)
        cdf = cdf_1*cdf_2
         
        if method is None or method == 'none':
            comp_cdf = cdf
        elif self.band_1_detection == True:
            comp = mask.mask_1.completeness(mag_1, method=method)
            comp_cdf = comp*cdf
        elif self.band_1_detection == False:
            comp =mask.mask_2.completeness(mag_2, method=method)
            comp_cdf = comp*cdf
        else:
            comp_1 = mask.mask_1.completeness(mag_1, method=method)
            comp_2 = mask.mask_2.completeness(mag_2, method=method)
            comp_cdf = comp_1*comp_2*cdf
         
        observable_fraction = (mass_pdf[np.newaxis]*comp_cdf).sum(axis=-1)
        return observable_fraction[mask.mask_roi_digi[mask.roi.pixel_interior_cut]]

    def observableFractionMMD(self, mask, distance_modulus, mass_min=0.1):
        # This can be done faster...
        logger.info('Calculating observable fraction from MMD')

        mmd = self.signalMMD(mask,distance_modulus)
        obs_frac = mmd.sum(axis=-1).sum(axis=-1)[mask.mask_roi_digi[mask.roi.pixel_interior_cut]]
        return obs_frac

    observable_fraction = observableFractionCMD
    observableFraction = observable_fraction

    def signalMMD(self, mask, distance_modulus, mass_min=0.1, nsigma=5, delta_mag=0.03, mass_steps=1000, method='step'):
        roi = mask.roi
       
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = self.sample(mass_steps=mass_steps,mass_min=mass_min,full_data_range=False)
        mag_1 = mag_1+distance_modulus
        mag_2 = mag_2+distance_modulus
         
        mask_1,mask_2 = mask.mask_roi_unique.T
         
        mag_err_1 = mask.photo_err_1(mask_1[:,np.newaxis]-mag_1)
        mag_err_2 = mask.photo_err_2(mask_2[:,np.newaxis]-mag_2)
         
        # Set mag_err for mask==0 to epsilon
        mag_err_1[mask_1==0] *= -np.inf
        mag_err_2[mask_2==0] *= -np.inf
         
        #edges_mag = np.arange(mask.roi.bins_mag[0] - (0.5*delta_mag),
        #                      mask.roi.bins_mag[-1] + (0.5*delta_mag),
        #                      delta_mag)
        #nedges = edges_mag.shape[0]

        nedges = np.rint((roi.bins_mag[-1]-roi.bins_mag[0])/delta_mag)+1
        edges_mag,delta_mag = np.linspace(roi.bins_mag[0],roi.bins_mag[-1],nedges,retstep=True)
        edges_mag_1 = edges_mag_2 = edges_mag
        nbins = nedges - 1
         
        mag_err_1_max = mag_err_1.max(axis=0)
        mag_err_2_max = mag_err_2.max(axis=0)
         
        max_idx_1 = np.searchsorted(edges_mag[:-1],mag_1+nsigma*mag_err_1_max)
        min_idx_1 = np.searchsorted(edges_mag[:-1],mag_1-nsigma*mag_err_1_max)
        max_idx_2 = np.searchsorted(edges_mag[:-1],mag_2+nsigma*mag_err_1_max)
        min_idx_2 = np.searchsorted(edges_mag[:-1],mag_2-nsigma*mag_err_1_max)
         
        # Select only isochrone values that will contribute to the MMD space
        sel = (max_idx_1>0)&(min_idx_1<nbins)&(max_idx_2>0)&(min_idx_2<nbins)
        if sel.sum() == 0:
            msg = 'No isochrone points in magnitude selection range'
            raise Exception(msg)
         
        mag_1,mag_2 = mag_1[sel],mag_2[sel]
        mag_err_1,mag_err_2 = mag_err_1[:,sel],mag_err_2[:,sel]
        mass_pdf = mass_pdf[sel]
        mag_err_1_max = mag_err_1.max(axis=0)
        mag_err_2_max = mag_err_2.max(axis=0)
        min_idx_1,max_idx_1 = min_idx_1[sel],max_idx_1[sel]
        min_idx_2,max_idx_2 = min_idx_2[sel],max_idx_2[sel]
         
        nmaglim,niso = mag_err_1.shape
         
        # Find valid indices in MMD space (can we avoid this loop?)
        nidx = ((max_idx_1-min_idx_1)*(max_idx_2-min_idx_2))
        mag_idx = np.arange(niso).repeat(nidx)
        bin_idx = np.zeros(nidx.sum(),dtype=int)
        ii = 0
        # ADW: Can we avoid this loop?
        for i in range(niso):
            x = np.ravel_multi_index(np.mgrid[min_idx_1[i]:max_idx_1[i],
                                              min_idx_2[i]:max_idx_2[i]],
                                     [nbins,nbins]).ravel()
            bin_idx[ii:ii+len(x)] = x
            ii += len(x)
         
        #idx = np.unique(idx)
        idx_1,idx_2 = np.unravel_index(bin_idx,[nbins,nbins])
         
        # Pre-compute the indexed arrays to save time at the cost of memory
        mag_1_idx,mag_2_idx = mag_1[mag_idx],mag_2[mag_idx]
        mag_err_1_idx,mag_err_2_idx = mag_err_1[:,mag_idx],mag_err_2[:,mag_idx]
        edges_mag_1_idx,edges_mag_2_idx = edges_mag[idx_1],edges_mag[idx_2]
         
        arg_mag_1_hi = (mag_1_idx - edges_mag_1_idx) / mag_err_1_idx
        arg_mag_1_lo = arg_mag_1_hi - delta_mag/mag_err_1_idx
        arg_mag_2_hi = (mag_2_idx - edges_mag_2_idx) / mag_err_2_idx
        arg_mag_2_lo = arg_mag_2_hi - delta_mag/mag_err_2_idx
         
        del mag_1_idx,mag_2_idx
        del mag_err_1_idx,mag_err_2_idx
        del edges_mag_1_idx,edges_mag_2_idx
         
        # This may become necessary with more maglim bins         
        ### # PDF is only ~nonzero for object-bin pairs within 5 sigma in both magnitudes  
        ### index_nonzero = np.nonzero((arg_mag_1_hi > -nsigma)*(arg_mag_1_lo < nsigma) \
        ###                                *(arg_mag_2_hi > -nsigma)*(arg_mag_2_lo < nsigma))
        ### idx_maglim,idx_iso,idx_idx = index_nonzero
        ### subidx = idx[idx_idx]
         
        pdf_val_1 = norm_cdf(arg_mag_1_hi)-norm_cdf(arg_mag_1_lo)
        pdf_val_2 = norm_cdf(arg_mag_2_hi)-norm_cdf(arg_mag_2_lo)
        pdf_val = pdf_val_1 * pdf_val_2
         
        # Deal with completeness
        if method is None or method == 'none':
            comp = None
        elif self.band_1_detection == True:
            comp=mask.completeness(mask_1[:,np.newaxis]-mag_1, method=method)
        elif self.band_1_detection == False:
            comp=mask.completeness(mask_2[:,np.newaxis]-mag_2, method=method)
        else:
            comp_1 = mask.completeness(mask_1[:,np.newaxis]-mag_1, method=method)
            comp_2 = mask.completeness(mask_2[:,np.newaxis]-mag_2, method=method)
            comp = comp_1*comp_2
         
        if comp is not None:
            comp_pdf_val = pdf_val*comp[:,mag_idx]
        else:
            comp_pdf_val = pdf_val
         
        # Deal with mass pdf values
        scaled_pdf_val = comp_pdf_val*mass_pdf[mag_idx]
         
        # Do the sum without creating the huge sparse array.
        label_idx = np.arange(nmaglim*nbins**2).reshape(nmaglim,nbins**2)
        labels = label_idx[:,bin_idx]
        sum_pdf = ndimage.sum(scaled_pdf_val,labels,label_idx.flat).reshape(nmaglim,nbins**2)
         
        # This is the clipping of the pdf at the maglim
        # Probably want to move this out of this function.
        final_pdf = sum_pdf.reshape(nmaglim,nbins,nbins)
         
        argmax_hi_1 = np.argmax((mask_1[:,np.newaxis] <= edges_mag[1:]),axis=1)
        argmax_hi_2 = np.argmax((mask_2[:,np.newaxis] <= edges_mag[1:]),axis=1)
         
        bin_frac_1 = (mask_1 - edges_mag[argmax_hi_1])/delta_mag
        bin_frac_2 = (mask_2 - edges_mag[argmax_hi_2])/delta_mag
         
        for i,(argmax_1,argmax_2) in enumerate(zip(argmax_hi_1,argmax_hi_2)):
            final_pdf[i,argmax_1,:] *= bin_frac_1[i]
            final_pdf[i,:,argmax_2] *= bin_frac_2[i]
            final_pdf[i,argmax_1+1:,:] = 0
            final_pdf[i,:,argmax_2+1:] = 0
         
        ## This is the actual data selection cut...
        #bins_2,bins_1 = np.meshgrid(edges_mag[:-1],edges_mag[:-1])
        #cut = (bins_1 < mask_1[:,np.newaxis,np.newaxis])*(bins_2 < mask_2[:,np.newaxis,np.newaxis])
        #final_pdf = sum_pdf.reshape(nmaglim,nbins,nbins)*cut
        return final_pdf

    def histo(self,distance_modulus=None,delta_mag=0.03,mass_steps=10000):
        """
        Histogram the isochrone in mag-mag space.
        """
        if distance_modulus is not None:
            self.distance_modulus = distance_modulus

        # Isochrone will be binned in next step, so can sample many points efficiently
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = self.sample(mass_steps=mass_steps)

        #logger.warning("Fudging intrinisic dispersion in isochrone.")
        #mag_1 += np.random.normal(scale=0.02,size=len(mag_1))
        #mag_2 += np.random.normal(scale=0.02,size=len(mag_2))

        bins_mag_1 = np.arange(self.mod+mag_1.min() - (0.5*delta_mag),
                               self.mod+mag_1.max() + (0.5*delta_mag),
                               delta_mag)
        bins_mag_2 = np.arange(self.mod+mag_2.min() - (0.5*delta_mag),
                               self.mod+mag_2.max() + (0.5*delta_mag),
                               delta_mag)        
 
        # ADW: Completeness needs to go in mass_pdf here
        histo_isochrone_pdf = np.histogram2d(self.mod + mag_1,
                                             self.mod + mag_2,
                                             bins=[bins_mag_1, bins_mag_2],
                                             weights=mass_pdf)[0]
 
        return histo_isochrone_pdf, bins_mag_1, bins_mag_2
 
    def pdf_mmd(self, lon, lat, mag_1, mag_2, distance_modulus, mask, delta_mag=0.03, mass_steps=1000):
        """
        Ok, now here comes the beauty of having the signal MMD.
        """
        logger.info('Running MMD pdf')
 
        roi = mask.roi
        mmd = self.signalMMD(mask,distance_modulus,delta_mag=delta_mag,mass_steps=mass_steps)
        
        # This is fragile, store this information somewhere else...
        nedges = np.rint((roi.bins_mag[-1]-roi.bins_mag[0])/delta_mag)+1
        edges_mag,delta_mag = np.linspace(roi.bins_mag[0],roi.bins_mag[-1],nedges,retstep=True)
                                    
        idx_mag_1 = np.searchsorted(edges_mag,mag_1)
        idx_mag_2 = np.searchsorted(edges_mag,mag_2)
 
        if np.any(idx_mag_1 > nedges) or np.any(idx_mag_1 == 0):
            msg = "Magnitude out of range..."
            raise Exception(msg)
        if np.any(idx_mag_2 > nedges) or np.any(idx_mag_2 == 0):
            msg = "Magnitude out of range..."
            raise Exception(msg)
 
        idx = mask.roi.indexROI(lon,lat)
        u_color = mmd[(mask.mask_roi_digi[idx],idx_mag_1,idx_mag_2)]
 
        # Remove the bin size to convert the pdf to units of mag^-2
        u_color /= delta_mag**2
 
        return u_color
 
    def pdf(self, mag_1, mag_2, mag_err_1, mag_err_2, distance_modulus, delta_mag=0.03, mass_steps=10000):
        """
        Compute isochrone probability for each catalog object.
 
        ADW: Still a little speed to be gained here (broadcasting)
 
        Units 
        """
        nsigma = 5.0

        # ADW: HACK FOR SYSTEMATIC UNCERTAINTY
        mag_err_1 = np.sqrt(mag_err_1**2 + 0.01**2)
        mag_err_2 = np.sqrt(mag_err_2**2 + 0.01**2)
 
        # Histogram of the isochrone
        histo_isochrone_pdf,bins_mag_1,bins_mag_2 = self.histo(distance_modulus,delta_mag,mass_steps)
         
        # Keep only isochrone bins that are within the color-magnitude space of the sample
        mag_1_mesh, mag_2_mesh = np.meshgrid(bins_mag_2[1:], bins_mag_1[1:])
        #pad = 1. # mag
         
        # pdf contribution only calculated out to nsigma,
        # so padding shouldn't be necessary.
        mag_1_max = np.max(mag_1+nsigma*mag_err_1)# +pad 
        mag_1_min = np.min(mag_1-nsigma*mag_err_1)# -pad 
        mag_2_max = np.max(mag_2+nsigma*mag_err_2)# +pad 
        mag_2_min = np.min(mag_2-nsigma*mag_err_2)# -pad 
         
        in_color_magnitude_space = ((mag_1_mesh>=mag_1_min)&(mag_1_mesh<=mag_1_max))
        in_color_magnitude_space*= ((mag_2_mesh>=mag_2_min)&(mag_2_mesh<=mag_2_max))
        histo_isochrone_pdf *= in_color_magnitude_space
 
        index_mag_1, index_mag_2 = np.nonzero(histo_isochrone_pdf)
        isochrone_pdf = histo_isochrone_pdf[index_mag_1, index_mag_2]
 
        n_catalog = len(mag_1)
        n_isochrone_bins = len(index_mag_1)
        ones = np.ones([n_catalog, n_isochrone_bins])
 
        mag_1_reshape = mag_1.reshape([n_catalog, 1])
        mag_err_1_reshape = mag_err_1.reshape([n_catalog, 1])
        mag_2_reshape = mag_2.reshape([n_catalog, 1])
        mag_err_2_reshape = mag_err_2.reshape([n_catalog, 1])
 
        # Calculate distance between each catalog object and isochrone bin
        # Assume normally distributed photometry uncertainties
        delta_mag_1_hi = (mag_1_reshape - bins_mag_1[index_mag_1])
        arg_mag_1_hi = (delta_mag_1_hi / mag_err_1_reshape)
        delta_mag_1_lo = (mag_1_reshape - bins_mag_1[index_mag_1 + 1])
        arg_mag_1_lo = (delta_mag_1_lo / mag_err_1_reshape)
        #pdf_mag_1 = (scipy.stats.norm.cdf(arg_mag_1_hi) - scipy.stats.norm.cdf(arg_mag_1_lo))
         
        delta_mag_2_hi = (mag_2_reshape - bins_mag_2[index_mag_2])
        arg_mag_2_hi = (delta_mag_2_hi / mag_err_2_reshape)
        delta_mag_2_lo = (mag_2_reshape - bins_mag_2[index_mag_2 + 1])
        arg_mag_2_lo = (delta_mag_2_lo / mag_err_2_reshape)
        #pdf_mag_2 = scipy.stats.norm.cdf(arg_mag_2_hi) - scipy.stats.norm.cdf(arg_mag_2_lo)
         
        # PDF is only ~nonzero for object-bin pairs within 5 sigma in both magnitudes  
        index_nonzero_0, index_nonzero_1 = np.nonzero((arg_mag_1_hi > -nsigma) \
                                                         *(arg_mag_1_lo < nsigma) \
                                                         *(arg_mag_2_hi > -nsigma) \
                                                         *(arg_mag_2_lo < nsigma))
        pdf_mag_1 = np.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_2 = np.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_1[index_nonzero_0,index_nonzero_1] = norm_cdf(arg_mag_1_hi[index_nonzero_0,
                                                                           index_nonzero_1]) \
                                                   - norm_cdf(arg_mag_1_lo[index_nonzero_0,
                                                                           index_nonzero_1])
        pdf_mag_2[index_nonzero_0,index_nonzero_1] = norm_cdf(arg_mag_2_hi[index_nonzero_0,
                                                                           index_nonzero_1]) \
                                                   - norm_cdf(arg_mag_2_lo[index_nonzero_0,
                                                                           index_nonzero_1])
         
        # Signal color probability is product of PDFs for each object-bin pair 
        # summed over isochrone bins
        u_color = np.sum(pdf_mag_1 * pdf_mag_2 * (isochrone_pdf * ones), axis=1)
 
        # Remove the bin size to convert the pdf to units of mag^-2
        u_color /= delta_mag**2
 
        return u_color
    
    # FIXME: Need to parallelize CMD and MMD formulation
 
    def pdfX(self, mask, mag_1, mag_2, mag_err_1, mag_err_2, distance_modulus, delta_mag=0.03, mass_steps=10000):
        """
        ADW: DEPRICATED (12/23/2014)
 
        Compute isochrone probability for each catalog object.
 
        Units 
        """
        nsigma = 5.0
 
        # Isochrone will be binned in next step, so can sample many points efficiently
        isochrone_mass_init,isochrone_mass_pdf,isochrone_mass_act,isochrone_mag_1,isochrone_mag_2 = self.sample(mass_steps=mass_steps)
 
        bins_mag_1 = np.arange(distance_modulus+isochrone_mag_1.min() - (0.5*delta_mag),
                               distance_modulus+isochrone_mag_1.max() + (0.5*delta_mag),
                               delta_mag)
        bins_mag_2 = np.arange(distance_modulus+isochrone_mag_2.min() - (0.5*delta_mag),
                               distance_modulus+isochrone_mag_2.max() + (0.5*delta_mag),
                               delta_mag)        
 
        # ADW: Restrict mag and color to range of mask with sufficient solid angle
        cmd_cut = ugali.utils.binning.take2D(mask.solid_angle_cmd,isochrone_mag_1-isochrone_mag_2,
                                             isochrone_mag_1+distance_modulus,
                                             mask.roi.bins_color, mask.roi.bins_mag) > 0
 
        # ADW: This does not account for the leakage out of CMD space
        histo_isochrone_pdf = np.histogram2d(distance_modulus + isochrone_mag_1[cmd_cut],
                                             distance_modulus + isochrone_mag_2[cmd_cut],
                                             bins=[bins_mag_1, bins_mag_2],
                                             weights=isochrone_mass_pdf[cmd_cut])[0]
 
 
        
        # Histogram of the isochrone
        #histo_isochrone_pdf,bins_mag_1,bins_mag_2 = self.histo(distance_modulus,delta_mag,mass_steps)
         
        # Keep only isochrone bins that are within the color-magnitude space of the sample
        mag_1_mesh, mag_2_mesh = np.meshgrid(bins_mag_2[1:], bins_mag_1[1:])
        #pad = 1. # mag
         
        ### # pdf contribution only calculated out to nsigma,
        ### # so padding shouldn't be necessary.
        ### mag_1_max = np.max(mag_1+nsigma*mag_err_1)# +pad 
        ### mag_1_min = np.min(mag_1-nsigma*mag_err_1)# -pad 
        ### mag_2_max = np.max(mag_2+nsigma*mag_err_2)# +pad 
        ### mag_2_min = np.min(mag_2-nsigma*mag_err_2)# -pad 
        ###  
        ### in_color_magnitude_space = ((mag_1_mesh>=mag_1_min)&(mag_1_mesh<=mag_1_max))
        ### in_color_magnitude_space*= ((mag_2_mesh>=mag_2_min)&(mag_2_mesh<=mag_2_max))
        ### histo_isochrone_pdf *= in_color_magnitude_space
 
        index_mag_1, index_mag_2 = np.nonzero(histo_isochrone_pdf)
        isochrone_pdf = histo_isochrone_pdf[index_mag_1, index_mag_2]
 
        n_catalog = len(mag_1)
        n_isochrone_bins = len(index_mag_1)
        ones = np.ones([n_catalog, n_isochrone_bins])
 
        mag_1_reshape = mag_1.reshape([n_catalog, 1])
        mag_err_1_reshape = mag_err_1.reshape([n_catalog, 1])
        mag_2_reshape = mag_2.reshape([n_catalog, 1])
        mag_err_2_reshape = mag_err_2.reshape([n_catalog, 1])
 
        # Calculate distance between each catalog object and isochrone bin
        # Assume normally distributed photometry uncertainties
        delta_mag_1_hi = (mag_1_reshape - bins_mag_1[index_mag_1])
        arg_mag_1_hi = (delta_mag_1_hi / mag_err_1_reshape)
        delta_mag_1_lo = (mag_1_reshape - bins_mag_1[index_mag_1 + 1])
        arg_mag_1_lo = (delta_mag_1_lo / mag_err_1_reshape)
        #pdf_mag_1 = (scipy.stats.norm.cdf(arg_mag_1_hi) - scipy.stats.norm.cdf(arg_mag_1_lo))
         
        delta_mag_2_hi = (mag_2_reshape - bins_mag_2[index_mag_2])
        arg_mag_2_hi = (delta_mag_2_hi / mag_err_2_reshape)
        delta_mag_2_lo = (mag_2_reshape - bins_mag_2[index_mag_2 + 1])
        arg_mag_2_lo = (delta_mag_2_lo / mag_err_2_reshape)
        #pdf_mag_2 = scipy.stats.norm.cdf(arg_mag_2_hi) - scipy.stats.norm.cdf(arg_mag_2_lo)
         
        # PDF is only ~nonzero for object-bin pairs within 5 sigma in both magnitudes  
        index_nonzero_0, index_nonzero_1 = np.nonzero((arg_mag_1_hi > -nsigma) \
                                                         *(arg_mag_1_lo < nsigma) \
                                                         *(arg_mag_2_hi > -nsigma) \
                                                         *(arg_mag_2_lo < nsigma))
        pdf_mag_1 = np.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_2 = np.zeros([n_catalog, n_isochrone_bins])
        pdf_mag_1[index_nonzero_0,index_nonzero_1] = scipy.stats.norm.cdf(arg_mag_1_hi[index_nonzero_0,
                                                                           index_nonzero_1]) \
                                                   - scipy.stats.norm.cdf(arg_mag_1_lo[index_nonzero_0,
                                                                           index_nonzero_1])
        pdf_mag_2[index_nonzero_0,index_nonzero_1] = scipy.stats.norm.cdf(arg_mag_2_hi[index_nonzero_0,
                                                                           index_nonzero_1]) \
                                                   - scipy.stats.norm.cdf(arg_mag_2_lo[index_nonzero_0,
                                                                           index_nonzero_1])
         
        # Signal color probability is product of PDFs for each object-bin pair 
        # summed over isochrone bins
        u_color = np.sum(pdf_mag_1 * pdf_mag_2 * (isochrone_pdf * ones), axis=1)
 
        # Remove the bin size to convert the pdf to units of mag^-2
        u_color /= delta_mag**2
 
        return u_color

    def separation(self,mag_1,mag_2,steps=10000):
        """ Separation in magnitude-magnitude space between points and isochrone """

        # http://stackoverflow.com/q/12653120/
        mag_1 = np.array(mag_1,copy=False,ndmin=1)
        mag_2 = np.array(mag_2,copy=False,ndmin=1)

        init,pdf,act,iso_mag_1,iso_mag_2 = self.sample(mass_steps=steps)
        iso_mag_1+=self.distance_modulus
        iso_mag_2+=self.distance_modulus

        iso_cut = (iso_mag_1<np.max(mag_1))&(iso_mag_1>np.min(mag_1)) | \
                  (iso_mag_2<np.max(mag_2))&(iso_mag_2>np.min(mag_2))
        iso_mag_1 = iso_mag_1[iso_cut]
        iso_mag_2 = iso_mag_2[iso_cut]
         
        dist_mag_1 = mag_1[:,np.newaxis]-iso_mag_1
        dist_mag_2 = mag_2[:,np.newaxis]-iso_mag_2
        
        return np.min(np.sqrt(dist_mag_1**2 + dist_mag_2**2),axis=1)

    def separation(self, mag_1, mag_2):
        """ Could speed this up..."""

        iso_mag_1 = self.mag_1 + self.distance_modulus
        iso_mag_2 = self.mag_2 + self.distance_modulus
        
        # First do the RGB
        if isinstance(self, DotterIsochrone):
            rgb_mag_1 = iso_mag_1
            rgb_mag_2 = iso_mag_2
        else:
            sel = self.stage <= 3
            rgb_mag_1 = iso_mag_1[sel]
            rgb_mag_2 = iso_mag_2[sel]

        def interp_iso(iso_mag_1,iso_mag_2,mag_1,mag_2):
            interp_1 = scipy.interpolate.interp1d(iso_mag_1,iso_mag_2,bounds_error=False)
            interp_2 = scipy.interpolate.interp1d(iso_mag_2,iso_mag_1,bounds_error=False)

            dy = interp_1(mag_1) - mag_2
            dx = interp_2(mag_2) - mag_1

            dmag_1 = np.fabs(dx*dy) / (dx**2 + dy**2) * dy
            dmag_2 = np.fabs(dx*dy) / (dx**2 + dy**2) * dx

            return dmag_1, dmag_2

        dmag_1,dmag_2 = interp_iso(rgb_mag_1,rgb_mag_2,mag_1,mag_2)

        # Then do the HB
        if not isinstance(self, DotterIsochrone):
            sel = self.stage > 3
            hb_mag_1 = iso_mag_1[sel]
            hb_mag_2 = iso_mag_2[sel]

            hb_dmag_1,hb_dmag_2 = interp_iso(hb_mag_1,hb_mag_2,mag_1,mag_2)

            dmag_1 = np.nanmin([dmag_1,hb_dmag_1],axis=0)
            dmag_2 = np.nanmin([dmag_2,hb_dmag_2],axis=0)

        #return dmag_1,dmag_2
        return np.sqrt(dmag_1**2 + dmag_2**2)


class PadovaIsochrone(Isochrone):
    _prefix = 'iso'
    _basename = '%(prefix)s_a%(age)04.1f_z%(z)0.5f.dat'
    #_dirname = '/u/ki/kadrlica/des/isochrones/v3/'
    _dirname =  os.path.join(get_iso_dir(),'padova')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage',4,'Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    def __init__(self,**kwargs):
        super(PadovaIsochrone,self).__init__(**kwargs)

        self.grid = self.create_grid()
        self.tree = self.create_tree(self.grid)
        self.agrid, self.zgrid = self.grid
        self.params['age'].set_bounds([self.agrid.min(),self.agrid.max()])
        self.params['metallicity'].set_bounds([self.zgrid.min(),self.zgrid.max()])  
        self.filename = None
        self._cache()

    def __str__(self,indent=0):
        ret = super(PadovaIsochrone,self).__str__(indent)
        filename = 'Filename: %s'%self.filename
        ret += '\n{0:>{2}}{1}'.format('',filename,indent+2)
        return ret

    @classmethod
    def params2filename(cls,age,metallicity):
        return cls._basename%dict(prefix=cls._prefix,age=age,z=metallicity)

    @classmethod
    def filename2params(cls,filename):
        #ADW: Could probably do something more clever so that parsing info
        #is stored in only one place...
        basename = os.path.basename(filename)
        prefix,a,z = os.path.splitext(basename)[0].split('_')
        if prefix != cls._prefix:
            msg = 'File prefix does not match: %s'%filename
            raise Exception(msg)
        age = float(a.strip('a'))
        metallicity = float(z.strip('z'))
        return age,metallicity

    def create_grid(self,abins=None,zbins=None):
        if abins is None and zbins is None:
            data = np.array([self.filename2params(f) for f in glob.glob(self.dirname+'/%s_*.dat'%(self._prefix))])
            if not len(data):
                msg = "No isochrone files found in: %s"%self.dirname
                raise Exception(msg)
            arange = np.unique(data[:,0])
            zrange = np.unique(data[:,1])
        elif abins is not None and zbins is not None:            
            # Age in units of Gyr
            arange = np.linspace(abins[0],abins[1],abins[2]+1)
            # Metallicity sampled logarithmically
            zrange = np.logspace(np.log10(zbins[0]),np.log10(zbins[1]),zbins[2]+1)
        else:
            msg = "Must specify both `abins` and `zbins` or neither"
            raise Exception(msg)
        aa,zz = np.meshgrid(arange,zrange)
        return aa.flatten(),zz.flatten()

    def create_tree(self,grid=None):
        if grid is None: grid = self.create_grid()
        return scipy.spatial.cKDTree(np.vstack(grid).T)

    def get_filename(self):
        dirname = self.dirname
        p = [self.age,self.metallicity]
        dist,idx = self.tree.query(p)
        age = self.grid[0][idx]
        z = self.grid[1][idx]
        return os.path.join(dirname,self.params2filename(age,z))

    def _cache(self,name=None):
        # For first call before init fully run
        if not hasattr(self,'tree'): return
        if name in ['distance_modulus']: return

        filename = self.get_filename()
        if filename != self.filename:
            self.filename = filename
            self._parse(self.filename)

    def _parse(self,filename):
        """
        Reads an isochrone file in the Padova (Marigo 2008) format and determines
        the age (log10 yrs and Gyr), metallicity (Z and [Fe/H]), and creates arrays with
        the initial stellar mass and corresponding magnitudes for each step along the isochrone.
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        if self.survey.lower() == 'des':
            columns = odict([
                    (3, ('mass_init',float)),
                    (4, ('mass_act',float)),
                    (5, ('log_lum',float)),
                    (10, ('g',float)),
                    (11, ('r',float)),
                    (12,('i',float)),
                    (13,('z',float)),
                    (14,('Y',float)),
                    (16,('stage',int)),
                    ])
        elif self.survey.lower() == 'sdss':
            columns = odict([
                    (3, ('mass_init',float)),
                    (4, ('mass_act',float)),
                    (5, ('log_lum',float)),
                    (9, ('u',float)),
                    (10, ('g',float)),
                    (11,('r',float)),
                    (12,('i',float)),
                    (13,('z',float)),
                    (16,('stage',int)),
                    ])
        else:
            logger.warning('did not recognize survey %s'%(survey))

        kwargs = dict(delimiter='\t',usecols=columns.keys(),dtype=columns.values())
        data = np.genfromtxt(filename,**kwargs)

        self.mass_init = data['mass_init']
        self.mass_act  = data['mass_act']
        self.luminosity = 10**data['log_lum']
        self.mag_1 = data[self.band_1]
        self.mag_2 = data[self.band_2]
        self.stage = data['stage']

        self.mass_init_upper_bound = np.max(self.mass_init)
        self.index = len(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2


class EmpiricalPadova(PadovaIsochrone):
    _prefix = 'iso'
    _basename = '%(prefix)s_a13.7_z0.00007.dat'
    _dirname =  os.path.join(get_iso_dir(),'empirical')

    defaults = (PadovaIsochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
    )

class M92(EmpiricalPadova):
    """ Empirical isochrone derived from the M92 ridgeline dereddened
    and transformed to the DES system.
    """
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
        ('age',              Parameter(13.7, [13.7, 13.7]) ),  # Gyr
        ('metallicity',      Parameter(7e-5,[7e-5,7e-5]) ),
    ])

    _prefix = 'm92'
    _basename = '%(prefix)s_a13.7_z0.00007.dat'

class DESDwarfs(EmpiricalPadova):
    """ Empirical isochrone derived from spectroscopic members of the
    DES dwarfs.
    """
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
        ('age',              Parameter(12.5, [12.5, 12.5]) ),  # Gyr
        ('metallicity',      Parameter(1e-4, [1e-4,1e-4]) ),
    ])

    _prefix = 'dsph'
    _basename = '%(prefix)s_a12.5_z0.00010.dat'

class Girardi2002(PadovaIsochrone):
    #_dirname = '/u/ki/kadrlica/des/isochrones/v5/'
    _dirname =  os.path.join(get_iso_dir(),'girardi2002')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
            des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (9, ('g',float)),
                (10, ('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('Y',float)),
                (15,('stage',object))
                ]),
            )
    
    def _parse(self,filename):
        """
        Reads an isochrone in the old Padova format (Girardi 2002,
        Marigo 2008) and determines the age (log10 yrs and Gyr),
        metallicity (Z and [Fe/H]), and creates arrays with the
        initial stellar mass and corresponding magnitudes for each
        step along the isochrone.
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError, e:
            logger.warning('did not recognize survey %s'%(survey))
            raise(e)

        kwargs = dict(delimiter='\t',usecols=columns.keys(),dtype=columns.values())
        data = np.genfromtxt(filename,**kwargs)

        self.mass_init = data['mass_init']
        self.mass_act  = data['mass_act']
        self.luminosity = 10**data['log_lum']
        self.mag_1 = data[self.band_1]
        self.mag_2 = data[self.band_2]
        self.stage = np.char.array(data['stage']).strip()
        for i,s in enumerate(self.stage):
            if i>0 and s=='' and self.stage[i-1]!='':
                self.stage[i] = self.stage[i-1]

        # Check where post-AGB isochrone data points begin
        self.mass_init_upper_bound = np.max(self.mass_init)
        if np.any(self.stage == 'LTP'):
            self.index = np.nonzero(self.stage == 'LTP')[0][0]
        else:
            self.index = len(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2

class Girardi2010(Girardi2002):
    #_dirname = '/u/ki/kadrlica/des/isochrones/v4/'
    _dirname =  os.path.join(get_iso_dir(),'girardi2010')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
            des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (9, ('g',float)),
                (10, ('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (13,('Y',float)),
                (19,('stage',object))
                ]),
            )

class OldPadovaIsochrone(Girardi2002):
    _prefix = 'isot'
    _basename = '%(prefix)sa%(age)iz%(z)g.dat'
    #_dirname = '/u/ki/kadrlica/des/isochrones/v0/' # won't work for SDSS...
    _dirname =  os.path.join(get_iso_dir(),'old_padova')

    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
        des = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (8, ('g',float)),
                (9, ('r',float)),
                (10,('i',float)),
                (11,('z',float)),
                (12,('Y',float)),
                (19,('stage',object)),
                ]),
        sdss = odict([
                (2, ('mass_init',float)),
                (3, ('mass_act',float)),
                (4, ('log_lum',float)),
                (8, ('u',float)),
                (9, ('g',float)),
                (10,('r',float)),
                (11,('i',float)),
                (12,('z',float)),
                (19,('stage',object)),
                ]),
        )

    @property
    def feh(self):
        # Anders & Grevesse 1989
        metallicity_solar = 0.019 
        feh = np.log10(self.metallicity / metallicity_solar)

    @classmethod
    def params2filename(cls,age,metallicity):
        z = 1e3 * metallicity
        a = 1e2 * np.log10(age*1e9) # Gyr
        return cls._basename%dict(prefix=cls._prefix,age=a,z=z)

    @classmethod
    def filename2params(cls,filename):
        #ADW: Could probably do something more clever so that parsing info
        #is stored in only one place...
        infile = os.path.basename(filename)
        log_age = 1.e-2 * float(infile.split('isota')[1].split('z')[0])
        metallicity = 1.e-3 * float(infile.split('z')[1].split('.dat')[0])
        age = 10**(log_age) / 1e9 # Gyr
        return age, metallicity

############################################################

class DotterIsochrone(PadovaIsochrone):
    """
    KCB: currently inheriting from PadovaIsochrone because there are 
    several useful functions where we would basically be copying code.
    """
    _dirname =  os.path.join(get_iso_dir(),'dotter')

    # KCB: What to do about horizontal branch?
    defaults = (Isochrone.defaults) + (
        ('dirname',_dirname,'Directory name for isochrone files'),
        ('hb_stage','BHeb','Horizontal branch stage name'),
        ('hb_spread',0.1,'Intrinisic spread added to horizontal branch'),
        )

    columns = dict(
            des = odict([
                (1, ('mass',float)),
                (4, ('log_lum',float)),
                (5, ('u',float)),
                (6, ('g',float)),
                (7, ('r',float)),
                (8,('i',float)),
                (9,('z',float))
                ]),
            )

    def _parse(self,filename):
        """
        Reads an isochrone in the Dotter format and determines the 
        age (log10 yrs and Gyr), metallicity (Z and [Fe/H]), and 
        creates arrays with the initial stellar mass and 
        corresponding magnitudes for each step along the isochrone.
        http://stellar.dartmouth.edu/models/isolf_new.html
        """
        try:
            columns = self.columns[self.survey.lower()]
        except KeyError, e:
            logger.warning('did not recognize survey %s'%(survey))
            raise(e)

        kwargs = dict(delimiter='',comments='#',usecols=columns.keys(),dtype=columns.values())
        data = np.genfromtxt(filename,**kwargs)

        # KCB: Not sure whether the mass in Dotter isochrone output
        # files is initial mass or current mass
        self.mass_init = data['mass']
        self.mass_act  = data['mass']
        self.luminosity = 10**data['log_lum']
        self.mag_1 = data[self.band_1]
        self.mag_2 = data[self.band_2]
        self.stage = numpy.tile('Main', len(data))
        
        # KCB: No post-AGB isochrone data points, right?
        self.mass_init_upper_bound = np.max(self.mass_init)

        self.mag = self.mag_1 if self.band_1_detection else self.mag_2
        self.color = self.mag_1 - self.mag_2

    @property
    def feh(self):
        # Dotter convention
        metallicity_solar = 0.02
        feh = np.log10(self.metallicity / metallicity_solar)

############################################################

class CompositeIsochrone(Isochrone):
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
    ])
    _mapping = odict([
            ('mod','distance_modulus'),
            ('a','age'),                 
            ('z','metallicity'),
            ])

    defaults = (Isochrone.defaults) + (
        ('type','PadovaIsochrone','Default type of isochrone to create'),
        ('weights',None,'Relative weights for each isochrone'),
        )
    
    def __init__(self, isochrones, **kwargs):
        super(CompositeIsochrone,self).__init__(**kwargs)

        self.isochrones = []
        for i in isochrones:
            if isinstance(i,Isochrone):
                iso = i
            else:
                name = i.pop('name',self.type)
                #iso = isochroneFactory(name=name,**i)
                iso = isochroneFactory(name=name,**i)
            # Tie the distance modulus
            iso.params['distance_modulus'] = self.params['distance_modulus']
            self.isochrones.append(iso)
        
        if self.weights is None: self.weights = np.ones(len(self.isochrones))
        self.weights /= np.sum(np.asarray(self.weights))
        self.weights = self.weights.tolist()

        if len(self.isochrones) != len(self.weights):
            msg = 'Length of isochrone and weight arrays must be equal'
            raise ValueError(msg)

    def __getitem__(self, key):
        return self.isochrones[key]
 
    def __str__(self,indent=0):
        ret = super(CompositeIsochrone,self).__str__(indent)
        ret += '\n{0:>{2}}{1}'.format('','Isochrones:',indent+2)
        for i in self:
            ret += '\n{0}'.format(i.__str__(indent+4))
        return ret

    @property
    def age(self):
        return np.array([i.age for i in self])

    @property
    def metallicity(self):
        return np.array([i.metallicity for i in self])

    def composite_decorator(func):
        """
        Decorator for wrapping functions that calculate a weighted sum
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            total = []
            for weight,iso in zip(self.weights,self.isochrones):
                subfunc = getattr(iso,func.__name__)
                total.append(weight*subfunc(*args,**kwargs))
            return np.sum(total,axis=0)
        return wrapper

    def sample(self, **kwargs):
        samples = [iso.sample(**kwargs) for iso in self.isochrones]
        for weight,sample in zip(self.weights,samples):
            sample[1] *= weight

        return np.hstack(samples)

    def separation(self, *args, **kwargs):
        separations = [iso.separation(*args,**kwargs) for iso in self.isochrones]
        return np.nanmin(separations,axis=0)
    
    def todict(self):
        ret = super(CompositeIsochrone,self).todict()
        ret['isochrones'] = [iso.todict() for iso in self.isochrones]
        return ret

    @composite_decorator
    def stellar_mass(self, *args, **kwargs): pass

    @composite_decorator
    def stellar_luminosity(self, *args, **kwargs): pass

    @composite_decorator
    def observable_fraction(self, *args, **kwargs): pass

    @composite_decorator
    def observableFractionX(self, *args, **kwargs): pass

    @composite_decorator
    def signalMMD(self, *args, **kwargs): pass

    # ADW: For temporary backwards compatibility
    stellarMass = stellar_mass
    stellarLuminosity = stellar_luminosity
    observableFraction = observable_fraction

### class OldCompositeIsochrone(CompositeIsochrone):
###     def __init__(self, isochrones, **kwargs):
###         super(CompositeIsochrone,self).__init__(**kwargs)
###  
###         self.isochrones = []
###         for i in isochrones:
###             a,z = OldPadovaIsochrone.filename2params(i)
###             iso = OldPadovaIsochrone(a=a,z=z)
###             # Tie the distance modulus
###             iso.params['distance_modulus'] = self.params['distance_modulus']
###             self.isochrones.append(iso)
###         
###         if self.weights is None: self.weights = np.ones(len(self.isochrones))
###         self.weights /= np.sum(self.weights)
###  
###         if len(self.isochrones) != len(self.weights):
###             msg = 'Length of isochrone and weight arrays must be equal'
###             raise ValueError(msg)

Padova = PadovaIsochrone
OldPadova = OldPadovaIsochrone
Composite = CompositeIsochrone
Dotter = DotterIsochrone
Bressan2012 = PadovaIsochrone

def factory(name, **kwargs):
    from ugali.utils.factory import factory
    return factory(name, module=__name__, **kwargs)

isochroneFactory = factory

#def factory(name, **kwargs):
#    """
#    Factory for creating isochrones. Arguments are 
#    passed directly to the constructor of the isochrone.
#    """
#    fn = lambda member: inspect.isclass(member) and member.__module__==__name__
#    classes = odict(inspect.getmembers(sys.modules[__name__], fn))
# 
#    if name not in classes.keys():
#        msg = "%s not found in isochrones:\n %s"%(name,isochrones.keys())
#        logger.error(msg)
#        msg = "Unrecognized class: %s"%name
#        raise Exception(msg)
# 
#    return isochrones[name](**kwargs)

def absolute_magnitude(distance_modulus,g,r,prob=None):
    """ Calculate the absolute magnitude from a set of bands """
    V = g - 0.487*(g - r) - 0.0249
        
    flux = np.sum(10**(-(V-distance_modulus)/2.5))
    Mv = -2.5*np.log10(flux)
    return Mv
