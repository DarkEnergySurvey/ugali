#!/usr/bin/env python
"""
Create and run an individual likelihood analysis.
"""
import os
import sys
from collections import OrderedDict as odict

import numpy
import numpy as np
import healpy
import fitsio

import ugali.utils.skymap
import ugali.analysis.loglike
from ugali.analysis.loglike import LogLikelihood, createSource, createObservation
from ugali.analysis.source import Source
from ugali.utils.parabola import Parabola

from ugali.utils.config import Config
from ugali.utils.logger import logger
from ugali.utils.healpix import superpixel, subpixel, pix2ang, ang2pix
from ugali.utils.healpix import write_partial_map

############################################################

class Scan(object):
    """
    The base of a likelihood analysis scan.

    FIXME: This really does nothing now and should be absorbed into
    GridSearch (confusing now).
    """
    def __init__(self, config, coords):
        self.config = Config(config)
        # Should only be one coordinate
        if len(coords) != 1: raise Exception('Must specify one coordinate.')
        self.lon,self.lat,radius = coords[0]
        self._setup()
        

    def _setup(self):
        self.observation = createObservation(self.config,self.lon,self.lat)
        self.source = createSource(self.config,'scan',lon=self.lon,lat=self.lat)
        loglike=LogLikelihood(self.config,self.observation,self.source)
        self.grid = GridSearch(self.config,loglike)

    @property
    def loglike(self):
        return self.grid.loglike

    def run(self, coords=None, debug=False):
        """
        Run the likelihood grid search
        """
        #self.grid.precompute()
        self.grid.search(coords=coords)
        return self.grid
        
    def write(self, outfile):
        self.grid.write(outfile)

def createGridSearch(config,lon,lat):
    config = Config(config)
    obs = createObservation(config,lon,lat)
    src = createSource(config,'scan',lon=lon,lat=lat)
    loglike=LogLikelihood(config,obs,src)
    return GridSearch(config,loglike)
    

############################################################

class GridSearch:

    def __init__(self, config, loglike): # What it should be...
        """
        Object to efficiently search over a grid of ROI positions.

        Parameters:
        -----------
        config  : Configuration object or filename.
        loglike : Log-likelihood object

        Returns:
        --------
        grid    : GridSearch instance
        """

        self.config = Config(config)
        self.loglike = loglike
        self.roi  = self.loglike.roi
        self.mask = self.loglike.mask

        logger.info(str(self.loglike))

        self.stellar_mass_conversion = self.loglike.source.stellar_mass()
        self.distance_modulus_array = np.asarray(self.config['scan']['distance_modulus_array'])

    def precompute(self, distance_modulus_array=None):
        """
        DEPRECATED: ADW 20170627

        Precompute color probabilities for background ('u_background')
        and signal ('u_color') for each star in catalog.  Precompute
        observable fraction in each ROI pixel.  # Precompute still
        operates over the full ROI, not just the likelihood region

        Parameters:
        -----------
        distance_modulus_array : Array of distance moduli 
        
        Returns:
        --------
        None
        """
        msg = "'%s.precompute': ADW 2017-09-20"%self.__class__.__name__
        DeprecationWarning(msg)

        if distance_modulus_array is not None:
            self.distance_modulus_array = distance_modulus_array
        else:
            self.distance_modulus_array = sel

        # Observable fraction for each pixel
        self.u_color_array = [[]] * len(self.distance_modulus_array)
        self.observable_fraction_sparse_array = [[]] * len(self.distance_modulus_array)

        logger.info('Looping over distance moduli in precompute ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            logger.info('  (%i/%i) Distance Modulus = %.2f ...'%(ii+1, len(self.distance_modulus_array), distance_modulus))

            self.u_color_array[ii] = False
            if self.config['scan']['color_lut_infile'] is not None:
                DeprecationWarning("'color_lut' is deprecated")
                logger.info('  Precomputing signal color from %s'%(self.config['scan']['color_lut_infile']))
                self.u_color_array[ii] = ugali.analysis.color_lut.readColorLUT(self.config['scan']['color_lut_infile'],
                                                                               distance_modulus,
                                                                               self.loglike.catalog.mag_1,
                                                                               self.loglike.catalog.mag_2,
                                                                               self.loglike.catalog.mag_err_1,
                                                                               self.loglike.catalog.mag_err_2)
            if not np.any(self.u_color_array[ii]):
                logger.info('  Precomputing signal color on the fly...')
                self.u_color_array[ii] = self.loglike.calc_signal_color(distance_modulus) 
            
            # Calculate over all pixels in ROI
            self.observable_fraction_sparse_array[ii] = self.loglike.calc_observable_fraction(distance_modulus)

        self.u_color_array = np.array(self.u_color_array)

    def search(self, coords=None, distance_modulus=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels and distance moduli in distance_modulus_array
        coords: (lon,lat)
        distance_modulus: scalar
        """
        nmoduli = len(self.distance_modulus_array)
        npixels    = len(self.roi.pixels_target)
        self.log_likelihood_sparse_array       = np.zeros([nmoduli, npixels])
        self.richness_sparse_array             = np.zeros([nmoduli, npixels])
        self.richness_lower_sparse_array       = np.zeros([nmoduli, npixels])
        self.richness_upper_sparse_array       = np.zeros([nmoduli, npixels])
        self.richness_upper_limit_sparse_array = np.zeros([nmoduli, npixels])
        self.stellar_mass_sparse_array         = np.zeros([nmoduli, npixels])
        self.fraction_observable_sparse_array  = np.zeros([nmoduli, npixels])

        # Specific pixel/distance_modulus
        coord_idx, distance_modulus_idx = None, None
        if coords is not None:
            # Match to nearest grid coordinate index
            coord_idx = self.roi.indexTarget(coords[0],coords[1])
        if distance_modulus is not None:
            # Match to nearest distance modulus index
            distance_modulus_idx=np.fabs(self.distance_modulus_array-distance_modulus).argmin()

        lon, lat = self.roi.pixels_target.lon, self.roi.pixels_target.lat
            
        logger.info('Looping over distance moduli in grid search ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):

            # Specific pixel
            if distance_modulus_idx is not None:
                if ii != distance_modulus_idx: continue

            logger.info('  (%-2i/%i) Distance Modulus=%.1f ...'%(ii+1,nmoduli,distance_modulus))

            # Set distance_modulus once to save time
            self.loglike.set_params(distance_modulus=distance_modulus)

            for jj in range(0, npixels):
                # Specific pixel
                if coord_idx is not None:
                    if jj != coord_idx: continue

                # Set kernel location
                self.loglike.set_params(lon=lon[jj],lat=lat[jj])
                # Doesn't re-sync distance_modulus each time
                self.loglike.sync_params()
                                         
                args = (jj+1, npixels, self.loglike.source.lon, self.loglike.source.lat)
                msg = '    (%-3i/%i) Candidate at (%.2f, %.2f) ... '%(args)

                self.log_likelihood_sparse_array[ii][jj], self.richness_sparse_array[ii][jj], parabola = self.loglike.fit_richness()
                self.stellar_mass_sparse_array[ii][jj] = self.stellar_mass_conversion * self.richness_sparse_array[ii][jj]
                self.fraction_observable_sparse_array[ii][jj] = self.loglike.f
                if self.config['scan']['full_pdf']:
                    #n_pdf_points = 100
                    #richness_range = parabola.profileUpperLimit(delta=25.) - self.richness_sparse_array[ii][jj]
                    #richness = np.linspace(max(0., self.richness_sparse_array[ii][jj] - richness_range),
                    #                          self.richness_sparse_array[ii][jj] + richness_range,
                    #                          n_pdf_points)
                    #if richness[0] > 0.:
                    #    richness = np.insert(richness, 0, 0.)
                    #    n_pdf_points += 1
                    # 
                    #log_likelihood = np.zeros(n_pdf_points)
                    #for kk in range(0, n_pdf_points):
                    #    log_likelihood[kk] = self.loglike.value(richness=richness[kk])
                    #parabola = ugali.utils.parabola.Parabola(richness, 2.*log_likelihood)
                    #self.richness_lower_sparse_array[ii][jj], self.richness_upper_sparse_array[ii][jj] = parabola.confidenceInterval(0.6827)
                    self.richness_lower_sparse_array[ii][jj], self.richness_upper_sparse_array[ii][jj] = self.loglike.richness_interval(0.6827)
                    
                    self.richness_upper_limit_sparse_array[ii][jj] = parabola.bayesianUpperLimit(0.95)

                    args = (
                        2. * self.log_likelihood_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_lower_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_upper_sparse_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_upper_limit_sparse_array[ii][jj]
                    )
                    msg += 'TS=%.1f, Stellar Mass=%.1f (%.1f -- %.1f @ 0.68 CL, < %.1f @ 0.95 CL)'%(args)
                else:
                    args = (
                        2. * self.log_likelihood_sparse_array[ii][jj], 
                        self.stellar_mass_conversion * self.richness_sparse_array[ii][jj],
                        self.fraction_observable_sparse_array[ii][jj]
                    )
                    msg += 'TS=%.1f, Stellar Mass=%.1f, Fraction=%.2g'%(args)
                logger.debug(msg)
                
                #if coords is not None and distance_modulus is not None:
                #    results = [self.richness_sparse_array[ii][jj],
                #               self.log_likelihood_sparse_array[ii][jj],
                #               self.richness_lower_sparse_array[ii][jj],
                #               self.richness_upper_sparse_array[ii][jj],
                #               self.richness_upper_limit_sparse_array[ii][jj],
                #               richness, log_likelihood, self.loglike.p, self.loglike.f]
                #    return results

            jj_max = self.log_likelihood_sparse_array[ii].argmax()
            args = (
                jj_max+1, npixels, lon[jj_max], lat[jj_max],
                2. * self.log_likelihood_sparse_array[ii][jj_max], 
                self.stellar_mass_conversion * self.richness_sparse_array[ii][jj_max]
            )
            msg = '  (%-3i/%i) Maximum at (%.2f, %.2f) ... TS=%.1f, Stellar Mass=%.1f'%(args)
            logger.info(msg)

    def mle(self):
        a = self.log_likelihood_sparse_array
        j,k = np.unravel_index(a.argmax(),a.shape)
        mle = odict()
        mle['richness'] = self.richness_sparse_array[j][k]
        mle['lon'] = self.roi.pixels_target.lon[k]
        mle['lat'] = self.roi.pixels_target.lat[k]
        mle['distance_modulus'] = self.distance_modulus_array[j]
        mle['extension'] = float(self.loglike.source.extension)
        mle['ellipticity'] = float(self.loglike.source.ellipticity)
        mle['position_angle'] = float(self.loglike.source.position_angle)
        # ADW: FIXME!
        try: 
            mle['age'] = np.mean(self.loglike.source.age)
            mle['metallicity'] = np.mean(self.loglike.source.metallicity)
        except AttributeError:
            mle['age'] = np.nan
            mle['metallicity'] = np.nan
            
        return mle

    def err(self):
        """
        A few rough approximations of the fit uncertainty. These
        values shouldn't be trusted for anything real (use MCMC instead).
        """
        # Initiallize error to nan
        err = odict(self.mle())
        err.update([(k,np.nan*np.ones(2)) for k in err.keys()])

        # Find the maximum likelihood
        a = self.log_likelihood_sparse_array
        j,k = np.unravel_index(a.argmax(),a.shape)

        self.loglike.set_params(distance_modulus=self.distance_modulus_array[j],
                                lon=self.roi.pixels_target.lon[k],
                                lat=self.roi.pixels_target.lat[k])
        self.loglike.sync_params()

        # Find the error in richness, starting at maximum
        lo,hi = np.array(self.loglike.richness_interval())
        err['richness'] = np.array([lo,hi])

        # ADW: This is a rough estimate of the distance uncertainty 
        # hacked to keep the maximum distance modulus on a grid index

        # This is a hack to get the confidence interval to play nice...
        # Require at least three points.
        if (a[:,k]>0).sum() >= 3:
            parabola = Parabola(np.insert(self.distance_modulus_array,0,0.), 
                                np.insert(a[:,k],0,0.) )
            lo,hi = np.array(parabola.confidenceInterval())
            err['distance_modulus'] = self.distance_modulus_array[j] + (hi-lo)/2.*np.array([-1.,1.])

        # ADW: Could estimate lon and lat from the grid.
        # This is just concept right now...
        if (a[j,:]>0).sum() >= 10:
            delta_ts = 2*(a[j,k] - a[j,:])
            pix = np.where(a[j,:][delta_ts < 2.71])[0]
            lons = self.roi.pixels_target.lon[pix]
            lats = self.roi.pixels_target.lat[pix]
            err['lon'] = np.array([ np.min(lons),np.max(lons)])
            err['lat'] = np.array([ np.min(lats),np.max(lats)])

        return err

    def write(self, outfile):
        """
        Save the likelihood results as a sparse HEALPix map.
        """
        data = odict()
        data['PIXEL']=self.roi.pixels_target
        # Full data output (too large for survey)
        if self.config['scan']['full_pdf']:
            data['LOG_LIKELIHOOD']=self.log_likelihood_sparse_array.T
            data['RICHNESS']=self.richness_sparse_array.T
            data['RICHNESS_LOWER']=self.richness_lower_sparse_array.T
            data['RICHNESS_UPPER']=self.richness_upper_sparse_array.T
            data['RICHNESS_LIMIT']=self.richness_upper_limit_sparse_array.T
            #data['STELLAR_MASS']=self.stellar_mass_sparse_array.T
            data['FRACTION_OBSERVABLE']=self.fraction_observable_sparse_array.T
        else:
            data['LOG_LIKELIHOOD']=self.log_likelihood_sparse_array.T
            data['RICHNESS']=self.richness_sparse_array.T
            data['FRACTION_OBSERVABLE']=self.fraction_observable_sparse_array.T

        # Convert to 32bit float
        for k in list(data.keys())[1:]:
            data[k] = data[k].astype('f4',copy=False)
            
        # Stellar mass can be calculated from STELLAR * RICHNESS
        header = odict()
        header['STELLAR']=round(self.stellar_mass_conversion,8)
        header['LKDNSIDE']=self.config['coords']['nside_likelihood']
        header['LKDPIX']=ang2pix(self.config['coords']['nside_likelihood'],
                                 self.roi.lon,self.roi.lat)
        header['NROI']=self.roi.inROI(self.loglike.catalog_roi.lon,
                                      self.loglike.catalog_roi.lat).sum()
        header['NANNULUS']=self.roi.inAnnulus(self.loglike.catalog_roi.lon,
                                              self.loglike.catalog_roi.lat).sum()
        header['NINSIDE']=self.roi.inInterior(self.loglike.catalog_roi.lon,
                                              self.loglike.catalog_roi.lat).sum()
        header['NTARGET']=self.roi.inTarget(self.loglike.catalog_roi.lon,
                                            self.loglike.catalog_roi.lat).sum()

        # Flatten if there is only a single distance modulus
        # ADW: Is this really what we want to do?
        if len(self.distance_modulus_array) == 1:
            for key in data:
                data[key] = data[key].flatten()

        logger.info("Writing %s..."%outfile)
        write_partial_map(outfile,data,
                          nside=self.config['coords']['nside_pixel'],
                          header=header,
                          clobber=True
                          )
        
        fitsio.write(outfile,
                     dict(DISTANCE_MODULUS=self.distance_modulus_array.astype('f4',copy=False)),
                     extname='DISTANCE_MODULUS',
                     clobber=False)

if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for executing the likelihood scan."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_argument('outfile',metavar='outfile.fits',help='Output fits file.')
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True,radius=False)
    opts = parser.parse_args()

    if len(opts.coords) != 1: 
        raise Exception('Must specify exactly one coordinate.')
    lon,lat,radius = opts.coords[0]

    grid = createGridSearch(opts.config,lon,lat)
    if not opts.debug:
        result = grid.search()
        grid.write(opts.outfile)
