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

from ugali.utils.batch import LSF
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
        self.source = self.loglike.source
        self.roi  = self.loglike.roi
        self.mask = self.loglike.mask

        logger.info(str(self.loglike))

        self.stellar_mass_conversion = self.source.stellar_mass()
        self.distance_modulus_array = np.asarray(self.config['scan']['distance_modulus_array'])
        self.extension_array = np.asarray(self.config['scan'].get('extension_array',[self.source.extension]))


    def search(self, coords=None, distance_modulus=None, extension=None, tolerance=1.e-2):
        """
        Organize a grid search over ROI target pixels, distance
        moduli, and extensions. If coords, distance_modulus, or
        extension is specified, then the nearest value in the
        predefined scan grid is used. ***This may be different than
        the input value.**

        Parameters
        ----------
        coords : (float,float)
            coordinate to search (matched to nearest scan value)
        distance_modulus : float
            distance modulus to search (matched to nearest scan value)
        extension : float
            extension to search (matched to nearest scan value)
        tolerance : float
            tolerance on richness maximization

        Returns
        -------
        None 
        """
        nmoduli = len(self.distance_modulus_array)
        npixels = len(self.roi.pixels_target)
        self.loglike_array              = np.zeros([nmoduli,npixels],dtype='f4')
        self.richness_array             = np.zeros([nmoduli,npixels],dtype='f4')
        self.stellar_mass_array         = np.zeros([nmoduli,npixels],dtype='f4')
        self.fraction_observable_array  = np.zeros([nmoduli,npixels],dtype='f4')
        self.extension_fit_array        = np.zeros([nmoduli,npixels],dtype='f4')
        if self.config['scan']['full_pdf']:
            # DEPRECATED: ADW 2019-04-27
            DeprecationWarning("'full_pdf' is deprecated.")
            self.richness_lower_array       = np.zeros([nmoduli,npixels],dtype='f4')
            self.richness_upper_array       = np.zeros([nmoduli,npixels],dtype='f4')
            self.richness_ulimit_array      = np.zeros([nmoduli,npixels],dtype='f4')

        # Specific pixel/distance_modulus
        coord_idx, distance_modulus_idx, extension_idx = None, None, None
        if coords is not None:
            # Match to nearest grid coordinate index
            coord_idx = self.roi.indexTarget(coords[0],coords[1])
        if distance_modulus is not None:
            # Match to nearest distance modulus index
            distance_modulus_idx=np.fabs(self.distance_modulus_array-distance_modulus).argmin()
        if extension is not None:
            # Match to nearest extension
            extension_idx=np.fabs(self.extension_array-extension).argmin()

        lon, lat = self.roi.pixels_target.lon, self.roi.pixels_target.lat
            
        logger.info('Looping over distance moduli in grid search ...')
        for ii, distance_modulus in enumerate(self.distance_modulus_array):
            # Specific distance
            if distance_modulus_idx is not None:
                if ii != distance_modulus_idx: continue

            logger.info('  (%-2i/%i) Distance Modulus=%.1f ...'%(ii+1,nmoduli,distance_modulus))

            # No objects, continue
            if len(self.loglike.catalog) == 0: 
                logger.warn("No catalog objects")
                continue

            # Set distance_modulus once to save time
            self.loglike.set_params(distance_modulus=distance_modulus)
            # Loop over pixels
            for jj in range(0, npixels):
                # Specific pixel
                if coord_idx is not None:
                    if jj != coord_idx: continue

                # Set kernel location
                self.loglike.set_params(lon=lon[jj],lat=lat[jj])

                loglike = 0
                # Loop over extensions
                for kk,ext in enumerate(self.extension_array):
                    # Specific extension
                    if extension_idx is not None:
                        if kk != extension_idx: continue

                    # Set extension
                    self.loglike.set_params(extension=ext)

                    # Doesn't re-sync distance_modulus each time
                    self.loglike.sync_params()

                    # Maximize the likelihood with respect to richness
                    loglike,rich,p = self.loglike.fit_richness()

                    if loglike < self.loglike_array[ii][jj]:
                        # No loglike increase, continue
                        continue

                    self.loglike_array[ii][jj],self.richness_array[ii][jj], parabola = loglike,rich,p
                    self.stellar_mass_array[ii][jj] = self.stellar_mass_conversion*self.richness_array[ii][jj]
                    self.fraction_observable_array[ii][jj] = self.loglike.f
                    self.extension_fit_array[ii][jj] = self.source.extension

                # ADW: Careful, we are leaving the extension at the
                # last value in the array, not at the maximum...

                # Debug output
                args = (jj+1, npixels, lon[jj], lat[jj],
                        2.*self.loglike_array[ii][jj], 
                        self.stellar_mass_array[ii][jj],
                        self.fraction_observable_array[ii][jj],
                        self.extension_fit_array[ii][jj]
                        )
                msg  = '    (%-3i/%i) Candidate at (%.2f, %.2f) ... '
                msg += 'TS=%.1f, Mstar=%.2g, ObsFrac=%.2g, Ext=%.2g'
                logger.debug(msg%args)

                """
                # This is debugging output
                if self.config['scan']['full_pdf']:
                    DeprecationWarning("'full_pdf' is deprecated.")
                    self.richness_lower_array[ii][jj], self.richness_upper_array[ii][jj] = self.loglike.richness_interval(0.6827)
                    
                    self.richness_ulimit_array[ii][jj] = parabola.bayesianUpperLimit(0.95)

                    args = (
                        2. * self.loglike_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_lower_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_upper_array[ii][jj],
                        self.stellar_mass_conversion*self.richness_ulimit_array[ii][jj]
                    )
                    msg = 'TS=%.1f, Stellar Mass=%.1f (%.1f -- %.1f @ 0.68 CL, < %.1f @ 0.95 CL)'%(args)
                    logger.debug(msg)
                """
                
            jj_max = self.loglike_array[ii].argmax()
            args = (
                jj_max+1, npixels, lon[jj_max], lat[jj_max],
                2. * self.loglike_array[ii][jj_max], 
                self.stellar_mass_conversion * self.richness_array[ii][jj_max],
                self.extension_fit_array[ii][jj_max]
            )
            msg = '  (%-3i/%i) Max at (%.2f, %.2f) : TS=%.1f, Mstar=%.2g, Ext=%.2f'%(args)
            logger.info(msg)

    def mle(self):
        a = self.loglike_array
        j,k = np.unravel_index(a.argmax(),a.shape)
        mle = odict()
        mle['richness'] = self.richness_array[j][k]
        mle['lon'] = self.roi.pixels_target.lon[k]
        mle['lat'] = self.roi.pixels_target.lat[k]
        mle['distance_modulus'] = self.distance_modulus_array[j]
        mle['extension'] = float(self.source.extension)
        mle['ellipticity'] = float(self.source.ellipticity)
        mle['position_angle'] = float(self.source.position_angle)
        # ADW: FIXME!
        try: 
            mle['age'] = np.mean(self.source.age)
            mle['metallicity'] = np.mean(self.source.metallicity)
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
        a = self.loglike_array
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
            DeprecationWarning("'full_pdf' is deprecated.")
            data['LOG_LIKELIHOOD']=self.loglike_array.T
            data['RICHNESS']=self.richness_array.T
            data['RICHNESS_LOWER']=self.richness_lower_array.T
            data['RICHNESS_UPPER']=self.richness_upper_array.T
            data['RICHNESS_LIMIT']=self.richness_ulimit_array.T
            #data['STELLAR_MASS']=self.stellar_mass_array.T
            data['FRACTION_OBSERVABLE']=self.fraction_observable_array.T
        else:
            data['LOG_LIKELIHOOD']=self.loglike_array.T
            data['RICHNESS']=self.richness_array.T
            data['FRACTION_OBSERVABLE']=self.fraction_observable_array.T

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
    parser.add_argument('-m','--mlimit',nargs='?',default=-1, type=int,
                        help='Memory limit (KB)')
    parser.add_debug()
    parser.add_verbose()
    parser.add_coords(required=True,radius=False)
    opts = parser.parse_args()

    if len(opts.coords) != 1: 
        raise Exception('Must specify exactly one coordinate.')
    lon,lat,radius = opts.coords[0]

    #if opts.mlimit > 0:
    if True:
        # Convert from KB to GB
        mlimit,_ = LSF.get_memory_limit()
        logger.info("Setting memory limit: %.1fGB"%(mlimit/1024.**3))
        soft,hard = LSF.set_memory_limit(mlimit)
        logger.info("Memory limit: %.1fGB"%(soft/1024.**3))

    grid = createGridSearch(opts.config,lon,lat)
    if not opts.debug:
        result = grid.search()
        grid.write(opts.outfile)
