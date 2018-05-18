#!/usr/bin/env python
"""
Module for simulation.
"""

import copy
import os

import numpy as np
import scipy.interpolate
import astropy.io.fits as pyfits
import healpy as hp
import numpy.lib.recfunctions as recfuncs
import fitsio

import ugali.observation.catalog
import ugali.observation.mask
import ugali.observation.roi
import ugali.utils.projector
import ugali.utils.stats
import ugali.analysis.scan

from ugali.utils.projector import gal2cel, cel2gal, sr2deg, mod2dist
from ugali.utils.healpix import ang2pix, pix2ang
from ugali.utils.logger import logger
from ugali.utils.config import Config

class Generator:
    """
    Class for generating the parameters of the simulation.
    """
    def __init__(self,config, seed=None):
        self.config = Config(config)
        self.seed = seed
        if self.seed is not None: np.random.seed(self.seed)

    def generate(self, size=1):
        params = dict(self.config['simulate']['params'])
        dtype = [(n,'>f4') for n in params.keys()]
        data = np.zeros(size,dtype=dtype)

        lon,lat = params.pop('lon'),params.pop('lat')
        data['lon'],data['lat'] = self.sky(lon,lat,size)

        for key,value in params.items():
            if value[-1] == 'linear':
                data[key] = self.linear(value[0],value[1],size)
            elif value[-1] == 'log':
                data[key] = self.logarithmic(value[0],value[1],size)
            else:
                raise Exception('...')
        return data

    def sky(self,lon=None,lat=None,size=1):
        logger.info("Generating %i random points..."%size)
        # Random longitue and latitude
        lon,lat = ugali.utils.stats.sky(lon,lat,size=10*size)
        # Random healpix coordinates inside footprint
        nside_pixel = self.config['coords']['nside_pixel']
        pixels = ang2pix(nside_pixel,lon,lat)
        if np.unique(pixels).size > 1:
            inside = ugali.utils.skymap.inFootprint(self.config,pixels,nside=nside_pixel)
        else:
            inside = np.ones(len(pixels),dtype=bool)
        return lon[inside][:size],lat[inside][:size]
        
    def linear(self,low,high,size):
        return np.random.uniform(low,high,size)
    
    def logarithmic(self,low,high,size):
        if low==0 and high==0:
            logger.warning("Can't sample logarithmically with boundary of zero.")
            return np.zeros(size)
        return 10**np.random.uniform(np.log10(low),np.log10(high),size)

    def detectability(self,**kwargs):
        """
        An a priori detectability proxy.
        """
        distance_modulus = kwargs.get('distance_modulus')
        distance = mod2dist(distance_modulus)
        stellar_mass = kwargs.get('stellar_mass')
        extension = kwargs.get('extension')

        # Normalized to 10^3 Msolar at mod=18
        norm = 10**3/mod2dist(18)**2
        detect = stellar_mass / distance**2
        detect /= norm

    def write(self, filename, data=None):
        if data is None: data = self.results
        logger.info("Writing %s..."%filename)
        if filename.endswith('.npy'):
            np.save(filename,data)
        elif filename.endswith('.fits'):
            # Copies data, so be careful..
            out = np.rec.array(data)
            out.dtype.names = np.char.upper(out.dtype.names)
            hdu = pyfits.new_table(out)
            hdu.writeto(filename,clobber=True)
        elif filename.endswith('.txt') or filename.endswith('.dat'):
            np.savetxt(filename,data)
        elif filename.endswith('.csv'):
            np.savetxt(filename,data,delimiter=',')
        else:
            raise Exception('Unrecognized file extension: %s'%filename)

    def run(self, outfile=None, size=None):
        if size is None: size = self.config['simulate']['size']
        data = self.generate(size)
        
        dtype=[('kernel','S18'),('ts','>f4'),('fit_kernel','S18'),('fit_ts','>f4'),
               ('fit_mass','>f4'),('fit_mass_err','>f4'),
               ('fit_distance','>f4'),('fit_distance_err','>f4')]
        results = np.array(np.nan*np.ones(size),dtype=dtype)
        results = recfuncs.merge_arrays([data,results],flatten=True,asrecarray=False,usemask=False)
        self.results = results

        if outfile: self.write(outfile,results)

        for i,d in enumerate(data): 
            params = dict(list(zip(data.dtype.names,d)))
            lon,lat = params['lon'],params['lat']
            distance_modulus = params['distance_modulus']

            logger.info('\n(%i/%i); (lon, lat) = (%.2f, %.2f)'%(i+1,len(data),lon,lat))
            roi = ugali.analysis.loglike.createROI(self.config,lon,lat)
            mask = ugali.analysis.loglike.createMask(self.config,roi)
            isochrone = ugali.analysis.loglike.createIsochrone(self.config)
            kernel = ugali.analysis.loglike.createKernel(self.config,lon,lat)
            pix = roi.indexTarget(lon,lat)

            simulator = Simulator(self.config,roi)
            #catalog   = simulator.simulate(seed=self.seed, **params)
            catalog   = simulator.simulate(**params)
            #print "Catalog annulus contains:",roi.inAnnulus(simulator.catalog.lon,simulator.catalog.lat).sum()
            logger.info("Simulated catalog annulus contains %i stars"%roi.inAnnulus(catalog.lon,catalog.lat).sum())

            if len(catalog.lon) < 1000:
                logger.error("Simulation contains too few objects; skipping...")
                continue

            """
            like = ugali.analysis.loglike.LogLikelihood(self.config, roi, mask, catalog, isochrone, kernel)
            like.set_params(distance_modulus=params['distance_modulus'])
            like.sync_params()
            results[i]['ts'] = 2*like.fit_richness()[0]
            print 'TS=',results[i]['ts'] 
            
            like2 = ugali.analysis.loglike.LogLikelihood(self.config, roi, mask, simulator.catalog, isochrone, kernel)
            like2.set_params(distance_modulus=params['distance_modulus'])
            like2.sync_params()
            print 'TS=',2*like2.fit_richness()[0]
            """
            #return simulator,like,like2

            # Index of closest distance modulus
            grid = ugali.analysis.scan.GridSearch(self.config,roi,mask,catalog,isochrone,kernel)

            self.catalog = catalog
            self.simulator = simulator
            self.grid = grid
            self.loglike = self.grid.loglike

            # ADW: Should allow fit_distance to float in order to model search procedure
            #fit_distance = float(distance_modulus)
            distance_idx = np.fabs(grid.distance_modulus_array-params['distance_modulus']).argmin()
            fit_distance = grid.distance_modulus_array[distance_idx]
            grid.search(coords=(lon,lat),distance_modulus=fit_distance)

            logger.info(str(self.loglike))

            mle = grid.mle()
            results[i]['kernel'] = simulator.kernel.name
            results[i]['fit_kernel'] = grid.loglike.kernel.name
            results[i]['ts'] = 2*grid.log_likelihood_sparse_array[distance_idx][pix]
            results[i]['fit_ts'] = 2*np.max(grid.log_likelihood_sparse_array[:,pix])
            results[i]['fit_mass'] = grid.stellar_mass_conversion*mle['richness']
            results[i]['fit_distance'] = fit_distance #mle['distance_modulus']

            err = grid.err()
            richness_err = (err['richness'][1]-err['richness'][0])/2.
            results[i]['fit_mass_err'] = grid.stellar_mass_conversion*richness_err

            distance_modulus_err = (err['distance_modulus'][1]-err['distance_modulus'][0])/2.
            results[i]['fit_distance_err'] = distance_modulus_err

            for d in dtype:
                logger.info('\t%s: %s'%(d[0], results[i][d[0]]))

            if i%self.config['simulate']['save']==0 and outfile: 
                self.write(outfile,results)

        if outfile: self.write(outfile,results)
            
        return results

############################################################

class Simulator(object):
    """
    Class for simulating catalog data.
    """
    
    def __init__(self, config, roi, **kwargs):
        self.config = ugali.utils.config.Config(config)
        self.roi = roi
        #np.random.seed(0)

        params = dict(self.config)
        if self.config['simulate'].get('isochrone') is None:
            params['simulate']['isochrone'] = params['isochrone']
        if self.config['simulate'].get('kernel') is None:
            params['simulate']['kernel'] = params['kernel']

        self.isochrone = ugali.analysis.loglike.createIsochrone(params)
        self.kernel = ugali.analysis.loglike.createKernel(params['simulate'],lon=self.roi.lon,lat=self.roi.lat)

        self.mask = ugali.analysis.loglike.createMask(self.config,self.roi)

        self._create_catalog(kwargs.get('catalog'))

        self.photo_err_1,self.photo_err_2 = self.mask.photo_err_1,self.mask.photo_err_2
        #self._photometricErrors()
        self._setup_subpix()
        #self._setup_cmd()

    def _create_catalog(self,catalog=None):
        """
        Bundle it.
        """
        if catalog is None:
            catalog = ugali.analysis.loglike.createCatalog(self.config,self.roi)
        cut = self.mask.restrictCatalogToObservableSpace(catalog)
        self.catalog = catalog.applyCut(cut)
        
    def _photometricErrors(self, n_per_bin=100, plot=False):
        """
        Realistic photometric errors estimated from catalog objects and mask.
        Extend below the magnitude threshold with a flat extrapolation.
        """

        self.catalog.spatialBin(self.roi)

        if len(self.catalog.mag_1) < n_per_bin:
            logger.warning("Catalog contains fewer objects than requested to calculate errors.")
            n_per_bin = int(len(self.catalog.mag_1) / 3)

        # Band 1
        mag_1_thresh = self.mask.mask_1.mask_roi_sparse[self.catalog.pixel_roi_index] - self.catalog.mag_1
        sorting_indices = np.argsort(mag_1_thresh)
        mag_1_thresh_sort = mag_1_thresh[sorting_indices]
        mag_err_1_sort = self.catalog.mag_err_1[sorting_indices]

        # ADW: Can't this be done with np.median(axis=?)
        mag_1_thresh_medians = []
        mag_err_1_medians = []
        for i in range(0, int(len(mag_1_thresh) / float(n_per_bin))):
            mag_1_thresh_medians.append(np.median(mag_1_thresh_sort[n_per_bin * i: n_per_bin * (i + 1)]))
            mag_err_1_medians.append(np.median(mag_err_1_sort[n_per_bin * i: n_per_bin * (i + 1)]))
        
        if mag_1_thresh_medians[0] > 0.:
            mag_1_thresh_medians = np.insert(mag_1_thresh_medians, 0, -99.)
            mag_err_1_medians = np.insert(mag_err_1_medians, 0, mag_err_1_medians[0])
        
        self.photo_err_1 = scipy.interpolate.interp1d(mag_1_thresh_medians, mag_err_1_medians,
                                                      bounds_error=False, fill_value=mag_err_1_medians[-1])

        # Band 2
        mag_2_thresh = self.mask.mask_2.mask_roi_sparse[self.catalog.pixel_roi_index] - self.catalog.mag_2
        sorting_indices = np.argsort(mag_2_thresh)
        mag_2_thresh_sort = mag_2_thresh[sorting_indices]
        mag_err_2_sort = self.catalog.mag_err_2[sorting_indices]

        mag_2_thresh_medians = []
        mag_err_2_medians = []
        for i in range(0, int(len(mag_2_thresh) / float(n_per_bin))):
            mag_2_thresh_medians.append(np.median(mag_2_thresh_sort[n_per_bin * i: n_per_bin * (i + 1)]))
            mag_err_2_medians.append(np.median(mag_err_2_sort[n_per_bin * i: n_per_bin * (i + 1)]))

        if mag_2_thresh_medians[0] > 0.:
            mag_2_thresh_medians = np.insert(mag_2_thresh_medians, 0, -99.)
            mag_err_2_medians = np.insert(mag_err_2_medians, 0, mag_err_2_medians[0])
        
        self.photo_err_2 = scipy.interpolate.interp1d(mag_2_thresh_medians, mag_err_2_medians,
                                                      bounds_error=False, fill_value=mag_err_2_medians[-1])


    def _setup_subpix(self,nside=2**16):
        """
        Subpixels for random position generation.
        """
        # Only setup once...
        if hasattr(self,'subpix'): return

        # Simulate over full ROI
        self.roi_radius  = self.config['coords']['roi_radius']

        # Setup background spatial stuff
        logger.info("Setup subpixels...")
        self.nside_pixel = self.config['coords']['nside_pixel']
        self.nside_subpixel = self.nside_pixel * 2**4 # Could be config parameter
        epsilon = np.degrees(hp.max_pixrad(self.nside_pixel)) # Pad roi radius to cover edge healpix
        subpix = ugali.utils.healpix.query_disc(self.nside_subpixel,self.roi.vec,self.roi_radius+epsilon)
        superpix = ugali.utils.healpix.superpixel(subpix,self.nside_subpixel,self.nside_pixel)
        self.subpix = subpix[np.in1d(superpix,self.roi.pixels)]

    def _setup_cmd(self,mode='cloud-in-cells'):
        """
        The purpose here is to create a more finely binned
        background CMD to sample from.
        """
        # Only setup once...
        if hasattr(self,'bkg_lambda'): return

        logger.info("Setup color...")
        # In the limit theta->0: 2*pi*(1-cos(theta)) -> pi*theta**2
        # (Remember to convert from sr to deg^2) 
        #solid_angle_roi = sr2deg(2*np.pi*(1-np.cos(np.radians(self.roi_radius))))
        solid_angle_roi = self.roi.area_pixel*len(self.roi.pixels)

        # Large CMD bins cause problems when simulating
        config = Config(self.config) 
        config['color']['n_bins'] *= 5 #10
        config['mag']['n_bins']   *= 1 #2
        #config['mask']['minimum_solid_angle'] = 0
        roi = ugali.analysis.loglike.createROI(config,self.roi.lon,self.roi.lat)
        mask = ugali.analysis.loglike.createMask(config,roi)

        self.bkg_centers_color  = roi.centers_color
        self.bkg_centers_mag    = roi.centers_mag

        # Background CMD has units: [objs / deg^2 / mag^2]
        cmd_background = mask.backgroundCMD(self.catalog,mode)
        
        self.bkg_lambda=cmd_background*solid_angle_roi*roi.delta_color*roi.delta_mag
        np.sum(self.bkg_lambda)

        # Clean up 
        del config, roi, mask


    def toy_background(self,mc_source_id=2,seed=None):
        """
        Quick uniform background generation.
        """

        logger.info("Running toy background simulation...")
        size = 20000
        nstar = np.random.poisson(size)
        #np.random.seed(0)
        logger.info("Simulating %i background stars..."%nstar)

        ### # Random points from roi pixels
        ### idx = np.random.randint(len(self.roi.pixels)-1,size=nstar)
        ### pix = self.roi.pixels[idx]

        # Random points drawn from subpixels
        logger.info("Generating uniform positions...")
        idx = np.random.randint(0,len(self.subpix)-1,size=nstar)
        lon,lat = pix2ang(self.nside_subpixel,self.subpix[idx])

        pix = ang2pix(self.nside_pixel, lon, lat)
        lon,lat = pix2ang(self.nside_pixel,pix)

        # Single color
        #mag_1 = 19.05*np.ones(len(pix))
        #mag_2 = 19.10*np.ones(len(pix))

        # Uniform in color
        logger.info("Generating uniform CMD...")
        mag_1 = np.random.uniform(self.config['mag']['min'],self.config['mag']['max'],size=nstar)
        color = np.random.uniform(self.config['color']['min'],self.config['color']['max'],size=nstar)

        mag_2 = mag_1 - color

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * np.ones(hp.nside2npix(self.nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * np.ones(hp.nside2npix(self.nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]
        
        #mag_err_1 = 1.0*np.ones(len(pix))
        #mag_err_2 = 1.0*np.ones(len(pix))
        mag_err_1 = self.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.photo_err_2(mag_lim_2 - mag_2)
        mc_source_id = mc_source_id * np.ones(len(mag_1))

        select = (mag_lim_1>mag_1)&(mag_lim_2>mag_2)

        hdu = ugali.observation.catalog.makeHDU(self.config,mag_1[select],mag_err_1[select],
                                                mag_2[select],mag_err_2[select],
                                                lon[select],lat[select],mc_source_id[select])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog


    def background(self,mc_source_id=2,seed=None):
        """
        Create a simulation of the background stellar population.
        Because some stars have been clipped to generate the CMD,
        this function tends to slightly underestimate (~1%) the 
        background as compared to the true catalog.

        The simulation of background object colors relies on the
        data-derived CMD. As such, it is a binned random generator
        and thus has some fundamental limitations.
        - The expected number of counts per bin is drawn ra

        There are a few limitations of this procedure:
        - Colors are drawn from the CMD of the background annulus
        - The number of stars per CMD bin is randomized according to the CMD
        - The colors/mags are then uniformly distributed within the bin
        - This leads to trouble with large bins when the cloud-in-cells 
        algorithm is applied to the simulated data
        - The positions are chosen randomly over the spherical cap of the ROI
        - Objects that are outside of the 

        WARNING: The cloud-in-cells method of generating
        the CMD leads to some difficulties since it disperses
        objects from high-density zones to low density zones.

        - Magnitudes are not randomized according to their errors
        """
        if seed is not None: np.random.seed(seed)
        self._setup_cmd()

        # Randomize the number of stars per bin according to Poisson distribution
        nstar_per_bin = np.random.poisson(lam=self.bkg_lambda)
        nstar = nstar_per_bin.sum()

        logger.info("Simulating %i background stars..."%nstar)

        if not self.config['simulate'].get('uniform'):
            logger.info("Generating colors from background CMD.")

            # Distribute the stars within each CMD bin
            delta_color = self.bkg_centers_color[1]-self.bkg_centers_color[0]
            delta_mag   = self.bkg_centers_mag[1]-self.bkg_centers_mag[0]

            # Distribute points within each color-mag bins
            xx,yy = np.meshgrid(self.bkg_centers_color,self.bkg_centers_mag)
            color = np.repeat(xx.flatten(),repeats=nstar_per_bin.flatten())
            color += np.random.uniform(-delta_color/2.,delta_color/2.,size=nstar)
            mag_1 = np.repeat(yy.flatten(),repeats=nstar_per_bin.flatten())
            mag_1 += np.random.uniform(-delta_mag/2.,delta_mag/2.,size=nstar)
        else:
            # Uniform color-magnitude distribution
            logger.info("Generating uniform CMD.")
            mag_1 = np.random.uniform(self.config['mag']['min'],self.config['mag']['max'],size=nstar)
            color = np.random.uniform(self.config['color']['min'],self.config['color']['max'],size=nstar)

        mag_2 = mag_1 - color

        # Random points drawn from healpix subpixels
        logger.info("Generating uniform positions...")
        idx = np.random.randint(0,len(self.subpix)-1,size=nstar)
        lon,lat = pix2ang(self.nside_subpixel,self.subpix[idx])

        nside_pixel = self.nside_pixel
        pix = ang2pix(nside_pixel, lon, lat)

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * np.ones(hp.nside2npix(nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * np.ones(hp.nside2npix(nside_pixel))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]

        mag_err_1 = self.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.photo_err_2(mag_lim_2 - mag_2)
        mc_source_id = mc_source_id * np.ones(len(mag_1))

        # ADW: Should magnitudes be randomized by the erros?
        #mag_1 += (np.random.normal(size=len(mag_1)) * mag_err_1)
        #mag_2 += (np.random.normal(size=len(mag_2)) * mag_err_2)

        select = (mag_lim_1>mag_1)&(mag_lim_2>mag_2)

        ### # Make sure objects lie within the original cmd (should be done later...)
        ### select &= (ugali.utils.binning.take2D(self.mask.solid_angle_cmd, color, mag_1,
        ###                                       self.roi.bins_color, self.roi.bins_mag) > 0)

        logger.info("Clipping %i simulated background stars..."%(~select).sum())
        
        hdu = ugali.observation.catalog.makeHDU(self.config,mag_1[select],mag_err_1[select],
                                                mag_2[select],mag_err_2[select],
                                                lon[select],lat[select],mc_source_id[select])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog

    def satellite(self,stellar_mass,distance_modulus,mc_source_id=1,seed=None,**kwargs):
        """
        Create a simulated satellite. Returns a catalog object.
        """
        if seed is not None: np.random.seed(seed)

        isochrone = kwargs.pop('isochrone',self.isochrone)
        kernel    = kwargs.pop('kernel',self.kernel)

        for k,v in kwargs.items():
            if k in kernel.params.keys(): setattr(kernel,k,v)

        mag_1, mag_2 = isochrone.simulate(stellar_mass, distance_modulus)
        lon, lat     = kernel.simulate(len(mag_1))
 
        logger.info("Simulating %i satellite stars..."%len(mag_1))
        pix = ang2pix(self.config['coords']['nside_pixel'], lon, lat)

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * np.ones(hp.nside2npix(self.config['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * np.ones(hp.nside2npix(self.config['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]

        mag_err_1 = self.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.photo_err_2(mag_lim_2 - mag_2)

        # Randomize magnitudes by their errors
        mag_obs_1 = mag_1+np.random.normal(size=len(mag_1))*mag_err_1
        mag_obs_2 = mag_2+np.random.normal(size=len(mag_2))*mag_err_2
        #mag_obs_1 = mag_1
        #mag_obs_2 = mag_2

        #select = np.logical_and(mag_obs_1 < mag_lim_1, mag_obs_2 < mag_lim_2)
        select = (mag_lim_1>mag_obs_1)&(mag_lim_2>mag_obs_2)

        # Make sure objects lie within the original cmd (should also be done later...)
        #select &= (ugali.utils.binning.take2D(self.mask.solid_angle_cmd, mag_obs_1 - mag_obs_2, mag_obs_1,self.roi.bins_color, self.roi.bins_mag) > 0)

        #return mag_1_obs[cut], mag_2_obs[cut], lon[cut], lat[cut]
        logger.info("Clipping %i simulated satellite stars..."%(~select).sum())
        mc_source_id = mc_source_id * np.ones(len(mag_1))
        
        hdu = ugali.observation.catalog.makeHDU(self.config,mag_obs_1[select],mag_err_1[select],
                                                mag_obs_2[select],mag_err_2[select], 
                                                lon[select],lat[select],mc_source_id[select])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog


    def satellite2(self,stellar_mass,distance_modulus,mc_source_id=1,seed=None,**kwargs):
        """
        Create a simulated satellite. Returns a catalog object.
        """
        if seed is not None: np.random.seed(seed)

        isochrone = kwargs.pop('isochrone',self.isochrone)
        kernel    = kwargs.pop('kernel',self.kernel)

        for k,v in kwargs.items():
            if k in kernel.params.keys(): setattr(kernel,k,v)

        mag_1, mag_2 = isochrone.simulate(stellar_mass, distance_modulus)
        lon, lat     = kernel.simulate(len(mag_1))

        logger.info("Simulating %i satellite stars..."%len(mag_1))
        pix = ang2pix(self.config['coords']['nside_pixel'], lon, lat)

        # There is probably a better way to do this step without creating the full HEALPix map
        mask = -1. * np.ones(hp.nside2npix(self.config['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_1.mask_roi_sparse
        mag_lim_1 = mask[pix]
        mask = -1. * np.ones(hp.nside2npix(self.config['coords']['nside_pixel']))
        mask[self.roi.pixels] = self.mask.mask_2.mask_roi_sparse
        mag_lim_2 = mask[pix]

        mag_err_1 = self.mask.photo_err_1(mag_lim_1 - mag_1)
        mag_err_2 = self.mask.photo_err_2(mag_lim_2 - mag_2)

        # Completeness is a function of true magnitude
        method = 'step'
        if method is None or method == 'none':
            comp = np.ones(len(mag_1))
        elif self.config['catalog']['band_1_detection']:
            comp=self.mask.completeness(mag_lim_1-mag_1, method=method)
        elif not self.config['catalog']['band_1_detection']:
            comp=self.mask.completeness(mag_lim_2-mag_2, method=method)
        else:
            comp_1 = self.mask.completeness(mag_lim_1-mag_1, method=method)
            comp_2 = self.mask.completeness(mag_lim_2-mag_2, method=method)
            comp = comp_1*comp_2
        accept = comp > 1 - np.random.uniform(size=len(mag_1))

        # Randomize magnitudes by their errors
        mag_obs_1 = mag_1 + (np.random.normal(size=len(mag_1))*mag_err_1)
        mag_obs_2 = mag_2 + (np.random.normal(size=len(mag_2))*mag_err_2)

        #select = np.logical_and(mag_obs_1 < mag_lim_1, mag_obs_2 < mag_lim_2)
        select = (mag_lim_1>mag_obs_1)&(mag_lim_2>mag_obs_2)&accept

        ### # Make sure objects lie within the original cmd (should also be done later...)
        ### select &= (ugali.utils.binning.take2D(self.mask.solid_angle_cmd, color, mag_1,
        ###                                       self.roi.bins_color, self.roi.bins_mag) > 0)

        #return mag_1_obs[cut], mag_2_obs[cut], lon[cut], lat[cut]
        logger.info("Clipping %i simulated satellite stars..."%(~select).sum())
        mc_source_id = mc_source_id * np.ones(len(mag_1))
        
        hdu = ugali.observation.catalog.makeHDU(self.config,mag_obs_1[select],mag_err_1[select],
                                                mag_obs_2[select],mag_err_2[select], 
                                                lon[select],lat[select],mc_source_id[select])
        catalog = ugali.observation.catalog.Catalog(self.config, data=hdu.data)
        return catalog


    def simulate(self, seed=None, **kwargs):
        if seed is not None: np.random.seed(seed)

        logger.info("Simulating object catalog...")
        catalogs = []
        #catalogs.append(self.toy_background(seed=seed))
        catalogs.append(self.background(seed=seed))
        catalogs.append(self.satellite(seed=seed,**kwargs))
        logger.info("Merging simulated catalogs...")
        catalog = ugali.observation.catalog.mergeCatalogs(catalogs)
        nsig = (catalog.mc_source_id == 1).sum()
        nbkg = (catalog.mc_source_id == 2).sum()
        logger.info("Simulated catalog contains: %i background stars"%nbkg)
        logger.info("Simulated catalog contains: %i satellite stars"%nsig)
        return catalog

    def makeHDU(self, mag_1, mag_err_1, mag_2, mag_err_2, lon, lat, mc_source_id):
        """
        Create a catalog fits file object based on input data.

        ADW: This should be combined with the write_membership
        function of loglike.
        """
        if self.config['catalog']['coordsys'].lower() == 'cel' \
           and self.config['coords']['coordsys'].lower() == 'gal':
            lon, lat = ugali.utils.projector.gal2cel(lon, lat)
        elif self.config['catalog']['coordsys'].lower() == 'gal' \
           and self.config['coords']['coordsys'].lower() == 'cel':
            lon, lat = ugali.utils.projector.cel2gal(lon, lat)

        columns = [
            pyfits.Column(name=self.config['catalog']['objid_field'],
                          format = 'D',array = np.arange(len(lon))),
            pyfits.Column(name=self.config['catalog']['lon_field'],
                          format = 'D',array = lon),
            pyfits.Column(name = self.config['catalog']['lat_field'],          
                          format = 'D',array = lat), 
            pyfits.Column(name = self.config['catalog']['mag_1_field'],        
                          format = 'E',array = mag_1),
            pyfits.Column(name = self.config['catalog']['mag_err_1_field'],    
                          format = 'E',array = mag_err_1),
            pyfits.Column(name = self.config['catalog']['mag_2_field'],        
                          format = 'E',array = mag_2),
            pyfits.Column(name = self.config['catalog']['mag_err_2_field'],    
                          format = 'E',array = mag_err_2),
            pyfits.Column(name = self.config['catalog']['mc_source_id_field'], 
                          format = 'I',array = mc_source_id),
        ]

        hdu = pyfits.new_table(columns)
        return hdu


    def write(self, outfile):
        """
        """
        pass

############################################################

class Analyzer(object):
    """
    Class for generating the parameters of the simulation.
    """
    def __init__(self, config, seed=None):
        self.config = Config(config)

    def create_population(self):
        if self.config['simulate']['popfile']:
            filename = os.path.join(self.config['simulate']['dirname'],self.config['simulate']['popfile'])
            population = fitsio.read(filename)
        else:
            size = self.config['simulate']['size']
            population = self.generate(size)
            
        self.population = population

        return self.population

    def write(self, filename, data=None):
        """ Write the output results """
        if data is None: data = self.results
        logger.info("Writing %s..."%filename)
        if filename.endswith('.npy'):
            np.save(filename,data)
        elif filename.endswith('.fits'):
            # Copies data, so be careful..
            out = np.rec.array(data)
            out.dtype.names = np.char.upper(out.dtype.names)
            hdu = pyfits.new_table(out)
            hdu.writeto(filename,clobber=True)
        elif filename.endswith('.txt') or filename.endswith('.dat'):
            np.savetxt(filename,data)
        elif filename.endswith('.csv'):
            np.savetxt(filename,data,delimiter=',')
        else:
            raise Exception('Unrecognized file extension: %s'%filename)

    def run(self, catalog=None, outfile=None):
        #if size is None: size = self.config['simulate']['size']
        #data = self.generate(size)
        data = self.create_population()
        size = len(data)

        dtype=[('kernel','S18'),('ts','>f4'),('fit_kernel','S18'),('fit_ts','>f4'),
               ('fit_mass','>f4'),('fit_mass_err','>f4'),
               ('fit_distance','>f4'),('fit_distance_err','>f4')]
        results = np.array(np.nan*np.ones(size),dtype=dtype)
        results = recfuncs.merge_arrays([data,results],flatten=True,asrecarray=False,usemask=False)
        self.results = results

        if outfile: self.write(outfile,results)

        for i,d in enumerate(data): 
            params = dict(list(zip(data.dtype.names,d)))
            lon,lat = params['ra'],params['dec']
            distance_modulus = params['distance_modulus']

            logger.info('\n(%i/%i); (lon, lat) = (%.2f, %.2f)'%(i+1,len(data),lon,lat))
            roi = ugali.analysis.loglike.createROI(self.config,lon,lat)
            mask = ugali.analysis.loglike.createMask(self.config,roi)
            isochrone = ugali.analysis.loglike.createIsochrone(self.config)
            kernel = ugali.analysis.loglike.createKernel(self.config,lon=lon,lat=lat)
            pix = roi.indexTarget(lon,lat)

            if not config['simulate']['catfile']:
                simulator = Simulator(self.config,roi)
                #catalog   = simulator.simulate(seed=self.seed, **params)
                catalog   = simulator.simulate(**params)
                #print "Catalog annulus contains:",roi.inAnnulus(simulator.catalog.lon,simulator.catalog.lat).sum()
            else:
                
                pass
            import pdb; pdb.set_trace()
            logger.info("Simulated catalog annulus contains %i stars"%roi.inAnnulus(catalog.lon,catalog.lat).sum())

            if len(catalog.lon) < 1000:
                logger.error("Simulation contains too few objects; skipping...")
                continue

            """
            like = ugali.analysis.loglike.LogLikelihood(self.config, roi, mask, catalog, isochrone, kernel)
            like.set_params(distance_modulus=params['distance_modulus'])
            like.sync_params()
            results[i]['ts'] = 2*like.fit_richness()[0]
            print 'TS=',results[i]['ts'] 
            
            like2 = ugali.analysis.loglike.LogLikelihood(self.config, roi, mask, simulator.catalog, isochrone, kernel)
            like2.set_params(distance_modulus=params['distance_modulus'])
            like2.sync_params()
            print 'TS=',2*like2.fit_richness()[0]
            """
            #return simulator,like,like2

            # Index of closest distance modulus
            grid = ugali.analysis.scan.GridSearch(self.config,roi,mask,catalog,isochrone,kernel)

            self.catalog = catalog
            self.simulator = simulator
            self.grid = grid
            self.loglike = self.grid.loglike

            # ADW: Should allow fit_distance to float in order to model search procedure
            #fit_distance = float(distance_modulus)
            distance_idx = np.fabs(grid.distance_modulus_array-params['distance_modulus']).argmin()
            fit_distance = grid.distance_modulus_array[distance_idx]
            grid.search(coords=(lon,lat),distance_modulus=fit_distance)

            logger.info(str(self.loglike))

            mle = grid.mle()
            results[i]['kernel'] = simulator.kernel.name
            results[i]['fit_kernel'] = grid.loglike.kernel.name
            results[i]['ts'] = 2*grid.log_likelihood_sparse_array[distance_idx][pix]
            results[i]['fit_ts'] = 2*np.max(grid.log_likelihood_sparse_array[:,pix])
            results[i]['fit_mass'] = grid.stellar_mass_conversion*mle['richness']
            results[i]['fit_distance'] = fit_distance #mle['distance_modulus']

            err = grid.err()
            richness_err = (err['richness'][1]-err['richness'][0])/2.
            results[i]['fit_mass_err'] = grid.stellar_mass_conversion*richness_err

            distance_modulus_err = (err['distance_modulus'][1]-err['distance_modulus'][0])/2.
            results[i]['fit_distance_err'] = distance_modulus_err

            for d in dtype:
                logger.info('\t%s: %s'%(d[0], results[i][d[0]]))

            if i%self.config['simulate']['save']==0 and outfile: 
                self.write(outfile,results)

        if outfile: self.write(outfile,results)
            
        return results
    


############################################################

def satellite(isochrone, kernel, stellar_mass, distance_modulus,**kwargs):
    """
    Wrapping the isochrone and kernel simulate functions.
    """
    mag_1, mag_2 = isochrone.simulate(stellar_mass, distance_modulus)
    lon, lat     = kernel.simulate(len(mag_1))

    return mag_1, mag_2, lon, lat

############################################################


if __name__ == "__main__":
    import ugali.utils.parser
    description = "Script for executing the likelihood scan."
    parser = ugali.utils.parser.Parser(description=description)
    parser.add_config()
    parser.add_argument('outfile',metavar='outfile.fits',help='Output fits file.')
    parser.add_debug()
    parser.add_verbose()
    parser.add_seed()

    opts = parser.parse_args()
    config = Config(opts.config)
    generator = Generator(config,opts.seed)
    sim,like1,like2 = generator.run(opts.outfile)
    
