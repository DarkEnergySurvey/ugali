"""
Classes and functions that handle masks (i.e., exposure depth). 

Classes
    Mask

Functions
    someFunction
"""

import os
import numpy as np
import healpy as hp
import scipy.signal

#import ugali.utils.plotting
import ugali.utils.binning
import ugali.utils.skymap

import ugali.observation.roi
from ugali.utils import healpix

from ugali.utils.logger import logger
from ugali.utils.healpix import ang2pix
from ugali.utils.config import Config
from ugali.utils.constants import MAGERR_PARAMS
############################################################

class Mask(object):
    """
    Contains maps of completeness depth in magnitudes for multiple observing bands, and associated products.
    """
    def __init__(self, config, roi):
        self.config = Config(config)
        self.roi = roi
        filenames = self.config.getFilenames()
        catalog_pixels = self.roi.getCatalogPixels()

        self.mask_1 = MaskBand(filenames['mask_1'][catalog_pixels],self.roi)
        self.mask_2 = MaskBand(filenames['mask_2'][catalog_pixels],self.roi)
        self._fracRoiSparse()

        self.minimum_solid_angle = self.config.params['mask']['minimum_solid_angle'] # deg^2

        # FIXME: Need to parallelize CMD and MMD formulation
        self._solidAngleCMD()
        self._pruneCMD(self.minimum_solid_angle)
        
        #self._solidAngleMMD()
        #self._pruneMMD(self.minimum_solid_angle)

        self._photometricErrors()

    @property
    def mask_roi_unique(self):
        """
        Assemble a set of unique magnitude tuples for the ROI
        """
        # There is no good inherent way in numpy to do this...
        # http://stackoverflow.com/q/16970982/

        # Also possible and simple:
        #return np.unique(zip(self.mask_1.mask_roi_sparse,self.mask_2.mask_roi_sparse))

        A = np.vstack([self.mask_1.mask_roi_sparse,self.mask_2.mask_roi_sparse]).T
        B = A[np.lexsort(A.T[::-1])]
        return B[np.concatenate(([True],np.any(B[1:]!=B[:-1],axis=1)))]

    @property
    def mask_roi_digi(self):
        """
        Get the index of the unique magnitude tuple for each pixel in the ROI.
        """
        # http://stackoverflow.com/q/24205045/#24206440
        A = np.vstack([self.mask_1.mask_roi_sparse,self.mask_2.mask_roi_sparse]).T
        B = self.mask_roi_unique

        AA = np.ascontiguousarray(A)
        BB = np.ascontiguousarray(B)
         
        dt = np.dtype((np.void, AA.dtype.itemsize * AA.shape[1]))
        a = AA.view(dt).ravel()
        b = BB.view(dt).ravel()
         
        idx = np.argsort(b)
        indices = np.searchsorted(b[idx],a)
        return idx[indices]

    @property
    def frac_annulus_sparse(self):
        return self.frac_roi_sparse[self.roi.pixel_annulus_cut]
     
    @property
    def frac_interior_sparse(self):
        return self.frac_roi_sparse[self.roi.pixel_interior_cut]

    def _fracRoiSparse(self):
        """
        Calculate an approximate pixel coverage fraction from the two masks.

        We have no way to know a priori how much the coverage of the
        two masks overlap in a give pixel. For example, masks that each
        have frac = 0.5 could have a combined frac = [0.0 to 0.5]. 
        The limits will be: 
          max:  min(frac1,frac2)
          min:  max((frac1+frac2)-1, 0.0)

        Sometimes we are lucky and our fracdet is actually already
        calculated for the two masks combined, so that the max
        condition is satisfied. That is what we will assume...
        """
        self.frac_roi_sparse = np.min([self.mask_1.frac_roi_sparse,self.mask_2.frac_roi_sparse],axis=0)
        return self.frac_roi_sparse

    def _solidAngleMMD(self):
        """
        Compute solid angle within the mask annulus (deg^2) as a
        function of mag_1 and mag_2
        """
        # Take upper corner of the magnitude bin
        mag_2,mag_1 = np.meshgrid(self.roi.bins_mag[1:],self.roi.bins_mag[1:])

        # Havent tested since adding fracdet
        unmasked_mag_1 = (self.mask_1.mask_annulus_sparse[:,np.newaxis]>mag_1[:,np.newaxis])
        unmasked_mag_2 = (self.mask_2.mask_annulus_sparse[:,np.newaxis]>mag_2[:,np.newaxis])
        n_unmasked_pixels = (unmasked_mag_1*unmasked_mag_2*self.frac_annulus_sparse).sum(axis=1)

        self.solid_angle_mmd = self.roi.area_pixel * n_unmasked_pixels

        if self.solid_angle_mmd.sum() == 0:
            msg = "Mask annulus contains no solid angle."
            logger.error(msg)
            raise Exception(msg)

    def _pruneMMD(self, minimum_solid_angle):
        """
        Remove regions of magnitude-magnitude space where the unmasked solid angle is
        statistically insufficient to estimate the background.

        INPUTS:
            solid_angle[1]: minimum solid angle (deg^2)
        """

        logger.info('Pruning mask based on minimum solid angle of %.2f deg^2'%(minimum_solid_angle))

        solid_angle_mmd = self.solid_angle_mmd*(self.solid_angle_mmd > minimum_solid_angle)
        if solid_angle_mmd.sum() == 0:
            msg = "Pruned mask contains no solid angle."
            logger.error(msg)
            raise Exception(msg)

        self.solid_angle_mmd = solid_angle_mmd

        # Compute which magnitudes the clipping correspond to
        index_mag_1, index_mag_2 = np.nonzero(self.solid_angle_mmd)
        self.mag_1_clip = self.roi.bins_mag[1:][np.max(index_mag_1)]
        self.mag_2_clip = self.roi.bins_mag[1:][np.max(index_mag_2)]

        logger.info('Clipping mask 1 at %.2f mag'%(self.mag_1_clip) )
        logger.info('Clipping mask 2 at %.2f mag'%(self.mag_2_clip) )
        self.mask_1.mask_roi_sparse = np.clip(self.mask_1.mask_roi_sparse, 0., self.mag_1_clip)
        self.mask_2.mask_roi_sparse = np.clip(self.mask_2.mask_roi_sparse, 0., self.mag_2_clip)

    def _solidAngleCMD(self):
        """
        Compute solid angle within the mask annulus (deg^2) as a
        function of color and magnitude.
        """

        self.solid_angle_cmd = np.zeros([len(self.roi.centers_mag),
                                            len(self.roi.centers_color)])

        for index_mag in np.arange(len(self.roi.centers_mag)):
            for index_color in np.arange(len(self.roi.centers_color)):
                # mag and color at bin center
                mag = self.roi.centers_mag[index_mag]
                color = self.roi.centers_color[index_color]

                if self.config.params['catalog']['band_1_detection']:
                    # Evaluating at the center of the color-mag bin, be consistent!
                    #mag_1 = self.roi.centers_mag[index_mag]
                    #color = self.roi.centers_color[index_color]
                    #mag_2 = mag_1 - color
                    # Evaluating at corner of the color-mag bin, be consistent!
                    mag_1 = mag + (0.5 * self.roi.delta_mag)
                    mag_2 = mag - color + (0.5 * self.roi.delta_color)
                else:
                    # Evaluating at the center of the color-mag bin, be consistent!
                    #mag_2 = self.roi.centers_mag[index_mag]
                    #color = self.roi.centers_color[index_color]
                    #mag_1 = mag_2 + color
                    # Evaluating at corner of the color-mag bin, be consistent!
                    mag_1 = mag + color + (0.5 * self.roi.delta_color)
                    mag_2 = mag + (0.5 * self.roi.delta_mag)

                # ADW: Is there a problem here?
                #self.solid_angle_cmd[index_mag, index_color] = self.roi.area_pixel * np.sum((self.mask_1.mask > mag_1) * (self.mask_2.mask > mag_2))

                # ADW: I think we want to keep pixels that are >= mag
                unmasked_mag_1 = (self.mask_1.mask_annulus_sparse >= mag_1)
                unmasked_mag_2 = (self.mask_2.mask_annulus_sparse >= mag_2)
                n_unmasked_pixels = np.sum(unmasked_mag_1*unmasked_mag_2*self.frac_annulus_sparse)

                #n_unmasked_pixels = np.sum((self.mask_1.mask_annulus_sparse > mag_1) \
                #                               * (self.mask_2.mask_annulus_sparse > mag_2))

                self.solid_angle_cmd[index_mag, index_color] = self.roi.area_pixel * n_unmasked_pixels
        if self.solid_angle_cmd.sum() == 0:
            msg = "Mask annulus contains no solid angle."
            logger.error(msg)
            raise Exception(msg)

        return self.solid_angle_cmd


    def _solidAngleCMD(self):
        """
        Compute solid angle within the mask annulus (deg^2) as a
        function of color and magnitude.

        Returns:
        --------
        solid_angle_cmd : 2d array
        """

        self.solid_angle_cmd = np.zeros([len(self.roi.centers_mag),
                                            len(self.roi.centers_color)])

        idx_mag,idx_color=np.where(self.solid_angle_cmd == 0)
        mag = self.roi.centers_mag[idx_mag]
        color = self.roi.centers_color[idx_color]

        if self.config.params['catalog']['band_1_detection']:
            # Evaluating at corner of the color-mag bin, be consistent!
            mag_1 = mag + (0.5 * self.roi.delta_mag)
            mag_2 = mag - color + (0.5 * self.roi.delta_color)
        else:
            # Evaluating at corner of the color-mag bin, be consistent!
            mag_1 = mag + color + (0.5 * self.roi.delta_color)
            mag_2 = mag + (0.5 * self.roi.delta_mag)

        n_unmasked_pixels = np.zeros_like(mag)
        for i in np.arange(len(mag_1)):
            unmasked_mag_1 = (self.mask_1.mask_annulus_sparse >= mag_1[i])
            unmasked_mag_2 = (self.mask_2.mask_annulus_sparse >= mag_2[i])
            n_unmasked_pixels[i] = np.sum(unmasked_mag_1 * unmasked_mag_2 *
                                          self.frac_annulus_sparse)

        self.solid_angle_cmd[idx_mag, idx_color] = self.roi.area_pixel * n_unmasked_pixels
        if self.solid_angle_cmd.sum() == 0:
            msg = "Mask annulus contains no solid angle."
            logger.error(msg)
            raise Exception(msg)

        return self.solid_angle_cmd
        
    def _pruneCMD(self, minimum_solid_angle):
        """
        Remove regions of color-magnitude space where the unmasked solid angle is
        statistically insufficient to estimate the background.

        ADW: Why are we clipping at the bin center instead of edge?

        INPUTS:
            solid_angle[1]: minimum solid angle (deg^2)
        """

        logger.info('Pruning mask based on minimum solid angle of %.2f deg^2'%(minimum_solid_angle))
        self.solid_angle_cmd *= self.solid_angle_cmd > minimum_solid_angle

        if self.solid_angle_cmd.sum() == 0:
            msg = "Pruned mask contains no solid angle."
            logger.error(msg)
            raise Exception(msg)

        # Compute which magnitudes the clipping correspond to
        index_mag, index_color = np.nonzero(self.solid_angle_cmd)
        mag = self.roi.centers_mag[index_mag]
        color = self.roi.centers_color[index_color]
        if self.config.params['catalog']['band_1_detection']:
            mag_1 = mag
            mag_2 = mag_1 - color
            self.mag_1_clip = np.max(mag_1) + (0.5 * self.roi.delta_mag)
            self.mag_2_clip = np.max(mag_2) + (0.5 * self.roi.delta_color)
        else:
            mag_2 = mag
            mag_1 = color + mag_2
            self.mag_1_clip = np.max(mag_1) + (0.5 * self.roi.delta_color)
            self.mag_2_clip = np.max(mag_2) + (0.5 * self.roi.delta_mag)

        logger.info('Clipping mask 1 at %.2f mag'%(self.mag_1_clip) )
        logger.info('Clipping mask 2 at %.2f mag'%(self.mag_2_clip) )
        self.mask_1.mask_roi_sparse = np.clip(self.mask_1.mask_roi_sparse, 0., self.mag_1_clip)
        self.mask_2.mask_roi_sparse = np.clip(self.mask_2.mask_roi_sparse, 0., self.mag_2_clip)
        

    def completeness(self, delta, method='step'):
        """
        Return the completeness as a function of magnitude.

        ADW: Eventually want a completeness mask to set overall efficiency.
        """
        delta = np.asarray(delta)
        if method == 'step':
            func = lambda delta: (delta > 0).astype(float)
        elif method == 'erf':
            # Trust the SDSS EDR???
            # 95% completeness: 
            def func(delta):
                # Efficiency at bright end (assumed to be 100%)
                e = 1.0
                # EDR says full width is ~0.5 mag
                width = 0.2 
                # This should be the halfway point in the curve
                return (e/2.0)*(1/np.sqrt(2*width))*(np.sqrt(2*width)-scipy.special.erf(-delta))
        elif method == 'flemming':
            # Functional form taken from Fleming et al. AJ 109, 1044 (1995)
            # http://adsabs.harvard.edu/abs/1995AJ....109.1044F
            # f = 1/2 [1 - alpha(V - Vlim)/sqrt(1 + alpha^2 (V - Vlim)^2)]
            # CAREFUL: This definition is for Vlim = 50% completeness
            def func(delta):
                alpha = 2.0
                return 0.5 * (1 - (alpha * delta)/np.sqrt(1+alpha**2 * delta**2))
        else:
            raise Exception('...')
        return func(delta)

    def _photometricErrors(self, catalog=None, n_per_bin=100):
        """
        Realistic photometric errors estimated from catalog objects and mask.
        Extend below the magnitude threshold with a flat extrapolation.
        """

        if catalog is None:
            # Simple proxy for photometric errors
            release = self.config['data']['release']
            band_1 = self.config['catalog'].get('mag_1_band')
            if not band_1: band_1 = self.config['isochrone']['mag_1_field']
            band_2 = self.config['catalog'].get('mag_2_band')
            if not band_2: band_2 = self.config['isochrone']['mag_2_field']
            
            DELMIN = 0.0
            pars_1 = MAGERR_PARAMS[release][band_1]
            
            def photo_err_1(delta):
                p = pars_1
                return np.clip(np.exp(p[0]*delta+p[1])+p[2], 0, np.exp(p[0]*(DELMIN)+p[1])+p[2])

            pars_2 = MAGERR_PARAMS[release][band_2]
            def photo_err_2(delta):
                p = pars_2
                return np.clip(np.exp(p[0]*delta+p[1])+p[2], 0, np.exp(p[0]*(DELMIN)+p[1])+p[2])

        else:
            catalog.spatialBin(self.roi)

            if len(catalog.mag_1) < n_per_bin:
                logger.warning("Catalog contains fewer objects than requested to calculate errors.")
                #n_per_bin = int(len(catalog.mag_1) / 3)
                return self._photometricErrors(catalog=None)
             
            # Band 1
            mag_1_thresh = self.mask_1.mask_roi_sparse[catalog.pixel_roi_index] - catalog.mag_1
            sorting_indices = np.argsort(mag_1_thresh)
            mag_1_thresh_sort = mag_1_thresh[sorting_indices]
            mag_err_1_sort = catalog.mag_err_1[sorting_indices]
             
            # ADW: Can't this be done with np.median(axis=?)
            mag_1_thresh_medians = []
            mag_err_1_medians = []
            for i in range(0, int(len(mag_1_thresh) / float(n_per_bin))):
                mag_1_thresh_medians.append(np.median(mag_1_thresh_sort[n_per_bin * i: n_per_bin * (i + 1)]))
                mag_err_1_medians.append(np.median(mag_err_1_sort[n_per_bin * i: n_per_bin * (i + 1)]))
             
            if mag_1_thresh_medians[0] > 0.:
                mag_1_thresh_medians = np.insert(mag_1_thresh_medians, 0, -99.)
                mag_err_1_medians = np.insert(mag_err_1_medians, 0, mag_err_1_medians[0])
             
            photo_err_1 = scipy.interpolate.interp1d(mag_1_thresh_medians, mag_err_1_medians,
                                                     bounds_error=False, fill_value=mag_err_1_medians[-1])
             
            # Band 2
            mag_2_thresh = self.mask_2.mask_roi_sparse[catalog.pixel_roi_index] - catalog.mag_2
            sorting_indices = np.argsort(mag_2_thresh)
            mag_2_thresh_sort = mag_2_thresh[sorting_indices]
            mag_err_2_sort = catalog.mag_err_2[sorting_indices]
             
            mag_2_thresh_medians = []
            mag_err_2_medians = []
            for i in range(0, int(len(mag_2_thresh) / float(n_per_bin))):
                mag_2_thresh_medians.append(np.median(mag_2_thresh_sort[n_per_bin * i: n_per_bin * (i + 1)]))
                mag_err_2_medians.append(np.median(mag_err_2_sort[n_per_bin * i: n_per_bin * (i + 1)]))
             
            if mag_2_thresh_medians[0] > 0.:
                mag_2_thresh_medians = np.insert(mag_2_thresh_medians, 0, -99.)
                mag_err_2_medians = np.insert(mag_err_2_medians, 0, mag_err_2_medians[0])
             
            photo_err_2 = scipy.interpolate.interp1d(mag_2_thresh_medians, mag_err_2_medians,
                                                     bounds_error=False, fill_value=mag_err_2_medians[-1])
             
        self.photo_err_1=photo_err_1
        self.photo_err_2=photo_err_2

        return self.photo_err_1, self.photo_err_2

    def plotSolidAngleCMD(self):
        """
        Solid angle within the mask as a function of color and magnitude.
        """
        msg = "'%s.plotSolidAngleCMD': ADW 2018-05-05"%self.__class__.__name__
        DeprecationWarning(msg)

        import ugali.utils.plotting        
        ugali.utils.plotting.twoDimensionalHistogram('mask', 'color', 'magnitude',
                                                     self.solid_angle_cmd,
                                                     self.roi.bins_color,
                                                     self.roi.bins_mag,
                                                     lim_x = [self.roi.bins_color[0],
                                                              self.roi.bins_color[-1]],
                                                     lim_y = [self.roi.bins_mag[-1],
                                                              self.roi.bins_mag[0]])

    def plotSolidAngleMMD(self):
        """
        Solid angle within the mask as a function of color and magnitude.
        """
        msg = "'%s.plotSolidAngleMMD': ADW 2018-05-05"%self.__class__.__name__
        DeprecationWarning(msg)

        import ugali.utils.plotting        

        ugali.utils.plotting.twoDimensionalHistogram('mask', 'mag_2', 'mag_1',
                                                     self.solid_angle_mmd,
                                                     self.roi.bins_mag,
                                                     self.roi.bins_mag,
                                                     lim_x = [self.roi.bins_mag[0],
                                                              self.roi.bins_mag[-1]],
                                                     lim_y = [self.roi.bins_mag[-1],
                                                              self.roi.bins_mag[0]])



    def backgroundMMD(self, catalog, method='cloud-in-cells', weights=None):
        """
        Generate an empirical background model in magnitude-magnitude space.
        
        INPUTS:
            catalog: Catalog object
        OUTPUTS:
            background
        """

        # Select objects in annulus
        cut_annulus = self.roi.inAnnulus(catalog.lon,catalog.lat)
        mag_1 = catalog.mag_1[cut_annulus]
        mag_2 = catalog.mag_2[cut_annulus]

        # Units are (deg^2)
        solid_angle = ugali.utils.binning.take2D(self.solid_angle_mmd, mag_2, mag_1,
                                                 self.roi.bins_mag, self.roi.bins_mag)

        # Weight each object before binning
        # Divide by solid angle and bin size in magnitudes to get number density 
        # [objs / deg^2 / mag^2]
        if weights is None:
            number_density = (solid_angle*self.roi.delta_mag**2)**(-1)
        else:
            number_density = weights*(solid_angle*self.roi.delta_mag**2)**(-1)

        method = str(method).lower()
        if method == 'cloud-in-cells':
            # Apply cloud-in-cells algorithm
            mmd_background = ugali.utils.binning.cloudInCells(mag_2,mag_1,
                                                              [self.roi.bins_mag,self.roi.bins_mag],
                                                              weights=number_density)[0]
        elif method == 'bootstrap':
            # Not implemented
            raise ValueError("Bootstrap method not implemented")
            mag_1 + (mag_1_err * np.random.normal(0, 1., len(mag_1)))
            mag_2 + (mag_2_err * np.random.normal(0, 1., len(mag_2)))

        elif method == 'histogram':
            # Apply raw histogram
            mmd_background = np.histogram2d(mag_1,mag_2,bins=[self.roi.bins_mag,self.roi.bins_mag],
                                            weights=number_density)[0]

        elif method == 'kde':
            # Gridded kernel density estimator
            logger.warning("### KDE not implemented properly")
            mmd_background = ugali.utils.binning.kernelDensity(mag_2,mag_1,
                                                               [self.roi.bins_mag,self.roi.bins_mag],
                                                               weights=number_density)[0]
        elif method == 'uniform':
            logger.warning("### WARNING: Uniform CMD")
            hist = np.histogram2d(mag_1,mag_2,bins=[self.roi.bins_mag,self.roi.bins_mag], 
                                  weights=number_density)[0]
            mmd_background = np.mean(hist)*np.ones(hist.shape)
            observable = (self.solid_angle_mmd > self.minimum_solid_angle)
            mmd_background *= observable
            return mmd_background
        else:
            raise ValueError("Unrecognized method: %s"%method)
        ## Account for the objects that spill out of the observable space
        ## But what about the objects that spill out to red colors??
        #for index_color in range(0, len(self.roi.centers_color)):
        #    for index_mag in range(0, len(self.roi.centers_mag)):
        #        if self.solid_angle_cmd[index_mag][index_color] < self.minimum_solid_angle:
        #            cmd_background[index_mag - 1][index_color] += cmd_background[index_mag][index_color]
        #            cmd_background[index_mag][index_color] = 0.
        #            break

        mmd_area = self.solid_angle_mmd*self.roi.delta_mag**2 # [deg^2 * mag^2]

        # ADW: This accounts for leakage to faint magnitudes
        # But what about the objects that spill out to red colors??
        # Maximum obsevable magnitude index for each color (uses the fact that
        # np.argmin returns first minimum (zero) instance found.
        # NOTE: More complicated maps may have holes causing problems

        observable = (self.solid_angle_mmd > self.minimum_solid_angle)
        index_mag_1 = observable.argmin(axis=0) - 1
        index_mag_2 = np.arange(len(self.roi.centers_mag))
        # Add the cumulative leakage back into the last bin of the CMD
        leakage = (mmd_background * ~observable).sum(axis=0)
        ### mmd_background[[index_mag_1,index_mag_2]] += leakage
        # Zero out all non-observable bins
        ### mmd_background *= observable

        # Avoid dividing by zero by setting empty bins to the value of the 
        # minimum filled bin of the CMD. This choice is arbitrary and 
        # could be replaced by a static minimum, some fraction of the 
        # CMD maximum, some median clipped minimum, etc. However, should 
        # be robust against outliers with very small values.
        min_mmd_background = max(mmd_background[mmd_background > 0.].min(),
                                 1e-4*mmd_background.max())
        mmd_background[observable] = mmd_background[observable].clip(min_mmd_background)

        ### # ADW: This is a fudge factor introduced to renormalize the CMD
        ### # to the number of input stars in the annulus. While leakage
        ### # will still smooth the distribution, it shouldn't result in 
        ### fudge_factor = len(mag) / float((cmd_background*cmd_area).sum())
        ### cmd_background *= fudge_factor

        return mmd_background

    def backgroundCMD(self, catalog, mode='cloud-in-cells', weights=None):
        """
        Generate an empirical background model in color-magnitude space.
        
        INPUTS:
            catalog: Catalog object
        OUTPUTS:
            background
        """

        # Select objects in annulus
        cut_annulus = self.roi.inAnnulus(catalog.lon,catalog.lat)
        color = catalog.color[cut_annulus]
        mag   = catalog.mag[cut_annulus]

        # Units are (deg^2)
        solid_angle = ugali.utils.binning.take2D(self.solid_angle_cmd, color, mag,
                                                 self.roi.bins_color, self.roi.bins_mag)

        # Weight each object before binning
        # Divide by solid angle and bin size in magnitudes to get number density 
        # [objs / deg^2 / mag^2]
        if weights is None:
            number_density = (solid_angle*self.roi.delta_color*self.roi.delta_mag)**(-1)
        else:
            number_density = weights*(solid_angle*self.roi.delta_color*self.roi.delta_mag)**(-1)

        mode = str(mode).lower()
        if mode == 'cloud-in-cells':
            # Apply cloud-in-cells algorithm
            cmd_background = ugali.utils.binning.cloudInCells(color,mag,
                                                              [self.roi.bins_color,self.roi.bins_mag],
                                                              weights=number_density)[0]
        elif mode == 'bootstrap':
            # Not implemented
            raise ValueError("Bootstrap mode not implemented")
            mag_1_array = catalog.mag_1
            mag_2_array = catalog.mag_2

            catalog.mag_1 + (catalog.mag_1_err * np.random.normal(0, 1., len(catalog.mag_1)))
            catalog.mag_2 + (catalog.mag_2_err * np.random.normal(0, 1., len(catalog.mag_2)))

        elif mode == 'histogram':
            # Apply raw histogram
            cmd_background = np.histogram2d(mag,color,bins=[self.roi.bins_mag,self.roi.bins_color],
                                            weights=number_density)[0]

        elif mode == 'kde':
            # Gridded kernel density estimator
            logger.warning("### KDE not implemented properly")
            cmd_background = ugali.utils.binning.kernelDensity(color,mag,
                                                               [self.roi.bins_color,self.roi.bins_mag],
                                                               weights=number_density)[0]
        elif mode == 'uniform':
            logger.warning("### WARNING: Uniform CMD")
            hist = np.histogram2d(mag,color,bins=[self.roi.bins_mag,self.roi.bins_color], weights=number_density)[0]
            cmd_background = np.mean(hist)*np.ones(hist.shape)
            observable = (self.solid_angle_cmd > self.minimum_solid_angle)
            cmd_background *= observable
            return cmd_background
        else:
            raise ValueError("Unrecognized mode: %s"%mode)
        ## Account for the objects that spill out of the observable space
        ## But what about the objects that spill out to red colors??
        #for index_color in range(0, len(self.roi.centers_color)):
        #    for index_mag in range(0, len(self.roi.centers_mag)):
        #        if self.solid_angle_cmd[index_mag][index_color] < self.minimum_solid_angle:
        #            cmd_background[index_mag - 1][index_color] += cmd_background[index_mag][index_color]
        #            cmd_background[index_mag][index_color] = 0.
        #            break

        cmd_area = self.solid_angle_cmd*self.roi.delta_color*self.roi.delta_mag # [deg^2 * mag^2]

        # ADW: This accounts for leakage to faint magnitudes
        # But what about the objects that spill out to red colors??
        # Maximum obsevable magnitude index for each color (uses the fact that
        # np.argmin returns first minimum (zero) instance found.
        # NOTE: More complicated maps may have holes causing problems

        observable = (self.solid_angle_cmd > self.minimum_solid_angle)
        index_mag = observable.argmin(axis=0) - 1
        index_color = np.arange(len(self.roi.centers_color))
        # Add the cumulative leakage back into the last bin of the CMD
        leakage = (cmd_background * ~observable).sum(axis=0)
        cmd_background[[index_mag,index_color]] += leakage
        # Zero out all non-observable bins
        cmd_background *= observable

        # Avoid dividing by zero by setting empty bins to the value of the 
        # minimum filled bin of the CMD. This choice is arbitrary and 
        # could be replaced by a static minimum, some fraction of the 
        # CMD maximum, some median clipped minimum, etc. However, should 
        # be robust against outliers with very small values.
        min_cmd_background = max(cmd_background[cmd_background > 0.].min(),
                                 1e-4*cmd_background.max())
        cmd_background[observable] = cmd_background[observable].clip(min_cmd_background)

        ### # ADW: This is a fudge factor introduced to renormalize the CMD
        ### # to the number of input stars in the annulus. While leakage
        ### # will still smooth the distribution, it shouldn't result in 
        ### fudge_factor = len(mag) / float((cmd_background*cmd_area).sum())
        ### cmd_background *= fudge_factor

        return cmd_background

    def restrictCatalogToObservableSpaceMMD(self, catalog):
        """
        Retain only the catalog objects which fall within the observable (i.e., unmasked) space.

        Parameters:
        catalog: a Catalog object
        Returns:
        sel    : boolean selection array where True means the object would be observable (i.e., unmasked).

        ADW: Careful, this function is fragile! The selection here should
             be the same as isochrone.observableFraction space. However,
             for technical reasons it is faster to do the calculation with
             broadcasting there.
        """

        # ADW: This creates a slope in color-magnitude space near the magnitude limit
        # i.e., if color=g-r then you can't have an object with g-r=1 and mag_r > mask_r-1
        # Depending on which is the detection band, this slope will appear at blue
        # or red colors. When it occurs at blue colors, it effects very few objects.
        # However, when occuring for red objects it can cut many objects. It is 
        # unclear that this is being correctly accounted for in the likelihood

        catalog.spatialBin(self.roi)
        sel_roi = (catalog.pixel_roi_index >= 0) # Objects outside ROI have pixel_roi_index of -1
        sel_mag_1 = catalog.mag_1 < self.mask_1.mask_roi_sparse[catalog.pixel_roi_index]
        sel_mag_2 = catalog.mag_2 < self.mask_2.mask_roi_sparse[catalog.pixel_roi_index]

        # and are located in the region of mag-mag space where background can be estimated
        sel_mmd = ugali.utils.binning.take2D(self.solid_angle_mmd,
                                             catalog.mag_2, catalog.mag_1,
                                             self.roi.bins_mag, self.roi.bins_mag) > 0.

        sel = np.all([sel_roi,sel_mag_1,sel_mag_2,sel_mmd], axis=0)
        return sel

    def restrictCatalogToObservableSpaceCMD(self, catalog):
        """
        Retain only the catalog objects which fall within the
        observable (i.e., unmasked) space.  NOTE: This returns a
        *selection* (i.e., objects are retained if the value of the
        output array is True).

        Parameters:
        catalog: a Catalog object
        Returns:
        sel    : boolean selection array where True means the object would be observable (i.e., unmasked).

        ADW: Careful, this function is fragile! The selection here should
             be the same as isochrone.observableFraction space. However,
             for technical reasons it is faster to do the calculation with
             broadcasting there.
        """

        # ADW: This creates a slope in color-magnitude space near the magnitude limit
        # i.e., if color=g-r then you can't have an object with g-r=1 and mag_r > mask_r-1
        # Depending on which is the detection band, this slope will appear at blue
        # or red colors. When it occurs at blue colors, it effects very few objects.
        # However, when occuring for red objects it can cut many objects. It is 
        # unclear that this is being correctly accounted for in the likelihood

        ### # Check that the objects fall in the color-magnitude space of the ROI
        ### # ADW: I think this is degenerate with the cut_cmd
        ### sel_mag = np.logical_and(catalog.mag > self.roi.bins_mag[0],
        ###                             catalog.mag < self.roi.bins_mag[-1])
        ### sel_color = np.logical_and(catalog.color > self.roi.bins_color[0],
        ###                               catalog.color < self.roi.bins_color[-1])

        # and are observable in the ROI-specific mask for both bands
        #if not hasattr(catalog, 'pixel_roi_index'): # TODO: An attempt to save computations, but not robust
        #    catalog.spatialBin(self.roi)
        catalog.spatialBin(self.roi)
        sel_roi = (catalog.pixel_roi_index >= 0) # Objects outside ROI have pixel_roi_index of -1
        sel_mag_1 = catalog.mag_1 < self.mask_1.mask_roi_sparse[catalog.pixel_roi_index]
        sel_mag_2 = catalog.mag_2 < self.mask_2.mask_roi_sparse[catalog.pixel_roi_index]

        # and are located in the region of color-magnitude space where background can be estimated
        sel_cmd = ugali.utils.binning.take2D(self.solid_angle_cmd,
                                             catalog.color, catalog.mag,
                                             self.roi.bins_color, self.roi.bins_mag) > 0.

        sel = np.all([sel_roi,sel_mag_1,sel_mag_2,sel_cmd], axis=0)
        return sel
    
    # FIXME: Need to parallelize CMD and MMD formulation
    restrictCatalogToObservableSpace = restrictCatalogToObservableSpaceCMD

############################################################

class MaskBand(object):
    """
    Map of liming magnitude for a single observing band.
    """

    def __init__(self, infiles, roi):
        """
        Parameters:
        -----------
        infiles : list of sparse healpix mask files
        roi : roi object

        Returns:
        --------
        mask : MaskBand object
        """
        self.roi = roi
        self.config = self.roi.config

        # ADW: It's overkill to make the full map just to slim it
        # down, but we don't have a great way to go from map pixels to
        # roi pixels.
        nside,pixel,maglim = healpix.read_partial_map(infiles,column='MAGLIM')
        self.nside = nside

        # Sparse maps of pixels in various ROI regions
        self.mask_roi_sparse = maglim[self.roi.pixels] 

        # Try to get the detection fraction
        self.frac_roi_sparse = (self.mask_roi_sparse > 0)
        try: 
            logger.debug("Reading FRACDET...")
            nside,pixel,frac=healpix.read_partial_map(infiles,column='FRACDET')
            # This clipping might gloss over bugs...
            fractype = self.config['mask'].get('fractype','binary')
            fracmin = self.config['mask'].get('fracmin',0.5)
            if fractype == 'binary':
                frac = np.where(frac < fracmin, 0.0, 1.0)
            elif fractype == 'full':
                frac = np.where(frac < fracmin, 0.0, frac)
            elif not fractype:
                pass
            else:
                msg = "Unrecognized fractype: %s"%fractype
                logger.warn(msg)
                
            self.frac_roi_sparse = np.clip(frac[self.roi.pixels],0.0,1.0)
        except ValueError as e:
            # No detection fraction present
            msg = "No 'FRACDET' column found in masks; assuming FRACDET = 1.0"
            logger.info(msg)

        # Explicitly zero the maglim of pixels with fracdet < fracmin
        self.mask_roi_sparse[self.frac_roi_sparse == 0] = 0.0

    #ADW: Safer and more robust (though slightly slower)
    @property
    def mask_annulus_sparse(self):
        return self.mask_roi_sparse[self.roi.pixel_annulus_cut]
     
    @property
    def mask_interior_sparse(self):
        return self.mask_roi_sparse[self.roi.pixel_interior_cut]

    @property
    def mask_roi_unique(self):
        return np.unique(self.mask_roi_sparse)

    @property
    def mask_roi_digi(self):
        """ Mapping from unique pixels to roi pixels """
        return np.digitize(self.mask_roi_sparse,bins=self.mask_roi_unique)-1

    @property
    def frac_annulus_sparse(self):
        return self.frac_roi_sparse[self.roi.pixel_annulus_cut]
     
    @property
    def frac_interior_sparse(self):
        return self.frac_roi_sparse[self.roi.pixel_interior_cut]

    def completeness(self, mags, method='step'):
        """
        Return the completeness as a function of magnitude.

        ADW: Eventually want a completeness mask to set overall efficiency.
        """
        if method == 'step':
            func = lambda x: (x < self.mask_roi_unique[:,np.newaxis]).astype(float)
        elif method == 'erf':
            # Trust the ERD???
            # 95% completeness: 
            def func(x):
                # Efficiency at bright end (assumed to be 100%)
                e = 1.0
                # SDSS EDR says full width is ~0.5 mag
                width = 0.2 
                # This should be the halfway point in the curve
                maglim = self.mask_roi_unique[:,np.newaxis]
                return (e/2.0)*(1/np.sqrt(2*width))*(np.sqrt(2*width)-scipy.special.erf((x-maglim)))
        else:
            raise Exception('...')

        return func(mags)
        
    def depth(self, lon, lat):
        """
        Magnitude limit at given image coordinates.
        """
        pass

    def plot(self):
        """
        Plot the magnitude depth.
        """
        msg = "'%s.plot': ADW 2018-05-05"%self.__class__.__name__
        DeprecationWarning(msg)

        import ugali.utils.plotting

        mask = hp.UNSEEN * np.ones(hp.nside2npix(self.nside))
        mask[self.roi.pixels] = self.mask_roi_sparse
        mask[mask == 0.] = hp.UNSEEN
        ugali.utils.plotting.zoomedHealpixMap('Completeness Depth',
                                              mask,
                                              self.roi.lon, self.roi.lat,
                                              self.roi.config.params['coords']['roi_radius'])


class CoverageBand(object):
    """
    Map of coverage fraction for a single observing band.
    """

    def __init__(self, infiles, roi):
        """
        Infile is a sparse HEALPix map fits file.
        """
        self.roi = roi
        mask = ugali.utils.skymap.readSparseHealpixMaps(infiles, field='COVERAGE')
        self.nside = hp.npix2nside(len(mask))
        # Sparse maps of pixels in various ROI regions
        self.mask_roi_sparse = mask[self.roi.pixels] 


############################################################

class SimpleMask(Mask):
    """
    Contains maps of completeness depth in magnitudes for multiple observing bands, and associated products.
    """
    def __init__(self, config, roi, maglim_1=23, maglim_2=23):
        self.config = Config(config)
        self.roi = roi

        self.mask_1 = SimpleMaskBand(maglim_1,self.roi)
        self.mask_2 = SimpleMaskBand(maglim_2,self.roi)
        
        self.minimum_solid_angle = self.config.params['mask']['minimum_solid_angle'] # deg^2

        # FIXME: Need to parallelize CMD and MMD formulation
        self._solidAngleCMD()
        self._pruneCMD(self.minimum_solid_angle)
        
        #self._solidAngleMMD()
        #self._pruneMMD(self.minimum_solid_angle)

        self._photometricErrors()

############################################################

class SimpleMaskBand(MaskBand):
    """
    Map of completeness depth in magnitudes for a single observing band.
    """

    def __init__(self, maglim, roi):
        """
        Infile is a sparse HEALPix map fits file.
        """
        self.roi = roi
        mask = maglim*np.ones(hp.nside2npix(self.roi.config['coords']['nside_pixel']))
        self.nside = hp.npix2nside(len(mask))
        # Sparse maps of pixels in various ROI regions
        self.mask_roi_sparse = mask[self.roi.pixels] 

############################################################
def simpleMask(config):

    #params = ugali.utils.(config, kwargs)

    roi = ugali.observation.roi.ROI(config)

    # De-project the bin centers to get magnitude depths

    mesh_x, mesh_y = np.meshgrid(roi.centers_x, roi.centers_y)
    r = np.sqrt(mesh_x**2 + mesh_y**2) # Think about x, y conventions here

    #z = (0. * (r > 1.)) + (21. * (r < 1.))
    #z = 21. - r
    #z = (21. - r) * (mesh_x > 0.) * (mesh_y < 0.)
    z = (21. - r) * np.logical_or(mesh_x > 0., mesh_y > 0.)

    return MaskBand(z, roi)
    
############################################################

def readMangleFile(infile, lon, lat, index = None):
    """
    DEPRECATED: 2018-05-04
    Mangle must be set up on your system.
    The index argument is a temporary file naming convention to avoid file conflicts.
    Coordinates must be given in the native coordinate system of the Mangle file.
    """
    msg = "'mask.readMangleFile': ADW 2018-05-05"
    DeprecationWarning(msg)

    if index is None:
        index = np.random.randint(0, 1.e10)
    
    coordinate_file = 'temp_coordinate_%010i.dat'%(index)
    maglim_file = 'temp_maglim_%010i.dat'%(index)

    writer = open(coordinate_file, 'w')
    for ii in range(0, len(lon)):
        writer.write('%12.5f%12.5f\n'%(lon[ii], lat[ii]))
    writer.close()

    os.system('polyid -W %s %s %s || exit'%(infile,
                                            coordinate_file,
                                            maglim_file))

    reader = open(maglim_file)
    lines = reader.readlines()
    reader.close()

    os.remove(maglim_file)
    os.remove(coordinate_file)

    maglim = []
    for ii in range(1, len(lines)):
        if len(lines[ii].split()) == 3:
            maglim.append(float(lines[ii].split()[2]))
        elif len(lines[ii].split()) == 2:
            maglim.append(0.) # Coordinates outside of the MANGLE ploygon
        elif len(lines[ii].split()) > 3:
            msg = 'Coordinate inside multiple polygons, masking that coordinate.'
            logger.warning(msg)
            maglim.append(0.)
        else:
            msg = 'Cannot parse maglim file, unexpected number of columns.'
            logger.error(msg)
            break
            
    maglim = np.array(maglim)
    return maglim

############################################################

def allSkyMask(infile, nside):
    msg = "'mask.allSkyMask': ADW 2018-05-05"
    DeprecationWarning(msg)
    lon, lat = ugali.utils.skymap.allSkyCoordinates(nside)
    maglim = readMangleFile(infile, lon, lat, index = None)
    return maglim

############################################################

def scale(mask, mag_scale, outfile=None):
    """
    Scale the completeness depth of a mask such that mag_new = mag + mag_scale.
    Input is a full HEALPix map.
    Optionally write out the scaled mask as an sparse HEALPix map.
    """
    msg = "'mask.scale': ADW 2018-05-05"
    DeprecationWarning(msg)
    mask_new = hp.UNSEEN * np.ones(len(mask))
    mask_new[mask == 0.] = 0.
    mask_new[mask > 0.] = mask[mask > 0.] + mag_scale

    if outfile is not None:
        pix = np.nonzero(mask_new > 0.)[0]
        data_dict = {'MAGLIM': mask_new[pix]}
        nside = hp.npix2nside(len(mask_new))
        ugali.utils.skymap.writeSparseHealpixMap(pix, data_dict, nside, outfile)

    return mask_new

############################################################

