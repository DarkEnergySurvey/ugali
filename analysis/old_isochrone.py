"""
Object for isochrone storage and basic calculations.

NOTE: only absolute magnitudes are used in the Isochrone class

ADW: There are some issues involved here. As we are generally
using a forward-folding likelihood technique, what we would
like to do is to convolve the isochrone model with the survey
response functions to derive a model of the observed distribution
of objects given a specific true isochrone. This convolution
involves two distinct parts: 1) the object completeness as a 
function of delta-magnitude (difference between magnitude and 
local limiting magnitude), 2) the magnitude dispersion (magnitude 
uncertainty) as a function of delta-magnitude. 

Since the survey response (i.e., depth) changes on the pixel scale, 
this would means deriving the convolved isochrone for every pixel 
in the interior of the ROI. Assuming a magnitude binning of 70x70, 
and a set of 3000 interior pixels, this is a 70x70x3000 matrix. 
However, the issue is that to generate this array you need to 
sample the isochrone at roughly 1000 points and sum these points.
Needless to say it is fairly intensive to calculate and store
this matrix. Things become much more reasonable if you only
calculate this matrix once for each unique magnitude limit,
but again this becomes difficult because you need each unique
limit in both magnitudes. 

"""

import sys
import collections
from collections import OrderedDict as odict

import numpy
import numpy as np
import scipy.interpolate
import scipy.stats
import scipy.ndimage as ndimage

import ugali.analysis.imf
#import ugali.observation.photometric_errors # Probably won't need this in the future since will be passed
from ugali.analysis.model import Model, Parameter
from ugali.utils.stats import norm_cdf

from ugali.utils.config import Config
from ugali.utils.logger import logger
############################################################

import warnings
warnings.simplefilter('module', DeprecationWarning)
msg = "Old isochrone is deprecated"
warnings.warn(msg,DeprecationWarning)
warnings.simplefilter('default', DeprecationWarning)


class Isochrone(Model):

    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
    ])
    _mapping = odict([])
    
    def __init__(self, config, infile, infile_format='padova', **kwargs):
        """
        Initialize an isochrone instance.
        """
        super(Isochrone,self).__init__(**kwargs)

        self.config = Config(config)
        self.infile = infile
        
        if infile_format.lower() == 'padova':
            self._parseIsochronePadova(self.config['data']['survey'])
        else:
            logger.warning('did not recognize infile format %s'%(infile_format))

        self.imf = ugali.analysis.imf.IMF(self.config['isochrone']['imf'])

        # Check where post-AGB isochrone data points begin
        self.mass_init_upper_bound = numpy.max(self.mass_init)
        if numpy.any(self.stage == 'LTP'):
            self.index = numpy.nonzero(self.stage == 'LTP')[0][0]
        else:
            self.index = len(self.mass_init)
        
        # Other housekeeping
        if self.config['catalog']['band_1_detection']:
            self.mag = self.mag_1
        else:
            self.mag = self.mag_2
        self.color = self.mag_1 - self.mag_2

        # Photometric error functions
        # This should actually be set in the future
        #self.photometric_error_func_1 = ugali.observation.photometric_errors.PhotometricErrors()
        #self.photometric_error_func_2 = ugali.observation.photometric_errors.PhotometricErrors()

        # Horizontal branch dispersion
        #self._setHorizontalBranch()
        self.horizontal_branch_dispersion = self.config['isochrone']['horizontal_branch_dispersion'] # mag
        self.horizontal_branch_stage      = self.config['isochrone']['horizontal_branch_stage']

    def plotCMD(self):
        """
        Show the color-magnitude diagram of isochrone points in absolute magnitudes.
        """
        import pylab
        import ugali.utils.plotting

        ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
                                                   self.color, self.mag)
        y_min, y_max = pylab.axis()[2], pylab.axis()[3]
        pylab.ylim(y_max, y_min)

    def _parseIsochronePadova2(self):
        """
        ADW: DEPRICATED (12/23/2014)

        Reads an isochrone file in the Padova format and returns the age (log yrs), metallicity (Z), and an
        array with initial stellar mass and corresponding magnitudes.
        """
        mass_init_field = self.config['isochrone']['mass_init_field']
        mag_1_field = self.config['isochrone']['mag_1_field']
        mag_2_field = self.config['isochrone']['mag_2_field']
        
        reader = open(self.infile)
        lines = reader.readlines()
        reader.close()

        index_dict = {}
        metallicity = -999
        log_age = -999
        mass_init = []
        mag_1 = []
        mag_2 = []
        for line in lines:
            parts = line.split()
            if line[0] == '#':
                if mass_init_field in line and mag_1_field in line and mag_2_field in line: 
                    for ii in range(0, len(parts)):
                        for field in [mass_init_field, mag_1_field, mag_2_field]:
                            if parts[ii] == field:
                                index_dict[field] = ii - 1
                elif 'Isochrone' in line:
                    metallicity = float(parts[4])
                    log_age = numpy.log10(float(parts[7]))
                else:
                    continue
            else:
                mass_init.append(float(parts[index_dict[mass_init_field]]))
                mag_1.append(float(parts[index_dict[mag_1_field]]))
                mag_2.append(float(parts[index_dict[mag_2_field]]))

        # Set basic properties
        self.log_age = log_age
        self.metallicity = metallicity
        self.mass_init = numpy.array(mass_init)
        self.mag_1 = numpy.array(mag_1)
        self.mag_2 = numpy.array(mag_2)

    # NOTE: Would a better solution be to convert data files into a uniform format?
    def _parseIsochronePadova(self, survey='DES'):
        """
        Reads an isochrone file in the Padova (Marigo 2008) format and determines
        the age (log10 yrs and Gyr), metallicity (Z and [Fe/H]), and creates arrays with
        the initial stellar mass and corresponding magnitudes for each step along the isochrone.
        http://stev.oapd.inaf.it/cgi-bin/cmd
        """
        mass_init_field = self.config['isochrone']['mass_init_field']
        mass_act_field = self.config['isochrone']['mass_act_field']
        luminosity_field = self.config['isochrone']['luminosity_field']
        mag_1_field = self.config['isochrone']['mag_1_field']
        mag_2_field = self.config['isochrone']['mag_2_field']
        stage_field = self.config['isochrone']['stage_field']

        if survey.lower() == 'des':
            index_dict = {mass_init_field: 1,
                          mass_act_field: 2,
                          luminosity_field: 3,
                          'g': 7,
                          'r': 8,
                          'i': 9,
                          'z': 10,
                          'Y': 11,
                          stage_field: 18}
        elif survey.lower() == 'sdss':
            index_dict = {mass_init_field: 1,
                          mass_act_field: 2,
                          luminosity_field: 3,
                          'u': 7,
                          'g': 8,
                          'r': 9,
                          'i': 10,
                          'z': 11,
                          stage_field: 18}
        else:
            logger.warning('did not recognize survey %s'%(survey))
    
        reader = open(self.infile)
        lines = reader.readlines()
        reader.close()

        log_age = 1.e-2 * float(self.infile.split('isota')[1].split('z')[0])
        metallicity = 1.e-3 * float(self.infile.split('z')[1].split('.dat')[0])
        age = 10**log_age # Gyr
        metallicity_solar = 0.019 # Anders & Grevesse 1989
        feh = numpy.log10(metallicity / metallicity_solar)

        self.color_data = {}
        for key in index_dict.keys():
            if key == mass_init_field:
                continue
            self.color_data[key] = []
    
        mass_init = []
        mass_act = []
        luminosity = []
        mag_1 = []
        mag_2 = []
        stage = []
        for line in lines:
            parts = line.split()
            if len(parts) < 11:
                continue
            mass_init.append(float(parts[index_dict[mass_init_field]]))
            mass_act.append(float(parts[index_dict[mass_act_field]]))
            luminosity.append(10.**float(parts[index_dict[luminosity_field]])) # Convert from log-space
            mag_1.append(float(parts[index_dict[mag_1_field]]))
            mag_2.append(float(parts[index_dict[mag_2_field]]))

            if len(parts) == index_dict[stage_field] + 1:
                stage.append(parts[index_dict[stage_field]])
            else:
                stage.append('')
            if len(stage) > 1:
                if stage[-1] == '' and stage[-2] != ['']:
                    stage[-1] = stage[-2]

            for key in index_dict.keys():
                if key in [mass_init_field, mass_act_field, luminosity_field, stage_field]:
                    continue
                self.color_data[key].append(float(parts[index_dict[key]]))

        for key in self.color_data.keys():
            self.color_data[key] = numpy.array(self.color_data[key])

        # Set basic properties
        self.log_age = log_age
        self.age = age
        self.metallicity = metallicity
        self.feh = feh
        self.mass_init = numpy.array(mass_init)
        self.mass_act = numpy.array(mass_act)
        self.luminosity = numpy.array(luminosity)
        self.mag_1 = numpy.array(mag_1)
        self.mag_2 = numpy.array(mag_2)
        self.stage = numpy.array(stage)

    def sample(self, mode='data', mass_steps=1000, mass_min=0.1, full_data_range=False):
        """
        Documentation here.
        """

        if full_data_range:
            # Generate points over full isochrone data range
            mass_init = self.mass_init
            mass_act = self.mass_act
            mag_1 = self.mag_1
            mag_2 = self.mag_2
        else:
            # Not generating points for the post-AGB stars,
            # but still count those stars towards the normalization
            mass_init = self.mass_init[0: self.index]
            mass_act = self.mass_act[0: self.index]
            mag_1 = self.mag_1[0: self.index]
            mag_2 = self.mag_2[0: self.index]
        
        # ADW: Assume that the isochrones are pre-sorted by mass_init
        # This avoids some numerical instability from points that have the same
        # mass_init value (discontinuities in the isochrone).
        # ADW: Might consider using numpy.interp for speed
        mass_act_interpolation = scipy.interpolate.interp1d(mass_init, mass_act,assume_sorted=True)
        mag_1_interpolation = scipy.interpolate.interp1d(mass_init, mag_1,assume_sorted=True)
        mag_2_interpolation = scipy.interpolate.interp1d(mass_init, mag_2,assume_sorted=True)

        if mode=='data':
            # Mass interpolation with uniform coverage between data points from isochrone file 
            mass_interpolation = scipy.interpolate.interp1d(range(0, len(mass_init)), mass_init)
            mass_array = mass_interpolation(numpy.linspace(0, len(mass_init) - 1, mass_steps + 1))
            d_mass = mass_array[1:] - mass_array[0: -1]
            mass_init_array = numpy.sqrt(mass_array[1:] * mass_array[0: -1])
            mass_pdf_array = d_mass * self.imf.pdf(mass_init_array, log_mode = False)
            mass_act_array = mass_act_interpolation(mass_init_array)
            mag_1_array = mag_1_interpolation(mass_init_array)
            mag_2_array = mag_2_interpolation(mass_init_array)
            
        # Horizontal branch dispersion
        if self.horizontal_branch_dispersion and numpy.any(self.stage==self.horizontal_branch_stage):
            logger.debug("Performing dispersion of horizontal branch...")
            mass_init_horizontal_branch_min = self.mass_init[self.stage==self.horizontal_branch_stage].min()
            mass_init_horizontal_branch_max = self.mass_init[self.stage==self.horizontal_branch_stage].max()
            cut = numpy.logical_and(mass_init_array > mass_init_horizontal_branch_min,
                                    mass_init_array < mass_init_horizontal_branch_max)

            if isinstance(self.horizontal_branch_dispersion,collections.Iterable):
                dispersion_array = self.horizontal_branch_dispersion
                n = len(dispersion_array)
            else:
                dispersion = self.horizontal_branch_dispersion
                spacing = 0.1
                n = int(round(2.0*self.horizontal_branch_dispersion/spacing))
                if n % 2 != 1: n += 1
                dispersion_array = numpy.linspace(-dispersion, dispersion, n)

            # Reset original values
            mass_pdf_array[cut] = mass_pdf_array[cut] / float(n)

            # Add dispersed values
            for dispersion in dispersion_array:
                if dispersion == 0.: continue
                msg = '%-6g%-6g%-6g'%(dispersion,cut.sum(),len(mass_init_array))
                logger.debug(msg)
                mass_init_array = numpy.append(mass_init_array, mass_init_array[cut]) 
                mass_pdf_array = numpy.append(mass_pdf_array, mass_pdf_array[cut])
                mass_act_array = numpy.append(mass_act_array, mass_act_array[cut]) 
                mag_1_array = numpy.append(mag_1_array, mag_1_array[cut] + dispersion)
                mag_2_array = numpy.append(mag_2_array, mag_2_array[cut] + dispersion)

        #ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
        #                                           mag_1_array - mag_2_array, mag_1_array)


        # Note that the mass_pdf_array is not generally normalized to unity
        # since the isochrone data range typically covers a different range
        # of initial masses
        #mass_pdf_array /= numpy.sum(mass_pdf_array) # ORIGINAL
        # Normalize to the number of stars in the satellite with mass > mass_min
        mass_pdf_array /= self.imf.integrate(mass_min, self.mass_init_upper_bound)

        return mass_init_array, mass_pdf_array, mass_act_array, mag_1_array, mag_2_array

    def stellarMass(self, mass_min=0.1, steps=10000):
        """
        Compute the stellar mass (M_Sol; average per star). PDF comes from IMF, but weight by actual stellar mass.
        """
        mass_max = self.mass_init_upper_bound
            
        d_log_mass = (numpy.log10(mass_max) - numpy.log10(mass_min)) / float(steps)
        log_mass = numpy.linspace(numpy.log10(mass_min), numpy.log10(mass_max), steps)
        mass = 10.**log_mass

        if mass_min < numpy.min(self.mass_init):
            mass_act_interpolation = scipy.interpolate.interp1d(numpy.insert(self.mass_init, 0, mass_min),
                                                                numpy.insert(self.mass_act, 0, mass_min))
        else:
           mass_act_interpolation = scipy.interpolate.interp1d(self.mass_init, self.mass_act) 

        mass_act = mass_act_interpolation(mass)
        return numpy.sum(mass_act * d_log_mass * self.imf.pdf(mass, log_mode=True))

    def stellarLuminosity(self, steps=10000):
        """
        Compute the stellar luminosity (L_Sol; average per star). PDF comes from IMF.
        The range of integration only covers the input isochrone data (no extrapolation used),
        but this seems like a sub-percent effect if the isochrone goes to 0.15 stellar masses for the
        old and metal-poor stellar populations of interest.

        Note that the stellar luminosity is very sensitive to the post-AGB population.
        """
        mass_min = numpy.min(self.mass_init)
        mass_max = self.mass_init_upper_bound
        
        d_log_mass = (numpy.log10(mass_max) - numpy.log10(mass_min)) / float(steps)
        log_mass = numpy.linspace(numpy.log10(mass_min), numpy.log10(mass_max), steps)
        mass = 10.**log_mass
        
        luminosity_interpolation = scipy.interpolate.interp1d(self.mass_init, self.luminosity,fill_value=0,bounds_error=False)
        luminosity = luminosity_interpolation(mass)

        return numpy.sum(luminosity * d_log_mass * self.imf.pdf(mass, log_mode=True))
    
    def addPhotometricErrors(self, mag_delta_1, mag_delta_2):
        """
        ADW: DEPRICATED (12/23/2014)

        Add photometric errors.

        mag_delta = mag_completeness - distance_modulus
        """

        mass_center_array, mass_pdf_array, mag_1_array, mag_2_array = self.sample()
        
        std_bins = numpy.linspace(-5, 5, 50)
        std_centers = std_bins[1:] - (0.5 * (std_bins[1] - std_bins[0]))
        norm_pdf_interval = scipy.stats.norm.cdf(std_bins[1:]) - scipy.stats.norm.cdf(std_bins[0: -1])

        xx, yy = numpy.meshgrid(range(0, len(norm_pdf_interval)), range(0, len(norm_pdf_interval)))
        norm_pdf_array = norm_pdf_interval[xx] * norm_pdf_interval[yy]

        n = len(mass_pdf_array)
    
        pdf = (mass_pdf_array.reshape(n, 1, 1) * norm_pdf_array).flatten()
        mag_1_observed = ((mag_1_array).reshape(n, 1, 1) \
                          + (self.photometric_error_func_1(mag_delta_1 - mag_1_array).reshape(n, 1, 1) \
                             * (std_centers[xx] - numpy.mean(std_centers[xx])))).flatten()
        mag_2_observed = ((mag_2_array).reshape(n, 1, 1) \
                          + (self.photometric_error_func_2(mag_delta_2 - mag_2_array).reshape(n, 1, 1) \
                             * (std_centers[yy] - numpy.mean(std_centers[yy])))).flatten()

        #pylab.figure()
        #color = mag_1_array - mag_2_array
        #left = numpy.min(color) - 0.1 * (numpy.max(color) - numpy.min(color))
        #right = numpy.max(color) + 0.1 * (numpy.max(color) - numpy.min(color))
        #bottom = numpy.min(mag_1_array) - 0.1 * (numpy.max(mag_1_array) - numpy.min(mag_1_array))
        #top = numpy.max(mag_1_array) + 0.1 * (numpy.max(mag_1_array) - numpy.min(mag_1_array))        
        #pylab.hexbin(mag_1_observed - mag_2_observed, mag_1_observed, bins='log', extent=[left, right, bottom, top])

        bins_color = numpy.linspace(-0.5, 1, 60 + 1)
        bins_mag = numpy.linspace(-4, 12, 160 + 1)

        histo = numpy.histogram2d(mag_1_observed - mag_2_observed,
                                  mag_1_observed,
                                  #mag_1_observed if self.config['catalog']['band_1_detection'] else mag_2_observed,
                                  [bins_color, bins_mag],
                                  weights = pdf)[0].transpose()

        #ugali.utils.plotting.twoDimensionalHistogram('test', 'color', 'mag',
        #                                             numpy.log10(histo + 1.e-10), bins_color, bins_mag,
        #                                             lim_x = [bins_color[0], bins_color[-1]])

        #return pdf, mag_1_observed, mag_2_observed
        return histo

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

        if self.config['catalog']['band_1_detection']:
            mag = mag_1_array
        else:
            mag = mag_2_array
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

        if self.config['catalog']['band_1_detection']:
            mag = mag_1
        else:
            mag = mag_2
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
        elif self.config['catalog']['band_1_detection']:
            comp = mask.mask_1.completeness(mag_1, method=method)
            comp_cdf = comp*cdf
        elif not self.config['catalog']['band_1_detection']:
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

    observableFraction = observableFractionCMD

    def normalizeWithMask(self, mask, distance_modulus, mass_steps=1000, mass_min=0.1, kernel=None, plot=False):
        """
        ADW: DEPRICATED (12/23/2014)

        Return fraction of stars that are observable as a spatial map with same dimensions as input mask.
        """
        mass_center_array, mass_pdf_array, mag_1_array, mag_2_array = self.sample(mass_steps=mass_steps)

        normalization = numpy.zeros(mask.mask_1.mask.shape)

        if mass_min < mass_center_array[0]:
            # Accounting for stars with masses below the isochrone data range
            fraction = self.imf.integrate(mass_center_array[0], mass_center_array[-1]) \
                       / self.imf.integrate(mass_min, mass_center_array[-1])
        else:
            # Restrict the isochrone data range to match the minimum stellar mass under consideration
            fraction = 1.
            cut = mass_center_array > mass_min
            mass_center_array = mass_center_array[cut]
            mass_pdf_array = mass_pdf_array[cut]
            mag_1_array = mag_1_array[cut]
            mag_2_array = mag_2_array[cut]

        for bin_y in range(0, len(mask.roi.centers_y)):
            for bin_x in range(0, len(mask.roi.centers_x)):
                mag_1_mask = mask.mask_1.mask[bin_y][bin_x]
                mag_2_mask = mask.mask_1.mask[bin_y][bin_x]
                normalization[bin_y][bin_x] = fraction * numpy.sum(mass_pdf_array \
                                                                   * (mag_1_array + distance_modulus < mag_1_mask) \
                                                                   * (mag_2_array + distance_modulus < mag_2_mask))

        if plot:
            import ugali.utils.plotting

            plot_title = 'Fraction of Stars (>%.2f M_{Sol}) Observable within Pixel'%(mass_min)
            ugali.utils.plotting.twoDimensionalHistogram(plot_title,
                                                         'x (deg)', 'y (deg)',
                                                         normalization,
                                                         mask.roi.bins_x,
                                                         mask.roi.bins_y,
                                                         lim_x = [mask.roi.bins_x[0], mask.roi.bins_x[-1]],
                                                         lim_y = [mask.roi.bins_y[0], mask.roi.bins_y[-1]])

        if kernel is not None:
            normalization_kernel = scipy.signal.convolve(normalization, kernel.kernel, mode='same')
            if plot:
                import ugali.utils.plotting
                plot_title = 'Fraction of Stars (>%.2f M_{Sol}) Observable within Kernel'%(mass_min)
                ugali.utils.plotting.twoDimensionalHistogram(plot_title,
                                                             'x (deg)', 'y (deg)',
                                                             normalization_kernel,
                                                             mask.roi.bins_x,
                                                             mask.roi.bins_y,
                                                             lim_x = [mask.roi.bins_x[0], mask.roi.bins_x[-1]],
                                                             lim_y = [mask.roi.bins_y[0], mask.roi.bins_y[-1]])
            return normalization, normalization_kernel
        else:
            return normalization
    
    def cmd_template(self, distance_modulus, mass_steps = 400):
        """
        ADW: DEPRICATED (12/23/2014)

        Documentation here.
        """
        
        mag_1_interpolation = scipy.interpolate.interp1d(self.mass_init, self.mag_1)
        mag_2_interpolation = scipy.interpolate.interp1d(self.mass_init, self.mag_2)

        # Mass interpolation with uniform coverage between data points from isochrone file 
        mass_interpolation = scipy.interpolate.interp1d(range(0, len(self.mass_init)), self.mass_init)
        mass_array = mass_interpolation(numpy.linspace(0, len(self.mass_init) - 1, mass_steps + 1))
        d_mass = mass_array[1:] - mass_array[0: -1]
        mass_center_array = numpy.sqrt(mass_array[1:] * mass_array[0: -1])
        mass_pdf_array = d_mass * self.imf.pdf(mass_center_array, log_mode = False)
        mag_1_array = mag_1_interpolation(mass_center_array) #+ distance_modulus
        mag_2_array = mag_2_interpolation(mass_center_array) #+ distance_modulus
        
        import ugali.utils.plotting
        ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
                                                   mag_1_array - mag_2_array, mag_1_array)

        raw_input('wait')

    
        # Next, weight by detection probability
        if self.config['catalog']['band_1_detection']:
            detection_pdf_array = self.irfs.completeness_function(mag_1_array)
        else:
            detection_pdf_array = self.irfs.completeness_function(mag_2_array)

        # Finally, account for photometric errors (assume normal distribution)
        std_bins = numpy.linspace(-5, 5, 50)
        std_centers = std_bins[1:] - (0.5 * (std_bins[1] - std_bins[0]))
        norm_pdf_interval = scipy.stats.norm.cdf(std_bins[1:]) - scipy.stats.norm.cdf(std_bins[0: -1])

        xx, yy = numpy.meshgrid(range(0, len(norm_pdf_interval)), range(0, len(norm_pdf_interval)))
        norm_pdf_array = norm_pdf_interval[xx] * norm_pdf_interval[yy]

        n = len(mass_pdf_array)
    
        pdf = ((mass_pdf_array * detection_pdf_array).reshape(n, 1, 1) * norm_pdf_array).flatten()
        mag_1_observed = ((mag_1_array).reshape(n, 1, 1) \
                          + (self.irfs.magerr_functions[0](mag_1_array).reshape(n, 1, 1) \
                             * (std_centers[xx] - numpy.mean(std_centers[xx])))).flatten()
        mag_2_observed = ((mag_2_array).reshape(n, 1, 1) \
                          + (self.irfs.magerr_functions[1](mag_2_array).reshape(n, 1, 1) \
                             * (std_centers[yy] - numpy.mean(std_centers[yy])))).flatten()

        return numpy.histogram2d(mag_1_observed - mag_2_observed,
                                 mag_1_observed if self.config['catalog']['band_1_detection'] else mag_2_observed,
                                 [self.irfs.color_bins, self.irfs.mag_bins],
                                 weights = pdf)[0]

    def _setHorizontalBranch(self, horizontal_branch_stage='BHeb', pad=0.5):
        """
        Define horizontal branch region of CMD space.
        """
        cut = self.stage == horizontal_branch_stage
        
        # Check that colors are monotonically increasing
        if numpy.sum(cut) < 2 or numpy.any(numpy.sort(self.color[cut]) != self.color[cut]):
            logger.warning('Colors are not monotonically increasing.')

        self.horizontal_branch_stage_lower_envelope = scipy.interpolate.interp1d(self.color[cut], self.mag[cut] - pad, bounds_error=False, fill_value=0.)
        self.horizontal_branch_stage_upper_envelope = scipy.interpolate.interp1d(self.color[cut], self.mag[cut] + pad, bounds_error=False, fill_value=0.)

        # Assert uniform density in CMD-space
        self.horizontal_branch_density = self.imf.integrate(self.mass_init[cut][0], self.mass_init[cut][-1]) / (2. * pad * (self.color[cut][-1] - self.color[cut][0])) # mag^-2

    def horizontalBranch(self, color, mag):
        """
        Horizontal branch weight in units of mag^-2 to mirror field density. 
        """
        if numpy.isscalar(color):
            if numpy.logical_and(mag > self.horizontal_branch_stage_lower_envelope(color),
                                 mag < self.horizontal_branch_stage_upper_envelope(color)):
                return self.horizontal_branch_density
            else:
                return 0.
            
        value = numpy.zeros(len(color))
        cut = numpy.logical_and(mag > self.horizontal_branch_stage_lower_envelope(color),
                                mag < self.horizontal_branch_stage_upper_envelope(color))
        value[cut] = self.horizontal_branch_density
        return value

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
        elif self.config['catalog']['band_1_detection']:
            comp=mask.completeness(mask_1[:,np.newaxis]-mag_1, method=method)
        elif not self.config['catalog']['band_1_detection']:
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


############################################################

class CompositeIsochrone(Model):
    _params = odict([
        ('distance_modulus', Parameter(15.0, [10.0, 30.0]) ),
    ])
    _mapping = odict([])

    def __init__(self, config, **kwargs):
        super(CompositeIsochrone,self).__init__(**kwargs)
        self.config = Config(config)

        self.isochrones = []
        for name in self.config['isochrone']['infiles']:
            self.isochrones.append(Isochrone(self.config, name))

        self.weights = self.config['isochrone']['weights']

        if len(self.isochrones) != len(self.weights):
            sys.exit('ERROR: size of isochrone array and weight array must be equal')

        # Make sure the composite isochrone is properly normalized
        self.weights /= numpy.sum(self.weights)

    def __getitem__(self, key):
        return self.isochrones[key]

    def __str__(self,indent=None):
        ret = super(CompositeIsochrone,self).__str__()
        ret += '\n  Isochrone Files:'
        for name in self.config['isochrone']['infiles']:
            ret += '\n    %s'%name
        return ret

    def sample(self, mass_steps=1000, mass_min=0.1, full_data_range=False):
        """
        Documentation here.
        """
        
        mass_init_array = []
        mass_pdf_array = []
        mass_act_array = []
        mag_1_array = []
        mag_2_array = []

        for weight,isochrone in zip(self.weights,self.isochrones):
            mass_init_array_single, mass_pdf_array_single,  mass_act_array_single, mag_1_array_single, mag_2_array_single = isochrone.sample(mass_steps=mass_steps, mass_min=mass_min, full_data_range=full_data_range)
            mass_init_array.append(mass_init_array_single)
            mass_pdf_array.append(weight * mass_pdf_array_single)
            mass_act_array.append(mass_act_array_single)
            mag_1_array.append(mag_1_array_single)
            mag_2_array.append(mag_2_array_single)

        mass_init_array = numpy.concatenate(mass_init_array)
        mass_pdf_array = numpy.concatenate(mass_pdf_array)
        mass_act_array = numpy.concatenate(mass_act_array)
        mag_1_array = numpy.concatenate(mag_1_array)
        mag_2_array = numpy.concatenate(mag_2_array)

        return mass_init_array, mass_pdf_array, mass_act_array, mag_1_array, mag_2_array


    def stellarMass(self):
        """
        Compute stellar mass (M_Sol) for composite stellar population. Average per star.
        """
        total = 0.
        for weight,isochrone in zip(self.weights,self.isochrones):
            total += weight * isochrone.stellarMass()
        return total

    def stellarLuminosity(self):
        """
        Compute the stellar luminosity (L_Sol) for composite stellar population. Average per star.
        """
        total = 0.
        for weight,isochrone in zip(self.weights,self.isochrones):
            total += weight * isochrone.stellarLuminosity()
        return total

    def observableFraction(self, mask, distance_modulus):
        """
        Calculated observable fraction for all isochrones.
        """
        total = []
        for weight, isochrone in zip(self.weights, self.isochrones):
            total.append(weight * isochrone.observableFraction(mask, distance_modulus))

        return numpy.sum(total,axis=0)

    def observableFractionX(self, mask, distance_modulus):
        """
        Calculated observable fraction for all isochrones.
        """
        total = []
        for weight, isochrone in zip(self.weights, self.isochrones):
            total.append(weight*isochrone.observableFractionX(mask, distance_modulus))

        return numpy.sum(total,axis=0)



    def simulate(self, stellar_mass, distance_modulus=0.):
        """
        Simulate observed magnitudes for satellite of given mass and distance.
        """
        # Total number of stars in system
        n = int(stellar_mass/self.stellarMass()) 
        mass_init, mass_pdf, mass_act, mag_1, mag_2 = self.sample()
        cdf = numpy.cumsum(mass_pdf)
        cdf = numpy.insert(cdf, 0, 0.)
        #ADW: CDF not normalized (because of minimum mass?)
        f = scipy.interpolate.interp1d(cdf, range(0, len(cdf)), bounds_error=False, fill_value=-1)
        index = numpy.floor(f(numpy.random.uniform(size=n))).astype(int)
        #print "WARNING: non-random isochrone simulation"
        #index = numpy.floor(f(np.linspace(0,1,n))).astype(int)
        index = index[index >= 0]
        return mag_1[index]+distance_modulus, mag_2[index]+distance_modulus

    def horizontalBranch(self, color, mag):
        """
        Horizontal branch weight in units of mag^-2 to mirror field density.
        """
        value = []
        for weight, isochrone in zip(self.weights, self.isochrones):
            value.append(weight * isochrone.horizontalBranch(color, mag))
        return numpy.sum(value, axis=0)


    def histo(self,distance_modulus,delta_mag=0.03,mass_steps=10000):
        """
        Histogram the isochrone in mag-mag space.
        """
        # Isochrone will be binned in next step, so can sample many points efficiently
        mass_init,mass_pdf,mass_act,mag_1,mag_2 = self.sample(mass_steps=mass_steps)

        bins_mag_1 = np.arange(distance_modulus+mag_1.min() - (0.5*delta_mag),
                               distance_modulus+mag_1.max() + (0.5*delta_mag),
                               delta_mag)
        bins_mag_2 = np.arange(distance_modulus+mag_2.min() - (0.5*delta_mag),
                               distance_modulus+mag_2.max() + (0.5*delta_mag),
                               delta_mag)        

        # ADW: Completeness needs to go in mass_pdf here
        histo_isochrone_pdf = np.histogram2d(distance_modulus + mag_1,
                                             distance_modulus + mag_2,
                                             bins=[bins_mag_1, bins_mag_2],
                                             weights=mass_pdf)[0]

        return histo_isochrone_pdf, bins_mag_1, bins_mag_2

    def signalMMD(self,mask,distance_modulus,**kwargs):
        value = []
        for weight, isochrone in zip(self.weights, self.isochrones):
            value.append(weight * isochrone.signalMMD(mask,distance_modulus,**kwargs))
        return numpy.sum(value, axis=0)

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
        index_nonzero_0, index_nonzero_1 = numpy.nonzero((arg_mag_1_hi > -nsigma) \
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
        index_nonzero_0, index_nonzero_1 = numpy.nonzero((arg_mag_1_hi > -nsigma) \
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
    
############################################################
