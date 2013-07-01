"""
Object for isochrone storage and basic calculations.

NOTE: only absolute magnitudes are used in the Isochrone class
"""

import numpy
import scipy.interpolate
import scipy.stats
import pylab

import ugali.analysis.imf
import ugali.observation.photometric_errors # Probably won't need this in the future since will be passed
import ugali.utils.plotting

############################################################

class Isochrone:

    def __init__(self, config, infile, infile_format='padova_des'):
        """
        Initialize an isochrone instance.
        """
        
        self.config = config
        self.infile = infile

        if infile_format.lower() == 'padova_des':
            self._parseIsochronePadovaDES()
        elif infile_format.lower() == 'padova':
            self._parseIsochronePadovaDES()
        else:
            print 'WARNING: did not recognize infile format %s'%(infile_format)

        self.imf = ugali.analysis.imf.IMF(self.config.params['isochrone']['imf'])

        # Check where post-AGB isochrone data points begin
        self.mass_init_upper_bound = numpy.max(self.mass_init)
        if numpy.any(self.stage == 'LTP'):
            self.index = numpy.nonzero(self.stage == 'LTP')[0][0]
        else:
            self.index = len(self.mass_init)
        
        # Other housekeeping
        if self.config.params['catalog']['band_1_detection']:
            self.mag = self.mag_1
        else:
            self.mag = self.mag_2
        self.color = self.mag_1 - self.mag_2

        # Photometric error functions
        # This should actually be set in the future
        #self.photometric_error_func_1 = ugali.observation.photometric_errors.PhotometricErrors()
        #self.photometric_error_func_2 = ugali.observation.photometric_errors.PhotometricErrors()

    def plotCMD(self):
        """
        Show the color-magnitude diagram of isochrone points in absolute magnitudes.
        """
        ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
                                                   self.color, self.mag)
        y_min, y_max = pylab.axis()[2], pylab.axis()[3]
        pylab.ylim(y_max, y_min)

    def _parseIsochronePadova(self):
        """
        Reads an isochrone file in the Padova format and returns the age (log yrs), metallicity (Z), and an
        array with initial stellar mass and corresponding magnitudes.
        """
        mass_init_field = self.config.params['isochrone']['mass_init_field']
        mag_1_field = self.config.params['isochrone']['mag_1_field']
        mag_2_field = self.config.params['isochrone']['mag_2_field']
        
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
    def _parseIsochronePadovaDES(self):
        """
        Reads an isochrone file in the Padova DES format and returns the age (log10 yrs), metallicity (Z), and an
        array with initial stellar mass and corresponding magnitudes.
        """
        mass_init_field = self.config.params['isochrone']['mass_init_field']
        mass_act_field = self.config.params['isochrone']['mass_act_field']
        mag_1_field = self.config.params['isochrone']['mag_1_field']
        mag_2_field = self.config.params['isochrone']['mag_2_field']
        stage_field = self.config.params['isochrone']['stage_field']
        
        index_dict = {mass_init_field: 1,
                      mass_act_field: 2,
                      'g': 7,
                      'r': 8,
                      'i': 9,
                      'z': 10,
                      'Y': 11,
                      stage_field: 18}
    
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
        mag_1 = []
        mag_2 = []
        stage = []
        for line in lines:
            parts = line.split()
            if len(parts) < 11:
                continue
            mass_init.append(float(parts[index_dict[mass_init_field]]))
            mass_act.append(float(parts[index_dict[mass_act_field]]))
            mag_1.append(float(parts[index_dict[mag_1_field]]))
            mag_2.append(float(parts[index_dict[mag_2_field]]))

            if len(parts) == index_dict[stage_field] + 1:
                stage.append(parts[index_dict[stage_field]])
            else:
                stage.append('')

            for key in index_dict.keys():
                if key in [mass_init_field, mass_act_field, stage_field]:
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
            # Not generating points for the post-AGN stars,
            # but still count those stars towards the normalization
            mass_init = self.mass_init[0: self.index]
            mass_act = self.mass_act[0: self.index]
            mag_1 = self.mag_1[0: self.index]
            mag_2 = self.mag_2[0: self.index]
        
        mass_act_interpolation = scipy.interpolate.interp1d(mass_init, mass_act)
        mag_1_interpolation = scipy.interpolate.interp1d(mass_init, mag_1)
        mag_2_interpolation = scipy.interpolate.interp1d(mass_init, mag_2)

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
        Compute the stellar mass. PDF comes from IMF, but weight by actual stellar mass.
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
        
    def addPhotometricErrors(self, mag_delta_1, mag_delta_2):
        """
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
                                  #mag_1_observed if self.config.params['catalog']['band_1_detection'] else mag_2_observed,
                                  [bins_color, bins_mag],
                                  weights = pdf)[0].transpose()

        #ugali.utils.plotting.twoDimensionalHistogram('test', 'color', 'mag',
        #                                             numpy.log10(histo + 1.e-10), bins_color, bins_mag,
        #                                             lim_x = [bins_color[0], bins_color[-1]])

        #return pdf, mag_1_observed, mag_2_observed
        return histo

    def observableFraction(self, mask, distance_modulus, mass_min=0.1):
        """
        Compute observable fraction of stars with masses greater than mass_min in each mask pixel
        """
        mass_init_array, mass_pdf_array, mass_act_array, mag_1_array, mag_2_array = self.sample(mass_min=mass_min,
                                                                                                full_data_range=False)
        if self.config.params['catalog']['band_1_detection']:
            mag = mag_1_array
        else:
            mag = mag_2_array
        color = mag_1_array - mag_2_array

        observable_fraction = numpy.zeros(len(mask.roi.pixels))

        for ii in range(0, len(mask.roi.pixels)):
            mag_1_mask = mask.mask_1.mask_roi[mask.roi.pixels[ii]]
            mag_2_mask = mask.mask_2.mask_roi[mask.roi.pixels[ii]]
            observable_fraction[ii] = numpy.sum(mass_pdf_array \
                                                * (mag_1_array + distance_modulus < mag_1_mask) \
                                                * (mag_2_array + distance_modulus < mag_2_mask) \
                                                * (mag + distance_modulus > mask.roi.bins_mag[0]) \
                                                * (mag + distance_modulus < mask.roi.bins_mag[-1]) \
                                                * (color > mask.roi.bins_color[0]) \
                                                * (color < mask.roi.bins_color[-1]))
        
        return observable_fraction

    def normalizeWithMask(self, mask, distance_modulus, mass_steps=1000, mass_min=0.1, kernel=None, plot=False):
        """
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
        
        ugali.utils.plotting.twoDimensionalScatter('test', 'color (mag)', 'mag (mag)',
                                                   mag_1_array - mag_2_array, mag_1_array)

        raw_input('wait')

    
        # Next, weight by detection probability
        if self.config.params['catalog']['band_1_detection']:
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
                                 mag_1_observed if self.config.params['catalog']['band_1_detection'] else mag_2_observed,
                                 [self.irfs.color_bins, self.irfs.mag_bins],
                                 weights = pdf)[0]

############################################################

class CompositeIsochrone:

    def __init__(self, isochrones, weights):

        self.isochrones = isochrones
        self.weights = weights

        if len(self.isochrones) != len(self.weights):
            sys.exit('ERROR: size of isochrone array and weight array must be equal')

        # Make sure the composite isochrone is properly normalized
        self.weights /= numpy.sum(self.weights)

    def sample(self, mass_steps=1000, mass_min=0.1, full_data_range=False):
        """

        """
        
        mass_init_array = []
        mass_pdf_array = []
        mass_act_array = []
        mag_1_array = []
        mag_2_array = []

        for ii in range(0, len(self.isochrones)):
            mass_init_array_single, mass_pdf_array_single,  mass_act_array_single, mag_1_array_single, mag_2_array_single = self.isochrones[ii].sample(mass_steps=mass_steps, mass_min=mass_min, full_data_range=full_data_range)
            mass_init_array.append(mass_init_array_single)
            mass_pdf_array.append(self.weights[ii] * mass_pdf_array_single)
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
        Compute stellar mass (M_Sol) for composite stellar population.
        """
        sum = 0.
        for ii in range(0, len(self.isochrones)):
            sum += self.weights[ii] * self.isochrones[ii].stellarMass()
        return sum

    def observableFraction(self, mask, distance_modulus):
        """

        """
        value = numpy.zeros(len(mask.roi.pixels))
        for ii in range(0, len(self.isochrones)):
            value += self.weights[ii] * self.isochrones[ii].observableFraction(mask, distance_modulus)
        return value

############################################################
