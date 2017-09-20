"""
DEPRECATD 2017-09-01: ADW: This code is deprecated (probably long
before the quoted date).

Functions to create and use a look-up table for the signal color probability distribution function.
"""

import time
import numpy
import scipy.signal
import astropy.io.fits as pyfits

import ugali.utils.config
import ugali.utils.binning
import ugali.isochrone

from ugali.utils.logger import logger

msg = "'color_lut': ADW 2017-09-20"
DeprecationWarning(msg)

############################################################

def writeColorLUT2(config,
                  outfile=None, isochrone=None, distance_modulus_array=None,
                  delta_mag=None, mag_err_array=None,
                  mass_steps=10000, plot=False):
    """
    Precompute a 4-dimensional signal color probability look-up table to speed up the likelihood evaluation.
    Inputs are a Config object (or file name), an Isochrone object,
    an array of distance moduli at which to evaluate the signal color probability,
    and an array of magnitude uncertainties which set the bin edges of those dimensions (zero implicity included).
    Finally there is an outfile name.
    """
    if plot: import ugali.utils.plotting

    if type(config) == str:
        config = ugali.utils.config.Config(config)
    if outfile is None:
        outfile = config.params['color_lut']['filename']
    if isochrone is None:
        isochrones = []
        for ii, name in enumerate(config.params['isochrone']['infiles']):
            isochrones.append(ugali.isochrone.Isochrone(config, name))
        isochrone = ugali.isochrone.CompositeIsochrone(isochrones, config.params['isochrone']['weights'])
    if distance_modulus_array is None:
        distance_modulus_array = config.params['color_lut']['distance_modulus_array']
    if delta_mag is None:
        delta_mag = config.params['color_lut']['delta_mag']
    if mag_err_array is None:
        mag_err_array = config.params['color_lut']['mag_err_array']

    mag_buffer = 0.5 # Safety buffer in magnitudes around the color-magnitude space defined by the ROI
    epsilon = 1.e-10
    if config.params['catalog']['band_1_detection']:
        bins_mag_1 = numpy.arange(config.params['mag']['min'] - mag_buffer,
                                  config.params['mag']['max'] + mag_buffer + epsilon,
                                  delta_mag)
        bins_mag_2 = numpy.arange(config.params['mag']['min'] - config.params['color']['max'] - mag_buffer,
                                  config.params['mag']['max'] - config.params['color']['min'] + mag_buffer + epsilon,
                                  delta_mag)
    else:
        bins_mag_1 = numpy.arange(config.params['mag']['min'] + config.params['color']['min'] - mag_buffer,
                                  config.params['mag']['max'] + config.params['color']['max'] + mag_buffer + epsilon,
                                  delta_mag)
        bins_mag_2 = numpy.arange(config.params['mag']['min'] - mag_buffer,
                                  config.params['mag']['max'] + mag_buffer + epsilon,
                                  delta_mag)

    # Output binning configuration
    #print config.params['catalog']['band_1_detection']
    #print config.params['mag']['min'], config.params['mag']['max']
    #print config.params['color']['min'], config.params['color']['max']

    #print bins_mag_1[0], bins_mag_1[-1], len(bins_mag_1)
    #print bins_mag_2[0], bins_mag_2[-1], len(bins_mag_2)

    isochrone_mass_init, isochrone_mass_pdf, isochrone_mass_act, isochrone_mag_1, isochrone_mag_2 = isochrone.sample(mass_steps=mass_steps)

    hdul = pyfits.HDUList()

    for index_distance_modulus, distance_modulus in enumerate(distance_modulus_array):

        logger.debug('(%i/%i)'%(index_distance_modulus, len(distance_modulus_array)))

        columns_array = []
        
        time_start = time.time()

        histo_isochrone_pdf = numpy.histogram2d(distance_modulus + isochrone_mag_1,
                                                distance_modulus + isochrone_mag_2,
                                                bins=[bins_mag_1, bins_mag_2],
                                                weights=isochrone_mass_pdf)[0]
        
        if plot:
            # Checked that axis are plotted correctly
            ugali.utils.plotting.twoDimensionalHistogram('Isochrone', 'mag_1', 'mag_2',
                                                         numpy.log10(histo_isochrone_pdf + epsilon).transpose(),
                                                         bins_mag_1, bins_mag_2,
                                                         lim_x=None, lim_y=None,
                                                         vmin=None, vmax=None)

            
        
        for index_mag_err_1, mag_err_1 in enumerate(mag_err_array):
            for index_mag_err_2, mag_err_2 in enumerate(mag_err_array):
                logger.debug('  Distance modulus = %.2f mag_err_1 = %.2f mag_err_2 = %.2f'%(distance_modulus, mag_err_1, mag_err_2))

                mag_1_sigma_step = delta_mag / mag_err_1
                n = int(numpy.ceil(4. / mag_1_sigma_step))
                mag_1_sigma = numpy.arange(-1. * (n + 0.5) * mag_1_sigma_step,
                                           ((n + 0.5) * mag_1_sigma_step) + epsilon,
                                           mag_1_sigma_step)
                mag_1_pdf_array = scipy.stats.norm.cdf(mag_1_sigma[1:]) - scipy.stats.norm.cdf(mag_1_sigma[0:-1])

                mag_2_sigma_step = delta_mag / mag_err_2
                n = int(numpy.ceil(4. / mag_2_sigma_step))
                mag_2_sigma = numpy.arange(-1. * (n + 0.5) * mag_2_sigma_step,
                                           ((n + 0.5) * mag_2_sigma_step) + epsilon,
                                           mag_2_sigma_step)
                mag_2_pdf_array = scipy.stats.norm.cdf(mag_2_sigma[1:]) - scipy.stats.norm.cdf(mag_2_sigma[0:-1])

                mag_1_pdf, mag_2_pdf = numpy.meshgrid(mag_2_pdf_array, mag_1_pdf_array)
                
                pdf = mag_1_pdf * mag_2_pdf

                histo_isochrone_pdf_convolve = scipy.signal.convolve2d(histo_isochrone_pdf, pdf, mode='same')

                if plot:
                    # Checked that axis are plotted correctly
                    ugali.utils.plotting.twoDimensionalHistogram('Convolved Isochrone', 'mag_1', 'mag_2',
                                                                 numpy.log10(histo_isochrone_pdf_convolve + epsilon).transpose(),
                                                                 bins_mag_1, bins_mag_2,
                                                                 lim_x=None, lim_y=None,
                                                                 vmin=None, vmax=None)

                columns_array.append(pyfits.Column(name = '%i%i'%(index_mag_err_1, index_mag_err_2),
                                                   format = '%iE'%(histo_isochrone_pdf_convolve.shape[1]),
                                                   array = histo_isochrone_pdf_convolve))

        hdu = pyfits.new_table(columns_array)
        hdu.header.update('DIST_MOD', distance_modulus)
        hdu.name = '%.2f'%(distance_modulus)
        hdul.append(hdu)

        time_end = time.time()
        logger.debug('%.2f s'%(time_end - time_start))

    # Store distance modulus info
    columns_array = [pyfits.Column(name = 'DISTANCE_MODULUS',
                                   format = 'E',
                                   array = distance_modulus_array)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'DISTANCE_MODULUS'
    hdul.append(hdu)

    # Store magnitude error info
    columns_array = [pyfits.Column(name = 'BINS_MAG_ERR',
                                   format = 'E',
                                   array = numpy.insert(mag_err_array, 0, 0.))]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_ERR'
    hdul.append(hdu)

    # Store magnitude 1 info
    columns_array = [pyfits.Column(name = 'BINS_MAG_1',
                                   format = 'E',
                                   array = bins_mag_1)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_1'
    hdul.append(hdu)

    # Store magnitude 2 info
    columns_array = [pyfits.Column(name = 'BINS_MAG_2',
                                   format = 'E',
                                   array = bins_mag_2)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_2'
    hdul.append(hdu)

    logger.info('Writing look-up table to %s'%(outfile))
    hdul.writeto(outfile, clobber = True)

############################################################

def writeColorLUT(config,
                  outfile=None, isochrone=None, distance_modulus_array=None,
                  delta_mag=None, mag_err_array=None,
                  mass_steps=1000000, plot=False):
    """
    Precompute a 4-dimensional signal color probability look-up table to speed up the likelihood evaluation.
    Inputs are a Config object (or file name), an Isochrone object,
    an array of distance moduli at which to evaluate the signal color probability,
    and an array of magnitude uncertainties which set the bin edges of those dimensions (zero implicity included).
    Finally there is an outfile name.
    """
    if plot: import ugali.utils.plotting
    if type(config) == str:
        config = ugali.utils.config.Config(config)
    if outfile is None:
        outfile = config.params['color_lut']['filename']
    if isochrone is None:
        isochrones = []
        for ii, name in enumerate(config.params['isochrone']['infiles']):
            isochrones.append(ugali.isochrone.Isochrone(config, name))
        isochrone = ugali.isochrone.CompositeIsochrone(isochrones, config.params['isochrone']['weights'])
    if distance_modulus_array is None:
        distance_modulus_array = config.params['color_lut']['distance_modulus_array']
    if delta_mag is None:
        delta_mag = config.params['color_lut']['delta_mag']
    if mag_err_array is None:
        mag_err_array = config.params['color_lut']['mag_err_array']

    mag_buffer = 0.5 # Safety buffer in magnitudes around the color-magnitude space defined by the ROI
    epsilon = 1.e-10
    if config.params['catalog']['band_1_detection']:
        bins_mag_1 = numpy.arange(config.params['mag']['min'] - mag_buffer,
                                  config.params['mag']['max'] + mag_buffer + epsilon,
                                  delta_mag)
        bins_mag_2 = numpy.arange(config.params['mag']['min'] - config.params['color']['max'] - mag_buffer,
                                  config.params['mag']['max'] - config.params['color']['min'] + mag_buffer + epsilon,
                                  delta_mag)
    else:
        bins_mag_1 = numpy.arange(config.params['mag']['min'] + config.params['color']['min'] - mag_buffer,
                                  config.params['mag']['max'] + config.params['color']['max'] + mag_buffer + epsilon,
                                  delta_mag)
        bins_mag_2 = numpy.arange(config.params['mag']['min'] - mag_buffer,
                                  config.params['mag']['max'] + mag_buffer + epsilon,
                                  delta_mag)

    # Output binning configuration
    #print config.params['catalog']['band_1_detection']
    #print config.params['mag']['min'], config.params['mag']['max']
    #print config.params['color']['min'], config.params['color']['max']

    #print bins_mag_1[0], bins_mag_1[-1], len(bins_mag_1)
    #print bins_mag_2[0], bins_mag_2[-1], len(bins_mag_2)

    isochrone_mass_init, isochrone_mass_pdf, isochrone_mass_act, isochrone_mag_1, isochrone_mag_2 = isochrone.sample(mass_steps=mass_steps)

    # make randoms
    randoms_1 = numpy.random.normal(0., 1., len(isochrone_mass_pdf))
    randoms_2 = numpy.random.normal(0., 1., len(isochrone_mass_pdf))

    hdul = pyfits.HDUList()

    for index_distance_modulus, distance_modulus in enumerate(distance_modulus_array):

        logger.debug('(%i/%i)'%(index_distance_modulus, len(distance_modulus_array)))

        columns_array = []
        
        time_start = time.time()
        
        for index_mag_err_1, mag_err_1 in enumerate(mag_err_array):
            for index_mag_err_2, mag_err_2 in enumerate(mag_err_array):
                logger.debug('  (%i/%i) Distance modulus = %.2f mag_err_1 = %.3f mag_err_2 = %.3f'%(index_mag_err_1 * len(mag_err_array) + index_mag_err_2,
                                                                                             len(mag_err_array)**2,
                                                                                             distance_modulus,
                                                                                             mag_err_1,
                                                                                             mag_err_2))
                
                # Add randoms
                histo_isochrone_pdf = numpy.histogram2d(distance_modulus + isochrone_mag_1 + randoms_1 * mag_err_1,
                                                        distance_modulus + isochrone_mag_2 + randoms_2 * mag_err_2,
                                                        bins=[bins_mag_1, bins_mag_2],
                                                        weights=isochrone_mass_pdf)[0]
                
                if plot:
                    # Checked that axis are plotted correctly
                    ugali.utils.plotting.twoDimensionalHistogram('Convolved Isochrone', 'mag_1', 'mag_2',
                                                                 numpy.log10(histo_isochrone_pdf + epsilon).transpose(),
                                                                 bins_mag_1, bins_mag_2,
                                                                 lim_x=None, lim_y=None,
                                                                 vmin=None, vmax=None)
                    raw_input('WAIT')

                columns_array.append(pyfits.Column(name = '%i%i'%(index_mag_err_1, index_mag_err_2),
                                                   format = '%iE'%(histo_isochrone_pdf.shape[1]),
                                                   array = histo_isochrone_pdf))

        hdu = pyfits.new_table(columns_array)
        hdu.header.update('DIST_MOD', distance_modulus)
        hdu.name = '%.2f'%(distance_modulus)
        hdul.append(hdu)

        time_end = time.time()
        logger.debug('%.2f s'%(time_end - time_start))

    # Store distance modulus info
    columns_array = [pyfits.Column(name = 'DISTANCE_MODULUS',
                                   format = 'E',
                                   array = distance_modulus_array)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'DISTANCE_MODULUS'
    hdul.append(hdu)

    # Store magnitude error info
    columns_array = [pyfits.Column(name = 'BINS_MAG_ERR',
                                   format = 'E',
                                   array = numpy.insert(mag_err_array, 0, 0.))]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_ERR'
    hdul.append(hdu)

    # Store magnitude 1 info
    columns_array = [pyfits.Column(name = 'BINS_MAG_1',
                                   format = 'E',
                                   array = bins_mag_1)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_1'
    hdul.append(hdu)

    # Store magnitude 2 info
    columns_array = [pyfits.Column(name = 'BINS_MAG_2',
                                   format = 'E',
                                   array = bins_mag_2)]
    hdu = pyfits.new_table(columns_array)
    hdu.name = 'BINS_MAG_2'
    hdul.append(hdu)

    logger.info('Writing look-up table to %s'%(outfile))
    hdul.writeto(outfile, clobber = True)

############################################################

def mergeColorLUT(infiles):
    """
    Tool to merge color look-up tables.
    """
    pass

############################################################

def readColorLUT(infile, distance_modulus, mag_1, mag_2, mag_err_1, mag_err_2):
    """
    Take in a color look-up table and return the signal color evaluated for each object.
    Consider making the argument a Catalog object rather than magnitudes and uncertainties.
    """

    reader = pyfits.open(infile)

    distance_modulus_array = reader['DISTANCE_MODULUS'].data.field('DISTANCE_MODULUS')
    if not numpy.any(numpy.fabs(distance_modulus_array - distance_modulus) < 1.e-3):
        logger.warning("Distance modulus %.2f not available in file %s"%(distance_modulus, infile))
        logger.warning('         available distance moduli:'+str(distance_modulus_array))
        return False

    distance_modulus_key = '%.2f'%(distance_modulus_array[numpy.argmin(numpy.fabs(distance_modulus_array - distance_modulus))])

    bins_mag_err = reader['BINS_MAG_ERR'].data.field('BINS_MAG_ERR') 
    bins_mag_1 = reader['BINS_MAG_1'].data.field('BINS_MAG_1') 
    bins_mag_2 = reader['BINS_MAG_2'].data.field('BINS_MAG_2') 
    # Note that magnitude uncertainty is always assigned by rounding up, is this the right thing to do?
    index_mag_err_1 = numpy.clip(numpy.digitize(mag_err_1, bins_mag_err) - 1,
                                 0, len(bins_mag_err) - 2)
    index_mag_err_2 = numpy.clip(numpy.digitize(mag_err_2, bins_mag_err) - 1,
                                 0, len(bins_mag_err) - 2)

    u_color = numpy.zeros(len(mag_1))

    for index_mag_err_1_select in range(0, len(bins_mag_err) - 1):
        for index_mag_err_2_select in range(0, len(bins_mag_err) - 1):
            cut = numpy.logical_and(index_mag_err_1 == index_mag_err_1_select,
                                    index_mag_err_2 == index_mag_err_2_select)
            if numpy.sum(cut) < 1:
                continue
            histo = reader[distance_modulus_key].data.field('%i%i'%(index_mag_err_1_select, index_mag_err_2_select))
            u_color[cut] = ugali.utils.binning.take2D(histo,
                                                      mag_2[cut], mag_1[cut],
                                                      bins_mag_2, bins_mag_1)
    
    reader.close()
    return u_color

############################################################
