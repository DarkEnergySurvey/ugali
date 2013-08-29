"""
Look-up table for color PDF.
"""

import time
import numpy
import scipy.signal
import pyfits
import pylab

import ugali.utils.plotting
import ugali.utils.binning

pylab.ion()

############################################################

def writeColorLUT(config, isochrone, distance_modulus_array, mag_err_array, mass_steps=10000, delta_mag=0.01, plot=False):
    """

    """
    mag_buffer = 0.5
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

    print config.params['catalog']['band_1_detection']
    print config.params['mag']['min'], config.params['mag']['max']
    print config.params['color']['min'], config.params['color']['max']

    print bins_mag_1[0], bins_mag_1[-1], len(bins_mag_1)
    print bins_mag_2[0], bins_mag_2[-1], len(bins_mag_2)

    isochrone_mass_init, isochrone_mass_pdf, isochrone_mass_act, isochrone_mag_1, isochrone_mag_2 = isochrone.sample(mass_steps=mass_steps)

    hdul = pyfits.HDUList()

    for distance_modulus in distance_modulus_array:

        columns_array = []

        time_start = time.time()

        histo_isochrone_pdf = numpy.histogram2d(distance_modulus + isochrone_mag_1,
                                                distance_modulus + isochrone_mag_2,
                                                bins=[bins_mag_1, bins_mag_2],
                                                weights=isochrone_mass_pdf)[0]

        print numpy.sum(histo_isochrone_pdf)

        #print histo_isochrone_pdf.shape

        if plot:
            # Checked that axis are plotted correctly
            ugali.utils.plotting.twoDimensionalHistogram('Isochrone', 'mag_1', 'mag_2',
                                                         numpy.log10(histo_isochrone_pdf + epsilon).transpose(),
                                                         bins_mag_1, bins_mag_2,
                                                         lim_x=None, lim_y=None,
                                                         vmin=None, vmax=None)

            
        
        for index_mag_err_1, mag_err_1 in enumerate(mag_err_array):
            for index_mag_err_2, mag_err_2 in enumerate(mag_err_array):
                print 'Distance modulus = %.2f mag_err_1 = %.2f mag_err_2 = %.2f'%(distance_modulus,
                                                                                   mag_err_1,
                                                                                   mag_err_2)

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

                #print numpy.sum(pdf)

                histo_isochrone_pdf_convolve = scipy.signal.convolve2d(histo_isochrone_pdf, pdf, mode='same')

                #print bins_mag_1.shape, bins_mag_2.shape
                #print histo_isochrone_pdf.shape
                #print pdf.shape

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
        print '%.2f s'%(time_end - time_start)

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
        
    outfile = 'test.fits'
    hdul.writeto(outfile, clobber = True)

############################################################

def readColorLUT(infile, distance_modulus, mag_1, mag_2, mag_err_1, mag_err_2):

    reader = pyfits.open(infile)

    print reader.info()

    distance_modulus_array = reader['DISTANCE_MODULUS'].data.field('DISTANCE_MODULUS')

    if not numpy.any(numpy.fabs(distance_modulus_array - distance_modulus) < 1.e-3):
        print 'ERROR: distance modulus %.2f not available in file %s'%(distance_modulus, infile)
        print '       available distance moduli:', distance_modulus_array

    distance_modulus_key = '%.2f'%(distance_modulus_array[numpy.argmin(numpy.fabs(distance_modulus_array - distance_modulus))])
    print distance_modulus_key

    bins_mag_err = reader['BINS_MAG_ERR'].data.field('BINS_MAG_ERR') 
    print bins_mag_err

    bins_mag_1 = reader['BINS_MAG_1'].data.field('BINS_MAG_1') 
    print bins_mag_1[0], bins_mag_1[-1], len(bins_mag_1)

    bins_mag_2 = reader['BINS_MAG_2'].data.field('BINS_MAG_2') 
    print bins_mag_2[0], bins_mag_2[-1], len(bins_mag_2)

    #index_mag_1 = numpy.clip(numpy.digitize(mag_1, bins_mag_1) - 1,
    #                             0, len(bins_mag_err) - 2)
    #index_mag_2 = numpy.clip(numpy.digitize(mag_2, bins_mag_2) - 1,
    #                             0, len(bins_mag_err) - 2)

    # Note that assigning error
    # Is this actually the right thing to do?
    index_mag_err_1 = numpy.clip(numpy.digitize(mag_err_1, bins_mag_err) - 1,
                                 0, len(bins_mag_err) - 2)
    index_mag_err_2 = numpy.clip(numpy.digitize(mag_err_2, bins_mag_err) - 1,
                                 0, len(bins_mag_err) - 2)

    u_color = numpy.zeros(len(mag_1))

    for index_mag_err_1_select in range(0, len(bins_mag_err) - 1):
        for index_mag_err_2_select in range(0, len(bins_mag_err) - 1):
            cut = numpy.logical_and(index_mag_err_1 == index_mag_err_1_select,
                                    index_mag_err_2 == index_mag_err_2_select)

            print numpy.sum(cut)

            if numpy.sum(cut) < 1:
                continue

            histo = reader[distance_modulus_key].data.field('%i%i'%(index_mag_err_1_select, index_mag_err_2_select))

            print '%i%i'%(index_mag_err_1_select, index_mag_err_2_select)
            print numpy.sum(histo)

            print mag_1[cut][0:10]
            print mag_2[cut][0:10]

            print histo.shape
            print bins_mag_1.shape
            print bins_mag_2.shape

            u_color[cut] = ugali.utils.binning.take2D(histo,
                                                      mag_2[cut], mag_1[cut],
                                                      bins_mag_2, bins_mag_1)
    
    reader.close()

    return u_color

#    def __init__(self, infile):
#
#        self.reader = 
#        
#        self.distance_modulus_array = 

############################################################
