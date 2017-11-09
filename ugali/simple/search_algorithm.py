#!/usr/bin/env python
"""
Simple binning search algorithm
"""
__author__ = "Keith Bechtol, Sidney Mau"

# Set the backend first!
import matplotlib
matplotlib.use('Agg')
import pylab

import sys
import os
import glob
import numpy
import numpy as np
from matplotlib import mlab
import pyfits
import healpy
import scipy.interpolate
from scipy import interpolate
import scipy.ndimage
import scipy.signal
import scipy.stats
import scipy.spatial

import ugali.utils.healpix
import ugali.utils.projector
import ugali.isochrone
import ugali.utils.plotting
import ugali.candidate.associate

from ugali.isochrone import factory as isochrone_factory
from astropy.coordinates import SkyCoord

import yaml

###########################################################

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nside   = cfg['nside']
datadir = cfg['datadir']

maglim_g = cfg['maglim_g']
maglim_r = cfg['maglim_r']

mag_g_dred_flag         = cfg['mag_g_dred_flag']
mag_r_dred_flag         = cfg['mag_r_dred_flag']
#mag_g_flag              = cfg['mag_g_flag']
#mag_r_flag              = cfg['mag_r_flag']
mag_err_g_flag          = cfg['mag_err_g_flag']
mag_err_r_flag          = cfg['mag_err_r_flag']
#extinction_g_flag       = cfg['extinction_g_flag']
#extinction_r_flag       = cfg['extinction_r_flag']
star_galaxy_classification = cfg['star_galaxy_classification']
#spread_model_r_flag     = cfg['spread_model_r_flag']
#spread_model_r_err_flag = cfg['spread_model_r_err_flag']
flags_g                 = cfg['flags_g']
flags_r                 = cfg['flags_r']

results_dir = os.path.join(os.getcwd(), cfg['results_dir'])
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

###########################################################

print matplotlib.get_backend()

############################################################

def cutIsochronePath(g, r, g_err, r_err, isochrone, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    if numpy.all(isochrone.stage == 'Main'):
        # Dotter case
        index_transition = len(isochrone.stage)
    else:
        # Other cases
        index_transition = numpy.nonzero(isochrone.stage > 3)[0][0] + 1    
    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]
    
    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

    cut = numpy.logical_or(cut_1, cut_2)

    mag_bins = numpy.arange(17., 24.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = numpy.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = numpy.median(numpy.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

############################################################

#def circle(x, y, r):
#    phi = numpy.linspace(0, 2 * numpy.pi, 1000)
#    return x + (r * numpy.cos(phi)), y + (r * numpy.sin(phi))

############################################################

try:
    ra_select, dec_select = float(sys.argv[1]), float(sys.argv[2])
except:
    sys.exit('ERROR! Coordinates not given in correct format.')

print 'Search coordinates: (RA, Dec) = (%.2f, %.2f)'%(ra_select, dec_select)

# Now cut for a single pixel
pix_nside_select = ugali.utils.healpix.angToPix(nside, ra_select, dec_select)
ra_select, dec_select = ugali.utils.healpix.pixToAng(nside, pix_nside_select)
pix_nside_neighbors = numpy.concatenate([[pix_nside_select], healpy.get_all_neighbours(nside, pix_nside_select)])

############################################################

data_array = []
for pix_nside in pix_nside_neighbors:
    #infile = '%s/cat_hpx_%05i.fits'%(datadir, pix_nside)
    infile = '%s/y3a2_ngmix_cm_%05i.fits'%(datadir, pix_nside)
    #infile = '%s/*_%05i.fits'%(datadir, pix_nside) # TODO - get this to work for adaptable directories
    if not os.path.exists(infile):
        continue
    reader = pyfits.open(infile)
    data_array.append(reader[1].data)
    reader.close()
print 'Assembling data...'
data = numpy.concatenate(data_array) # TODO reduce this to just use needed columns so there is no excessive use of memory

# De-redden magnitudes
#try:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_dred_flag], data[mag_r_dred_flag]])
#except:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag] - data[extinction_g_flag], data[mag_r_flag] - data[extinction_r_flag]])
#except:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag], data[mag_r_flag]])

print 'Found %i objects...'%(len(data))

############################################################

print 'Applying cuts...'
#cut = (data[flags_g] < 4) & (data[flags_r] < 4) \
#      & (data['WAVG_MAG_PSF_DRED_G'] < 24.0) \
#      & ((data['WAVG_MAG_PSF_DRED_G'] - data['WAVG_MAG_PSF_DRED_R']) < 1.) \
#      & (numpy.fabs(data[spread_model_r_flag]) < 0.003 + data[spread_model_r_err_flag])
#
#cut_gal = (data[flags_g] < 4) & (data[flags_r] < 4) \
#          & (data['WAVG_MAG_PSF_DRED_G'] < 24.0) \
#          & ((data['WAVG_MAG_PSF_DRED_G'] - data['WAVG_MAG_PSF_DRED_R']) < 1.) \
#          & (numpy.fabs(data[spread_model_r_flag]) > 0.003 + data[spread_model_r_err_flag])

cut = (data[star_galaxy_classification] >= 0) & (data[star_galaxy_classification] <= 1)
cut_gal = (data[star_galaxy_classification] > 1)

data_gal = data[cut_gal]
data = data[cut]

print '%i star-like objects in ROI...'%(len(data))
print '%i galaxy-like objects in ROI...'%(len(data_gal))

############################################################

def searchByDistance(data, distance_modulus, ra_select, dec_select, magnitude_threshold=23.):
    """
    Idea: 
    Send a data extension that goes to faint magnitudes, e.g., g < 24.
    Use the whole region to identify hotspots using a slightly brighter 
    magnitude threshold, e.g., g < 23, so not susceptible to variations 
    in depth. Then compute the local field density using a small annulus 
    around each individual hotspot, e.g., radius 0.3 to 0.5 deg.
    """

    print 'Distance = %.1f kpc (m-M = %.1f)'%(ugali.utils.projector.distanceModulusToDistance(distance_modulus), distance_modulus)

    dirname = '/home/s1/kadrlica/.ugali/isochrones/des/dotter2016/'
    iso = ugali.isochrone.factory('Dotter', hb_spread=0, dirname=dirname)
    iso.age = 12.
    iso.metallicity = 0.0001
    iso.distance_modulus = distance_modulus

    cut = cutIsochronePath(data[mag_g_dred_flag], data[mag_r_dred_flag], data[mag_err_g_flag], data[mag_err_r_flag], iso, radius=0.1)
    data = data[cut]
    cut_magnitude_threshold = (data[mag_g_dred_flag] < magnitude_threshold)

    print '%i objects left after isochrone cut...'%(len(data))

    ###

    proj = ugali.utils.projector.Projector(ra_select, dec_select)
    x, y = proj.sphereToImage(data['RA'][cut_magnitude_threshold], data['DEC'][cut_magnitude_threshold]) # Trimmed magnitude range for hotspot finding
    x_full, y_full = proj.sphereToImage(data['RA'], data['DEC']) # Full magnitude range for significance evaluation
    delta_x = 0.01
    area = delta_x**2
    smoothing = 2. / 60. # Was 3 arcmin
    bins = numpy.arange(-8., 8. + 1.e-10, delta_x)
    centers = 0.5 * (bins[0: -1] + bins[1:])
    yy, xx = numpy.meshgrid(centers, centers)

    h = numpy.histogram2d(x, y, bins=[bins, bins])[0]

    #pylab.figure('raw')
    #pylab.imshow(h.T, interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower', cmap='binary')
    #pylab.xlim([8, -8])
    #pylab.ylim([-8, 8])
    #pylab.colorbar()

    #npix_256 = healpy.nside2npix(256)
    #pix_256 = ugali.utils.healpix.angToPix(256, data['RA'], data['DEC'])
    #m_256 = numpy.histogram(pix_256, numpy.arange(npix_256 + 1))[0].astype(float)
    #healpy.mollview(m_256)

    #pylab.figure()
    #pylab.scatter(x, y, edgecolor='none', s=1)
    #pylab.xlim([5, -5])
    #pylab.ylim([-5, 5])

    h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)

    #cut_goodcoverage = (data['NEPOCHS_G'][cut_magnitude_threshold] >= 2) & (data['NEPOCHS_R'][cut_magnitude_threshold] >= 2)
    # expect NEPOCHS to be good in DES data

    delta_x_coverage = 0.1
    area_coverage = (delta_x_coverage)**2
    bins_coverage = numpy.arange(-5., 5. + 1.e-10, delta_x_coverage)
    h_coverage = numpy.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
    #h_goodcoverage = numpy.histogram2d(x[cut_goodcoverage], y[cut_goodcoverage], bins=[bins_coverage, bins_coverage])[0]
    h_goodcoverage = numpy.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]

    n_goodcoverage = h_coverage[h_goodcoverage > 0].flatten()
    #pylab.figure('poisson')
    #pylab.clf()
    #pylab.hist(n_goodcoverage, bins=numpy.arange(20) - 0.5, normed=True)
    #pylab.scatter(numpy.arange(20), scipy.stats.poisson.pmf(numpy.arange(20), mu=numpy.median(n_goodcoverage)), c='red', zorder=10)

    #print numpy.histogram(n_goodcoverage, bins=numpy.arange(20), normed=True)
    #print scipy.stats.poisson.pmf(numpy.arange(20), mu=numpy.median(n_goodcoverage))
    #print 

    #characteristic_density = numpy.mean(n_goodcoverage) / area_coverage # per square degree
    characteristic_density = numpy.median(n_goodcoverage) / area_coverage # per square degree

    #vmax = min(3. * characteristic_density * area, numpy.max(h_g))
    #
    #pylab.figure('smooth')
    #pylab.clf()
    #pylab.imshow(h_g.T,
    #             interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower', cmap='gist_heat', vmax=vmax)
    #pylab.colorbar()
    #pylab.xlim([8, -8])
    #pylab.ylim([-8, 8])
    #pylab.xlabel(r'$\Delta$ RA (deg)')
    #pylab.ylabel(r'$\Delta$ Dec (deg)')
    #pylab.title('(RA, Dec, mu) = (%.2f, %.2f, %.2f)'%(ra_select, dec_select, distance_modulus))

    factor_array = numpy.arange(1., 5., 0.05)
    rara, decdec = proj.imageToSphere(xx.flatten(), yy.flatten())
    cutcut = (ugali.utils.healpix.angToPix(nside, rara, decdec) == pix_nside_select).reshape(xx.shape)
    threshold_density = 5 * characteristic_density * area
    for factor in factor_array:
        h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > (area * characteristic_density * factor))
        #print 'factor', factor, n_region, n_region < 10
        if n_region < 10:
            threshold_density = area * characteristic_density * factor
            break
    
    h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > threshold_density)
    #pylab.figure('regions')
    #pylab.imshow(h_region.T,
    #             interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower')
    #pylab.colorbar()
    #pylab.xlim([8, -8])
    #pylab.ylim([-8, 8])

    ra_peak_array = []
    dec_peak_array = []
    r_peak_array = []
    sig_peak_array = []
    distance_modulus_array = []

    pylab.figure('sig')
    pylab.clf()
    for index in range(1, n_region + 1):
        index_peak = numpy.argmax(h_g * (h_region == index))
        x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]
        #print index, numpy.max(h_g * (h_region == index))
        #pylab.figure('regions')
        #pylab.scatter(x_peak, y_peak, marker='x', c='white')

        # Compute the local characteristic density using the full magnitude range
        area_field = numpy.pi * (0.5**2 - 0.3**2)
        angsep_peak = numpy.sqrt((x_full - x_peak)**2 + (y_full - y_peak)**2)
        n_field = numpy.sum((angsep_peak > 0.3) & (angsep_peak < 0.5))
        characteristic_density_local = n_field / area_field

        # If not good azimuthal coverage, revert
        cut_annulus = (angsep_peak > 0.3) & (angsep_peak < 0.5) 
        phi = numpy.degrees(numpy.arctan2(y_full[cut_annulus] - y_peak, x_full[cut_annulus] - x_peak))
        h = numpy.histogram(phi, bins=numpy.linspace(-180., 180., 13))[0]
        if numpy.sum(h > 0) < 10 or numpy.sum(h > 0.5 * numpy.median(h)) < 10:
            angsep_peak = numpy.sqrt((x - x_peak)**2 + (y - y_peak)**2)
            characteristic_density_local = characteristic_density

        size_array = numpy.arange(0.01, 0.3, 0.01)
        #size_array = numpy.array([0.04])
        sig_array = numpy.tile(0., len(size_array))
        for ii in range(0, len(size_array)):
            n_peak = numpy.sum(angsep_peak < size_array[ii])
            n_model = characteristic_density_local * (numpy.pi * size_array[ii]**2)
            sig_array[ii] = scipy.stats.norm.isf(scipy.stats.poisson.sf(n_peak, n_model))
            if sig_array[ii] > 25:
                sig_array[ii] = 25.

        #pylab.figure('sig')
        #pylab.plot(size_array, sig_array)

        ra_peak, dec_peak = proj.imageToSphere(x_peak, y_peak)
        r_peak = size_array[numpy.argmax(sig_array)]
        if numpy.max(sig_array) == 25.:
            r_peak = 0.5

        print 'Candidate:', x_peak, y_peak, r_peak, numpy.max(sig_array), ra_peak, dec_peak
        if numpy.max(sig_array) > 5.:
            #x_circle, y_circle = circle(x_peak, y_peak, r_peak)
            #pylab.figure('smooth')
            #pylab.plot(x_circle, y_circle, c='gray')
            #pylab.text(x_peak - r_peak, y_peak + r_peak, r'%.2f $\sigma$'%(numpy.max(sig_array)), color='gray')
    
            ra_peak_array.append(ra_peak)
            dec_peak_array.append(dec_peak)
            r_peak_array.append(r_peak)
            sig_peak_array.append(numpy.max(sig_array))
            distance_modulus_array.append(distance_modulus)

        #raw_input('WAIT')

    return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array

############################################################

def diagnostic(data, data_gal, ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, age=12., metallicity=0.0001):

    # Dotter isochrones
    dirname = '/home/s1/kadrlica/.ugali/isochrones/des/dotter2016/'
    iso = ugali.isochrone.factory('Dotter', hb_spread=0, dirname=dirname)
    iso.age = age
    iso.metallicity = metallicity
    iso.distance_modulus = distance_modulus

    ## Padova isochrones
    #dirname_alt = '/home/s1/kadrlica/.ugali/isochrones/des/bressan2012/' #padova/'
    #iso_alt = ugali.isochrone.factory('Padova', hb_spread=0, dirname=dirname_alt)
    #iso_alt.age = age
    #iso_alt.metallicity = metallicity
    #iso_alt.distance_modulus = distance_modulus

    cut_iso, g_iso, gr_iso_min, gr_iso_max = cutIsochronePath(data[mag_g_dred_flag], 
                                                              data[mag_r_dred_flag], 
                                                              data[mag_err_g_flag], 
                                                              data[mag_err_r_flag], 
                                                              iso, 
                                                              radius=0.1, 
                                                              return_all=True)
    
    #cut_iso_gal = cutIsochronePath(data_gal['MAG_PSF_G'], # TODO: should these be WAVG? and if so, should these then also be dereddened flags?
    #                               data_gal['MAG_PSF_R'],
    #                               data_gal['MAGERR_PSF_G'],
    #                               data_gal['MAGERR_PSF_R'],
    #                               iso,
    #                               radius=0.1,
    #                               return_all=False)
    cut_iso_gal = cutIsochronePath(data_gal[mag_g_dred_flag],
                                   data_gal[mag_r_dred_flag],
                                   data_gal[mag_err_g_flag],
                                   data_gal[mag_err_r_flag],
                                   iso,
                                   radius=0.1,
                                   return_all=False)
    
    proj = ugali.utils.projector.Projector(ra_peak, dec_peak)
    x, y = proj.sphereToImage(data['RA'][cut_iso], data['DEC'][cut_iso])
    x_gal, y_gal = proj.sphereToImage(data_gal['RA'][cut_iso_gal], data_gal['DEC'][cut_iso_gal])

###########################################################

    angsep = ugali.utils.projector.angsep(ra_peak, dec_peak, data['RA'], data['DEC'])
    cut_inner = (angsep < r_peak)
    cut_annulus = (angsep > 0.5) & (angsep < 1.)

    angsep_gal = ugali.utils.projector.angsep(ra_peak, dec_peak, data_gal['RA'], data_gal['DEC'])
    cut_inner_gal = (angsep_gal < r_peak)
    cut_annulus_gal = (angsep_gal > 0.5) & (angsep_gal < 1.)

#    ##########
#
#    # Check for possible associations
#    glon_peak, glat_peak = ugali.utils.projector.celToGal(ra_peak, dec_peak)
#    catalog_array = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14','ExtraDwarfs','ExtraClusters']
#    catalog = ugali.candidate.associate.SourceCatalog()
#    for catalog_name in catalog_array:
#        catalog += ugali.candidate.associate.catalogFactory(catalog_name)
#
#    idx1, idx2, sep = catalog.match(glon_peak, glat_peak, tol=0.5, nnearest=1)
#    match = catalog[idx2]
#    if len(match) > 0:
#        association_string = '; %s at %.3f deg'%(match[0]['name'], sep)
#    else:
#        association_string = '; no association within 0.5 deg'

############################################################

distance_modulus_search_array = numpy.arange(16., 24., 0.5)

ra_peak_array = []
dec_peak_array = [] 
r_peak_array = []
sig_peak_array = []
distance_modulus_array = []
for distance_modulus in distance_modulus_search_array:
    ra_peak, dec_peak, r_peak, sig_peak, distance_modulus = searchByDistance(data, distance_modulus, ra_select, dec_select)
    ra_peak_array.append(ra_peak)
    dec_peak_array.append(dec_peak)
    r_peak_array.append(r_peak)
    sig_peak_array.append(sig_peak)
    distance_modulus_array.append(distance_modulus)

ra_peak_array = numpy.concatenate(ra_peak_array)
dec_peak_array = numpy.concatenate(dec_peak_array)
r_peak_array = numpy.concatenate(r_peak_array)
sig_peak_array = numpy.concatenate(sig_peak_array)
distance_modulus_array = numpy.concatenate(distance_modulus_array)

index_sort = numpy.argsort(sig_peak_array)[::-1]
ra_peak_array = ra_peak_array[index_sort]
dec_peak_array = dec_peak_array[index_sort]
r_peak_array = r_peak_array[index_sort]
sig_peak_array = sig_peak_array[index_sort]
distance_modulus_array = distance_modulus_array[index_sort]

for ii in range(0, len(sig_peak_array)):
    if sig_peak_array[ii] < 0:
        continue
    angsep = ugali.utils.projector.angsep(ra_peak_array[ii], dec_peak_array[ii], ra_peak_array, dec_peak_array)
    sig_peak_array[(angsep < r_peak_array[ii]) & (numpy.arange(len(sig_peak_array)) > ii)] = -1.


ra_peak_array = ra_peak_array[sig_peak_array > 0.]
dec_peak_array = dec_peak_array[sig_peak_array > 0.]
r_peak_array = r_peak_array[sig_peak_array > 0.]
distance_modulus_array = distance_modulus_array[sig_peak_array > 0.]
sig_peak_array = sig_peak_array[sig_peak_array > 0.] # Update the sig_peak_array last!

for ii in range(0, len(sig_peak_array)):
    print '%.2f sigma; (RA, Dec, d) = (%.2f, %.2f); r = %.2f deg; d = %.1f, mu = %.2f mag)'%(sig_peak_array[ii], 
                 ra_peak_array[ii], 
                 dec_peak_array[ii], 
                 r_peak_array[ii],
                 ugali.utils.projector.distanceModulusToDistance(distance_modulus_array[ii]),
                 distance_modulus_array[ii])

    if (sig_peak_array[ii] > 5.5) & (r_peak_array[ii] < 0.28):
        diagnostic(data, data_gal, ra_peak_array[ii], dec_peak_array[ii], r_peak_array[ii], sig_peak_array[ii], distance_modulus_array[ii])


outfile = '%s/results_nside_%s_%i.csv'%(results_dir, nside, pix_nside_select)
writer = open(outfile, 'w')
#writer.write('sig, ra, dec, distance_modulus, r\n')
for ii in range(0, len(sig_peak_array)):
    writer.write('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f\n'%(sig_peak_array[ii], ra_peak_array[ii], dec_peak_array[ii], distance_modulus_array[ii], r_peak_array[ii]))
writer.close()
