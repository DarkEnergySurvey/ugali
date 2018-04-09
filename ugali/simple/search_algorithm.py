#!/usr/bin/env python
"""
Simple binning search algorithm
"""
__author__ = "Keith Bechtol, Sidney Mau"

# Set the backend first!
import matplotlib
matplotlib.use('Agg') # May need to disable this option for interactive plotting
import pylab

import sys
import os
import glob
import numpy as np
from matplotlib import mlab
#import astropy.io.fits as pyfits
import astropy.io.fits as pyfits
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

#pylab.ion() # For interactive plotting

###########################################################

#with open('config_dev_v1.yaml', 'r') as ymlfile:
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

print(matplotlib.get_backend())

############################################################

def cutIsochronePath(g, r, g_err, r_err, isochrone, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    if np.all(isochrone.stage == 'Main'):
        # Dotter case
        index_transition = len(isochrone.stage)
    else:
        # Other cases
        index_transition = np.nonzero(isochrone.stage > 3)[0][0] + 1    
    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]
    
    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))

    cut = np.logical_or(cut_1, cut_2)

    mag_bins = np.arange(17., 24.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = np.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = np.median(np.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

############################################################

def circle(x, y, r):
    phi = np.linspace(0, 2 * np.pi, 1000)
    return x + (r * np.cos(phi)), y + (r * np.sin(phi))

############################################################

try:
    ra_select, dec_select = float(sys.argv[1]), float(sys.argv[2])
except:
    sys.exit('ERROR! Coordinates not given in correct format.')

print('Search coordinates: (RA, Dec) = (%.2f, %.2f)'%(ra_select, dec_select))

# Now cut for a single pixel
pix_nside_select = ugali.utils.healpix.angToPix(nside, ra_select, dec_select)
ra_select, dec_select = ugali.utils.healpix.pixToAng(nside, pix_nside_select)
pix_nside_neighbors = np.concatenate([[pix_nside_select], healpy.get_all_neighbours(nside, pix_nside_select)])

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
print('Assembling data...')
data = np.concatenate(data_array) # TODO reduce this to just use needed columns so there is no excessive use of memory

# De-redden magnitudes
#try:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_dred_flag], data[mag_r_dred_flag]])
#except:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag] - data[extinction_g_flag], data[mag_r_flag] - data[extinction_r_flag]])
#except:
#    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag], data[mag_r_flag]])

print('Found %i objects...'%(len(data)))

############################################################

print('Applying cuts...')
#cut = (data[flags_g] < 4) & (data[flags_r] < 4) \
#      & (data['WAVG_MAG_PSF_DRED_G'] < 24.0) \
#      & ((data['WAVG_MAG_PSF_DRED_G'] - data['WAVG_MAG_PSF_DRED_R']) < 1.) \
#      & (np.fabs(data[spread_model_r_flag]) < 0.003 + data[spread_model_r_err_flag])
#
#cut_gal = (data[flags_g] < 4) & (data[flags_r] < 4) \
#          & (data['WAVG_MAG_PSF_DRED_G'] < 24.0) \
#          & ((data['WAVG_MAG_PSF_DRED_G'] - data['WAVG_MAG_PSF_DRED_R']) < 1.) \
#          & (np.fabs(data[spread_model_r_flag]) > 0.003 + data[spread_model_r_err_flag])

#cut = (data[star_galaxy_classification] >= 0) & (data[star_galaxy_classification] <= 2)
#cut_gal = (data[star_galaxy_classification] > 2)
cut = (data[star_galaxy_classification] >= 0) & (data[star_galaxy_classification] <= 1)
cut_gal = (data[star_galaxy_classification] > 1)

data_gal = data[cut_gal]
data = data[cut]

print('%i star-like objects in ROI...'%(len(data)))
print('%i galaxy-like objects in ROI...'%(len(data_gal)))

############################################################

if (cfg['fracdet'] is not None) and (cfg['fracdet'].lower().strip() != 'none') and (cfg['fracdet'] != ''):
    print('Reading fracdet map %s ...'%(cfg['fracdet']))
    fracdet = healpy.read_map(cfg['fracdet'])
else:
    print('No fracdet map specified ...')
    fracdet = None

############################################################

def searchByDistance(nside, data, distance_modulus, ra_select, dec_select, magnitude_threshold=24.5, plot=False, fracdet=None):
    """
    Idea: 
    Send a data extension that goes to faint magnitudes, e.g., g < 24.
    Use the whole region to identify hotspots using a slightly brighter 
    magnitude threshold, e.g., g < 23, so not susceptible to variations 
    in depth. Then compute the local field density using a small annulus 
    around each individual hotspot, e.g., radius 0.3 to 0.5 deg.

    plot = True enables diagnostic plotting for testings
    fracdet corresponds to a fracdet map (numpy array, assumed to be EQUATORIAL and RING)
    """

    SCALE = 2.75 * (healpy.nside2resol(nside, arcmin=True) / 60.) # deg, scale for 2D histogram and various plotting
    
    print('Distance = %.1f kpc (m-M = %.1f)'%(ugali.utils.projector.distanceModulusToDistance(distance_modulus), distance_modulus))

    dirname = '/home/s1/kadrlica/.ugali/isochrones/des/dotter2016/'
    #dirname = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/ugalidir/isochrones/des/dotter2016/'
    iso = ugali.isochrone.factory('Dotter', hb_spread=0, dirname=dirname)
    iso.age = 12.
    iso.metallicity = 0.0001
    iso.distance_modulus = distance_modulus

    cut = cutIsochronePath(data[mag_g_dred_flag], data[mag_r_dred_flag], data[mag_err_g_flag], data[mag_err_r_flag], iso, radius=0.1)
    data = data[cut]
    cut_magnitude_threshold = (data[mag_g_dred_flag] < magnitude_threshold)

    print('%i objects left after isochrone cut...'%(len(data)))

    ###

    proj = ugali.utils.projector.Projector(ra_select, dec_select)
    x, y = proj.sphereToImage(data['RA'][cut_magnitude_threshold], data['DEC'][cut_magnitude_threshold]) # Trimmed magnitude range for hotspot finding
    x_full, y_full = proj.sphereToImage(data['RA'], data['DEC']) # In we want to use full magnitude range for significance evaluation
    delta_x = 0.01
    area = delta_x**2
    smoothing = 2. / 60. # Was 3 arcmin
    bins = np.arange(-1. * SCALE, SCALE + 1.e-10, delta_x)
    centers = 0.5 * (bins[0: -1] + bins[1:])
    yy, xx = np.meshgrid(centers, centers)

    h = np.histogram2d(x, y, bins=[bins, bins])[0]

    if plot:
        #pylab.figure('raw')
        #pylab.clf()
        #pylab.imshow(h.T, interpolation='nearest', extent=[-1. * SCALE, SCALE, -1. * SCALE, SCALE], origin='lower', cmap='binary')
        #pylab.xlim([SCALE, -1. * SCALE])
        #pylab.ylim([-1. * SCALE, SCALE])
        #pylab.colorbar()

        ##import skymap
        ##s = skymap.Skymap(projection='laea',llcrnrlon=340,llcrnrlat=-60,urcrnrlon=360,urcrnrlat=-50,lon_0 =355,lat_0=-55,celestial=False)
        ##s.draw_hpxmap(fracdet)
        #pix = skymap.healpix.ang2disc(355,-63,1)
        #pix = skymap.healpix.ang2disc(4096,355,-63,1)
        #s = skymap.Skymap()
        #s.draw_hpxmap(m[pix],pix,4096)
        #s.zoom_to_fit()
        #s.zoom_to_fit(4096,m[pix],pix)

        reso = 0.25
        pylab.figure('gnom')
        pylab.clf()
        healpy.gnomview(fracdet, fig='gnom', rot=(ra_select, dec_select, 0.), reso=reso, xsize=(2. * SCALE * 60. / reso), 
                        cmap='Greens', title='Fracdet') #binary
        healpy.projscatter(data['RA'], data['DEC'], edgecolor='none', c='red', lonlat=True, s=2)

    h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)

    #cut_goodcoverage = (data['NEPOCHS_G'][cut_magnitude_threshold] >= 2) & (data['NEPOCHS_R'][cut_magnitude_threshold] >= 2)
    # expect NEPOCHS to be good in DES data

    delta_x_coverage = 0.1
    area_coverage = (delta_x_coverage)**2
    bins_coverage = np.arange(-5., 5. + 1.e-10, delta_x_coverage)
    h_coverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
    #h_goodcoverage = np.histogram2d(x[cut_goodcoverage], y[cut_goodcoverage], bins=[bins_coverage, bins_coverage])[0]
    h_goodcoverage = np.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]

    n_goodcoverage = h_coverage[h_goodcoverage > 0].flatten()

    #characteristic_density = np.mean(n_goodcoverage) / area_coverage # per square degree
    characteristic_density = np.median(n_goodcoverage) / area_coverage # per square degree
    print('Characteristic density = %.1f deg^-2'%(characteristic_density))

    # Use pixels with fracdet ~1.0 to estimate the characteristic density
    if fracdet is not None:
        fracdet_zero = np.tile(0., len(fracdet))
        cut = (fracdet != healpy.UNSEEN)
        fracdet_zero[cut] = fracdet[cut]

        nside_fracdet = healpy.npix2nside(len(fracdet))
        
        subpix_region_array = []
        for pix in np.unique(ugali.utils.healpix.angToPix(nside, data['RA'], data['DEC'])):
            subpix_region_array.append(ugali.utils.healpix.subpixel(pix, nside, nside_fracdet))
        subpix_region_array = np.concatenate(subpix_region_array)

        # Compute mean fracdet in the region so that this is available as a correction factor
        cut = (fracdet[subpix_region_array] != healpy.UNSEEN)
        mean_fracdet = np.mean(fracdet[subpix_region_array[cut]])

        subpix_region_array = subpix_region_array[fracdet[subpix_region_array] > 0.99]
        subpix = ugali.utils.healpix.angToPix(nside_fracdet, 
                                              data['RA'][cut_magnitude_threshold], 
                                              data['DEC'][cut_magnitude_threshold]) # Remember to apply mag threshold to objects
        characteristic_density_fracdet = float(np.sum(np.in1d(subpix, subpix_region_array))) \
                                         / (healpy.nside2pixarea(nside_fracdet, degrees=True) * len(subpix_region_array)) # deg^-2
        print('Characteristic density fracdet = %.1f deg^-2'%(characteristic_density_fracdet))
        
        # Correct the characteristic density by the mean fracdet value
        characteristic_density_raw = 1. * characteristic_density
        characteristic_density /= mean_fracdet 
        print('Characteristic density (fracdet corrected) = %.1f deg^-2'%(characteristic_density))

    if plot:
        pylab.figure('poisson')
        pylab.clf()
        n_max = np.max(h_coverage)
        pylab.hist(n_goodcoverage, bins=np.arange(n_max) - 0.5, color='blue', histtype='step', lw=2, normed=True)
        pylab.scatter(np.arange(n_max), scipy.stats.poisson.pmf(np.arange(n_max), mu=np.median(n_goodcoverage)), c='red', edgecolor='none', zorder=10)
        #pylab.plot(np.arange(n_max), scipy.stats.poisson.pmf(np.arange(n_max), mu=np.median(n_goodcoverage)), c='red', lw=2, zorder=10)
        pylab.axvline(characteristic_density * area_coverage, c='black', ls='--')
        if fracdet is not None:
            pylab.axvline(characteristic_density_raw * area_coverage, c='orange', ls='--')
            pylab.axvline(characteristic_density_fracdet * area_coverage, c='green', ls='--')
        pylab.xlabel('Counts per %.3f deg^-2 pixel'%(area_coverage))
        pylab.ylabel('PDF')

    if plot:
        vmax = min(3. * characteristic_density * area, np.max(h_g))
    
        pylab.figure('smooth')
        pylab.clf()
        pylab.imshow(h_g.T,
                     interpolation='nearest', extent=[-1. * SCALE, SCALE, -1. * SCALE, SCALE], origin='lower', cmap='gist_heat', vmax=vmax)
        pylab.colorbar()
        pylab.xlim([SCALE, -1. * SCALE])
        pylab.ylim([-1. * SCALE, SCALE])
        pylab.xlabel(r'$\Delta$ RA (deg)')
        pylab.ylabel(r'$\Delta$ Dec (deg)')
        pylab.title('(RA, Dec, mu) = (%.2f, %.2f, %.2f)'%(ra_select, dec_select, distance_modulus))

    factor_array = np.arange(1., 5., 0.05)
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
    h_region = np.ma.array(h_region, mask=(h_region < 1))
    if plot:
        pylab.figure('regions')
        pylab.clf()
        pylab.imshow(h_region.T,
                     interpolation='nearest', extent=[-1. * SCALE, SCALE, -1. * SCALE, SCALE], origin='lower')
        pylab.colorbar()
        pylab.xlim([SCALE, -1. * SCALE])
        pylab.ylim([-1. * SCALE, SCALE])

    ra_peak_array = []
    dec_peak_array = []
    r_peak_array = []
    sig_peak_array = []
    distance_modulus_array = []

    pylab.figure('sig')
    pylab.clf()
    for index in range(1, n_region + 1):
        index_peak = np.argmax(h_g * (h_region == index))
        x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]
        #print index, np.max(h_g * (h_region == index))
        if plot:
            pylab.figure('regions')
            #pylab.scatter(x_peak, y_peak, marker='x', c='white')
            pylab.scatter(x_peak, y_peak, marker='o', c='none', edgecolor='black', s=50)

        
        #angsep_peak = np.sqrt((x_full - x_peak)**2 + (y_full - y_peak)**2) # Use full magnitude range, NOT TESTED!!!
        angsep_peak = np.sqrt((x - x_peak)**2 + (y - y_peak)**2) # Impose magnitude threshold

        # Compute the local characteristic density
        # If fracdet map is available, use that information to either compute local density,
        # or in regions of spotty coverage, use the typical density of the region
        if fracdet is not None:
            ra_peak, dec_peak = proj.imageToSphere(x_peak, y_peak)
            subpix_all = ugali.utils.healpix.angToDisc(nside_fracdet, ra_peak, dec_peak, 0.5)
            subpix_inner = ugali.utils.healpix.angToDisc(nside_fracdet, ra_peak, dec_peak, 0.3)
            subpix_annulus = subpix_all[~np.in1d(subpix_all, subpix_inner)]
            mean_fracdet = np.mean(fracdet_zero[subpix_annulus])
            print('mean_fracdet', mean_fracdet)
            if mean_fracdet < 0.5:
                characteristic_density_local = characteristic_density
                print('characteristic_density_local baseline', characteristic_density_local)
            else:
                # Check pixels in annulus with complete coverage
                subpix_annulus_region = np.intersect1d(subpix_region_array, subpix_annulus)
                print(float(len(subpix_annulus_region)) / len(subpix_annulus))
                if (float(len(subpix_annulus_region)) / len(subpix_annulus)) < 0.25:
                    characteristic_density_local = characteristic_density
                    print('characteristic_density_local spotty', characteristic_density_local)
                else:
                    characteristic_density_local = float(np.sum(np.in1d(subpix, subpix_annulus_region))) \
                                                   / (healpy.nside2pixarea(nside_fracdet, degrees=True) * len(subpix_annulus_region)) # deg^-2
                    print('characteristic_density_local cleaned up', characteristic_density_local)
        else:
            # Compute the local characteristic density
            area_field = np.pi * (0.5**2 - 0.3**2)
            n_field = np.sum((angsep_peak > 0.3) & (angsep_peak < 0.5))
            characteristic_density_local = n_field / area_field

            # If not good azimuthal coverage, revert
            cut_annulus = (angsep_peak > 0.3) & (angsep_peak < 0.5) 
            #phi = np.degrees(np.arctan2(y_full[cut_annulus] - y_peak, x_full[cut_annulus] - x_peak)) # Use full magnitude range, NOT TESTED!!!
            phi = np.degrees(np.arctan2(y[cut_annulus] - y_peak, x[cut_annulus] - x_peak)) # Impose magnitude threshold
            h = np.histogram(phi, bins=np.linspace(-180., 180., 13))[0]
            if np.sum(h > 0) < 10 or np.sum(h > 0.5 * np.median(h)) < 10:
                #angsep_peak = np.sqrt((x - x_peak)**2 + (y - y_peak)**2)
                characteristic_density_local = characteristic_density

        print('Characteristic density local = %.1f deg^-2'%(characteristic_density_local))

        size_array = np.arange(0.01, 0.3, 0.01)
        sig_array = np.tile(0., len(size_array))
        for ii in range(0, len(size_array)):
            n_peak = np.sum(angsep_peak < size_array[ii])
            n_model = characteristic_density_local * (np.pi * size_array[ii]**2)
            sig_array[ii] = scipy.stats.norm.isf(scipy.stats.poisson.sf(n_peak, n_model))
            if sig_array[ii] > 25:
                sig_array[ii] = 25. # Set a maximum significance value

        if plot:
            pylab.figure('sig')
            pylab.plot(size_array, sig_array)
            pylab.xlabel('Radius used to compute significance (deg)')
            pylab.ylabel('Detection significance')

        ra_peak, dec_peak = proj.imageToSphere(x_peak, y_peak)
        r_peak = size_array[np.argmax(sig_array)]
        if np.max(sig_array) == 25.:
            r_peak = 0.5

        #print 'Candidate:', x_peak, y_peak, r_peak, np.max(sig_array), ra_peak, dec_peak
        print('Candidate: %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f'%(x_peak, y_peak, r_peak, np.max(sig_array), ra_peak, dec_peak))
        if np.max(sig_array) > 5.:
            if plot:
                x_circle, y_circle = circle(x_peak, y_peak, r_peak)
                pylab.figure('smooth')
                pylab.plot(x_circle, y_circle, c='gray')
                pylab.text(x_peak - r_peak, y_peak + r_peak, r'%.2f $\sigma$'%(np.max(sig_array)), color='gray')
    
            ra_peak_array.append(ra_peak)
            dec_peak_array.append(dec_peak)
            r_peak_array.append(r_peak)
            sig_peak_array.append(np.max(sig_array))
            distance_modulus_array.append(distance_modulus)

    if plot:
        input('Plots are ready...')

    return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array

############################################################

def diagnostic(data, data_gal, ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, age=12., metallicity=0.0001):

    # Dotter isochrones
    dirname = '/home/s1/kadrlica/.ugali/isochrones/des/dotter2016/'
    #dirname = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/ugalidir/isochrones/des/dotter2016/'
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

distance_modulus_search_array = np.arange(16., 24., 0.5)

ra_peak_array = []
dec_peak_array = [] 
r_peak_array = []
sig_peak_array = []
distance_modulus_array = []
for distance_modulus in distance_modulus_search_array:
    ra_peak, dec_peak, r_peak, sig_peak, distance_modulus = searchByDistance(nside, data, distance_modulus, ra_select, dec_select, fracdet=fracdet)
    ra_peak_array.append(ra_peak)
    dec_peak_array.append(dec_peak)
    r_peak_array.append(r_peak)
    sig_peak_array.append(sig_peak)
    distance_modulus_array.append(distance_modulus)

ra_peak_array = np.concatenate(ra_peak_array)
dec_peak_array = np.concatenate(dec_peak_array)
r_peak_array = np.concatenate(r_peak_array)
sig_peak_array = np.concatenate(sig_peak_array)
distance_modulus_array = np.concatenate(distance_modulus_array)

# Sort peaks according to significance
index_sort = np.argsort(sig_peak_array)[::-1]
ra_peak_array = ra_peak_array[index_sort]
dec_peak_array = dec_peak_array[index_sort]
r_peak_array = r_peak_array[index_sort]
sig_peak_array = sig_peak_array[index_sort]
distance_modulus_array = distance_modulus_array[index_sort]

# Collect overlapping peaks
for ii in range(0, len(sig_peak_array)):
    if sig_peak_array[ii] < 0:
        continue
    angsep = ugali.utils.projector.angsep(ra_peak_array[ii], dec_peak_array[ii], ra_peak_array, dec_peak_array)
    sig_peak_array[(angsep < r_peak_array[ii]) & (np.arange(len(sig_peak_array)) > ii)] = -1.

# Prune the list of peaks
ra_peak_array = ra_peak_array[sig_peak_array > 0.]
dec_peak_array = dec_peak_array[sig_peak_array > 0.]
r_peak_array = r_peak_array[sig_peak_array > 0.]
distance_modulus_array = distance_modulus_array[sig_peak_array > 0.]
sig_peak_array = sig_peak_array[sig_peak_array > 0.] # Update the sig_peak_array last!

for ii in range(0, len(sig_peak_array)):
    print('%.2f sigma; (RA, Dec, d) = (%.2f, %.2f); r = %.2f deg; d = %.1f, mu = %.2f mag)'%(sig_peak_array[ii], 
                 ra_peak_array[ii], 
                 dec_peak_array[ii], 
                 r_peak_array[ii],
                 ugali.utils.projector.distanceModulusToDistance(distance_modulus_array[ii]),
                 distance_modulus_array[ii]))

    if (sig_peak_array[ii] > 5.5) & (r_peak_array[ii] < 0.28):
        diagnostic(data, data_gal, ra_peak_array[ii], dec_peak_array[ii], r_peak_array[ii], sig_peak_array[ii], distance_modulus_array[ii])


outfile = '%s/results_nside_%s_%i.csv'%(results_dir, nside, pix_nside_select)
writer = open(outfile, 'w')
#writer.write('sig, ra, dec, distance_modulus, r\n')
for ii in range(0, len(sig_peak_array)):
    writer.write('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f\n'%(sig_peak_array[ii], 
                                                             ra_peak_array[ii], 
                                                             dec_peak_array[ii], 
                                                             distance_modulus_array[ii], 
                                                             r_peak_array[ii]))
writer.close()
