"""
Currently this is more set up as a standalone script.
"""

import os
import collections
import numpy as np
import scipy.interpolate
import healpy
import astropy.io.fits as pyfits
import pylab

import ugali.utils.projector
import ugali.utils.healpix
import ugali.analysis.source
import ugali.analysis.imf
import ugali.analysis.results
import ugali.simulation.population
from ugali.isochrone import factory as isochrone_factory

pylab.ion()

############################################################

def getCompleteness():
    # This is an easy place to make an error
    infile = 'y3a2_stellar_classification_summary_ext2.csv'
    d = np.recfromcsv(infile)

    x = d['mag_r']
    y = d['eff_star']

    x = np.insert(x, 0, 16.)
    y = np.insert(y, 0, y[0])

    f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=0.)

    return f

############################################################

def getPhotoError():
    infile = 'photo_error_model.csv'
    d = np.recfromcsv(infile)

    x = d['mag']
    y = d['log_mag_err']

    x = np.insert(x, 0, -10.)
    y = np.insert(y, 0, y[0])

    f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=1.)

    return f

############################################################

def magToFlux(mag):
    """
    Convert from an AB magnitude to a flux (Jy)
    """
    return 3631. * 10**(-0.4 * mag)

############################################################

def fluxToMag(flux):
    """
    Convert from flux (Jy) to AB magnitude
    """
    return -2.5 * np.log10(flux / 3631.)

############################################################

def getFluxError(mag, mag_error):
    return magToFlux(mag) * mag_error / 1.0857362

############################################################

def meanFracdet(map_fracdet, lon_population, lat_population, radius_population):
    """
    Compute the mean fracdet within circular aperture (radius specified in decimal degrees)

    lon, lat, and radius are taken to be arrays of the same length
    """
    nside_fracdet = healpy.npix2nside(map_fracdet)
    map_fracdet_zero = np.where(map_fracdet >= 0., map_fracdet, 0.)
    fracdet_population = np.empty(len(lon))
    for ii in range(0, len(lon)):
        fracdet_population[ii] = np.mean(map_fracdet_zero[ugali.utils.healpix.ang2disc(nside_fracdet, 
                                                                                       lon_population[ii], 
                                                                                       lat_population[ii], 
                                                                                       radius_population,
                                                                                       inclusive=True)])
    return fracdet_population

############################################################

def simSatellite(lon_centroid, lat_centroid, distance, stellar_mass, r_physical, 
                 m_maglim_1, m_maglim_2,
                 plot=False, title='test'):
    """
    Simulate a single satellite. This is currently only valid for band_1 = g and band_2 = r.
    r_physical is azimuthally averaged half-light radius, kpc
    """

    # Probably don't want to parse every time
    completeness = getCompleteness()
    log_photo_error = getPhotoError()

    # Not exactly sure where these numbers came from, this should be edited
    #maglim_g = 24.282 
    #maglim_r = 23.952

    s = ugali.analysis.source.Source()

    # Following McConnachie 2012, ellipticity = 1 - (b/a) , where a is semi-major axis and b is semi-minor axis
    
    r_h = np.degrees(np.arcsin(r_physical / distance)) # Azimuthally averaged half-light radius
    ellipticity = 0.3 # Semi-arbitrary default for testing purposes
    #ellipticity = np.clip(scipy.stats.norm.rvs(loc=0.3, scale=0.15, size=1), 0.1, 0.8) # Based loosely on https://arxiv.org/abs/0805.2945
    position_angle = np.random.uniform(0., 180.) # Random position angle (deg)
    a_h = r_h / np.sqrt(1. - ellipticity) # semi-major axis (deg)
    
    flag_too_big = False
    if a_h >= 0.5:
        print 'Too big!'
        a_h = 0.5
        flag_too_big = True
        
    # Elliptical kernels take the "extension" as the semi-major axis
    ker = ugali.kernel.EllipticalPlummer(lon=lon_centroid, lat=lat_centroid, extension=a_h, ellipticity=ellipticity, position_angle=position_angle)
    s.set_kernel(ker)
    
    distance_modulus = ugali.utils.projector.distanceToDistanceModulus(distance)
    iso = isochrone_factory('Bressan2012', survey='des', age=12, z=0.0001, distance_modulus=distance_modulus)
    s.set_isochrone(iso)

    mag_1, mag_2 = s.isochrone.simulate(stellar_mass) # Simulate takes stellar mass as an argument, NOT richness

    lon, lat = s.kernel.sample_lonlat(len(mag_2))

    nside = healpy.npix2nside(len(m_maglim_1)) # Assuming that the two maglim maps have same resolution
    pix = ugali.utils.healpix.angToPix(nside, lon, lat)
    maglim_1 = m_maglim_1[pix]
    maglim_2 = m_maglim_2[pix]

    mag_1_error = 0.01 + 10**(log_photo_error(mag_1 - maglim_1))
    mag_2_error = 0.01 + 10**(log_photo_error(mag_2 - maglim_2))

    # It would be better to convert to a flux uncertainty and then transform back to a magnitude
    #mag_1_meas = mag_1 + np.random.normal(scale=mag_1_error)
    #mag_2_meas = mag_2 + np.random.normal(scale=mag_2_error)
    flux_1_meas = magToFlux(mag_1) + np.random.normal(scale=getFluxError(mag_1, mag_1_error))
    mag_1_meas = np.where(flux_1_meas > 0., fluxToMag(flux_1_meas), 99.)
    flux_2_meas = magToFlux(mag_2) + np.random.normal(scale=getFluxError(mag_2, mag_2_error))
    mag_2_meas = np.where(flux_2_meas > 0., fluxToMag(flux_2_meas), 99.)

    cut_detect = (np.random.uniform(size=len(mag_2)) < completeness(mag_2))

    n_g24 = np.sum(cut_detect & (mag_1 < 24.))
    print '  n_sim = %i'%(len(mag_1))
    print '  n_detect = %i'%(np.sum(cut_detect))
    print '  n_g24 = %i'%(n_g24)
    
    richness = stellar_mass / s.isochrone.stellarMass()
    #abs_mag = s.isochrone.absolute_magnitude()
    #abs_mag_martin = s.isochrone.absolute_magnitude_martin(richness=richness, n_trials=10)[0] # 100 trials seems to be sufficient for rough estimate
    #print 'abs_mag_martin = %.2f mag'%(abs_mag_martin)

    # The more clever thing to do would be to sum up the actual simulated stars
    v = mag_1 - 0.487*(mag_1 - mag_2) - 0.0249 # See https://github.com/DarkEnergySurvey/ugali/blob/master/ugali/isochrone/model.py
    flux = np.sum(10**(-v/2.5))
    abs_mag = -2.5*np.log10(flux) - distance_modulus

    #print abs_mag, abs_mag_martin

    #distance = ugali.utils.projector.distanceModulusToDistance(distance_modulus)
    #r_h = extension * np.sqrt(1. - ellipticity) # Azimuthally averaged half-light radius
    r_physical = distance * np.tan(np.radians(r_h)) # Azimuthally averaged half-light radius, kpc
    #print 'distance = %.3f kpc'%(distance)
    #print 'r_physical = %.3f kpc'%(r_physical)
    surface_brightness = ugali.analysis.results.surfaceBrightness(abs_mag, r_physical, distance) # Average within azimuthally averaged half-light radius
    #print 'surface_brightness = %.3f mag arcsec^-2'%(surface_brightness)
    
    if plot:
        n_sigma_p = np.sum(cut_detect & (mag_1 < 23.))

        pylab.figure(figsize=(6., 6.))
        pylab.scatter(mag_1_meas[cut_detect] - mag_2_meas[cut_detect], mag_1_meas[cut_detect], edgecolor='none', c='black', s=5)
        pylab.xlim(-0.5, 1.)
        pylab.ylim(26., 16.)
        pylab.xlabel('g - r')
        pylab.ylabel('g')
        pylab.title('Number of stars with g < 23: %i'%(n_sigma_p))
        pylab.savefig('y3_sat_sim_cmd_%s.png'%(title), dpi=150.)
        
        print 'n_Sigma_p = %i'%(n_sigma_p)
        raw_input('WAIT')
        
    if flag_too_big:
        # This is a kludge to remove these satellites. fragile!!
        n_g24 = 1.e6

    return lon[cut_detect], lat[cut_detect], mag_1_meas[cut_detect], mag_2_meas[cut_detect], mag_1_error[cut_detect], mag_2_error[cut_detect], n_g24, abs_mag, surface_brightness, ellipticity, position_angle

############################################################

mc_source_id_start = 1
assert mc_source_id_start >= 1, "Starting mc_source_id must be greater than or equal to 1" 
n = 5000 # Number of satellites to simulation; 100 for testing, 1000 for partial testing, 5000 for bulk production
n_chunk = 100
assert n % n_chunk == 0, "Total number of satellites must be divisible by the chunk size"
nside_pix = 256 # NSIDE = 128 -> 27.5 arcmin, NSIDE = 256 -> 13.7 arcmin 
#tag = '_v2_n_%i'%(n)
tag = '_v3'

infile_fracdet = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz'
#infile_badregions = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_badregions_mask_v1.0.fits.gz'
infile_foreground = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_foreground_mask_v2.0.fits.gz'

infile_maglim_g = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_gold_1.0_cmv02-001_v1_nside4096_nest_g_depth.fits.gz'
infile_maglim_r = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_gold_1.0_cmv02-001_v1_nside4096_nest_r_depth.fits.gz'

infile_density = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a2/skymap/des_y3a2_stellar_density_map_g_23_cel_nside_128.npy'
m_density = np.load(infile_density)
nside_density = healpy.npix2nside(len(m_density))

m_fracdet = healpy.read_map(infile_fracdet, nest=False)
nside_fracdet = healpy.npix2nside(len(m_fracdet))

m_maglim_g = healpy.read_map(infile_maglim_g, nest=False)
m_maglim_r = healpy.read_map(infile_maglim_r, nest=False)

##### ONLY FOR TESTING
"""
print 'DOING SOMETHING ODD'
m_fracdet = np.tile(1., healpy.nside2npix(4096))
nside_fracdet = 4096
m_maglim_g = np.tile(24., healpy.nside2npix(4096))
m_maglim_r = np.tile(24., healpy.nside2npix(4096))
print 'DONE DOING SOMETHING ODD'
"""
#####

#m_foreground = healpy.read_map(infile_foreground)

mask = (m_fracdet > 0.5) 

###########
"""
# Tuc III
distance = 25.
stellar_mass = 0.8e3
r_physical = 0.044 # kpc
simSatellite(0., 0., distance, stellar_mass, r_physical, m_maglim_g, m_maglim_r, plot=True, title='tuc_iii')

# Gru II
distance = 53.
stellar_mass = 3.4e3
r_physical = 0.093 # kpc
simSatellite(0., 0., distance, stellar_mass, r_physical, m_maglim_g, m_maglim_r, plot=True, title='gru_ii')
"""
##########

# r_physical is azimuthally-averaged half-light radius, kpc
simulation_area, lon_population, lat_population, distance_population, stellar_mass_population, r_physical_population = ugali.simulation.population.satellitePopulation(mask, nside_pix, n)
n_g24_population = np.empty(n)
abs_mag_population = np.empty(n)
surface_brightness_population = np.empty(n)
ellipticity_population = np.empty(n)
position_angle_population = np.empty(n)
#mc_source_id_population = np.arange(n) + 1 # Indexed starting from 1
mc_source_id_population = np.arange(mc_source_id_start, mc_source_id_start + n)

lon_array = []
lat_array = []
mag_1_array = []
mag_2_array = []
mag_1_error_array = []
mag_2_error_array = []
mc_source_id_array = []
#for ii in range(0, n):
for ii, mc_source_id in enumerate(mc_source_id_population):

    print '  Simulating satellite (%i/%i) ... MC_SOURCE_ID = %i'%(ii + 1, n, mc_source_id)

    lon, lat, mag_1, mag_2, mag_1_error, mag_2_error, n_g24, abs_mag, surface_brightness, ellipticity, position_angle = simSatellite(lon_population[ii], 
                                                                                                                                     lat_population[ii], 
                                                                                                                                     distance_population[ii], 
                                                                                                                                     stellar_mass_population[ii], 
                                                                                                                                     r_physical_population[ii],
                                                                                                                                     m_maglim_g,
                                                                                                                                     m_maglim_r)

    print '  ', len(lon)

    n_g24_population[ii] = n_g24
    abs_mag_population[ii] = abs_mag
    surface_brightness_population[ii] = surface_brightness
    ellipticity_population[ii] = ellipticity
    position_angle_population[ii] = position_angle

    lon_array.append(lon)
    lat_array.append(lat)
    mag_1_array.append(mag_1)
    mag_2_array.append(mag_2)
    mag_1_error_array.append(mag_1_error)
    mag_2_error_array.append(mag_2_error)
    #mc_source_id_array.append(np.tile(ii + 1, len(lon))) # Indexed starting from 1
    mc_source_id_array.append(np.tile(mc_source_id, len(lon)))

# Concatenate all the arrays

lon_array = np.concatenate(lon_array)
lat_array = np.concatenate(lat_array)
mag_1_array = np.concatenate(mag_1_array)
mag_2_array = np.concatenate(mag_2_array)
mag_1_error_array = np.concatenate(mag_1_error_array)
mag_2_error_array = np.concatenate(mag_2_error_array)
mc_source_id_array = np.concatenate(mc_source_id_array)

# Now do the masking all at once

pix_array = ugali.utils.healpix.angToPix(nside_fracdet, lon_array, lat_array)

cut_fracdet = (np.random.uniform(size=len(lon_array)) < m_fracdet[pix_array])

lon_array = lon_array[cut_fracdet]
lat_array = lat_array[cut_fracdet]
mag_1_array = mag_1_array[cut_fracdet]
mag_2_array = mag_2_array[cut_fracdet]
mag_1_error_array = mag_1_error_array[cut_fracdet]
mag_2_error_array = mag_2_error_array[cut_fracdet]
mc_source_id_array = mc_source_id_array[cut_fracdet]

# Cut out the entries that are easily detectable

cut_easy = ((surface_brightness_population < 25.) & (n_g24_population > 100.)) \
           | ((surface_brightness_population < 30.) & (n_g24_population > 1.e4)) \
           | (n_g24_population > 1.e5)
cut_hard = (surface_brightness_population > 35.) | (n_g24_population < 1.)
cut_difficulty = ~cut_easy & ~cut_hard
lon_population = lon_population[cut_difficulty]
lat_population = lat_population[cut_difficulty]
distance_population = distance_population[cut_difficulty]
stellar_mass_population = stellar_mass_population[cut_difficulty]
r_physical_population = r_physical_population[cut_difficulty]
n_g24_population = n_g24_population[cut_difficulty]
abs_mag_population = abs_mag_population[cut_difficulty]
surface_brightness_population = surface_brightness_population[cut_difficulty]
ellipticity_population = ellipticity_population[cut_difficulty]
position_angle_population = position_angle_population[cut_difficulty]
mc_source_id_population = mc_source_id_population[cut_difficulty]

cut_difficulty = np.in1d(mc_source_id_array, mc_source_id_population)
lon_array = lon_array[cut_difficulty]
lat_array = lat_array[cut_difficulty]
mag_1_array = mag_1_array[cut_difficulty]
mag_2_array = mag_2_array[cut_difficulty]
mag_1_error_array = mag_1_error_array[cut_difficulty]
mag_2_error_array = mag_2_error_array[cut_difficulty]
mc_source_id_array = mc_source_id_array[cut_difficulty]

# Create bonus columns

distance_modulus_population = ugali.utils.projector.distanceToDistanceModulus(distance_population)
hpix_32_population = ugali.utils.healpix.angToPix(32, lon_population, lat_population) # Make sure this matches the dataset

# Local stellar density
pixarea = healpy.nside2pixarea(nside_density, degrees=True) * 60.**2 # arcmin^2
density_population = m_density[ugali.utils.healpix.angToPix(nside_density, lon_population, lat_population)] / pixarea # arcmin^-2

# Average fracdet within the azimuthally averaged half-light radius
m_fracdet_zero = np.where(m_fracdet >= 0., m_fracdet, 0.)
fracdet_population = np.empty(len(mc_source_id_population))
for ii in range(0, len(mc_source_id_population)):
    r_half = np.degrees(np.arctan2(r_physical_population[ii], distance_population[ii])) # Azimuthally averaged half-light radius in degrees
    fracdet_population[ii] = np.mean(m_fracdet_zero[ugali.utils.healpix.ang2disc(nside_fracdet, 
                                                                                 lon_population[ii], 
                                                                                 lat_population[ii], 
                                                                                 r_half,
                                                                                 inclusive=True)])

# Catalog output file

# for ii in range(0, len(d.formats)): print '\'%s\': [ , \'%s\'],'%(d.names[ii], d.formats[ii])

# See: 
# https://github.com/sidneymau/simple/blob/master/search_algorithm.py 
# https://github.com/sidneymau/simple/blob/master/config.yaml
# /home/s1/kadrlica/projects/y3a2/dsphs/v2/skim/ , e.g., /home/s1/kadrlica/projects/y3a2/dsphs/v2/skim/y3a2_ngmix_cm_11755.fits

default_array = np.tile(-9999., len(mc_source_id_array)) # To recognize that those values are synthetic filler

# Column name, data, fits format
key_map = {'CM_MAG_ERR_G': [mag_1_error_array, 'D'],
           'CM_MAG_ERR_R': [mag_2_error_array, 'D'],
           'CM_MAG_G': [mag_1_array, 'D'],
           'CM_MAG_R': [mag_2_array, 'D'],
           'CM_T': [default_array, 'D'],
           'CM_T_ERR': [default_array, 'D'],
           'COADD_OBJECT_ID': [default_array, 'K'],
           'DEC': [lat_array, 'D'],
           'FLAGS': [default_array, 'K'],
           'PSF_MAG_ERR_G': [mag_1_error_array, 'D'],
           'PSF_MAG_ERR_R': [mag_2_error_array, 'D'],
           'PSF_MAG_G': [mag_1_array, 'D'],
           'PSF_MAG_R': [mag_2_array, 'D'],
           'RA': [lon_array, 'D'],
           'SEXTRACTOR_FLAGS_G': [np.tile(0, len(mc_source_id_array)), 'I'],
           'SEXTRACTOR_FLAGS_R': [np.tile(0, len(mc_source_id_array)), 'I'],
           'WAVG_MAG_PSF_G': [mag_1_array, 'E'],
           'WAVG_MAG_PSF_R': [mag_2_array, 'E'],
           'WAVG_MAGERR_PSF_G': [mag_1_error_array, 'E'],
           'WAVG_MAGERR_PSF_R': [mag_2_error_array, 'E'],
           'WAVG_SPREAD_MODEL_I': [default_array, 'E'],
           'WAVG_SPREADERR_MODEL_I': [default_array, 'E'],
           'EXT_SFD98_G': [default_array, 'E'],
           'EXT_SFD98_R': [default_array, 'E'],
           'CM_MAG_SFD_G': [mag_1_array, 'D'],
           'CM_MAG_SFD_R': [mag_2_array, 'D'],
           'FLAG_FOOTPRINT': [np.tile(1, len(mc_source_id_array)), 'J'],
           'FLAG_FOREGROUND': [np.tile(0, len(mc_source_id_array)), 'J'],
           'EXTENDED_CLASS_MASH': [np.tile(0, len(mc_source_id_array)), 'K'],
           'PSF_MAG_SFD_G': [mag_1_array, 'D'],
           'PSF_MAG_SFD_R': [mag_2_array, 'D'],
           'WAVG_MAG_PSF_SFD_G': [mag_1_array, 'E'],
           'WAVG_MAG_PSF_SFD_R': [mag_2_array, 'E']}
key_map['MC_SOURCE_ID'] = [mc_source_id_array, 'K']

columns = []
for key in key_map:
    columns.append(pyfits.Column(name=key, format=key_map[key][1], array=key_map[key][0]))
tbhdu = pyfits.BinTableHDU.from_columns(columns)
tbhdu.header.set('AREA', simulation_area, 'Simulation area (deg^2)')
#tbhdu.writeto('sim_catalog%s.fits'%(tag), clobber=True)
for mc_source_id_chunk in np.split(np.arange(mc_source_id_start, mc_source_id_start + n), n / n_chunk):
    print '  writing MC_SOURCE_ID values from %i to %i'%(mc_source_id_chunk[0], mc_source_id_chunk[-1])
    cut_chunk = np.in1d(mc_source_id_array, mc_source_id_chunk)
    outfile = 'sim_catalog%s_mc_source_id_%06i-%06i.fits'%(tag, mc_source_id_chunk[0], mc_source_id_chunk[-1])
    #tbhdu[cut_chunk].tbhdu.writeto('sim_catalog%s_mc_source_id_%06i-%06i.fits'%(tag, mc_source_id_chunk[0], mc_source_id_chunk[-1]), clobber=True)
    header = tbhdu.header
    pyfits.writeto(outfile, tbhdu.data[cut_chunk], header, clobber=True)

"""
tbhdu = pyfits.BinTableHDU.from_columns([
    pyfits.Column(name='ra', format='E', array=lon_array, unit='deg'),
    pyfits.Column(name='dec', format='E', array=lat_array, unit='deg'),
    pyfits.Column(name='mag_g', format='E', array=mag_1_array, unit='mag'),
    pyfits.Column(name='mag_r', format='E', array=mag_2_array, unit='mag'),
    pyfits.Column(name='magerr_g', format='E', array=mag_1_error_array, unit='mag'),
    pyfits.Column(name='magerr_r', format='E', array=mag_2_error_array, unit='mag'),
    pyfits.Column(name='mc_source_id', format='K', array=mc_source_id_array, unit=''),
])
tbhdu.header.set('AREA', simulation_area, 'Simulation area (deg^2)')
tbhdu.writeto('sim_catalog.fits', clobber=True)
"""

# Population output file

tbhdu = pyfits.BinTableHDU.from_columns([
    pyfits.Column(name='RA', format='E', array=lon_population, unit='deg'),
    pyfits.Column(name='DEC', format='E', array=lat_population, unit='deg'),
    pyfits.Column(name='DISTANCE', format='E', array=distance_population, unit='kpc'),
    pyfits.Column(name='DISTANCE_MODULUS', format='E', array=distance_modulus_population, unit='kpc'),
    pyfits.Column(name='STELLAR_MASS', format='E', array=stellar_mass_population, unit='m_solar'),
    pyfits.Column(name='R_PHYSICAL', format='E', array=r_physical_population, unit='kpc'),
    pyfits.Column(name='N_G24', format='J', array=n_g24_population, unit=''),
    pyfits.Column(name='ABS_MAG', format='E', array=abs_mag_population, unit='mag'),
    pyfits.Column(name='SURFACE_BRIGHTNESS', format='E', array=surface_brightness_population, unit='mag arcsec^-2'),
    pyfits.Column(name='ELLIPTICITY', format='E', array=ellipticity_population, unit=''),
    pyfits.Column(name='POSITION_ANGLE', format='E', array=position_angle_population, unit='deg'),
    pyfits.Column(name='MC_SOURCE_ID', format='K', array=mc_source_id_population, unit=''),
    pyfits.Column(name='HPIX_32', format='E', array=hpix_32_population, unit=''),
    pyfits.Column(name='DENSITY', format='E', array=density_population, unit='arcmin^-2'),
    pyfits.Column(name='FRACDET', format='E', array=fracdet_population, unit='')
])
tbhdu.header.set('AREA', simulation_area, 'Simulation area (deg^2)')
tbhdu.writeto('sim_population%s.fits'%(tag), clobber=True)

# 5284.2452461023322

# Mask output file

outfile_mask = 'sim_mask_cel_nside_%i%s.fits'%(healpy.npix2nside(len(mask)), tag)
healpy.write_map(outfile_mask, mask.astype(int), nest=True, coord='C', overwrite=True)
os.system('gzip -f %s'%(outfile_mask))
