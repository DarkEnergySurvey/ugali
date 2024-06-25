#!/usr/bin/env python
"""
Currently this is more set up as a standalone script.
"""
import os
import copy
import collections
from collections import OrderedDict as odict
import yaml
import numpy as np
import scipy.interpolate
import healpy as hp
import fitsio
import astropy.io.fits as pyfits

import ugali.utils.projector
import ugali.utils.healpix
import ugali.analysis.source
import ugali.analysis.kernel
import ugali.analysis.imf
import ugali.analysis.results
import ugali.simulation.population
from ugali.isochrone import factory as isochrone_factory
from ugali.utils.healpix import read_map

############################################################

def getCompleteness(config):
    # This is an easy place to make an error - make sure to get the right version
    #infile = 'y3a2_stellar_classification_summary_ext2.csv'
    infile = config['completeness']
    d = np.recfromcsv(infile)

    x = d['mag_r']
    y = d['eff_star']

    # Extend to saturation
    if min(x) > 16:
        x = np.insert(x, 0, 16.)
        y = np.insert(y, 0, y[0])

    # Make efficiency go to zero at faint end
    if y[-1] > 0:
        x = np.insert(x, -1, x[-1]+1)
        y = np.insert(y, -1, 0)

    f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=0.)

    return f

############################################################

def getPhotoError(config):
    """Photometric error model based on the delta-mag from the maglim and
    the photometric uncertainty estimated from the data

    Parameters
    ----------
    config : configuration dictionary

    Returns
    -------
    interp : interpolation function (mag_err as a function of delta_mag)
    """
    #infile = 'photo_error_model.csv'
    infile = config['photo_error']
    d = np.recfromcsv(infile)

    x = d['delta_mag']
    y = d['log_mag_err']

    # Extend on the bright end
    if min(x) > -10.0:
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
    nside_fracdet = hp.npix2nside(len(map_fracdet))
    map_fracdet_zero = np.where(map_fracdet >= 0., map_fracdet, 0.)
    fracdet_population = np.empty(len(lon_population))
    for ii in range(0, len(lon_population)):
        fracdet_population[ii] = np.mean(map_fracdet_zero[ugali.utils.healpix.ang2disc(nside_fracdet, 
                                                                                       lon_population[ii], 
                                                                                       lat_population[ii], 
                                                                                       radius_population if np.isscalar(radius_population) else radius_population[ii],
                                                                                       inclusive=True)])
    return fracdet_population

############################################################

def catsimSatellite(config, lon_centroid, lat_centroid, distance, stellar_mass, r_physical, 
                    ellipticity, position_angle, age, metal, 
                    m_maglim_1, m_maglim_2, m_ebv,
                    plot=False, title='test'):
    """ Simulate a single satellite. This is currently only valid for
    band_1 = g and band_2 = r.  r_physical is azimuthally averaged
    half-light radius, kpc

    Parameters
    ----------
    config : configuration
    lon_centroid : longitude centroid (deg)
    lat_centroid : latitude centroid (deg)
    distance : distance (kpc)
    stellar_mass : stellar mass (Msun)
    r_physical : azimuthally averaged physical half-light radius (kpc)
    ellipticity : ellipticity [0, 1]
    position_angle : position angle (deg)
    age : age (Gyr)
    metal : metallicity
    m_maglim_1 : mask of magnitude limit in band 2
    m_maglim_2 : mask of magnitude limit in band 2
    m_ebv : mask of E(B-V) values
    plot : Plot the output [False]
    
    Returns
    -------
    satellite : ordered dictionary of satellite star output
    """

    # Probably don't want to parse every time
    completeness = getCompleteness(config)
    log_photo_error = getPhotoError(config)

    src = ugali.analysis.source.Source()

    # Azimuthally averaged projected half-light radius (deg)
    r_h = np.degrees(np.arcsin(r_physical / distance)) 
    # Elliptical half-light radius along semi-major axis (deg)
    a_h = r_h / np.sqrt(1. - ellipticity) 

    # Create the kernel without extension
    ker = ugali.analysis.kernel.EllipticalPlummer(lon=lon_centroid, lat=lat_centroid, 
                                                  ellipticity=ellipticity, position_angle=position_angle)
    # Apply a max extension cut
    flag_too_extended = False
    max_extension = 5.0 # deg
    if a_h >= max_extension:
        print('Too extended: a_h = %.2f'%(a_h))
        a_h = max_extension
        flag_too_extended = True
    # Elliptical kernels take the "extension" as the semi-major axis
    extension = a_h # Elliptical half-light radius
    ker.setp('extension', value=a_h, bounds=[0.0,max_extension])
    src.set_kernel(ker)

    # Create the isochrone
    distance_modulus = ugali.utils.projector.dist2mod(distance)
    iso = isochrone_factory(config.get('isochrone','Bressan2012'), 
                            survey=config['survey'], age=age, z=metal, 
                            distance_modulus=distance_modulus)
    src.set_isochrone(iso)
    # Simulate takes stellar mass as an argument, NOT richness
    mag_1, mag_2 = src.isochrone.simulate(stellar_mass) 

    # Generate the positions of stars
    lon, lat = src.kernel.sample_lonlat(len(mag_2))

    nside = hp.npix2nside(len(m_maglim_1)) # Assuming that the two maglim maps have same resolution
    pix = ugali.utils.healpix.ang2pix(nside, lon, lat)
    maglim_1 = m_maglim_1[pix]
    maglim_2 = m_maglim_2[pix]
    if config['survey'] == 'des':
        # DES Y3 Gold fiducial
        mag_extinction_1 = 3.186 * m_ebv[pix]
        mag_extinction_2 = 2.140 * m_ebv[pix]
    elif config['survey'] == 'ps1':
        # From Table 6 in Schlafly 2011 with Rv = 3.1
        # http://iopscience.iop.org/article/10.1088/0004-637X/737/2/103/pdf
        mag_extinction_1 = 3.172 * m_ebv[pix]
        mag_extinction_2 = 2.271 * m_ebv[pix]
    elif config['survey'] == 'lsst_dp0':
        # From Table 6 in Schlafly 2011 with Rv = 3.1
        # http://iopscience.iop.org/article/10.1088/0004-637X/737/2/103/pdf
        mag_extinction_1 = 3.237 * m_ebv[pix]
        mag_extinction_2 = 2.273 * m_ebv[pix] 
    else:
        msg = "Unrecognized survey: %s"%config['survey']
        raise Exception(msg)
    
    # Photometric uncertainties are larger in the presence of interstellar dust reddening
    mag_1_error = 0.01 + 10**(log_photo_error((mag_1 + mag_extinction_1) - maglim_1))
    mag_2_error = 0.01 + 10**(log_photo_error((mag_2 + mag_extinction_2) - maglim_2))

    # It would be better to convert to a flux uncertainty and then transform back to a magnitude
    #mag_1_meas = mag_1 + np.random.normal(scale=mag_1_error)
    #mag_2_meas = mag_2 + np.random.normal(scale=mag_2_error)
    flux_1_meas = magToFlux(mag_1) + np.random.normal(scale=getFluxError(mag_1, mag_1_error))
    mag_1_meas = np.where(flux_1_meas > 0., fluxToMag(flux_1_meas), 99.)
    flux_2_meas = magToFlux(mag_2) + np.random.normal(scale=getFluxError(mag_2, mag_2_error))
    mag_2_meas = np.where(flux_2_meas > 0., fluxToMag(flux_2_meas), 99.)

    # In the HSC SXDS ultra-deep field:
    # mean maglim_r_sof_gold_2.0 = 23.46
    # median maglim_r_sof_gold_2.0 = 23.47
    # m = healpy.read_map('/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a1/data/maps/y3a2_gold_1.0_cmv02-001_v1_nside4096_nest_r_depth.fits.gz')
    # np.mean(m[ugali.utils.healpix.angToDisc(4096, 34.55, -4.83, 0.75)])
    # np.median(m[ugali.utils.healpix.angToDisc(4096, 34.55, -4.83, 0.75)])

    # Includes penalty for interstellar extinction and also include variations in depth
    if config['survey'] == 'des':
        cut_detect = (np.random.uniform(size=len(mag_2)) < completeness(mag_2 + mag_extinction_2 + (23.46 - np.clip(maglim_2, 20., 26.))))
    elif config['survey'] == 'ps1':
        cut_detect = (np.random.uniform(size=len(mag_2)) < completeness(mag_2 + mag_extinction_2))
    elif config['survey'] == 'lsst_dp0':
        cut_detect = (np.random.uniform(size=len(mag_2)) < completeness(mag_2 + mag_extinction_2 + (25.0 - np.clip(maglim_2, 20., 26.)))) # Using the psuedo mag depth of 25 for now
    else:
        msg = "Unrecognized survey: %s"%config['survey']
        raise Exception(msg)

    n_g22 = np.sum(cut_detect & (mag_1 < 22.))
    n_g24 = np.sum(cut_detect & (mag_1 < 24.))
    print('  n_sim = %i, n_detect = %i, n_g24 = %i, n_g22 = %i'%(len(mag_1),np.sum(cut_detect),n_g24,n_g22))
    
    richness = stellar_mass / src.isochrone.stellarMass()
    #abs_mag = src.isochrone.absolute_magnitude()
    #abs_mag_martin = src.isochrone.absolute_magnitude_martin(richness=richness, n_trials=10)[0] # 100 trials seems to be sufficient for rough estimate
    #print 'abs_mag_martin = %.2f mag'%(abs_mag_martin)

    # The more clever thing is to sum the simulated stars
    if config['survey'] == 'des':
        # See https://github.com/DarkEnergySurvey/ugali/blob/master/ugali/isochrone/model.py
        v = mag_1 - 0.487*(mag_1 - mag_2) - 0.0249 
    elif config['survey'] == 'ps1':
        # https://arxiv.org/pdf/1706.06147.pdf
        # V - g = C_0 + C_1 * (g - r)
        C_0 = -0.017
        C_1 = -0.508
        v = mag_1 + C_0 + C_1 * (mag_1 - mag_2)
    elif config['survey'] == 'lsst_dp0':
        # Numbers are just placeholders for now, need to figure out exact ones
        C_0 = -0.02
        C_1 = -0.50
        v = mag_1 + C_0 + C_1 * (mag_1 - mag_2)
    else:
        msg = "Unrecognized survey: %s"%config['survey']
        raise Exception(msg)

    flux = np.sum(10**(-v/2.5))
    abs_mag = -2.5*np.log10(flux) - distance_modulus

    # Realized surface brightness within azimuthally averaged half-light radius
    surface_brightness = ugali.analysis.results.surfaceBrightness(abs_mag, r_physical, distance) 

    if plot:
        import pylab
        pylab.ion()

        n_sigma_p = np.sum(cut_detect & (mag_1 < 23.))

        pylab.figure(figsize=(6., 6.))
        pylab.scatter(mag_1_meas[cut_detect] - mag_2_meas[cut_detect], mag_1_meas[cut_detect], edgecolor='none', c='black', s=5)
        pylab.xlim(-0.5, 1.)
        pylab.ylim(26., 16.)
        pylab.xlabel('g - r')
        pylab.ylabel('g')
        pylab.title('Number of stars with g < 23: %i'%(n_sigma_p))
        pylab.savefig('y3_sat_sim_cmd_%s.png'%('test'), dpi=150.)
        
        print('n_Sigma_p = %i'%(n_sigma_p))
        raw_input('WAIT')
        
    satellite=odict(lon=lon[cut_detect], lat=lat[cut_detect], 
                    mag_1=mag_1_meas[cut_detect], mag_2=mag_2_meas[cut_detect], 
                    mag_1_error=mag_1_error[cut_detect], mag_2_error=mag_2_error[cut_detect], 
                    mag_extinction_1=mag_extinction_1[cut_detect], 
                    mag_extinction_2=mag_extinction_2[cut_detect], 
                    n_g22=n_g22, n_g24=n_g24, abs_mag=abs_mag, surface_brightness=surface_brightness, 
                    extension=extension, flag_too_extended=flag_too_extended)

    return satellite

############################################################

#from memory_profiler import profile
#@profile
def catsimPopulation(config, tag, mc_source_id_start=1, n=5000, n_chunk=100, 
                     known_dwarfs=False):
    """ Create a population of satellites and then simulate the stellar distributions for each.

    Parameters
    ----------
    config : configuration file or dictionary
    tag    : output name
    mc_source_id_start : starting value of source id
    n       : number of satellites to simulate [5000]
    n_chunk : number of satellites written in a file chunk

    Returns
    -------
    None
    """

    assert mc_source_id_start >= 1, "Starting mc_source_id must be >= 1" 
    assert n % n_chunk == 0, "Total number of satellites must be divisible by the chunk size"
    nside_pix = 256 # NSIDE = 128 -> 27.5 arcmin, NSIDE = 256 -> 13.7 arcmin 
    
    if not os.path.exists(tag): os.makedirs(tag)

    if isinstance(config,str): config = yaml.load(open(config))
    assert config['survey'] in ['des', 'ps1', 'lsst_dp0']

    infile_ebv = config['ebv']
    infile_fracdet = config['fracdet']
    infile_maglim_g = config['maglim_g']
    infile_maglim_r = config['maglim_r']
    infile_density = config['stellar_density']

    range_distance      = config.get('range_distance',[5., 500.])
    range_stellar_mass  = config.get('range_stellar_mass',[1.e1, 1.e6])
    range_r_physical    = config.get('range_r_physical',[1.e-3, 2.0])
    range_ellipticity   = config.get('range_ellipticity',[0.1, 0.8])
    range_position_angle= config.get('range_position_angle',[0.0, 180.0])
    choice_age          = config.get('choice_age',[10., 12.0, 13.5])
    choice_metal        = config.get('choice_metal',[0.00010,0.00020])
    dwarf_file          = config.get('known_dwarfs',None)

    m_density = np.load(infile_density)
    nside_density = hp.npix2nside(len(m_density))
    m_fracdet = read_map(infile_fracdet, nest=False) #.astype(np.float16)
    nside_fracdet = hp.npix2nside(len(m_fracdet))

    m_maglim_g = read_map(infile_maglim_g, nest=False) #.astype(np.float16)
    m_maglim_r = read_map(infile_maglim_r, nest=False) #.astype(np.float16)

    m_ebv = read_map(infile_ebv, nest=False) #.astype(np.float16)
    
    #m_foreground = healpy.read_map(infile_foreground)

    mask = (m_fracdet > 0.5)

    if known_dwarfs:
        # Simulate from known dwarfs
        if dwarf_file is None: raise Exception("Must provide known_dwarf file")
        print("Simulating dwarfs from: %s"%dwarf_file)
        area, population = ugali.simulation.population.knownPopulation(dwarf_file, mask, nside_pix, n)
    else:
        # r_physical is azimuthally-averaged half-light radius, kpc
        kwargs = dict(range_distance = range_distance,
                      range_stellar_mass = range_stellar_mass,
                      range_r_physical = range_r_physical,
                      range_ellipticity=[0.1, 0.8],
                      range_position_angle=[0.0, 180.0],
                      choice_age=[10., 12.0, 13.5],
                      choice_metal=[0.00010,0.00020],
                      plot=False)
        area, population = ugali.simulation.population.satellitePopulation(mask, nside_pix, n, **kwargs)
        # Hack to duplicate RA,DEC
        print("Hacking lon,lat...")
        for idx in np.array_split(np.arange(len(population)), 10):
            population['lon'][idx] = population['lon'][idx[0]]
            population['lat'][idx] = population['lat'][idx[0]]
        
    population['id'] += mc_source_id_start
    simulation_area = area

    n_g22_population = np.tile(np.nan, n)
    n_g24_population = np.tile(np.nan, n)
    abs_mag_population = np.tile(np.nan, n)
    surface_brightness_population = np.tile(np.nan, n)
    extension_population = np.tile(np.nan, n)
    difficulty_population = np.tile(0, n)

    lon_array = []
    lat_array = []
    mag_1_array = []
    mag_2_array = []
    mag_1_error_array = []
    mag_2_error_array = []
    mag_extinction_1_array = []
    mag_extinction_2_array = []
    mc_source_id_array = []
    for ii, mc_source_id in enumerate(population['id']):
        print('Simulating satellite (%i/%i) ... mc_source_id = %i'%(ii + 1, n, mc_source_id))
        print('  distance=%(distance).2e, stellar_mass=%(stellar_mass).2e, r_physical=%(r_physical).2e'%(population[ii]))
        satellite = catsimSatellite(config, population[ii]['lon'], population[ii]['lat'], 
                                    population[ii]['distance'], population[ii]['stellar_mass'], 
                                    population[ii]['r_physical'],population[ii]['ellipticity'],
                                    population[ii]['position_angle'],population[ii]['age'],
                                    population[ii]['metallicity'],
                                    m_maglim_g, m_maglim_r, m_ebv)
        
        n_g22_population[ii] = satellite['n_g22']
        n_g24_population[ii] = satellite['n_g24']
        abs_mag_population[ii] = satellite['abs_mag']
        extension_population[ii] = satellite['extension']
        surface_brightness_population[ii] = satellite['surface_brightness']

        # These objects are too extended and are not simulated
        if (satellite['flag_too_extended']):
            difficulty_population[ii] |= 0b0001

        # We assume that these objects would be easily detected and
        # remove them to reduce data volume
        if config['survey'] == 'lsst_dp0':
            if (surface_brightness_population[ii]<23.5)&(n_g24_population[ii]>1e2):
                difficulty_population[ii] |= 0b0010
        else:
            if (surface_brightness_population[ii]<23.5)&(n_g22_population[ii]>1e3):
                difficulty_population[ii] |= 0b0010

        # ADW 2019-08-31: I don't think these were implmented
        #if (surface_brightness_population[ii]<25.)&(n_g22_population[ii]>1e2):
        #    difficulty_population[ii] |= 0b0010
        #if (surface_brightness_population[ii]<28.)&(n_g22_population[ii]>1e4):
        #    difficulty_population[ii] |= 0b0100
        #if (surface_brightness_population[ii]<30.)&(n_g22_population[ii]>1e5):
        #    difficulty_population[ii] |= 0b1000
        
        # ADW: 2019-08-31: These were Keith's original cuts, which were too aggressive
        #cut_easy = (surface_brightness_population[ii]<25.)&(n_g22_population[ii]>1.e2) \
        #           | ((surface_brightness_population[ii] < 30.) & (n_g24_population[ii] > 1.e4)) \
        #           | ((surface_brightness_population[ii] < 31.) & (n_g24_population[ii] > 1.e5))
        #cut_hard = (surface_brightness_population[ii] > 35.) | (n_g24_population[ii] < 1.)
        #cut_difficulty_population[ii] = ~cut_easy & ~cut_hard
        #if cut_easy:
        #    difficulty_population[ii] += 1 # TOO EASY
        #if cut_hard:
        #    difficulty_population[ii] += 2 # TOO HARD
        #if flag_too_extended:
        #    difficulty_population[ii] += 3 # TOO EXTENDED

        # Only write satellites that aren't flagged
        if difficulty_population[ii] == 0:
            lon_array.append(satellite['lon'])
            lat_array.append(satellite['lat'])
            mag_1_array.append(satellite['mag_1'])
            mag_2_array.append(satellite['mag_2'])
            mag_1_error_array.append(satellite['mag_1_error'])
            mag_2_error_array.append(satellite['mag_2_error'])
            mag_extinction_1_array.append(satellite['mag_extinction_1'])
            mag_extinction_2_array.append(satellite['mag_extinction_2'])
            mc_source_id_array.append(np.tile(mc_source_id, len(satellite['lon'])))
        else:
            print('  difficulty=%i; satellite not simulated...'%difficulty_population[ii])

    # Concatenate the arrays
    print("Concatenating arrays...")
    lon_array = np.concatenate(lon_array)
    lat_array = np.concatenate(lat_array)
    mag_1_array = np.concatenate(mag_1_array)
    mag_2_array = np.concatenate(mag_2_array)
    mag_1_error_array = np.concatenate(mag_1_error_array)
    mag_2_error_array = np.concatenate(mag_2_error_array)
    mag_extinction_1_array = np.concatenate(mag_extinction_1_array)
    mag_extinction_2_array = np.concatenate(mag_extinction_2_array)
    mc_source_id_array = np.concatenate(mc_source_id_array)

    # Now do the masking all at once
    print("Fracdet masking...")
    pix_array = ugali.utils.healpix.ang2pix(nside_fracdet, lon_array, lat_array)
    cut_fracdet = (np.random.uniform(size=len(lon_array)) < m_fracdet[pix_array])

    lon_array = lon_array[cut_fracdet]
    lat_array = lat_array[cut_fracdet]
    mag_1_array = mag_1_array[cut_fracdet]
    mag_2_array = mag_2_array[cut_fracdet]
    mag_1_error_array = mag_1_error_array[cut_fracdet]
    mag_2_error_array = mag_2_error_array[cut_fracdet]
    mag_extinction_1_array = mag_extinction_1_array[cut_fracdet]
    mag_extinction_2_array = mag_extinction_2_array[cut_fracdet]
    mc_source_id_array = mc_source_id_array[cut_fracdet]

    # Create bonus columns
    print("Creating bonus columns...")
    distance_modulus_population = ugali.utils.projector.dist2mod(population['distance'])
    hpix_32_population = ugali.utils.healpix.ang2pix(32, population['lon'], population['lat']) # Make sure this matches the dataset

    # Local stellar density
    pixarea = hp.nside2pixarea(nside_density, degrees=True) * 60.**2 # arcmin^2
    density_population = m_density[ugali.utils.healpix.ang2pix(nside_density, population['lon'], population['lat'])] / pixarea # arcmin^-2

    # Average fracdet within the azimuthally averaged half-light radius
    #m_fracdet_zero = np.where(m_fracdet >= 0., m_fracdet, 0.)
    #m_fracdet_zero = m_fracdet

    # Azimuthally averaged half-light radius in degrees
    r_half = np.degrees(np.arctan2(population['r_physical'], population['distance']))
    fracdet_half_population = meanFracdet(m_fracdet, population['lon'], population['lat'], r_half)
    fracdet_core_population = meanFracdet(m_fracdet, population['lon'], population['lat'], 0.1)
    fracdet_wide_population = meanFracdet(m_fracdet, population['lon'], population['lat'], 0.5)

    # Magnitude limits
    nside_maglim = hp.npix2nside(len(m_maglim_g))
    pix_population = ugali.utils.healpix.ang2pix(nside_maglim, population['lon'], population['lat'])
    maglim_g_population = m_maglim_g[pix_population]
    maglim_r_population = m_maglim_r[pix_population]
    
    # E(B-V)
    nside_ebv = hp.npix2nside(len(m_ebv))
    pix_population = ugali.utils.healpix.ang2pix(nside_ebv, population['lon'], population['lat'])
    ebv_population = m_ebv[pix_population]

    # Survey
    survey_population = np.tile(config['survey'], len(population))

    # Number of surviving catalog stars
    n_catalog_population = np.histogram(mc_source_id_array, bins=np.arange(population['id'][0] - 0.5, population['id'][-1] + 0.51))[0]

    # Faked-up coadd_object_ids
    coadd_object_id_array = []
    for mc_source_id in population['id']:
        coadd_object_id_array.append((1000000 * mc_source_id) + 1 + np.arange(np.sum(mc_source_id == mc_source_id_array)))
    # Assign negative numbers to distinguish from real objects
    coadd_object_id_array = -1 * np.concatenate(coadd_object_id_array) 

    # Object ID assignment can get messed up if there are duplicate population ids
    assert len(coadd_object_id_array) == len(mc_source_id_array)

    # Population metadata output file
    tbhdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name='RA',                 format='E', array=population['lon'], unit='deg'),
        pyfits.Column(name='DEC',                format='E', array=population['lat'], unit='deg'),
        pyfits.Column(name='DISTANCE',           format='E', array=population['distance'], unit='kpc'),
        pyfits.Column(name='DISTANCE_MODULUS',   format='E', array=distance_modulus_population, unit='kpc'),
        pyfits.Column(name='STELLAR_MASS',       format='E', array=population['stellar_mass'], unit='Msun'),
        pyfits.Column(name='R_PHYSICAL',         format='E', array=population['r_physical'], unit='kpc'),
        pyfits.Column(name='N_G22',              format='J', array=n_g22_population, unit=''),
        pyfits.Column(name='N_G24',              format='J', array=n_g24_population, unit=''),
        pyfits.Column(name='N_CATALOG',          format='J', array=n_catalog_population, unit=''),
        pyfits.Column(name='DIFFICULTY',         format='J', array=difficulty_population, unit=''),
        pyfits.Column(name='ABS_MAG',            format='E', array=abs_mag_population, unit='mag'),
        pyfits.Column(name='SURFACE_BRIGHTNESS', format='E', array=surface_brightness_population, unit='mag arcsec^-2'),
        pyfits.Column(name='EXTENSION',          format='E', array=extension_population, unit='deg'),
        pyfits.Column(name='ELLIPTICITY',        format='E', array=population['ellipticity'], unit=''),
        pyfits.Column(name='POSITION_ANGLE',     format='E', array=population['position_angle'], unit='deg'),
        pyfits.Column(name='AGE',                format='E', array=population['age'], unit='Gyr'),
        pyfits.Column(name='METAL_Z',            format='E', array=population['metallicity'], unit=''),
        pyfits.Column(name='MC_SOURCE_ID',       format='K', array=population['id'], unit=''),
        pyfits.Column(name='HPIX_32',            format='E', array=hpix_32_population, unit=''),
        pyfits.Column(name='DENSITY',            format='E', array=density_population, unit='arcmin^-2'),
        pyfits.Column(name='FRACDET_HALF',       format='E', array=fracdet_half_population, unit=''),
        pyfits.Column(name='FRACDET_CORE',       format='E', array=fracdet_core_population, unit=''),
        pyfits.Column(name='FRACDET_WIDE',       format='E', array=fracdet_wide_population, unit=''),
        pyfits.Column(name='MAGLIM_G',           format='E', array=maglim_g_population, unit='mag'),
        pyfits.Column(name='MAGLIM_R',           format='E', array=maglim_r_population, unit='mag'),
        pyfits.Column(name='EBV',                format='E', array=ebv_population, unit='mag'),
        pyfits.Column(name='SURVEY',             format='A12', array=survey_population, unit=''),
    ])
    tbhdu.header.set('AREA', simulation_area, 'Simulation area (deg^2)')
    print("Writing population metadata file...")
    filename = '%s/sim_population_%s_mc_source_id_%07i-%07i.fits'%(tag, tag, mc_source_id_start, mc_source_id_start + n - 1)
    tbhdu.writeto(filename, overwrite=True)

    # Write simulated catalogs

    # Simulated catalog output needs to match the real data
    #   https://github.com/sidneymau/simple/blob/master/search_algorithm.py 
    #   https://github.com/sidneymau/simple/blob/master/config.yaml
    #   /home/s1/kadrlica/projects/y3a2/dsphs/v2/skim/ 
    #   e.g., /home/s1/kadrlica/projects/y3a2/dsphs/v2/skim/y3a2_ngmix_cm_11755.fits
    # for ii in range(0, len(d.formats)): print '\'%s\': [ , \'%s\'],'%(d.names[ii], d.formats[ii])

    default_array = np.tile(-9999., len(mc_source_id_array))
    if config['survey'] == 'des':
        # Y3 Gold v2.0
        key_map = odict([
                ('COADD_OBJECT_ID',         [coadd_object_id_array, 'K']),
                ('RA',                      [lon_array, 'D']),
                ('DEC',                     [lat_array, 'D']),
                ('SOF_PSF_MAG_CORRECTED_G', [mag_1_array, 'D']),
                ('SOF_PSF_MAG_CORRECTED_R', [mag_2_array, 'D']),
                ('SOF_PSF_MAG_ERR_G',       [mag_1_error_array, 'D']),
                ('SOF_PSF_MAG_ERR_R',       [mag_2_error_array, 'D']),
                ('A_SED_SFD98_G',           [mag_extinction_1_array, 'E']),
                ('A_SED_SFD98_R',           [mag_extinction_2_array, 'E']),
                ('WAVG_MAG_PSF_G',          [mag_1_array+mag_extinction_1_array, 'E']),
                ('WAVG_MAG_PSF_R',          [mag_2_array+mag_extinction_2_array, 'E']),
                ('WAVG_MAGERR_PSF_G',       [mag_1_error_array, 'E']),
                ('WAVG_MAGERR_PSF_R',       [mag_2_error_array, 'E']),
                ('WAVG_SPREAD_MODEL_I',     [default_array, 'E']),
                ('WAVG_SPREADERR_MODEL_I',  [default_array, 'E']),
                ('SOF_CM_T',                [default_array, 'D']),
                ('SOF_CM_T_ERR',            [default_array, 'D']),
                ('FLAGS_GOLD',              [np.tile(0, len(mc_source_id_array)), 'J']),
                ('EXTENDED_CLASS_MASH_SOF', [np.tile(0, len(mc_source_id_array)), 'I']),
                ])
    elif config['survey'] == 'ps1':
        # PS1
        key_map = odict([
                ('OBJID',                   [coadd_object_id_array, 'K']),
                ('RA',                      [lon_array, 'D']),
                ('DEC',                     [lat_array, 'D']),
                #('UNIQUEPSPSOBID',          [coadd_object_id_array, 'K']),
                #('OBJINFOFLAG',             [default_array, 'E']),
                #('QUALITYFLAG',             [np.tile(16, len(mc_source_id_array)), 'I']),
                #('NSTACKDETECTIONS',        [np.tile(99, len(mc_source_id_array)), 'I']),
                #('NDETECTIONS',             [np.tile(99, len(mc_source_id_array)), 'I']),
                #('NG',                      [default_array, 'E']),
                #('NR',                      [default_array, 'E']),
                #('NI',                      [default_array, 'E']),
                ('GFPSFMAG',                 [mag_1_array+mag_extinction_1_array, 'E']),
                ('RFPSFMAG',                 [mag_2_array+mag_extinction_2_array, 'E']),
                #('IFPSFMAG',                 [np.tile(0., len(mc_source_id_array)), 'E'], # Pass star selection
                ('GFPSFMAGERR',              [mag_1_error_array, 'E']),
                ('RFPSFMAGERR',              [mag_2_error_array, 'E']),
                #('IFPSFMAGERR',              [default_array, 'E']),
                #('GFKRONMAG',                [mag_1_array, 'E']),
                #('RFKRONMAG',                [mag_2_array, 'E']),
                #('IFKRONMAG',                [np.tile(0., len(mc_source_id_array)), 'E'], # Pass star selection
                #('GFKRONMAGERR',             [mag_1_error_array, 'E']),
                #('RFKRONMAGERR',             [mag_2_error_array, 'E']),
                #('IFKRONMAGERR',             [default_array, 'E']),
                #('GFLAGS',                   [np.tile(0, len(mc_source_id_array)), 'I']),
                #('RFLAGS',                   [np.tile(0, len(mc_source_id_array)), 'I']),
                #('IFLAGS',                   [np.tile(0, len(mc_source_id_array)), 'I']),
                #('GINFOFLAG',                [np.tile(0, len(mc_source_id_array)), 'I']),
                #('RINFOFLAG',                [np.tile(0, len(mc_source_id_array)), 'I']),
                #('IINFOFLAG',                [np.tile(0, len(mc_source_id_array)), 'I']),
                #('GINFOFLAG2',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('RINFOFLAG2',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('IINFOFLAG2',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('GINFOFLAG3',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('RINFOFLAG3',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('IINFOFLAG3',               [np.tile(0, len(mc_source_id_array)), 'I']),
                #('PRIMARYDETECTION',         [default_array, 'E']),
                #('BESTDETECTION',            [default_array, 'E']),
                #('EBV',                      [default_array, 'E']),
                #('EXTSFD_G',                 [mag_extinction_1_array 'E']),
                #('EXTSFD_R',                 [mag_extinction_2_array, 'E']),
                #('EXTSFD_I',                 [default_array, 'E']),
                ('GFPSFMAG_SFD',             [mag_1_array, 'E']),
                ('RFPSFMAG_SFD',             [mag_2_array, 'E']),
                ('EXTENDED_CLASS',           [np.tile(0, len(mc_source_id_array)), 'I']),
                ])
    elif config['survey'] == 'lsst_dp0': # Keys make to match those in the GCRCatalog native_quantities
        key_map = odict([
                ('objectId', [coadd_object_id_array, 'K']),
                ('ra', [lon_array, 'D']),
                ('dec', [lat_array, 'D']),
                ('mag_g', [mag_1_array+mag_extinction_1_array, 'E']),
                ('mag_r', [mag_2_array+mag_extinction_2_array, 'E']),
                ('magerr_g', [mag_1_error_array, 'D']),
                ('magerr_r', [mag_2_error_array, 'D']),
                ('mag_corrected_g', [mag_1_array, 'D']), 
                ('mag_corrected_r', [mag_2_array, 'D']), 
                ('extended_class', [np.tile(0, len(mc_source_id_array)), 'I']), 
                ])
    else:
        msg = "Unrecognized survey: %s"%config['survey']
        raise Exception(msg)

    key_map['MC_SOURCE_ID'] = [mc_source_id_array, 'K']

    print("Writing catalog files...")
    for mc_source_id_chunk in np.split(np.arange(mc_source_id_start, mc_source_id_start + n), n//n_chunk):
        outfile = '%s/sim_catalog_%s_mc_source_id_%07i-%07i.fits'%(tag, tag, mc_source_id_chunk[0], mc_source_id_chunk[-1])
        print('  '+outfile)
        sel = np.in1d(mc_source_id_array, mc_source_id_chunk)
        columns = [pyfits.Column(name=k, format=v[1], array=v[0][sel]) for k,v in key_map.items()]
        tbhdu = pyfits.BinTableHDU.from_columns(columns)
        tbhdu.header.set('AREA', simulation_area, 'Simulation area (deg^2)')
        tbhdu.header.set('IDMIN',mc_source_id_chunk[0], 'Minimum MC_SOURCE_ID')
        tbhdu.header.set('IDMAX',mc_source_id_chunk[-1], 'Maximum MC_SOURCE_ID')
        tbhdu.writeto(outfile, overwrite=True)

    # Mask output file
    print("Writing population mask file...")
    outfile_mask = '%s/sim_mask_%s_cel_nside_%i.fits'%(tag, tag, hp.npix2nside(len(mask)))
    if not os.path.exists(outfile_mask):
        hp.write_map(outfile_mask, mask.astype(int), nest=True, coord='C', overwrite=True)
        os.system('gzip -f %s'%(outfile_mask))

############################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simulate at Milky Way satellite population.')
    parser.add_argument('config',
                        help='Configuration file')
    parser.add_argument('-s','--section',required=True,
                        choices=['des','ps1','lsst_dc2','lsst_dp0'],
                        help='Config section for simulation parameters')
    parser.add_argument('--tag',required=True,
                        help='Descriptive tag for the simulation run')
    parser.add_argument('--start', dest='mc_source_id_start', type=int, default=1,
                        help='MC_SOURCE_ID start')
    parser.add_argument('--size', dest='n', type=int, default=5000,
                        help='Number of satellites to start')
    parser.add_argument('--chunk', dest='n_chunk', type=int, default=100,
                        help="Number of MC_SOURCE_ID's per catalog output file")
    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help="Random seed")
    parser.add_argument('--dwarfs', dest='dwarfs', action='store_true', 
                        help="Simulate from known dwarfs")
    args = parser.parse_args()

    if args.seed is not None: 
        print("Setting random seed: %i"%args.seed)
        np.random.seed(args.seed)

    # Load the config and select the survey section
    config = yaml.safe_load(open(args.config))[args.section]
    
    #catsimPopulation(tag, mc_source_id_start=mc_source_id_start, n=n, n_chunk=n_chunk)
    catsimPopulation(config, args.tag, mc_source_id_start=args.mc_source_id_start, n=args.n, n_chunk=args.n_chunk, known_dwarfs=args.dwarfs)

############################################################

"""
# Tuc III
distance = 25.
stellar_mass = 0.8e3
r_physical = 0.044 # kpc
catsimSatellite(0., 0., distance, stellar_mass, r_physical, m_maglim_g, m_maglim_r, plot=True, title='tuc_iii')
    
# Gru II
distance = 53.
stellar_mass = 3.4e3
r_physical = 0.093 # kpc
catsimSatellite(0., 0., distance, stellar_mass, r_physical, m_maglim_g, m_maglim_r, plot=True, title='gru_ii')
"""

############################################################
