"""
Tool to generate a population of simulated satellite properties.
"""
from collections import OrderedDict as odict
import numpy as np
import pylab
import pandas as pd

import ugali.utils.config
import ugali.utils.projector
import ugali.utils.skymap
import ugali.analysis.kernel
import ugali.observation.catalog
import ugali.isochrone

############################################################
NAMES = ['id','name','lon','lat','distance','stellar_mass','r_physical',
         'ellipticity','position_angle','age','metallicity']
DTYPE = [('id',int)] + [('name','S8')] + [(n,float) for n in NAMES[2:]]

def satellitePopulation(mask, nside_pix, size,
                        range_distance=[5., 500.],
                        range_stellar_mass=[1.e1, 1.e6],
                        range_r_physical=[1.e-3, 2.],
                        range_ellipticity=[0.1,0.8],
                        range_position_angle=[0,180],
                        choice_age=[10., 12.0, 13.5],
                        choice_metal=[0.00010,0.00020],
                        plot=False):
    """
    Create a population of randomly placed satellites within a
    survey mask.  Satellites are distributed uniformly in
    log(distance/kpc), uniformly in log(stellar_mass/Msun), and
    uniformly in physical half-light radius log(r_physical/kpc). The
    ranges can be set by the user.

    Returns the simulated area (deg^2) as well as the lon (deg), lat
    (deg), distance modulus, stellar mass (M_sol), and half-light
    radius (deg) for each satellite

    Parameters:
    -----------
    mask      : the survey mask of available area
    nside_pix : coarse resolution npix for avoiding small gaps in survey
    size      : number of satellites to simulate
    range_distance     : heliocentric distance range (kpc)
    range_stellar_mass : stellar mass range (Msun)
    range_r_physical   : projected physical half-light radius (kpc)
    range_ellipticity : elliptictiy [0,1]
    range_position_angle : position angle (deg)
    choice_age : choices for age
    choice_metal : choices for metallicity
    
    Returns:
    --------
    area, population : area of sky covered and population of objects
    """

    population = np.recarray(size,dtype=DTYPE)
    population.fill(np.nan)
    population['name'] = ''

    # Source ID
    population['id'] = np.arange(size)

    # Distance (kpc)
    population['distance'] = 10**np.random.uniform(np.log10(range_distance[0]),
                                                   np.log10(range_distance[1]),
                                                   size)
    # Stellar mass (Msun)
    population['stellar_mass'] = 10**np.random.uniform(np.log10(range_stellar_mass[0]), 
                                                       np.log10(range_stellar_mass[1]), 
                                                       size)
    
    # Physical half-light radius (kpc)
    population['r_physical'] = 10**np.random.uniform(np.log10(range_r_physical[0]), 
                                                     np.log10(range_r_physical[1]), 
                                                     size)

    # Ellipticity [e = 1 - (b/a)] with semi-major axis a and semi-minor axis b
    # See http://iopscience.iop.org/article/10.3847/1538-4357/833/2/167/pdf
    # Based loosely on https://arxiv.org/abs/0805.2945
    population['ellipticity'] = np.random.uniform(range_ellipticity[0], 
                                                  range_ellipticity[1],
                                                  size)

    # Random position angle (deg)
    population['position_angle'] = np.random.uniform(range_position_angle[0],
                                                     range_position_angle[1],
                                                     size) 

    # Age (Gyr)
    population['age'] = np.random.choice(choice_age,size)
    # Metallicity
    population['metallicity'] = np.random.choice(choice_metal,size)

    # Call positions last because while loop has a variable number of
    # calls to np.random (thus not preserving seed information)
    lon, lat, area = ugali.utils.skymap.randomPositions(mask, nside_pix, n=size)
    population['lon'] = lon
    population['lat'] = lat

    return area, population

############################################################

def satellitePopulationOrig(config, n,
                            range_distance_modulus=[16.5, 24.],
                            range_stellar_mass=[1.e2, 1.e5],
                            range_r_physical=[5.e-3, 1.],
                            mode='mask',
                            plot=False):
    """
    Create a population of `n` randomly placed satellites within a
    survey mask or catalog specified in the config file.  Satellites
    are distributed uniformly in distance modulus, uniformly in
    log(stellar_mass / Msun), and uniformly in
    log(r_physical/kpc). The ranges can be set by the user.

    Returns the simulated area (deg^2) as well as the
    Parameters
    ----------
    config : configuration file or object
    n      : number of satellites to simulate
    range_distance_modulus : range of distance modulus to sample
    range_stellar_mass : range of stellar mass to sample (Msun)
    range_r_physical : range of azimuthally averaged half light radius (kpc)
    mode : how to sample the area['mask','catalog']

    Returns
    -------
    lon (deg), lat (deg), distance modulus, stellar mass (Msun), and
    half-light radius (kpc) for each satellite
    """
    msg = "'satellitePopulationOrig': ADW 2019-09-01"
    DeprecationWarning(msg)

    if type(config) == str:
        config = ugali.utils.config.Config(config)

    if mode == 'mask':
        mask_1 = ugali.utils.skymap.readSparseHealpixMap(config.params['mask']['infile_1'], 'MAGLIM')
        mask_2 = ugali.utils.skymap.readSparseHealpixMap(config.params['mask']['infile_2'], 'MAGLIM')
        input = (mask_1 > 0.) * (mask_2 > 0.)
    elif mode == 'catalog':
        catalog = ugali.observation.catalog.Catalog(config)
        input = np.array([catalog.lon, catalog.lat])
    
    lon, lat, simulation_area = ugali.utils.skymap.randomPositions(input,
                                                                   config.params['coords']['nside_likelihood_segmentation'],
                                                                   n=n)
    distance_modulus = np.random.uniform(range_distance_modulus[0], 
                                         range_distance_modulus[1], 
                                         n)
    stellar_mass = 10**np.random.uniform(np.log10(range_stellar_mass[0]), 
                                         np.log10(range_stellar_mass[1]), 
                                         n)
    
    half_light_radius_physical = 10**np.random.uniform(np.log10(range_r_physical[0]),
                                                       np.log10(range_r_physical[1]),
                                                       n) # kpc

    half_light_radius = np.degrees(np.arcsin(half_light_radius_physical \
                                             / ugali.utils.projector.distanceModulusToDistance(distance_modulus)))
    
    # One choice of theory prior
    #half_light_radius_physical = ugali.analysis.kernel.halfLightRadius(stellar_mass) # kpc
    #half_light_radius = np.degrees(np.arcsin(half_light_radius_physical \
    #                                               / ugali.utils.projector.distanceModulusToDistance(distance_modulus)))

    if plot:
        pylab.figure()
        #pylab.scatter(lon, lat, c=distance_modulus, s=500 * half_light_radius)
        #pylab.colorbar()
        pylab.scatter(lon, lat, edgecolors='none')
        xmin, xmax = pylab.xlim() # Reverse azimuthal axis
        pylab.xlim([xmax, xmin])
        pylab.title('Random Positions in Survey Footprint')
        pylab.xlabel('Longitude (deg)')
        pylab.ylabel('Latitude (deg)')

        pylab.figure()
        pylab.scatter(stellar_mass, ugali.utils.projector.distanceModulusToDistance(distance_modulus),
                      c=(60. * half_light_radius), s=500 * half_light_radius, edgecolors='none')
        pylab.xscale('log')
        pylab.yscale('log')
        pylab.xlim([0.5 * range_stellar_mass[0], 2. * range_stellar_mass[1]])
        pylab.colorbar()
        pylab.title('Half-light Radius (arcmin)')
        pylab.xlabel('Stellar Mass (arcmin)')
        pylab.ylabel('Distance (kpc)')

    return simulation_area, lon, lat, distance_modulus, stellar_mass, half_light_radius 

############################################################

def interpolate_absolute_magnitude():
    iso = ugali.isochrone.factory('Bressan2012',age=12,z=0.00010)

    stellar_mass,abs_mag = [],[]
    for richness in np.logspace(1,8,25):
        stellar_mass += [iso.stellar_mass()*richness]
        if stellar_mass[-1] < 1e3:
            abs_mag += [iso.absolute_magnitude_martin(richness)[0]]
        else:
            abs_mag += [iso.absolute_magnitude(richness)]

    return abs_mag,stellar_mass

def knownPopulation(dwarfs, mask, nside_pix, size):
    """ Sample parameters from a known population .
    
    Parameters
    ----------
    dwarfs : known dwarfs to sample at
    mask      : the survey mask of available area
    nside_pix : coarse resolution npix for avoiding small gaps in survey
    size      : number of satellites to simulate; will be broken into 
                size//len(dwarfs) per dwarf

    Returns
    -------
    area, population
    """
    # generated from the interpolation function above.
    abs_mag_interp = [5.7574, 4.5429, 3.5881, 2.7379, 1.8594, 0.9984, 0.0245, 
                    -0.851, -1.691, -2.495, -3.343, -4.072, -4.801, -5.530, 
                    -6.259, -6.988, -7.718, -8.447, -9.176, -9.905, -10.63, 
                    -11.36, -12.09, -12.82, -13.55][::-1]

    stellar_mass_interp = [2.363705510, 4.626579555, 9.055797468, 17.72529075, 
                    34.69445217, 67.90890082, 132.9209289, 260.1716878, 
                    509.2449149, 996.7663490, 1951.012421, 3818.798128, 
                    7474.693131, 14630.52917, 28636.94603, 56052.29096, 
                    109713.4910, 214746.8000, 420332.8841, 822735.1162, 
                    1610373.818, 3152051.957, 6169642.994, 12076100.01, 
                    23637055.10][::-1]

    if isinstance(dwarfs,str):
        # Note that the random seed will be dependent on the order of dwarfs
        dwarfs = pd.read_csv(dwarfs).to_records(index=False)

    nsim = size // len(dwarfs)
    nlast = nsim + size % len(dwarfs)

    print('Calculating coarse footprint mask...')
    coarse_mask = ugali.utils.skymap.coarseFootprint(mask, nside_pix)

    results = []
    for i,dwarf in enumerate(dwarfs):
        print(dwarf['name'])
        kwargs = dict()
        kwargs['range_distance']   = [dwarf['distance'],dwarf['distance']]
        r_physical = dwarf['r12']/1000.
        kwargs['range_r_physical'] = [r_physical,r_physical]
        stellar_mass = np.interp(dwarf['M_V'],abs_mag_interp,stellar_mass_interp)
        print("WARNING: Zeroing stellar mass...")
        stellar_mass = 0.01
        kwargs['range_stellar_mass'] = [stellar_mass, stellar_mass]      
        kwargs['range_ellipticity'] = [dwarf['ellipticity'],dwarf['ellipticity']]
        kwargs['range_position_angle']=[0.0, 180.0]
        kwargs['choice_age']=[10., 12.0, 13.5]
        kwargs['choice_metal']=[0.00010,0.00020]

        num = nsim if i < (len(dwarfs)-1) else nlast
        area,pop = satellitePopulation(coarse_mask, nside_pix, num, **kwargs)
        pop['name'] = dwarf['abbreviation']
        print("WARNING: Fixing location...")
        pop['lon'] = dwarf['ra']
        pop['lat'] = dwarf['dec']
        results += [pop]

    population = np.hstack(results)
    population['id'] = np.arange(size)
    return area, population

def plot_population(population):
    # ADW: DEPRECATED: 2019-09-01
    pylab.figure()
    #pylab.scatter(lon, lat, c=distance_modulus, s=500 * half_light_radius)
    #pylab.colorbar()
    pylab.scatter(lon, lat, edgecolors='none')
    xmin, xmax = pylab.xlim() # Reverse azimuthal axis
    pylab.xlim([xmax, xmin])
    pylab.title('Random Positions in Survey Footprint')
    pylab.xlabel('Longitude (deg)')
    pylab.ylabel('Latitude (deg)')

    pylab.figure()
    pylab.scatter(stellar_mass, distance,c=r_physical,
                  s=500 * r_physical, edgecolors='none')
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.xlim([0.5 * range_stellar_mass[0], 2. * range_stellar_mass[1]])
    pylab.colorbar()
    pylab.title('Half-light Radius (arcmin)')
    pylab.xlabel('Stellar Mass (arcmin)')
    pylab.ylabel('Distance (kpc)')
