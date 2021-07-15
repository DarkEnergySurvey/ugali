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
