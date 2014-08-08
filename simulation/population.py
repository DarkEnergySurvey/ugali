"""
Tool to generate a population of simulated satellite properties.
"""

import numpy
import pylab

import ugali.utils.config
import ugali.utils.projector
import ugali.utils.skymap
import ugali.analysis.kernel
import ugali.observation.catalog

pylab.ion()

############################################################

def satellitePopulation(config, n,
                        range_distance_modulus=[16.5, 24.],
                        range_stellar_mass=[1.e2, 1.e5],
                        mode='mask',
                        plot=False):
    """
    Create a population of n randomly placed satellites within a survey mask or catalog specified in the config file.
    Satellites are uniformly placed in distance modulus, and uniformly generated in log(stellar_mass) (M_sol).
    The ranges can be set by the user.

    Returns the simulated area (deg^2) as well as the
    lon (deg), lat (deg), distance modulus, stellar mass (M_sol), and half-light radius (deg) for each satellite
    """
    
    if type(config) == str:
        config = ugali.utils.config.Config(config)

    if mode == 'mask':
        mask_1 = ugali.utils.skymap.readSparseHealpixMap(config.params['mask']['infile_1'], 'MAGLIM')
        mask_2 = ugali.utils.skymap.readSparseHealpixMap(config.params['mask']['infile_2'], 'MAGLIM')
        input = (mask_1 > 0.) * (mask_2 > 0.)
    elif mode == 'catalog':
        catalog = ugali.observation.catalog.Catalog(config)
        input = numpy.array([catalog.lon, catalog.lat])
    
    lon, lat, simulation_area = ugali.utils.skymap.randomPositions(input,
                                                                   config.params['coords']['nside_likelihood_segmentation'],
                                                                   n=n)
    distance_modulus = numpy.random.uniform(range_distance_modulus[0], range_distance_modulus[1], n)
    stellar_mass = 10**numpy.random.uniform(numpy.log10(range_stellar_mass[0]), numpy.log10(range_stellar_mass[1]), n)
    half_light_radius_physical = ugali.analysis.kernel.halfLightRadius(stellar_mass) # kpc
    half_light_radius = numpy.degrees(numpy.arcsin(half_light_radius_physical \
                                                   / ugali.utils.projector.distanceModulusToDistance(distance_modulus)))

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
