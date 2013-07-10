"""
Documentation.
"""

import numpy

import ugali.analysis.farm
import ugali.observation.catalog
import ugali.simulation.simulator

############################################################

def validateSatellite(config, isochrone, kernel, stellar_mass, distance_modulus, trials=1):
    """

    """

    print '=== Validate Satellite ==='

    catalog_base = ugali.observation.catalog.Catalog(config)
    coords = (kernel.lon, kernel.lat)
    simulator = ugali.simulation.simulator.Simulator(config, kernel.lon, kernel.lat)

    results = {'richness': [],
               'log_likelihood': []}

    for ii in range(0, trials):

        # Simulate
        catalog_satellite = simulator.satellite(isochrone, kernel, stellar_mass, distance_modulus, mc_source_id=1)
        catalog_merge = ugali.observation.catalog.mergeCatalogs([catalog_base, catalog_satellite])

        # Analyze
        farm = ugali.analysis.farm.Farm(config, catalog=catalog_merge)
        likelihood = farm.farmLikelihoodFromCatalog(local=True, coords=coords)
        likelihood.precomputeGridSearch([distance_modulus])
        richness, log_likelihood, richness_upper_limit, richness_raw, log_likelihood_raw, p, f = likelihood.gridSearch(coords=coords, distance_modulus_index=0)

        results['richness'].append(richness)
        results['log_likelihood'].append(log_likelihood)

    return results

############################################################
