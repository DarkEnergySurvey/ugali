"""
Documentation.
"""

import numpy
import pylab

import ugali.analysis.farm
import ugali.observation.catalog
import ugali.simulation.simulator
import ugali.utils.bayesian_efficiency

pylab.ion()

############################################################

def validateSatellite(config, isochrone, kernel, stellar_mass, distance_modulus, trials=1, debug=False):
    """
    Tool for MC validation studies -- specifically to create multiple realizations of
    a satellite given an CompositeIsochrone object, Kernel object, stellar mass (M_sol) for normalization,
    and distance_modulus.
    """
    print '=== Validate Satellite ==='

    catalog_base = ugali.observation.catalog.Catalog(config)
    coords = (kernel.lon, kernel.lat)
    simulator = ugali.simulation.simulator.Simulator(config, kernel.lon, kernel.lat)

    results = {'richness': [],
               'log_likelihood': [],
               'richness_lim': [],
               'f': [],
               'stellar_mass': []}

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
        results['richness_lim'].append(richness_upper_limit)
        results['f'].append(f)
        results['stellar_mass'].append(richness * isochrone.stellarMass())

        if debug:
            return likelihood, richness, log_likelihood, richness_upper_limit, richness_raw, log_likelihood_raw, p, f

    return results

############################################################

def validateMembership(likelihood, p, mc_source_id=1):
    """
    Plot membership probabilities and MC source ids together.
    """

    cut_mc_source_id = (likelihood.catalog.mc_source_id == mc_source_id)

    # Spatial

    projector = ugali.utils.projector.Projector(likelihood.kernel.lon, likelihood.kernel.lat)
    x, y = projector.sphereToImage(likelihood.catalog.lon, likelihood.catalog.lat)

    pylab.figure()
    pylab.scatter(x[cut_mc_source_id], y[cut_mc_source_id], c='gray', s=100, edgecolors='none')
    pylab.scatter(x, y, c=p, edgecolors='none')
    pylab.colorbar()
    pylab.title('Membership Probability')
    pylab.xlabel(r'$\delta$ Lon (deg)')
    pylab.ylabel(r'$\delta$ Lat (deg)')

    # Spectral

    pylab.figure()
    pylab.scatter(likelihood.catalog.color[cut_mc_source_id], likelihood.catalog.mag[cut_mc_source_id], c='gray', s=100, edgecolors='none')
    pylab.scatter(likelihood.catalog.color, likelihood.catalog.mag, c=p, edgecolors='none')
    pylab.colorbar()
    pylab.title('Membership Probability')
    pylab.xlim(likelihood.roi.bins_color[0], likelihood.roi.bins_color[-1])
    pylab.ylim(likelihood.roi.bins_mag[-1], likelihood.roi.bins_mag[0])
    pylab.xlabel('Color (mag)')
    pylab.ylabel('Magnitude (mag)')

    # Membership accuracy

    p_array = []
    frac_members_array = []
    frac_members_array_low = []
    frac_members_array_high = []

    probability_bins = numpy.arange(0.1, 1. + 1.e-10, 0.1)
    for ii in range(0, len(probability_bins) - 1):

        if not numpy.any(numpy.logical_and(p >= probability_bins[ii], p < probability_bins[ii + 1])):
            continue

        p_array.append(0.5 * (probability_bins[ii] + probability_bins[ii + 1]))

        n_members = numpy.sum(numpy.all([likelihood.catalog.mc_source_id == mc_source_id,
                                         p >= probability_bins[ii],
                                         p < probability_bins[ii + 1]], axis=0))
        n_not_members = numpy.sum(numpy.all([likelihood.catalog.mc_source_id != mc_source_id,
                                             p >= probability_bins[ii],
                                             p < probability_bins[ii + 1]], axis=0))

        frac_members, (frac_members_low, frac_members_high) = ugali.utils.bayesian_efficiency.confidenceInterval(n_members + n_not_members,
                                                                                                                 n_members,
                                                                                                                 0.68)
        frac_members_array.append(frac_members)
        frac_members_array_low.append(frac_members - frac_members_low)
        frac_members_array_high.append(frac_members_high - frac_members)
        
    pylab.figure()
    pylab.plot([0,1], [0,1], c='red')
    pylab.errorbar(p_array, frac_members_array, yerr=[frac_members_array_low, frac_members_array_high],
                   marker='o', color='blue', linestyle='none')
    pylab.xlim(0, 1)
    pylab.ylim(0, 1)
    pylab.xlabel('Membership Probability')
    pylab.ylabel('Fraction of True Satellite Members')
        
############################################################
