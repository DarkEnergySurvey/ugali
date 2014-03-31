"""
Documentation.
"""

import numpy
import pylab

import ugali.analysis.farm
import ugali.observation.catalog
import ugali.observation.roi
import ugali.observation.mask
import ugali.simulation.simulator
import ugali.utils.bayesian_efficiency

from ugali.utils.logger import logger

pylab.ion()

############################################################

def validateSatellite(config, isochrone, kernel, stellar_mass, distance_modulus, trials=1, debug=False, seed=0):
    """
    Tool for simple MC validation studies -- specifically to create multiple realizations of
    a satellite given an CompositeIsochrone object, Kernel object, stellar mass (M_sol) for normalization,
    and distance_modulus.
    """
    logger.info('=== Validate Satellite ===')

    config.params['kernel']['params'] = [kernel.r_h] # TODO: Need better solution to update size??
    logger.debug('Using Plummer profile spatial model with half-light radius %.2f deg'%(config.params['kernel']['params'][0]))
    roi = ugali.observation.roi.ROI(config, kernel.lon, kernel.lat)
    simulator = ugali.simulation.simulator.Simulator(config, roi=roi)
    catalog_base = ugali.observation.catalog.Catalog(config,roi=roi)
    mask = ugali.observation.mask.Mask(config, roi)

    coords = (kernel.lon, kernel.lat)
    
    results = {'mc_lon': [],
               'mc_lat': [],
               'mc_distance_modulus': [],
               'mc_stellar_mass': [],
               'mc_radius': [],
               'richness': [],
               'log_likelihood': [],
               'richness_lower': [],
               'richness_upper': [],
               'richness_limit': [],
               'f': [],
               'stellar_mass': []}

    numpy.random.seed(seed)

    for ii in range(0, trials):
        logger.info('=== Running Satellite %i ==='%ii)

        # Simulate
        catalog_satellite = simulator.satellite(isochrone, kernel, stellar_mass, distance_modulus, mc_source_id=1)
        #catalog_bootstrap = catalog_base.bootstrap()
        #catalog_merge = ugali.observation.catalog.mergeCatalogs([catalog_bootstrap, catalog_satellite])
        catalog_background = simulator.satellite(mc_source_id=1)
        catalog_merge = ugali.observation.catalog.mergeCatalogs([catalog_background, catalog_satellite])

        # Analyze
        likelihood = ugali.analysis.likelihood.Likelihood(config, roi, mask, catalog_merge, isochrone, kernel)
                                                               
        likelihood.precomputeGridSearch([distance_modulus])
        richness, log_likelihood, richness_lower, richness_upper, richness_upper_limit, richness_raw, log_likelihood_raw, p, f = likelihood.gridSearch(coords=coords, distance_modulus_index=0)

        results['mc_lon'].append(kernel.lon)
        results['mc_lat'].append(kernel.lat)
        results['mc_distance_modulus'].append(distance_modulus)
        results['mc_stellar_mass'].append(stellar_mass)
        results['mc_radius'].append(kernel.r_h)
        results['richness'].append(richness)
        results['log_likelihood'].append(log_likelihood)
        results['richness_lower'].append(richness_lower)
        results['richness_upper'].append(richness_upper)
        results['richness_limit'].append(richness_upper_limit)
        results['f'].append(f)
        results['stellar_mass'].append(richness * isochrone.stellarMass())

        logger.info('MC Stellar Mass = %.2f, Measured Stellar Mass = %.2f'%(stellar_mass,richness * isochrone.stellarMass()))
        if debug:
            return likelihood, richness, log_likelihood, richness_lower, richness_upper, richness_upper_limit, richness_raw, log_likelihood_raw, p, f

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

    probability_bins = numpy.arange(0., 1. + 1.e-10, 0.1)
    for ii in range(0, len(probability_bins) - 1):

        if not numpy.any(numpy.logical_and(p >= probability_bins[ii], p < probability_bins[ii + 1])):
            continue
        
        #p_array.append(0.5 * (probability_bins[ii] + probability_bins[ii + 1]))
        cut = numpy.logical_and(p >= probability_bins[ii], p < probability_bins[ii + 1])
        p_array.append(numpy.median(p[cut]))

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

    # Where does richness come from?
    
    p_array = []
    p_sum_array = []
    n_members_array = []

    for ii in range(0, len(probability_bins) - 1):
        p_array.append(0.5 * (probability_bins[ii] + probability_bins[ii + 1]))
        p_sum_array.append(numpy.sum(p[numpy.logical_and(p >= probability_bins[ii],
                                                         p < probability_bins[ii + 1])]))
        n_members_array.append(numpy.sum(numpy.all([likelihood.catalog.mc_source_id == mc_source_id,
                                                    p >= probability_bins[ii],
                                                    p < probability_bins[ii + 1]], axis=0)))
        
    pylab.figure()
    pylab.scatter(p_array, p_sum_array, marker='o', c='blue')
    pylab.scatter(p_array, n_members_array, marker='o', c='red')
    pylab.xlim(0, 1)
    #pylab.ylim(0, 1)
    pylab.xlabel('Membership Probability')
    pylab.ylabel('Membership Probability Sum')

    # Purity and completeness

    x = numpy.linspace(0., 1., 1001) 
    purity = []
    completeness = []

    purity_reconstructed = []
    completeness_reconstructed = []

    for ii in range(0, len(x)):
        cut = p > (1 - x[ii])

        if numpy.sum(cut) < 1:
            purity.append(1.)
            completeness.append(0.)
            purity_reconstructed.append(1.)
            completeness_reconstructed.append(0.)
            continue
        
        purity_cut = numpy.logical_and(cut, likelihood.catalog.mc_source_id == mc_source_id)
        completeness_cut = likelihood.catalog.mc_source_id == mc_source_id
        
        purity.append(float(numpy.sum(purity_cut)) / numpy.sum(cut))
        completeness.append(float(numpy.sum(purity_cut)) / numpy.sum(completeness_cut))

        purity_reconstructed.append(numpy.mean(p[cut]))
        completeness_reconstructed.append(numpy.sum(p[cut]) / numpy.sum(p))
        
    pylab.figure()
    pylab.plot(x, purity, c='red', label='Purity')
    pylab.plot(x, completeness, c='blue', label='Completeness')
    pylab.plot(x, purity_reconstructed, c='red', linestyle='--', label='Purity Reconstructed')
    pylab.plot(x, completeness_reconstructed, c='blue', linestyle='--', label='Completeness Reconstructed')
    pylab.xlim(0, 1)
    pylab.ylim(0, 1)
    pylab.xlabel('1 - Membership Probability')
    #pylab.ylabel('Purity or Completeness')
    pylab.legend(loc='lower center')
        
############################################################
