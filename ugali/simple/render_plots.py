#!/usr/bin/env python
"""
Arrange and produce plots
"""
__author__ = "Sidney Mau"

# Set the backend first!
import matplotlib
matplotlib.use('Agg')

import os
import yaml
import numpy as np
import numpy
import pylab as plt
from matplotlib import gridspec
from multiprocessing import Pool

import ugali.utils.projector
import ugali.candidate.associate
import diagnostic_plots

print(matplotlib.get_backend())

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

nside = cfg['nside']
datadir = cfg['datadir']
candidate_list = cfg['candidate_list']

save_dir = os.path.join(os.getcwd(), cfg['save_dir'])
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

candidate_list = np.genfromtxt(candidate_list, delimiter=',', names=['sig', 'ra', 'dec', 'distance_modulus', 'r'])[1:] #, 'association', 'association_angsep'])[1:]
candidate_list = candidate_list[candidate_list['sig'] > 5.5] # only plot hotspots of sufficent significance
# try with sig > 10., 5.5, etc...

#################################################################

print('{} candidates found...').format(len(candidate_list))

def renderPlot(candidate):
    """
    Make plots
    """

    fig = plt.figure(figsize=(20, 17))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    gs = gridspec.GridSpec(3, 3)

    print('Analyzing candidate {}/{}...').format(candidate+1, len(candidate_list))

    #sig, targ_ra, targ_dec, mod, r = candidate_list[candidate]
    sig = round(candidate_list[candidate]['sig'], 2)
    targ_ra = round(candidate_list[candidate]['ra'], 2)
    targ_dec = round(candidate_list[candidate]['dec'], 2)
    mod = candidate_list[candidate]['distance_modulus']
    data, iso, g_radius, nbhd = diagnostic_plots.analysis(targ_ra, targ_dec, mod)

    print('Making diagnostic plots for (RA, Dec) = ({}, {})...').format(targ_ra, targ_dec)

    fig.add_subplot(gs[0,0])
    diagnostic_plots.densityPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, 'stars')

    fig.add_subplot(gs[1,0])
    diagnostic_plots.densityPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, 'galaxies')

    fig.add_subplot(gs[2,0])
    diagnostic_plots.densityPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, 'blue_stars')

    fig.add_subplot(gs[0,1])
    diagnostic_plots.cmPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, 'stars')

    fig.add_subplot(gs[1,1])
    diagnostic_plots.cmPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, 'galaxies')

    fig.add_subplot(gs[0,2])
    diagnostic_plots.hessPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd)

    fig.add_subplot(gs[1,2])
    diagnostic_plots.starPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd)

    fig.add_subplot(gs[2,1:3])
    diagnostic_plots.radialPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd)

    #try:
    # Check for possible associations
    glon_peak, glat_peak = ugali.utils.projector.celToGal(targ_ra, targ_dec)
    catalog_array = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14','ExtraDwarfs','ExtraClusters']
    catalog = ugali.candidate.associate.SourceCatalog()
    for catalog_name in catalog_array:
        catalog += ugali.candidate.associate.catalogFactory(catalog_name)

    idx1, idx2, sep = catalog.match(glon_peak, glat_peak, tol=0.5, nnearest=1)
    match = catalog[idx2]
    if len(match) > 0:
        association_string = '{} at {:.3f} deg'.format(match[0]['name'], float(sep))
    else:
        association_string = 'No association within 0.5 deg'
    #except:
    #    association_string = 'Association search error'

    plt.suptitle('{}\n'.format(association_string) + r'($\alpha$, $\delta$, $\mu$, $\sigma$) = ({}, {}, {}, {})'.format(targ_ra, targ_dec, mod, sig), fontsize=24)

    file_name = 'candidate_{}_{}'.format(targ_ra, targ_dec)
    plt.savefig(save_dir+'/'+file_name+'.png',  bbox_inches='tight')
    plt.close()

#renderPlot(0)
if __name__ == '__main__':
    pool = Pool(20)
    index = list(range(len(candidate_list)))
    pool.map(renderPlot, index)
