#!/usr/bin/env python
"""
Diagnostic plot functions
"""
__author__ = "Sidney Mau"

import os

import fitsio as fits
from astropy.coordinates import SkyCoord
from ugali.utils import healpix
from ugali.isochrone import factory as isochrone_factory
import ugali.utils.projector
import ugali.utils.plotting
import healpy

import numpy as np
from operator import add
from scipy import interpolate
from scipy.signal import argrelextrema
import scipy.ndimage
import utils

import pylab as plt
import astropy.io.fits as pyfits
import matplotlib
from matplotlib import mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable

import yaml

################################################################################

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

#COLUMNS = ['RA', 'DEC']
#COLUMNS+= ['WAVG_MAG_PSF_G', 'WAVG_MAG_PSF_R']
#COLUMNS+= ['EXTINCTION_G', 'EXTINCTION_R']
#COLUMNS+= ['WAVG_SPREAD_MODEL_R', 'SPREADERR_MODEL_R']
#COLUMNS+= ['WAVG_MAGERR_PSF_G', 'WAVG_MAGERR_PSF_R']
#COLUMNS+= ['MAGERR_PSF_G', 'MAGERR_PSF_R']

################################################################################

def star_filter(data):
    """Selects stars from the data set"""
    filter = (data[star_galaxy_classification] >= 0) & (data[star_galaxy_classification] <= 1)
    #filter = (data[spread_model_r_flag] < 0.003)
    return filter

def galaxy_filter(data):
    """Selects galaxies from the data set"""
    #filter = (data[spread_model_r_flag] > 0.005)
    filter = (data[star_galaxy_classification] > 1)
    return filter

def blue_star_filter(data):
    """Selects blue stars from the data set"""
    filter = (star_filter(data)) & ((data[mag_g_dred_flag] - data[mag_r_dred_flag]) < 0.4) # 0.2
    return filter

#filter_r = mag_r <= 23 # Filters out background galaxy contamination
#filter not working propery

################################################################################

def analysis(targ_ra, targ_dec, mod):
    """Analyze a candidate"""

    #hdisc = healpix.ang2disc(nside, targ_ra, targ_dec, 1, inclusive=True)
    #files = []
    #for i in hdisc:
    #    if os.path.exists(datadir+'/cat_hpx_'+str(i).zfill(5)+'.fits'):
    #        files.append(datadir+'/cat_hpx_'+str(i).zfill(5)+'.fits')
    #
    ##data = utils.load_infiles(files, columns=['RA', 'DEC', 'WAVG_MAG_PSF_G', 'WAVG_MAG_PSF_R', 'EXTINCTION_G', 'EXTINCTION_R', 'WAVG_SPREAD_MODEL_R', 'SPREADERR_MODEL_R', 'WAVG_MAGERR_PSF_G', 'WAVG_MAGERR_PSF_R', 'MAGERR_PSF_G', 'MAGERR_PSF_R'])
    #data = utils.load_infiles(files, columns=COLUMNS)

    pix_nside_select = ugali.utils.healpix.angToPix(nside, targ_ra, targ_dec)
    ra_select, dec_select = ugali.utils.healpix.pixToAng(nside, pix_nside_select)
    pix_nside_neighbors = np.concatenate([[pix_nside_select], healpy.get_all_neighbours(nside, pix_nside_select)])
    data_array = []
    for pix_nside in pix_nside_neighbors:
        #infile = '%s/cat_hpx_%05i.fits'%(datadir, pix_nside)
        infile = '%s/y3a2_ngmix_cm_%05i.fits'%(datadir, pix_nside)
        #infile = '%s/*_%05i.fits'%(datadir, pix_nside) # TODO: get to work
        if not os.path.exists(infile):
            continue
        reader = pyfits.open(infile)
        data_array.append(reader[1].data)
        reader.close()
    print('Assembling data...')
    data = np.concatenate(data_array)
    print('Found {} objects...').format(len(data))
    print('Loading data...')

    ## De-redden magnitudes
    #try:
    #    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_dred_flag], data[mag_r_dred_flag]])
    ##except:
    ##    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag] - data[extinction_g_flag], data[mag_r_flag] - data[extinction_r_flag]])
    #except:
    #    data = mlab.rec_append_fields(data, ['WAVG_MAG_PSF_DRED_G', 'WAVG_MAG_PSF_DRED_R'], [data[mag_g_flag], data[mag_r_flag]])
    #
    #mag_g = data['WAVG_MAG_PSF_DRED_G']
    #mag_r = data['WAVG_MAG_PSF_DRED_R']
    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    iso = isochrone_factory('Bressan2012', age=12, z=0.0001, distance_modulus=mod)

    # g_radius estimate
    filter_s = star_filter(data)

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    iso_filter = (iso.separation(mag_g, mag_r) < 0.1)

    angsep = ugali.utils.projector.angsep(targ_ra, targ_dec, data['RA'], data['DEC'])

    bins = np.linspace(0, 0.4, 21) # deg
    centers = 0.5*(bins[1:] + bins[0:-1])
    area = np.pi*(bins[1:]**2 - bins[0:-1]**2) * 60**2
    hist = np.histogram(angsep[(angsep < 0.4) & filter_s & iso_filter], bins=bins)[0] # counts

    f_interp = interpolate.interp1d(np.linspace(centers[0], centers[-1], len(hist)), hist/area, 'cubic')
    f_range = np.linspace(centers[0], centers[-1], 1000)
    f_val = f_interp(f_range)

    pairs = list(zip(f_range, f_val))

    peak = max(pairs[:len(pairs)/4], key=lambda x: x[1]) # find peak within first quarter

    def peak_index(pairs, peak):
        for i in range(len(pairs)):
            if pairs[i] == peak:
                return i

    osc = int(0.04/0.4*1000) # +/- 0.04 (rounded down) deg oscillation about local extremum
    relmin = argrelextrema(f_val, np.less, order=osc)[0]

    try:
        if len(relmin) > 0:
            #half_point = f_range[relmin[0]] # TODO rename
            i = 0
            while ((f_range[relmin[i]] <= f_range[peak_index(pairs,peak)]) & (i <= len(relmin)-1)):
                i+=1
            half_point = f_range[relmin[i]]
        elif len(relmin) == 0:
            half_peak = (peak[1] + np.mean(f_val[len(f_val)/4:]))/2. # normalized to background (after first quarter)
            #half_peak = np.mean(f_val[len(f_val)/4:])
            half_pairs = []
            for i in pairs[peak_index(pairs, peak):len(pairs)/2]: # start after peak, stay within first quarter
                if i != peak:
                    half_pairs.append((i[0], abs(i[1]-half_peak)))
            half_point = min(half_pairs, key=lambda x: x[1])[0] # deg
    except:
        half_point = 0.1 # fixed value to catch errors

    g_min = 0.5/60. # deg
    g_max = 12./60. # deg

    if half_point < g_min:
        g_radius = g_min
    elif half_point > g_max:
        g_radius = g_max
    else:
        g_radius = half_point # deg

    #c1 = SkyCoord(targ_ra, targ_dec, unit='deg') # frame is ICRS
    #nbhd = c1.separation(SkyCoord(data['RA'], data['DEC'], unit='deg')).deg < g_radius # selects objects inside the galactic radius
    angsep = ugali.utils.projector.angsep(targ_ra, targ_dec, data['RA'], data['DEC'])
    nbhd = (angsep < g_radius)

    return(data, iso, g_radius, nbhd)

def densityPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, type):
    """Stellar density plot"""

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    if type == 'stars':
        filter = star_filter(data)
        plt.title('Stellar Density')
    elif type == 'galaxies':
        filter = galaxy_filter(data)
        plt.title('Galactic Density')
    elif type == 'blue_stars':
        filter = blue_star_filter(data)
        plt.title('Blue Stellar Density')

    iso_filter = (iso.separation(mag_g, mag_r) < 0.1)

    # projection of image
    proj = ugali.utils.projector.Projector(targ_ra, targ_dec)
    x, y = proj.sphereToImage(data[filter & iso_filter]['RA'], data[filter & iso_filter]['DEC']) # filter & iso_filter

    bound = 0.5 #1.
    steps = 100.
    bins = np.linspace(-bound, bound, steps)

    signal = np.histogram2d(x, y, bins=[bins, bins])[0]

    sigma = 0.01 * (0.25 * np.arctan(0.25*g_radius*60. - 1.5) + 1.3) # full range, arctan
    
    convolution = scipy.ndimage.filters.gaussian_filter(signal, sigma/(bound/steps))
    plt.pcolormesh(bins, bins, convolution.T, cmap='Greys')

    plt.xlim(bound, -bound)
    plt.ylim(-bound, bound)
    plt.gca().set_aspect('equal')
    plt.xlabel(r'$\Delta \alpha$ (deg)')
    plt.ylabel(r'$\Delta \delta$ (deg)')

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '5%', pad=0)
    plt.colorbar(cax=cax)

def starPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd):
    """Star bin plot"""

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    filter = star_filter(data)

    iso_filter = (iso.separation(mag_g, mag_r) < 0.1)

    # projection of image
    proj = ugali.utils.projector.Projector(targ_ra, targ_dec)
    x, y = proj.sphereToImage(data[filter & iso_filter]['RA'], data[filter & iso_filter]['DEC'])

    plt.scatter(x, y, edgecolor='none', s=3, c='black')
    plt.xlim(0.2, -0.2)
    plt.ylim(-0.2, 0.2)
    plt.gca().set_aspect('equal')
    plt.xlabel(r'$\Delta \alpha$ (deg)')
    plt.ylabel(r'$\Delta \delta$ (deg)')

    plt.title('Stars')

def cmPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd, type):
    """Color-magnitude plot"""

    angsep = ugali.utils.projector.angsep(targ_ra, targ_dec, data['RA'], data['DEC'])
    annulus = (angsep > g_radius) & (angsep < 1.)

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    if type == 'stars':
        filter = star_filter(data)
        plt.title('Stellar Color-Magnitude')
    elif type == 'galaxies':
        filter = galaxy_filter(data)
        plt.title('Galactic Color-Magnitude')

    iso_filter = (iso.separation(mag_g, mag_r) < 0.1)

    # Plot background objects
    plt.scatter(mag_g[filter & annulus] - mag_r[filter & annulus], mag_g[filter & annulus], c='k', alpha=0.1, edgecolor='none', s=1)

    # Plot isochrone
    ugali.utils.plotting.drawIsochrone(iso, lw=2, label='{} Gyr, z = {}'.format(iso.age, iso.metallicity))

    # Plot objects in nbhd
    plt.scatter(mag_g[filter & nbhd] - mag_r[filter & nbhd], mag_g[filter & nbhd], c='g', s=5, label='r < {:.3f}$^\circ$'.format(g_radius))

    # Plot objects in nbhd and near isochrone
    plt.scatter(mag_g[filter & nbhd & iso_filter] - mag_r[filter & nbhd & iso_filter], mag_g[filter & nbhd & iso_filter], c='r', s=5, label='$\Delta$CM < 0.1')

    plt.axis([-0.5, 1, 16, 24])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1./4.)
    plt.legend(loc='upper left')
    plt.xlabel('g-r (mag)')
    plt.ylabel('g (mag)')

def hessPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd):
    """Hess plot"""

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    filter_s = star_filter(data)

    plt.title('Hess')

    c1 = SkyCoord(targ_ra, targ_dec, unit='deg')

    r_near = 2.*g_radius # annulus begins at 3*g_radius away from centroid
    r_far = np.sqrt(5.)*g_radius # annulus has same area as inner area

    inner = (c1.separation(SkyCoord(data['RA'], data['DEC'], unit='deg')).deg < g_radius)
    outer = (c1.separation(SkyCoord(data['RA'], data['DEC'], unit='deg')).deg > r_near) & (c1.separation(SkyCoord(data['RA'], data['DEC'], unit='deg')).deg < r_far)

    xbins = np.arange(-0.5, 1.1, 0.1)
    ybins = np.arange(16., 24.5, 0.5)

    foreground = np.histogram2d(mag_g[inner & filter_s] - mag_r[inner & filter_s], mag_g[inner & filter_s], bins=[xbins, ybins])
    background = np.histogram2d(mag_g[outer & filter_s] - mag_r[outer & filter_s], mag_g[outer & filter_s], bins=[xbins, ybins])

    fg = foreground[0].T
    bg = background[0].T

    fg_abs = np.absolute(fg)
    bg_abs = np.absolute(bg)

    mask_abs = fg_abs + bg_abs
    mask_abs[mask_abs == 0.] = np.nan # mask signficiant zeroes

    signal = fg - bg
    signal = np.ma.array(signal, mask=np.isnan(mask_abs)) # mask nan

    cmap = matplotlib.cm.viridis
    cmap.set_bad('w', 1.)
    plt.pcolormesh(xbins, ybins, signal, cmap=cmap)

    plt.colorbar()

    ugali.utils.plotting.drawIsochrone(iso, lw=2, c='k', zorder=10, label='Isocrhone')

    plt.axis([-0.5, 1.0, 16, 24])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1./4.)
    plt.xlabel('g-r (mag)')
    plt.ylabel('g (mag)')

    #ax = plt.gca()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size = '5%', pad=0)
    #plt.colorbar(cax=cax)

def radialPlot(targ_ra, targ_dec, data, iso, g_radius, nbhd):
    """Radial distribution plot"""

    mag_g = data[mag_g_dred_flag]
    mag_r = data[mag_r_dred_flag]

    filter_s = star_filter(data)
    filter_g = galaxy_filter(data)

    plt.title('Radial Distribution')

    angsep = ugali.utils.projector.angsep(targ_ra, targ_dec, data['RA'], data['DEC'])

    # Isochrone filtered/unfiltered
    iso_seln_f = (iso.separation(mag_g, mag_r) < 0.1)
    iso_seln_u = (iso.separation(mag_g, mag_r) >= 0.1)

    bins = np.linspace(0, 0.4, 21) # deg
    centers = 0.5*(bins[1:] + bins[0:-1])
    area = np.pi*(bins[1:]**2 - bins[0:-1]**2) * 60**2

    def interp_values(type, seln):
        if type == 'stars':
            filter = star_filter(data)
        elif type == 'galaxies':
            filter = galaxy_filter(data)

        if seln == 'f':
            iso_filter = iso_seln_f
        elif seln == 'u':
            iso_filter = iso_seln_u

        hist = np.histogram(angsep[(angsep < 0.4) & filter & iso_filter], bins=bins)[0] # counts

        f_interp = interpolate.interp1d(np.linspace(centers[0], centers[-1], len(hist)), hist/area, 'cubic')
        f_range = np.linspace(centers[0], centers[-1], 1000)
        f_val = f_interp(f_range)

        return(f_range, f_val)

    def value_errors(type, seln):
        if type == 'stars':
            filter = star_filter(data)
        elif type == 'galaxies':
            filter = galaxy_filter(data)
        if seln == 'f':
            iso_filter = iso_seln_f
        elif seln == 'u':
            iso_filter = iso_seln_u

        hist = np.histogram(angsep[(angsep < 0.4) & filter & iso_filter], bins=bins)[0] # counts

        val = hist/area
        yerr = np.sqrt(hist)/area

        return(val, yerr)

    f_range, f_val = interp_values('stars', 'f')
    pairs = list(zip(f_range, f_val))
    peak = max(pairs[:len(pairs)/4], key=lambda x: x[1]) # find peak within first quarter
    def peak_index(pairs, peak):
        for i in range(len(pairs)):
            if pairs[i] == peak:
                return i

    plt.axvline(x=f_range[peak_index(pairs,peak)], color='m', label='peak')

    plt.axvline(x=g_radius, color='r', label='g_radius')

    f_range, f_val = interp_values('galaxies', 'f')
    plt.plot(f_range, f_val, '-g', label='Filtered Galaxies')

    f_range, f_val = interp_values('stars', 'u')
    plt.plot(f_range, f_val, '-k', alpha=0.25, label='Unfiltered Stars')

    val, y_err = value_errors('stars', 'f')
    plt.plot(centers, val, '.b')
    plt.errorbar(centers, val, yerr=y_err, fmt='none', ecolor='b', elinewidth=1, capsize=5)

    f_range, f_val = interp_values('stars', 'f')
    plt.plot(f_range, f_val, '-b', label='Filtered Stars')

    ymax = plt.ylim()[1]
    plt.annotate(r'$\approx %0.1f$'+str(round(g_radius, 3))+'$^\circ$', (g_radius*1.1, ymax/50.), color='red', bbox=dict(boxstyle='round,pad=0.0', fc='white', alpha=0.75, ec='white', lw=0))
    plt.xlim(bins[0], bins[-1])
    plt.ylim(0., ymax)
    plt.legend(loc='upper right')
    plt.xlabel('Angular Separation (deg)')
    plt.ylabel('Denisty (arcmin$^{-2})$')

#def maglim_plot(targ_ra, targ_dec, data, iso, band):
#    """Maglim plots"""
#
#    reso = 0.5
#    xsize = 2.*60./reso
#
#    if band == 'g':
#        reader = pyfits.open(maglim_g)
#        m_maglim_g = reader[1].data.field('I').flatten()
#        reader.close()
#        m_maglim_g[np.isnan(m_maglim_g)] = healpy.UNSEEN
#        #healpy.gnomview(m_maglim_g, fig='summary', rot=(targ_ra, targ_dec, 0.), reso=reso, xsize=xsize, title='maglim g (S/N =10)', sub=(3, 4, 11))
#        healpy.gnomview(m_maglim_g, rot=(targ_ra, targ_dec, 0.), reso=reso, xsize=xsize, title='maglim g (S/N =10)', sub=(3, 4, 8))
#    elif band == 'r':
#        reader = pyfits.open(maglim_r)
#        m_maglim_r = reader[1].data.field('I').flatten()
#        reader.close()
#        m_maglim_r[np.isnan(m_maglim_r)] = healpy.UNSEEN
#        healpy.gnomview(m_maglim_r, rot=(targ_ra, targ_dec, 0.), reso=reso, xsize=xsize, title='maglim r (S/N =10)', sub=(3, 4, 12))
