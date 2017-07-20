#!/usr/bin/env python

# Set the backend first!
import matplotlib
matplotlib.use('Agg')
import pylab

import sys
import os
import glob
import numpy
import pyfits
import healpy
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.stats
import scipy.spatial

import ugali.utils.healpix
import ugali.utils.projector
import ugali.analysis.isochrone
import ugali.utils.plotting
import ugali.candidate.associate

print matplotlib.get_backend()

#pylab.ioff()

############################################################

def cutIsochronePath(g, r, g_err, r_err, isochrone, radius=0.1, return_all=False):
    """
    Cut to identify objects within isochrone cookie-cutter.
    """
    if numpy.all(isochrone.stage == 'Main'):
        # Dotter case
        index_transition = len(isochrone.stage)
    else:
        # Other cases
        index_transition = numpy.nonzero(isochrone.stage > 3)[0][0] + 1    
    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]
    
    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = numpy.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < numpy.sqrt(0.1**2 + r_err**2 + g_err**2))

    cut = numpy.logical_or(cut_1, cut_2)
    
    #print f_isochrone(17.)
    #print numpy.sum(cut)
    #pylab.figure('isochrone_cut')
    #pylab.scatter(g[0:1000], g[0:1000] - r[0:1000], edgecolor='none', s=1)
    #pylab.scatter(g[cut][0:1000], g[cut][0:1000] - r[cut][0:1000], edgecolor='none', s=1)
    #pylab.scatter(g[~cut][0:1000], g[~cut][0:1000] - r[~cut][0:1000], edgecolor='none', s=1)
    #pylab.plot(mag_1_rgb, mag_1_rgb - mag_2_rgb)
    #raw_input('hold up')

    mag_bins = numpy.arange(17., 24.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = numpy.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = numpy.median(numpy.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    if return_all:
        return cut, mag_centers[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) + magerr)[f_isochrone(mag_centers) < 100], (f_isochrone(mag_centers) - magerr)[f_isochrone(mag_centers) < 100]
    else:
        return cut

############################################################

def circle(x, y, r):
    phi = numpy.linspace(0, 2 * numpy.pi, 1000)
    return x + (r * numpy.cos(phi)), y + (r * numpy.sin(phi))

############################################################

try:
    ra_select, dec_select = float(sys.argv[1]), float(sys.argv[2])
except:
    sys.exit('ERROR! Coordinates not given in correct format.')

print 'Search coordinates: (RA, Dec) = (%.2f, %.2f)'%(ra_select, dec_select)

# Now cut for a single pixel
#ra_select, dec_select = 359.04, -59.63 # Tuc III
#ra_select, dec_select = 0.668, -60.888 # Tuc IV
#ra_select, dec_select = 82.85, -28.03 # Col I
#ra_select, dec_select = 38.00, -34.00 # Fornax
#ra_select, dec_select = 331.06, -46.47 # Gru II
#ra_select, dec_select = 354.37, -63.26
pix_16_select = ugali.utils.healpix.angToPix(16, ra_select, dec_select)
ra_select, dec_select = ugali.utils.healpix.pixToAng(16, pix_16_select)
pix_16_neighbors = numpy.concatenate([[pix_16_select], healpy.get_all_neighbours(16, pix_16_select)])

############################################################

#datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/hpx/cat'
datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/v6/hpx' # v6
data_array = []
for pix_16 in pix_16_neighbors:
    infile = '%s/cat_hpx_%05i.fits'%(datadir, pix_16)
    if not os.path.exists(infile):
        continue
    reader = pyfits.open(infile)
    data_array.append(reader[1].data)
    reader.close()
print 'Assembling data...'
data = numpy.concatenate(data_array)

print 'Found %i objects...'%(len(data))

############################################################

print 'Applying cuts...'
cut = (data['FLAGS_G'] < 4) & (data['FLAGS_R'] < 4) \
      & (data['QSLR_FLAG_G'] == 0) & (data['QSLR_FLAG_R'] == 0) \
      & (data['WAVG_MAG_PSF_G'] < 24.0) \
      & ((data['WAVG_MAG_PSF_G'] - data['WAVG_MAG_PSF_R']) < 1.) \
      & (numpy.fabs(data['WAVG_SPREAD_MODEL_R']) < 0.003 + data['SPREADERR_MODEL_R'])

cut_gal = (data['FLAGS_G'] < 4) & (data['FLAGS_R'] < 4) \
          & (data['QSLR_FLAG_G'] == 0) & (data['QSLR_FLAG_R'] == 0) \
          & (data['WAVG_MAG_PSF_G'] < 24.0) \
          & ((data['WAVG_MAG_PSF_G'] - data['WAVG_MAG_PSF_R']) < 1.) \
          & (numpy.fabs(data['WAVG_SPREAD_MODEL_R']) > 0.003 + data['SPREADERR_MODEL_R'])

# Separate selections for stars and galaxies
data_gal = data[cut_gal]
data = data[cut]

print '%i star-like objects in ROI...'%(len(data))
print '%i galaxy-like objects in ROI...'%(len(data_gal))

############################################################

def searchByDistance(data, distance_modulus, ra_select, dec_select, magnitude_threshold=23.):
    """
    Idea: 
    Send a data extension that goes to faint magnitudes, e.g., g < 24.
    Use the whole region to identify hotspots using a slightly brighter 
    magnitude threshold, e.g., g < 23, so not susceptible to variations 
    in depth. Then compute the local field density using a small annulus 
    around each individual hotspot, e.g., radius 0.3 to 0.5 deg.
    """

    print 'Distance = %.1f kpc (m-M = %.1f)'%(ugali.utils.projector.distanceModulusToDistance(distance_modulus), distance_modulus)

    #dirname = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/isochrones/v2/' # KCB
    #dirname = '/project/kicp/bechtol/des/mw_substructure/isochrones/v2/'
    #iso = ugali.analysis.isochrone.isochroneFactory('Padova', hb_spread=0, dirname=dirname)
    dirname = '/project/kicp/bechtol/des/mw_substructure/isochrones/dotter/dotter_v4/'
    iso = ugali.analysis.isochrone.isochroneFactory('Dotter', hb_spread=0, dirname=dirname)
    iso.age = 12.
    iso.metallicity = 0.0001
    #iso.distance_modulus = 21.3 # Col I
    #iso.distance_modulus = 18.6 # Gru II
    iso.distance_modulus = distance_modulus

    cut = cutIsochronePath(data['WAVG_MAG_PSF_G'], data['WAVG_MAG_PSF_R'], data['WAVG_MAGERR_PSF_G'], data['WAVG_MAGERR_PSF_R'], iso, radius=0.1)
    data = data[cut]
    cut_magnitude_threshold = (data['WAVG_MAG_PSF_G'] < magnitude_threshold)

    print '%i objects left after isochrone cut...'%(len(data))

    ###

    proj = ugali.utils.projector.Projector(ra_select, dec_select)
    x, y = proj.sphereToImage(data['RA'][cut_magnitude_threshold], data['DEC'][cut_magnitude_threshold]) # Trimmed magnitude range for hotspot finding
    x_full, y_full = proj.sphereToImage(data['RA'], data['DEC']) # Full magnitude range for significance evaluation
    delta_x = 0.01
    area = delta_x**2
    smoothing = 2. / 60. # Was 3 arcmin
    bins = numpy.arange(-8., 8. + 1.e-10, delta_x)
    centers = 0.5 * (bins[0: -1] + bins[1:])
    yy, xx = numpy.meshgrid(centers, centers)

    h = numpy.histogram2d(x, y, bins=[bins, bins])[0]

    #pylab.figure('raw')
    #pylab.imshow(h.T, interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower', cmap='binary')
    #pylab.xlim([8, -8])
    #pylab.ylim([-8, 8])
    #pylab.colorbar()

    #npix_256 = healpy.nside2npix(256)
    #pix_256 = ugali.utils.healpix.angToPix(256, data['RA'], data['DEC'])
    #m_256 = numpy.histogram(pix_256, numpy.arange(npix_256 + 1))[0].astype(float)
    #healpy.mollview(m_256)

    #pylab.figure()
    #pylab.scatter(x, y, edgecolor='none', s=1)
    #pylab.xlim([5, -5])
    #pylab.ylim([-5, 5])

    h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)

    cut_goodcoverage = (data['NEPOCHS_G'][cut_magnitude_threshold] >= 2) & (data['NEPOCHS_R'][cut_magnitude_threshold] >= 2)

    delta_x_coverage = 0.1
    area_coverage = (delta_x_coverage)**2
    bins_coverage = numpy.arange(-5., 5. + 1.e-10, delta_x_coverage)
    h_coverage = numpy.histogram2d(x, y, bins=[bins_coverage, bins_coverage])[0]
    h_goodcoverage = numpy.histogram2d(x[cut_goodcoverage], y[cut_goodcoverage], bins=[bins_coverage, bins_coverage])[0]

    n_goodcoverage = h_coverage[h_goodcoverage > 0].flatten()
    pylab.figure('poisson')
    pylab.clf()
    pylab.hist(n_goodcoverage, bins=numpy.arange(20) - 0.5, normed=True)
    pylab.scatter(numpy.arange(20), scipy.stats.poisson.pmf(numpy.arange(20), mu=numpy.median(n_goodcoverage)), c='red', zorder=10)

    #print numpy.histogram(n_goodcoverage, bins=numpy.arange(20), normed=True)
    #print scipy.stats.poisson.pmf(numpy.arange(20), mu=numpy.median(n_goodcoverage))
    #print 

    #characteristic_density = numpy.mean(n_goodcoverage) / area_coverage # per square degree
    characteristic_density = numpy.median(n_goodcoverage) / area_coverage # per square degree

    vmax = min(3. * characteristic_density * area, numpy.max(h_g))

    pylab.figure('smooth')
    pylab.clf()
    pylab.imshow(h_g.T,
                 interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower', cmap='gist_heat', vmax=vmax)
    pylab.colorbar()
    pylab.xlim([8, -8])
    pylab.ylim([-8, 8])
    pylab.xlabel(r'$\Delta$ RA (deg)')
    pylab.ylabel(r'$\Delta$ Dec (deg)')
    pylab.title('(RA, Dec, mu) = (%.2f, %.2f, %.2f)'%(ra_select, dec_select, distance_modulus))

    factor_array = numpy.arange(1., 5., 0.05)
    rara, decdec = proj.imageToSphere(xx.flatten(), yy.flatten())
    cutcut = (ugali.utils.healpix.angToPix(16, rara, decdec) == pix_16_select).reshape(xx.shape)
    threshold_density = 5 * characteristic_density * area
    for factor in factor_array:
        h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > (area * characteristic_density * factor))
        #print 'factor', factor, n_region, n_region < 10
        if n_region < 10:
            threshold_density = area * characteristic_density * factor
            break
    
    h_region, n_region = scipy.ndimage.measurements.label((h_g * cutcut) > threshold_density)
    #pylab.figure('regions')
    #pylab.imshow(h_region.T,
    #             interpolation='nearest', extent=[-8, 8, -8, 8], origin='lower')
    #pylab.colorbar()
    #pylab.xlim([8, -8])
    #pylab.ylim([-8, 8])

    ra_peak_array = []
    dec_peak_array = []
    r_peak_array = []
    sig_peak_array = []
    distance_modulus_array = []

    pylab.figure('sig')
    pylab.clf()
    for index in range(1, n_region + 1):
        index_peak = numpy.argmax(h_g * (h_region == index))
        x_peak, y_peak = xx.flatten()[index_peak], yy.flatten()[index_peak]
        #print index, numpy.max(h_g * (h_region == index))
        #pylab.figure('regions')
        #pylab.scatter(x_peak, y_peak, marker='x', c='white')

        # Compute the local characteristic density using the full magnitude range
        area_field = numpy.pi * (0.5**2 - 0.3**2)
        angsep_peak = numpy.sqrt((x_full - x_peak)**2 + (y_full - y_peak)**2)
        n_field = numpy.sum((angsep_peak > 0.3) & (angsep_peak < 0.5))
        characteristic_density_local = n_field / area_field

        # If not good azimuthal coverage, revert
        cut_annulus = (angsep_peak > 0.3) & (angsep_peak < 0.5) 
        phi = numpy.degrees(numpy.arctan2(y_full[cut_annulus] - y_peak, x_full[cut_annulus] - x_peak))
        h = numpy.histogram(phi, bins=numpy.linspace(-180., 180., 13))[0]
        if numpy.sum(h > 0) < 10 or numpy.sum(h > 0.5 * numpy.median(h)) < 10:
            angsep_peak = numpy.sqrt((x - x_peak)**2 + (y - y_peak)**2)
            characteristic_density_local = characteristic_density

        size_array = numpy.arange(0.01, 0.3, 0.01)
        #size_array = numpy.array([0.04])
        sig_array = numpy.tile(0., len(size_array))
        for ii in range(0, len(size_array)):
            n_peak = numpy.sum(angsep_peak < size_array[ii])
            n_model = characteristic_density_local * (numpy.pi * size_array[ii]**2)
            sig_array[ii] = scipy.stats.norm.isf(scipy.stats.poisson.sf(n_peak, n_model))
            if sig_array[ii] > 25:
                sig_array[ii] = 25.

        pylab.figure('sig')
        pylab.plot(size_array, sig_array)

        ra_peak, dec_peak = proj.imageToSphere(x_peak, y_peak)
        r_peak = size_array[numpy.argmax(sig_array)]
        if numpy.max(sig_array) == 25.:
            r_peak = 0.5

        print 'Candidate:', x_peak, y_peak, r_peak, numpy.max(sig_array), ra_peak, dec_peak
        if numpy.max(sig_array) > 5.:
            x_circle, y_circle = circle(x_peak, y_peak, r_peak)
            pylab.figure('smooth')
            pylab.plot(x_circle, y_circle, c='gray')
            pylab.text(x_peak - r_peak, y_peak + r_peak, r'%.2f $\sigma$'%(numpy.max(sig_array)), color='gray')
    
            ra_peak_array.append(ra_peak)
            dec_peak_array.append(dec_peak)
            r_peak_array.append(r_peak)
            sig_peak_array.append(numpy.max(sig_array))
            distance_modulus_array.append(distance_modulus)

        #raw_input('WAIT')

    return ra_peak_array, dec_peak_array, r_peak_array, sig_peak_array, distance_modulus_array

############################################################

def diagnostic(data, data_gal, ra_peak, dec_peak, r_peak, sig_peak, distance_modulus, age=12., metallicity=0.0001, savedir=None):
    #dirname = '/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/isochrones/v2/' # KCB
    #dirname = '/project/kicp/bechtol/des/mw_substructure/isochrones/v2/'
    #iso = ugali.analysis.isochrone.isochroneFactory('Padova', hb_spread=0, dirname=dirname)
    dirname = '/project/kicp/bechtol/des/mw_substructure/isochrones/dotter/dotter_v4/'
    iso = ugali.analysis.isochrone.isochroneFactory('Dotter', hb_spread=0, dirname=dirname)
    iso.age = age
    iso.metallicity = metallicity
    iso.distance_modulus = distance_modulus

    dirname_alt = '/project/kicp/bechtol/des/mw_substructure/isochrones/v2/'
    iso_alt = ugali.analysis.isochrone.isochroneFactory('Padova', hb_spread=0, dirname=dirname_alt)
    iso_alt.age = age
    iso_alt.metallicity = metallicity
    iso_alt.distance_modulus = distance_modulus

    cut_iso, g_iso, gr_iso_min, gr_iso_max = cutIsochronePath(data['WAVG_MAG_PSF_G'], 
                                                              data['WAVG_MAG_PSF_R'], 
                                                              data['WAVG_MAGERR_PSF_G'], 
                                                              data['WAVG_MAGERR_PSF_R'], 
                                                              iso, 
                                                              radius=0.1, 
                                                              return_all=True)
    
    cut_iso_gal = cutIsochronePath(data_gal['MAG_PSF_G'], 
                                   data_gal['MAG_PSF_R'], 
                                   data_gal['MAGERR_PSF_G'], 
                                   data_gal['MAGERR_PSF_R'], 
                                   iso, 
                                   radius=0.1, 
                                   return_all=False)
    
    proj = ugali.utils.projector.Projector(ra_peak, dec_peak)
    x, y = proj.sphereToImage(data['RA'][cut_iso], data['DEC'][cut_iso])
    x_gal, y_gal = proj.sphereToImage(data_gal['RA'][cut_iso_gal], data_gal['DEC'][cut_iso_gal])

    angsep = ugali.utils.projector.angsep(ra_peak, dec_peak, data['RA'], data['DEC'])
    cut_inner = (angsep < r_peak)
    cut_annulus = (angsep > 0.5) & (angsep < 1.)

    angsep_gal = ugali.utils.projector.angsep(ra_peak, dec_peak, data_gal['RA'], data_gal['DEC'])
    cut_inner_gal = (angsep_gal < r_peak)
    cut_annulus_gal = (angsep_gal > 0.5) & (angsep_gal < 1.)

    ##########

    # Check for possible associations
    glon_peak, glat_peak = ugali.utils.projector.celToGal(ra_peak, dec_peak)
    catalog_array = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14','ExtraDwarfs','ExtraClusters']
    catalog = ugali.candidate.associate.SourceCatalog()
    for catalog_name in catalog_array:
        catalog += ugali.candidate.associate.catalogFactory(catalog_name)

    idx1, idx2, sep = catalog.match(glon_peak, glat_peak, tol=0.5, nnearest=1)
    match = catalog[idx2]
    if len(match) > 0:
        association_string = '; %s at %.3f deg'%(match[0]['name'], sep)
    else:
        association_string = '; no association within 0.5 deg'

    ##########

    fig = pylab.figure('summary', figsize=(18, 15))
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    pylab.clf()

    pylab.suptitle(r'(RA, Dec, m - M) = (%.2f, %.2f, %.2f); Significance = %.2f $\sigma$%s'%(ra_peak, dec_peak, distance_modulus, sig_peak, association_string), fontsize=20)

    pylab.subplot(3, 3, 1)
    delta_x = 0.01
    smoothing = 0.03
    bins = numpy.arange(-1., 1. + 1.e-10, delta_x)
    h = numpy.histogram2d(x, y, bins=[bins, bins])[0]
    h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)
    pylab.imshow(h_g.T, interpolation='nearest', extent=[-1., 1., 1, -1], cmap='binary', vmin=0.5 * numpy.median(h_g))
    #pylab.xlim(0.5, -0.5)
    #pylab.ylim(-0.5, 0.5)
    pylab.xlim(1., -1.)
    pylab.ylim(-1., 1.)
    pylab.xlabel(r'$\Delta$ RA (deg)')
    pylab.ylabel(r'$\Delta$ Dec (deg)')
    pylab.title('Stars')

    pylab.subplot(3, 3, 4)
    h = numpy.histogram2d(x_gal, y_gal, bins=[bins, bins])[0]
    h_g = scipy.ndimage.filters.gaussian_filter(h, smoothing / delta_x)
    pylab.imshow(h_g.T, interpolation='nearest', extent=[-1., 1., 1, -1], cmap='binary', vmin=0.5 * numpy.median(h_g))
    #pylab.xlim(0.5, -0.5)
    #pylab.ylim(-0.5, 0.5)
    pylab.xlim(1., -1.)
    pylab.ylim(-1., 1.)
    pylab.xlabel(r'$\Delta$ RA (deg)')
    pylab.ylabel(r'$\Delta$ Dec (deg)')
    pylab.title('Galaxies')

    ##########

    pylab.subplot(3, 3, 2)
    pylab.scatter(data['WAVG_MAG_PSF_G'][cut_annulus] - data['WAVG_MAG_PSF_R'][cut_annulus], data['WAVG_MAG_PSF_G'][cut_annulus], 
                  c='0.', alpha=0.1, edgecolor='none', s=3)
    pylab.scatter(data['WAVG_MAG_PSF_G'][cut_inner] - data['WAVG_MAG_PSF_R'][cut_inner], data['WAVG_MAG_PSF_G'][cut_inner], 
                  c='red', edgecolor='none', label='r < %.2f deg'%(r_peak))
    ugali.utils.plotting.drawIsochrone(iso, lw=2, c='blue', zorder=10)
    pylab.plot(gr_iso_max, g_iso, lw=1, c='blue', zorder=11, label='%.1f Gyr, z = %.4f'%(age, metallicity))
    pylab.plot(gr_iso_min, g_iso, lw=1, c='blue', zorder=12)
    ugali.utils.plotting.drawIsochrone(iso_alt, lw=2, c='magenta', zorder=9)
    pylab.xlim(-0.5, 1.)
    pylab.ylim(24., 16.)
    pylab.xlabel('g - r (mag)')
    pylab.ylabel('g (mag)')
    pylab.title('Stars')
    pylab.legend(loc='upper left', frameon=True, scatterpoints=1, fontsize=10)

    pylab.subplot(3, 3, 5)
    pylab.scatter(data_gal['WAVG_MAG_PSF_G'][cut_annulus_gal] - data_gal['WAVG_MAG_PSF_R'][cut_annulus_gal], data_gal['WAVG_MAG_PSF_G'][cut_annulus_gal], 
                  c='0.', alpha=0.1, edgecolor='none', s=3)
    pylab.scatter(data_gal['WAVG_MAG_PSF_G'][cut_inner_gal] - data_gal['WAVG_MAG_PSF_R'][cut_inner_gal], data_gal['WAVG_MAG_PSF_G'][cut_inner_gal], 
                  c='red', edgecolor='none', label='r < %.2f deg'%(r_peak))
    ugali.utils.plotting.drawIsochrone(iso, lw=2, c='blue', zorder=10)
    pylab.plot(gr_iso_max, g_iso, lw=1, c='blue', zorder=11, label='%.1f Gyr, z = %.4f'%(age, metallicity))
    pylab.plot(gr_iso_min, g_iso, lw=1, c='blue', zorder=12)
    ugali.utils.plotting.drawIsochrone(iso_alt, lw=2, c='magenta', zorder=9)
    pylab.xlim(-0.5, 1.)
    pylab.ylim(24., 16.)
    pylab.xlabel('g - r (mag)')
    pylab.ylabel('g (mag)')
    pylab.title('Galaxies')
    pylab.legend(loc='upper left', frameon=True, scatterpoints=1, fontsize=10)

    ##########

    bins = numpy.arange(0, 0.4 + 1.e-10, 0.04)
    centers = 0.5 * (bins[1:] + bins[0:-1])
    area = numpy.pi * (bins[1:]**2 - bins[0:-1]**2) * 60**2

    bins_narrow = numpy.arange(0, 0.4 + 1.e-10, 0.02)
    centers_narrow = 0.5 * (bins_narrow[1:] + bins_narrow[0:-1])
    area_narrow = numpy.pi * (bins_narrow[1:]**2 - bins_narrow[0:-1]**2) * 60**2

    h = numpy.histogram(angsep[(angsep < 0.4) & cut_iso], bins=bins)[0]
    h_out = numpy.histogram(angsep[(angsep < 0.4) & (~cut_iso)], bins=bins)[0]
    h_narrow = numpy.histogram(angsep[(angsep < 0.4) & cut_iso], bins=bins_narrow)[0]
    
    pylab.subplot(3, 3, 3)
    pylab.plot(centers, h_out / area, c='gray', label='Iso. Out')
    pylab.errorbar(centers, h_out / area, yerr=(numpy.sqrt(h_out) / area), ecolor='gray', c='gray')
    pylab.scatter(centers, h_out / area, edgecolor='none', c='gray')
    pylab.plot(centers, h / area, c='blue', label='Iso. In')
    pylab.errorbar(centers, h / area, yerr=(numpy.sqrt(h) / area), ecolor='blue', c='blue')
    pylab.scatter(centers, h / area, edgecolor='none', c='blue')
    pylab.plot(centers_narrow, h_narrow / area_narrow, c='magenta', label='Iso. In')
    pylab.errorbar(centers_narrow, h_narrow / area_narrow, yerr=(numpy.sqrt(h_narrow) / area_narrow), ecolor='magenta', c='magenta')
    pylab.scatter(centers_narrow, h_narrow / area_narrow, edgecolor='none', c='magenta')

    pylab.xlabel('Angsep (deg)')
    pylab.ylabel('Stars per Square Arcmin')
    pylab.xlim(0., 0.4)
    ymax = pylab.ylim()[1]
    pylab.ylim(0, ymax)
    pylab.legend(loc='upper right', frameon=True)
    pylab.title('Stars')
    
    h = numpy.histogram(angsep_gal[(angsep_gal < 0.4) & cut_iso_gal], bins=bins)[0]
    h_out = numpy.histogram(angsep_gal[(angsep_gal < 0.4) & (~cut_iso_gal)], bins=bins)[0]
    pylab.subplot(3, 3, 6)
    pylab.plot(centers, h_out / area, c='gray', label='Iso. Out')
    pylab.errorbar(centers, h_out / area, yerr=(numpy.sqrt(h_out) / area), ecolor='gray', c='gray')
    pylab.scatter(centers, h_out / area, edgecolor='none', c='gray')
    pylab.plot(centers, h / area, c='blue', label='Iso. In')
    pylab.errorbar(centers, h / area, yerr=(numpy.sqrt(h) / area), ecolor='blue', c='blue')
    pylab.scatter(centers, h / area, edgecolor='none', c='blue')
    pylab.xlabel('Angsep (deg)')
    pylab.ylabel('Galaxies per Square Arcmin')
    pylab.xlim(0., 0.4)
    ymax = pylab.ylim()[1]
    pylab.ylim(0, ymax)
    pylab.legend(loc='upper right', frameon=True)
    pylab.title('Galaxies')

    ##########

    pylab.subplot(3, 3, 7)
    pylab.scatter(x, y, edgecolor='none', s=3, c='black')
    pylab.xlim(0.2, -0.2)
    pylab.ylim(-0.2, 0.2)
    pylab.xlabel(r'$\Delta$ RA (deg)')
    pylab.ylabel(r'$\Delta$ Dec (deg)')
    pylab.title('Stars')

    ##########

    reader = pyfits.open('/project/kicp/bechtol/des/mw_substructure/y2n/data/maglim/v6/y2q1_maglim_g_n1024_ring.fits.gz')
    m_maglim_g = reader[1].data.field('I').flatten()
    reader.close()
    m_maglim_g[numpy.isnan(m_maglim_g)] = healpy.UNSEEN

    reader = pyfits.open('/project/kicp/bechtol/des/mw_substructure/y2n/data/maglim/v6/y2q1_maglim_r_n1024_ring.fits.gz')
    m_maglim_r = reader[1].data.field('I').flatten()
    reader.close()
    m_maglim_r[numpy.isnan(m_maglim_r)] = healpy.UNSEEN

    reso = 0.5
    xsize = 2. * 60. / reso

    #pylab.subplot(3, 3, 8)
    healpy.gnomview(m_maglim_g, fig='summary', rot=(ra_peak, dec_peak, 0.), reso=reso, xsize=xsize, title='maglim g (S/N =10)', sub=(3, 3, 8))

    #pylab.subplot(3, 3, 9)
    healpy.gnomview(m_maglim_r, fig='summary', rot=(ra_peak, dec_peak, 0.), reso=reso, xsize=xsize, title='maglim r (S/N =10)', sub=(3, 3, 9))

    if savedir is not None:
        pylab.savefig('%s/candidate_%.2f_%.2f.png'%(savedir, ra_peak, dec_peak), dpi=150, bbox_inches='tight')
    
    ##########

    """
    pylab.figure('sanity')
    pylab.clf()
    bins = numpy.arange(-0.5, 0.5 + 1.e-10, 0.025)
    pylab.hist(data['WAVG_MAG_PSF_G'][cut_annulus] - data['WAVG_MAG_AUTO_G'][cut_annulus], bins=bins, color='green', histtype='step', lw=2, normed=True, label='g annulus')
    pylab.hist(data['WAVG_MAG_PSF_R'][cut_annulus] - data['WAVG_MAG_AUTO_R'][cut_annulus], bins=bins, color='orange', histtype='step', lw=2, normed=True, label='r annulus')
    pylab.hist(data['WAVG_MAG_PSF_G'][cut_inner] - data['WAVG_MAG_AUTO_G'][cut_inner], bins=bins, color='blue', histtype='step', lw=2, normed=True, label='g inner')
    pylab.hist(data['WAVG_MAG_PSF_R'][cut_inner] - data['WAVG_MAG_AUTO_R'][cut_inner], bins=bins, color='red', histtype='step', lw=2, normed=True, label='r inner')
    pylab.legend(loc='upper right', frameon=False)
    pylab.xlabel('WAVG_MAG_PSF - WAVG_MAG_AUTO (mag)')
    pylab.ylabel('PDF')
    pylab.title('(RA, Dec, m - M) = (%.2f, %.2f, %.2f)'%(ra_peak, dec_peak, distance_modulus))
    if savedir is not None:
        pylab.savefig('%s/sanity_%.2f_%.2f.png'%(savedir, ra_peak, dec_peak), dpi=250, bbox_inches='tight')
    """

    #raw_input('WAIT')

############################################################

distance_modulus_search_array = numpy.arange(16., 24., 0.5)
#distance_modulus_search_array = [21.3]
#distance_modulus_search_array = [18.5]
#distance_modulus_search_array = [19.5] # Just a test
#distance_modulus_search_array = [17.5, 19.5] # Just a test

ra_peak_array = []
dec_peak_array = [] 
r_peak_array = []
sig_peak_array = []
distance_modulus_array = []
for distance_modulus in distance_modulus_search_array:
    ra_peak, dec_peak, r_peak, sig_peak, distance_modulus = searchByDistance(data, distance_modulus, ra_select, dec_select)
    ra_peak_array.append(ra_peak)
    dec_peak_array.append(dec_peak)
    r_peak_array.append(r_peak)
    sig_peak_array.append(sig_peak)
    distance_modulus_array.append(distance_modulus)
    #raw_input('WAIT')

ra_peak_array = numpy.concatenate(ra_peak_array)
dec_peak_array = numpy.concatenate(dec_peak_array)
r_peak_array = numpy.concatenate(r_peak_array)
sig_peak_array = numpy.concatenate(sig_peak_array)
distance_modulus_array = numpy.concatenate(distance_modulus_array)

index_sort = numpy.argsort(sig_peak_array)[::-1]
ra_peak_array = ra_peak_array[index_sort]
dec_peak_array = dec_peak_array[index_sort]
r_peak_array = r_peak_array[index_sort]
sig_peak_array = sig_peak_array[index_sort]
distance_modulus_array = distance_modulus_array[index_sort]

#for ii in range(0, len(sig_peak_array)):
#    print '%.2f   %.2f   %.1f   %.2f'%(ra_peak_array[ii], dec_peak_array[ii], distance_modulus_array[ii], sig_peak_array[ii])
#raw_input('WAIT')

for ii in range(0, len(sig_peak_array)):
    if sig_peak_array[ii] < 0:
        continue
    angsep = ugali.utils.projector.angsep(ra_peak_array[ii], dec_peak_array[ii], ra_peak_array, dec_peak_array)
    sig_peak_array[(angsep < r_peak_array[ii]) & (numpy.arange(len(sig_peak_array)) > ii)] = -1.


ra_peak_array = ra_peak_array[sig_peak_array > 0.]
dec_peak_array = dec_peak_array[sig_peak_array > 0.]
r_peak_array = r_peak_array[sig_peak_array > 0.]
distance_modulus_array = distance_modulus_array[sig_peak_array > 0.]
sig_peak_array = sig_peak_array[sig_peak_array > 0.] # Update the sig_peak_array last!

#pylab.figure()
#pylab.scatter(ra_peak_array, dec_peak_array, c=distance_modulus_array, alpha=0.5)

for ii in range(0, len(sig_peak_array)):
    print '%.2f sigma; (RA, Dec, d) = (%.2f, %.2f); r = %.2f deg; d = %.1f, mu = %.2f mag)'%(sig_peak_array[ii], 
                                                                                             ra_peak_array[ii], 
                                                                                             dec_peak_array[ii], 
                                                                                             r_peak_array[ii],
                                                                                             ugali.utils.projector.distanceModulusToDistance(distance_modulus_array[ii]),
                                                                                             distance_modulus_array[ii])

    if (sig_peak_array[ii] > 5.5) & (r_peak_array[ii] < 0.28):
        diagnostic(data, data_gal, ra_peak_array[ii], dec_peak_array[ii], r_peak_array[ii], sig_peak_array[ii], distance_modulus_array[ii], savedir='figs_v8')


results_dir = 'results_v8'
outfile = '%s/results_nside_16_%i.csv'%(results_dir, pix_16_select)
writer = open(outfile, 'w')
writer.write('sig, ra, dec, distance_modulus, r\n')
for ii in range(0, len(sig_peak_array)):
    writer.write('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f\n'%(sig_peak_array[ii], ra_peak_array[ii], dec_peak_array[ii], distance_modulus_array[ii], r_peak_array[ii]))
writer.close()


#proj = ugali.utils.projector.Projector(ra_select, dec_select)
#print proj.imageToSphere(-4.54, -3.18)
