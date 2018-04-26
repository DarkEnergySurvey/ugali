import os
import glob
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.patches as patches
import pylab

pylab.ion()

##########

def wrap(x):
    x_return = x
    x_return[x > 180.] = x[x > 180.] - 360.
    return x_return

##########

def getCatalogFile(catalog_dir, mc_source_id):
    """
    Inputs:
        catalog_dir = string corresponding to directory containing the stellar catalog infiles
        mc_source_id = integer corresponding the target MC_SOURCE_ID value
    Outputs:
        catalog_infile = string corresponding to filename of stellar catalog containing mc_source_id
    """
    catalog_infiles = sorted(glob.glob(catalog_dir + '/*catalog*.fits'))
    mc_source_id_array = []
    catalog_infile_index_array = []
    for ii, catalog_infile in enumerate(catalog_infiles):
        mc_source_id_min = int(os.path.basename(catalog_infile).split('.')[0].split('mc_source_id_')[-1].split('-')[0])
        mc_source_id_max = int(os.path.basename(catalog_infile).split('.')[0].split('mc_source_id_')[-1].split('-')[1])
        assert (mc_source_id_max > mc_source_id_min) & (mc_source_id_min >= 1), 'Found invalue MC_SOURCE_ID values in filenames'
        mc_source_id_array.append(np.arange(mc_source_id_min, mc_source_id_max + 1))
        catalog_infile_index_array.append(np.tile(ii, 1 + (mc_source_id_max - mc_source_id_min)))

    mc_source_id_array = np.concatenate(mc_source_id_array)
    catalog_infile_index_array = np.concatenate(catalog_infile_index_array)

    assert len(mc_source_id_array) == len(np.unique(mc_source_id_array)), 'Found non-unique MC_SOURCE_ID values in filenames'
    assert np.in1d(mc_source_id, mc_source_id_array), 'Requested MC_SOURCE_ID value not among files'
    mc_source_id_index = np.nonzero(mc_source_id == mc_source_id_array)[0]
    return catalog_infiles[catalog_infile_index_array[mc_source_id_index]]

##########

def getCatalog(catalog_dir):
    catalog_infiles = sorted(glob.glob(catalog_dir + '/*catalog*.fits'))
    data_array = []
    for catalog_infile in catalog_infiles:
        print '  Reading %s ...'%(catalog_infile)
        reader = pyfits.open(catalog_infile)
        data_array.append(reader[1].data)
        reader.close()
    print '  Merging ...'
    return np.concatenate(data_array)

##########

save = False
dpi = 150

infile_population = 'v3/sim_population_v3.fits'
reader_population = pyfits.open(infile_population)
data_population = reader_population[1].data
print len(data_population)
data_population = data_population #[0:500]
reader_population.close()

#catalog_dir = 
#glob.glob('*catalog*.fits')


#infile_catalog = 'sim_catalog_v2_n_5000.fits.gz'
#reader_catalog = pyfits.open(infile_catalog)
#data_catalog = reader_catalog[1].data
#reader_catalog.close()

data_catalog = getCatalog('v3')

"""
pylab.figure()
pylab.scatter(wrap(data_catalog['RA']), data_catalog['DEC'], c=data_catalog['MC_SOURCE_ID'], s=1, cmap='jet')
colorbar = pylab.colorbar()
colorbar.set_label('MC Source ID')
pylab.xlabel('RA (deg)')
pylab.ylabel('Dec (deg)')
pylab.xlim(pylab.xlim()[::-1])
if save:
    pylab.savefig('sim_population_coordinates.png', dpi=dpi)
"""
"""
pylab.figure()
#pylab.yscale('log')
pylab.hist(data_catalog['PSF_MAG_SFD_G'], bins=np.arange(16., 30., 0.1), color='blue', alpha=0.5, label='g-band') # mag_g
pylab.hist(data_catalog['PSF_MAG_SFD_R'], bins=np.arange(16., 30., 0.1), color='green', alpha=0.5, label='r-band') # mag_r
pylab.xlabel('Magnitude')
pylab.ylabel('Counts')
pylab.legend(loc='upper left')
if save:
    pylab.savefig('sim_population_magnitudes.png', dpi=dpi)

for band, color in zip(['g', 'r'], ['blue', 'green']):
    pylab.figure()
    pylab.yscale('log')
    #pylab.scatter(data_catalog['mag_g'], data_catalog['magerr_g'], color='blue', marker='.', s=1, label='g-band')
    #pylab.scatter(data_catalog['mag_r'], data_catalog['magerr_r'], color='green', marker='.', s=1, label='r-band')
    pylab.scatter(data_catalog['PSF_MAG_SFD_%s'%(band)], data_catalog['PSF_MAG_ERR_%s'%(band)], color=color, marker='.', s=1, alpha=0.1) # mag_%s
    pylab.xlabel('%s Magnitude'%(band))
    pylab.ylabel('%s Magnitude Uncertainty'%(band))
    pylab.legend(loc='upper left', scatterpoints=1, markerscale=10)
    if save:
        pylab.savefig('sim_population_magnitude_errors_%s.png'%(band), dpi=dpi)



pylab.figure()
pylab.xscale('log')
pylab.scatter(1.e3 * data_population['r_physical'], data_population['abs_mag'], c=data_population['surface_brightness'], s=10)
colorbar = pylab.colorbar()
colorbar.set_label('Surface Brightness (mag arcsec^-2)')
pylab.xlim(1., 3.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Half-light Radius (pc)')
pylab.ylabel('M_V (mag)')
if save:
    pylab.savefig('sim_population_coordinates_size_luminosity.png', dpi=dpi)

pylab.figure()
pylab.xscale('log')
pylab.scatter(data_population['distance'], data_population['abs_mag'], c=data_population['n_g24'], vmin=0, vmax=3.e2, s=10)
colorbar = pylab.colorbar()
colorbar.set_label('Number of Stars with g < 24 mag')
pylab.xlim(3., 1.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Distance (kpc)')
pylab.ylabel('M_V (mag)')
if save:
    pylab.savefig('sim_population_distance_luminosity.png', dpi=dpi)

pylab.figure()
pylab.xscale('log')
pylab.scatter(data_population['n_g24'], data_population['surface_brightness'], s=10)
pylab.xlim(0.1, 1.e6)
pylab.ylim(35., 25.)
pylab.xlabel('Number of Stars with g < 24 mag')
pylab.ylabel('Surface Brightness (mag arcsec^-2)')
if save:
    pylab.savefig('sim_population_n_g24_surface_brightness.png', dpi=dpi)
"""
"""
pylab.figure()
counts_array = []
for index in data_population['MC_SOURCE_ID']:
    counts_array.append(np.sum(data_catalog['MC_SOURCE_ID'] == index))
pylab.scatter(counts_array, data_population['n_g24'])
"""
"""
cut = (data_catalog['MC_SOURCE_ID'] == data_population['MC_SOURCE_ID'][np.argmax(data_population['n_g24'])])

pylab.figure()
pylab.scatter(data_catalog['PSF_MAG_SFD_G'][cut] - data_catalog['PSF_MAG_SFD_R'][cut], 
              data_catalog['PSF_MAG_SFD_G'][cut],
              marker='.')
pylab.ylim(pylab.ylim()[::-1])
"""

pylab.figure()
pylab.scatter(wrap(data_population['RA']), data_population['DEC'], c=data_population['FRACDET'], vmin=0., vmax=1.)
colorbar = pylab.colorbar()
colorbar.set_label('Average Fracdet within Azimuthally Averaged Half-light Radius')
pylab.xlabel('RA (deg)')
pylab.ylabel('Dec (deg)')
pylab.xlim(pylab.xlim()[::-1])

pylab.figure()
pylab.scatter(wrap(data_population['RA']), data_population['DEC'], c=data_population['DENSITY'], vmax=1.)
colorbar = pylab.colorbar()
colorbar.set_label('Local Stellar Density (arcmin^-2)')
pylab.xlabel('RA (deg)')
pylab.ylabel('Dec (deg)')
pylab.xlim(pylab.xlim()[::-1])

##########
"""
counts_mc_source_id = np.histogram(data_catalog['MC_SOURCE_ID'],bins=np.arange(np.max(data_catalog['MC_SOURCE_ID']) + 1))[0]
pylab.figure()
pylab.yscale('log', nonposy='clip')
counts, edges = pylab.hist(counts_mc_source_id, bins=np.linspace(0, np.max(counts_mc_source_id) + 1, 101))[0:2]

centers = 0.5 * (edges[:-1] + edges[1:])
pylab.xlabel('Catalog Stars per Satellite')
pylab.ylabel('Number of Satellites')

#pylab.figure()
#pylab.scatter(centers, centers * counts)
"""
##########

print "Machine learning"

save = False

import sklearn.gaussian_process
import sklearn.neighbors
import sklearn.svm

x = np.vstack([np.log10(data_population['distance']), data_population['abs_mag'], np.log10(data_population['r_physical'])]).T
y = (data_population['surface_brightness'] < 29.) & (data_population['n_g24'] >= 10.)

#classifier = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0))
classifier = sklearn.neighbors.KNeighborsClassifier(2, weights='uniform')
#classifier = sklearn.neighbors.KNeighborsClassifier(2, weights='distance') 
#classifier = sklearn.svm.SVC(gamma=2, C=1)

classifier.fit(x, y)

y_pred = classifier.predict_proba(x)

pylab.figure()
pylab.xscale('log')
pylab.scatter(1.e3 * data_population['r_physical'][y], data_population['abs_mag'][y], c='black', marker='x', label='Detected')
pylab.scatter(1.e3 * data_population['r_physical'], data_population['abs_mag'], c=y_pred[:,1], vmin=0., vmax=1., s=10)
colorbar = pylab.colorbar()
colorbar.set_label('ML Predicted Detection Probability')
pylab.xlim(1., 3.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Half-light Radius (pc)')
pylab.ylabel('M_V (mag)')
pylab.legend(loc='upper left')
if save:
    pylab.savefig('sim_population_coordinates_size_luminosity_prediction.png', dpi=dpi)

pylab.figure()
pylab.xscale('log')
pylab.scatter(data_population['distance'][y], data_population['abs_mag'][y], c='black', marker='x', label='Detected')
pylab.scatter(data_population['distance'], data_population['abs_mag'], c=y_pred[:,1], vmin=0., vmax=1., s=10)
colorbar = pylab.colorbar()
colorbar.set_label('ML Predicted Detection Probability')
pylab.xlim(3., 1.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Distance (kpc)')
pylab.ylabel('M_V (mag)')
pylab.legend(loc='upper left')
if save:
    pylab.savefig('sim_population_distance_luminosity_prediction.png', dpi=dpi)

pylab.figure()
pylab.xscale('log')
#patches.Rectangle((10., 29.), 1.e2, -1., fc='black', alpha=0.1)
pylab.gca().add_patch(patches.Rectangle((10., 25.), 1.e6, 4., fc='black', alpha=0.2, zorder=-999, label='Detection Region'))
pylab.scatter(data_population['n_g24'][y], data_population['surface_brightness'][y], c='black', marker='x', label='Detected')
pylab.scatter(data_population['n_g24'], data_population['surface_brightness'], c=y_pred[:,1], vmin=0., vmax=1., s=10)
colorbar = pylab.colorbar()
colorbar.set_label('ML Predicted Detection Probability')
pylab.xlim(0.1, 1.e6)
pylab.ylim(35., 25.)
pylab.xlabel('Number of Stars with g < 24 mag')
pylab.ylabel('Surface Brightness (mag arcsec^-2)')
pylab.legend(loc='lower right')
if save:
    pylab.savefig('sim_population_n_g24_surface_brightness_prediction.png', dpi=dpi)

bins = np.linspace(0., 1., 10 + 1)
centers = np.empty(len(bins) - 1)
bin_prob = np.empty(len(bins) - 1)
for ii in range(0, len(centers)):
    cut = (y_pred[:,1] > bins[ii]) & (y_pred[:,1] < bins[ii + 1])
    centers[ii] = np.mean(y_pred[:,1][cut])
    bin_prob[ii] = np.mean(y[cut].astype(float))
pylab.figure()
pylab.plot(centers, bin_prob, c='black')
pylab.scatter(centers, bin_prob, c='black')
pylab.xlabel('ML Predicted Detection Probability')
pylab.ylabel('Fraction Detected')
pylab.xlim(0., 1.)
pylab.ylim(0., 1.)

bins = np.linspace(0., 1., 41.)
pylab.figure()
pylab.hist(y_pred[:,1][y], bins=bins, color='green', alpha=0.5, normed=True, label='Detected')
pylab.hist(y_pred[:,1][~y], bins=bins, color='red', alpha=0.5, normed=True, label='Not Detected')
pylab.xlabel('ML Predicted Detection Probability')
pylab.ylabel('PDF')
pylab.legend(loc='upper center')
