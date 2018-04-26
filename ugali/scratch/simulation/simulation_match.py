import numpy as np
import astropy.io.fits as pyfits
import pylab

import ugali.utils.projector
import ugali.utils.bayesian_efficiency

pylab.ion()

##########

def wrap(x):
    x_return = x
    x_return[x > 180.] = x[x > 180.] - 360.
    return x_return

##########

data_actual = np.recfromcsv('/Users/keithbechtol/Documents/DES/projects/mw_substructure/local_volume_database/LVDB_combination_pace_19_april_2018.csv')
distance_actual = ugali.utils.projector.distanceModulusToDistance(data_actual['distance_modulus'])
r_physical_actual = distance_actual * np.tan(np.radians(data_actual['rhalf_major'] / 60.)) * np.sqrt(1. - data_actual['ellipticity']) # Azimuthally averaged half-light radius in kpc; note that rhalf_major in arcmin
abs_mag_actual = data_actual['apparent_magnitude'] - data_actual['distance_modulus']

cut_hsc = np.in1d(data_actual['galaxy_key'], ['virgo_1', 'cetus_3'])

#########

save = True

#data_search = np.recfromcsv('results_12apr2018.csv')
reader_search = pyfits.open('/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y3a2/simple/sid/simulations_v2/candidate_list.fits')
data_search = reader_search[1].data
reader_search.close()

significance_threshold = 7.
data_search = data_search[data_search['SIG'] >= significance_threshold]

#reader_sim = pyfits.open('sim_population_12apr2018.fits')
reader_sim = pyfits.open('v3/sim_population_v3.fits')
data_sim = reader_sim[1].data
reader_sim.close()

print len(data_sim)

"""
match_search, match_sim, angsep = ugali.utils.projector.match(data_search['ra'], data_search['dec'], data_sim['RA'], data_sim['DEC'], tol=1.)
print len(match_search)
print match_sim
print angsep
"""
"""
pylab.figure()
pylab.scatter(wrap(data_sim['RA']), data_sim['DEC'], c='black', s=2, label='Sim')
pylab.scatter(wrap(data_search['ra']), data_search['dec'], c='red', label='Detection')
pylab.xlabel('RA')
pylab.ylabel('Dec')
pylab.legend(loc='center right')
pylab.xlim(pylab.xlim()[::-1])
"""
#index = np.nonzero(data_sim['MC_SOURCE_ID'] == 8)[0][0]
#print data_sim['RA'][index], data_sim['DEC'][index]

#cut_detected = np.where(np.in1d(np.arange(len(data_sim)), match_sim), True, False)
cut_detect = np.in1d(data_sim['MC_SOURCE_ID'], data_search['MC_SOURCE_ID'])

cut_why_not = (data_sim['surface_brightness'] < 27.) & (data_sim['n_g24'] > 50.) & ~cut_detect # First version
#cut_why_not = (data_sim['surface_brightness'] < 30.) & (data_sim['n_g24'] > 25.) & ~cut_detect # First version

#color = {'detect': 'black',
#         'nondetect': '0.75',
#         'why_not': 'none',
#         'actual': 'yellow',
#         'hsc': 'lime'}
color = {'detect': 'Red',
         'nondetect': 'Gold',
         'why_not': 'none',
         'actual': 'DodgerBlue',
         'hsc': 'lime'}
#color = {'detect': 'SteelBlue',
#         'nondetect': 'FireBrick',
#         'why_not': 'none',
#         'actual': 'Khaki',
#         'hsc': 'white'}
#color = {'detect': 'LightBlue ',
#         'nondetect': 'LightCoral',
#         'why_not': 'none',
#         'actual': 'yellow',
#         'hsc': 'lime'}
size = {'detect': 5,
        'nondetect': 5,
        'why_not': 35,
        'actual': None,
        'hsc': None}
marker = {'detect': 'o',
          'nondetect': 'o',
          'why_not': 'o',
          'actual': 's',
          'hsc': 's'}
alpha = {'detect': None,
         'nondetect': None,
         'why_not': None,
         'actual': None,
         'hsc': None}
edgecolor = {'detect': None,
             'nondetect': None,
             'why_not': 'magenta',
             'actual': 'black',
             'hsc': 'black'}


pylab.figure()
pylab.scatter(wrap(data_sim['RA'][~cut_detect]), data_sim['DEC'][~cut_detect], 
              c=color['nondetect'], s=size['nondetect'], marker=marker['nondetect'], edgecolor=edgecolor['nondetect'], alpha=alpha['nondetect'], label='Non-detection')
pylab.scatter(wrap(data_sim['ra'][cut_why_not]), data_sim['dec'][cut_why_not], 
              c=color['why_not'], s=size['why_not'], marker=marker['why_not'], edgecolor=edgecolor['why_not'], alpha=alpha['why_not'], label='Why Non-Detection')
pylab.scatter(wrap(data_sim['ra'][cut_detect]), data_sim['dec'][cut_detect], 
              c=color['detect'], s=size['detect'], marker=marker['detect'], edgecolor=edgecolor['detect'], alpha=alpha['detect'], label='Detection')
pylab.xlabel('RA')
pylab.ylabel('Dec')
pylab.legend(loc='center right', markerscale=2)
pylab.xlim(pylab.xlim()[::-1])
if save:
    pylab.savefig('sim_match_map.pdf')

pylab.figure()
pylab.xscale('log')
pylab.scatter(1.e3 * data_sim['r_physical'][~cut_detect], data_sim['abs_mag'][~cut_detect], 
              c=color['nondetect'], s=size['nondetect'], marker=marker['nondetect'], edgecolor=edgecolor['nondetect'], alpha=alpha['nondetect'], label='Non-detection')
#pylab.scatter(1.e3 * data_sim['r_physical'][cut_why_not], data_sim['abs_mag'][cut_why_not], 
#              c=color['why_not'], s=size['why_not'], marker=marker['why_not'], edgecolor=edgecolor['why_not'], alpha=alpha['why_not'], label='Why Non-Detection')
pylab.scatter(1.e3 * data_sim['r_physical'][cut_detect], data_sim['abs_mag'][cut_detect], 
              c=color['detect'], s=size['detect'], marker=marker['detect'], edgecolor=edgecolor['detect'], alpha=alpha['detect'], label='Detection')
pylab.scatter(1.e3 * r_physical_actual, abs_mag_actual, 
              c=color['actual'], s=size['actual'], marker=marker['actual'], edgecolor=edgecolor['actual'], alpha=alpha['actual'], label='Actual MW Satellites')
pylab.scatter(1.e3 * r_physical_actual[cut_hsc], abs_mag_actual[cut_hsc], 
              c=color['hsc'], s=size['hsc'], marker=marker['hsc'], edgecolor=edgecolor['hsc'], alpha=alpha['hsc'], label='Actual MW Satellites: HSC')
pylab.ylim(4., -10)
pylab.xlabel('Half-light Radius (pc)')
pylab.ylabel('M_V')
pylab.legend(loc='lower right', markerscale=2)
pylab.title('Detection Threshold: Sig > %.2f '%(significance_threshold))
if save:
    pylab.savefig('sim_match_size_luminosity.pdf')

pylab.figure()
pylab.xscale('log')
pylab.scatter(data_sim['distance'][~cut_detect], data_sim['abs_mag'][~cut_detect], 
              c=color['nondetect'], s=size['nondetect'], marker=marker['nondetect'], edgecolor=edgecolor['nondetect'], alpha=alpha['nondetect'], label='Non-detection')
#pylab.scatter(data_sim['distance'][cut_why_not], data_sim['abs_mag'][cut_why_not], 
#              c=color['why_not'], s=size['why_not'], marker=marker['why_not'], edgecolor=edgecolor['why_not'], alpha=alpha['why_not'], label='Why Non-Detection')
pylab.scatter(data_sim['distance'][cut_detect], data_sim['abs_mag'][cut_detect], 
              c=color['detect'], s=size['detect'], marker=marker['detect'], edgecolor=edgecolor['detect'], alpha=alpha['detect'], label='Detection')
pylab.scatter(distance_actual, abs_mag_actual, 
              c=color['actual'], s=size['actual'], marker=marker['actual'], edgecolor=edgecolor['actual'], alpha=alpha['actual'], label='Actual MW Satellites')
pylab.scatter(distance_actual[cut_hsc], abs_mag_actual[cut_hsc], 
              c=color['hsc'], s=size['hsc'], marker=marker['hsc'], edgecolor=edgecolor['hsc'], alpha=alpha['hsc'], label='Actual MW Satellites: HSC')
pylab.ylim(4., -10)
pylab.xlabel('Distance (kpc)')
pylab.ylabel('M_V')
pylab.legend(loc='lower right', markerscale=2)
pylab.title('Detection Threshold: Sig > %.2f '%(significance_threshold))
if save:
    pylab.savefig('sim_match_distance_luminosity.pdf')

print '%20s%20s%20s%20s%20s%20s'%('mc_source_id', 'ra', 'dec', 'distance_modulus', 'fracdet', 'density')
for index in np.nonzero(cut_why_not)[0]:
    print '%20i%20.3f%20.3f%20.3f%20.3f%20.3f'%(data_sim['mc_source_id'][index], 
                                                data_sim['ra'][index], data_sim['dec'][index], 
                                                data_sim['distance_modulus'][index],
                                                data_sim['fracdet'][index],
                                                data_sim['density'][index])
    
############################################################

#save = True

fit = False

import sklearn.gaussian_process
import sklearn.neighbors
import sklearn.svm
import pickle
import os
import time

#x = np.vstack([np.log10(data_population['distance']), data_population['abs_mag'], np.log10(data_population['r_physical'])]).T
#y = (data_population['surface_brightness'] < 29.) & (data_population['n_g24'] >= 10.)
x = np.vstack([np.log10(data_sim['distance']), data_sim['abs_mag'], np.log10(data_sim['r_physical'])]).T
y = cut_detect
cut_train = np.arange(len(x)) < 0.8 * len(x) # 0.8, 130 sec

classifier_file = 'trained_classifier.txt'
if fit:
    print 'Training the machine learning classifier. This may take a while ...'
    t_start = time.time()
    classifier = sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(0.5))
    #classifier = sklearn.neighbors.KNeighborsClassifier(3, weights='uniform')
    #classifier = sklearn.neighbors.KNeighborsClassifier(2, weights='distance') 
    #classifier = sklearn.svm.SVC(gamma=2, C=1)
    classifier.fit(x[cut_train], y[cut_train])
    t_end = time.time()
    print '  ... training took %.2f seconds'%(t_end - t_start)
    # Save the trained classifier
    classifier_data = pickle.dumps(classifier)
    writer = open(classifier_file, 'w')
    writer.write(classifier_data)
    writer.close()
    print 'Saving machine learning classifier to %s ...'%(classifier_file)
    os.system('gzip %s'%(classifier_file))
else:
    print 'Loading machine learning classifier from %s ...'%(classifier_file)
    if os.path.exists(classifier_file + '.gz') and not os.path.exists(classifier_file):
        print '  Unzipping...'
        os.system('gunzip -k %s.gz'%(classifier_file))
    reader = open(classifier_file)
    classifier_data = ''.join(reader.readlines())
    reader.close()
    classifier = pickle.loads(classifier_data)

y_pred = classifier.predict_proba(x[~cut_train])

#y_pred_load = np.load('y_pred.npy')

title = 'N_train = %i ; N_test = %i'%(np.sum(cut_train), np.sum(~cut_train))

#cmap = 'viridis'
#cmap = 'RdYlBu'
#cmap = 'summer'
#cmap = 'coolwarm'
#cmap = 'autumn'
import matplotlib
#cmap = matplotlib.colors.ListedColormap(['FireBrick', 'LightCoral', 'LightBlue', 'SteelBlue'])
#cmap = matplotlib.colors.ListedColormap(['Red', 'OrangeRed', 'DarkOrange', 'Orange', 'Gold'])
cmap = matplotlib.colors.ListedColormap(['Gold', 'Orange', 'DarkOrange', 'OrangeRed', 'Red'])

pylab.figure()
pylab.xscale('log')
#pylab.scatter(1.e3 * data_sim['r_physical'][y], data_sim['abs_mag'][y], c='black', marker='x', label='Detected')
pylab.scatter(1.e3 * data_sim['r_physical'][cut_train], data_sim['abs_mag'][cut_train], c=cut_detect[cut_train].astype(int), vmin=0., vmax=1., s=size['detect'], cmap=cmap, label=None) 
pylab.scatter(1.e3 * data_sim['r_physical'][~cut_train], data_sim['abs_mag'][~cut_train], c=y_pred[:,1], edgecolor='black', vmin=0., vmax=1., s=(3 * size['detect']), cmap=cmap, label=None) 
colorbar = pylab.colorbar()
colorbar.set_label('ML Predicted Detection Probability')
pylab.scatter(0., 0., s=(3 * size['detect']), c='none', edgecolor='black', label='Test')
pylab.scatter(1.e3 * r_physical_actual, abs_mag_actual, 
              c=color['actual'], s=size['actual'], marker=marker['actual'], edgecolor=edgecolor['actual'], alpha=alpha['actual'], label='Actual MW Satellites')
pylab.scatter(1.e3 * r_physical_actual[cut_hsc], abs_mag_actual[cut_hsc], 
              c=color['hsc'], s=size['hsc'], marker=marker['hsc'], edgecolor=edgecolor['hsc'], alpha=alpha['hsc'], label='Actual MW Satellites: HSC')
pylab.xlim(1., 3.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Half-light Radius (pc)')
pylab.ylabel('M_V (mag)')
pylab.legend(loc='upper left', markerscale=2)
pylab.title(title)
if save:
    pylab.savefig('sim_match_size_luminosity_prediction.pdf')#, dpi=dpi)

###

pylab.figure()
pylab.xscale('log')
#pylab.scatter(data_sim['distance'][y], data_sim['abs_mag'][y], c='black', marker='x', label='Detected')
pylab.scatter(data_sim['distance'][cut_train], data_sim['abs_mag'][cut_train], c=cut_detect[cut_train].astype(int), vmin=0., vmax=1., s=size['detect'], cmap=cmap, label=None)
pylab.scatter(data_sim['distance'][~cut_train], data_sim['abs_mag'][~cut_train], c=y_pred[:,1], edgecolor='black', vmin=0., vmax=1., s=(3 * size['detect']), cmap=cmap, label=None)
colorbar = pylab.colorbar()
colorbar.set_label('ML Predicted Detection Probability')
pylab.scatter(0., 0., s=(3 * size['detect']), c='none', edgecolor='black', label='Test')
pylab.scatter(distance_actual, abs_mag_actual, 
              c=color['actual'], s=size['actual'], marker=marker['actual'], edgecolor=edgecolor['actual'], alpha=alpha['actual'], label='Actual MW Satellites')
pylab.scatter(distance_actual[cut_hsc], abs_mag_actual[cut_hsc], 
              c=color['hsc'], s=size['hsc'], marker=marker['hsc'], edgecolor=edgecolor['hsc'], alpha=alpha['hsc'], label='Actual MW Satellites: HSC')
pylab.xlim(3., 1.e3)
pylab.ylim(2., -12.)
pylab.xlabel('Distance (kpc)')
pylab.ylabel('M_V (mag)')
pylab.legend(loc='upper left', markerscale=2)
pylab.title(title)
if save:
    pylab.savefig('sim_match_distance_luminosity_prediction.pdf')#, dpi=dpi)

###

bins = np.linspace(0., 1., 41.)
pylab.figure()
pylab.hist(y_pred[:,1][cut_detect[~cut_train]], bins=bins, color=color['detect'], alpha=0.5, normed=True, label='Detected')
pylab.hist(y_pred[:,1][cut_detect[~cut_train]], bins=bins, color=color['detect'], normed=True, histtype='step', lw=2)
pylab.hist(y_pred[:,1][~cut_detect[~cut_train]], bins=bins, color=color['nondetect'], alpha=0.5, normed=True, label='Not Detected')
pylab.hist(y_pred[:,1][~cut_detect[~cut_train]], bins=bins, color='DarkOrange', normed=True, histtype='step', lw=2)
pylab.xlabel('ML Predicted Detection Probability')
pylab.ylabel('PDF')
pylab.legend(loc='upper center', markerscale=2)
pylab.title(title)
if save:
    pylab.savefig('sim_match_hist_prediction.pdf')#, dpi=dpi)

###

bins = np.linspace(0., 1., 10 + 1)
centers = np.empty(len(bins) - 1)
bin_prob = np.empty(len(bins) - 1)
bin_prob_err_hi = np.empty(len(bins) - 1)
bin_prob_err_lo = np.empty(len(bins) - 1)
bin_counts = np.empty(len(bins) - 1)
for ii in range(0, len(centers)):
    cut_bin = (y_pred[:,1] > bins[ii]) & (y_pred[:,1] < bins[ii + 1])
    centers[ii] = np.mean(y_pred[:,1][cut_bin])
    
    n_trials = np.sum(cut_bin)
    n_successes = np.sum(cut_detect[~cut_train] & cut_bin)
    efficiency, errorbar = ugali.utils.bayesian_efficiency.bayesianInterval(n_trials, n_successes, errorbar=True)
    bin_prob[ii] = efficiency
    bin_prob_err_hi[ii] = errorbar[1]
    bin_prob_err_lo[ii] = errorbar[0]
    #bin_prob[ii] = float(np.sum(cut_detect[~cut_train] & cut_bin)) / np.sum(cut_bin)
    #bin_prob[ii] = np.mean(y[cut].astype(float))
    bin_counts[ii] = np.sum(cut_bin)
    

pylab.figure()
pylab.plot([0., 1.], [0., 1.], c='black', ls='--')
pylab.errorbar(centers, bin_prob, yerr=[bin_prob_err_lo, bin_prob_err_hi], c='red')
pylab.plot(centers, bin_prob, c='red')
pylab.scatter(centers, bin_prob, c=bin_counts, edgecolor='red', s=50, cmap='Reds', zorder=999)
colorbar = pylab.colorbar()
colorbar.set_label('Counts')
pylab.xlabel('ML Predicted Detection Probability')
pylab.ylabel('Fraction Detected')
pylab.xlim(0., 1.)
pylab.ylim(0., 1.)
pylab.title(title)
if save:
    pylab.savefig('sim_match_scatter_prediction.pdf')#, dpi=dpi)

# ROC curve
#n = 101
#threshold_array = np.linspace(0., 1., n)
#tpr = np.empty(n)
#fpr = np.empty(n)
#for ii, threshold in enumerate(threshold_array):
    
