import glob
import sys
import os
import numpy
import pylab
import healpy
import pyfits

import ugali.utils.healpix
#import uagli.utils.projector
import ugali.candidate.associate

pylab.ion()

############################################################
"""
datadir = '/Users/keithbechtol/Documents/DES/projects/calibration/Y2N/data/catalog'
infiles = glob.glob('%s/starcat_y2n_gr_v3_00000*.fits'%(datadir))
data_array = []
for infile in infiles:
    print infile
    reader = pyfits.open(infile)
    data_array.append(reader[1].data)
    reader.close()
print 'Assembling data...'
data = numpy.concatenate(data_array)

############################################################

print 'Applying cuts...'
cut = (data['WAVG_MAG_PSF_G'] < 23.5) \
      & ((data['WAVG_MAG_PSF_G'] - data['WAVG_MAG_PSF_R']) < 1.) \
      & (numpy.fabs(data['SPREAD_MODEL_R']) < 0.003 + data['SPREADERR_MODEL_R'])
#& (numpy.fabs(data['WAVG_SPREAD_MODEL_R']) < 0.003 + data['SPREADERR_MODEL_R'])
data = data[cut]

############################################################

print 'Pixelizing...'
npix_256 = healpy.nside2npix(256)
pix_256 = ugali.utils.healpix.angToPix(256, data['RA'], data['DEC'])
m_256 = numpy.histogram(pix_256, numpy.arange(npix_256 + 1))[0].astype(float)
m_256[m_256 == 0.] = healpy.UNSEEN 
"""
############################################################

save = False

infiles = glob.glob('results_midway_v8/*.csv')

ra_array = []
dec_array = []
sig_array = []
r_array = []
distance_modulus_array = []
for infile in infiles:
    data = numpy.recfromcsv(infile)
    if data.shape == (0,):
        continue

    if data['ra'].shape == ():
        ra_array.append([data['ra']])
        dec_array.append([data['dec']])
        sig_array.append([data['sig']])
        r_array.append([data['r']])
        distance_modulus_array.append([data['distance_modulus']])
    else:
        ra_array.append(data['ra'])
        dec_array.append(data['dec'])
        sig_array.append(data['sig'])
        r_array.append(data['r'])
        distance_modulus_array.append(data['distance_modulus'])

sig_array = numpy.concatenate(sig_array)
index_sort = numpy.argsort(sig_array)
sig_array = sig_array[index_sort]
ra_array = numpy.concatenate(ra_array)[index_sort]
dec_array = numpy.concatenate(dec_array)[index_sort]
r_array = numpy.concatenate(r_array)[index_sort]
distance_modulus_array = numpy.concatenate(distance_modulus_array)[index_sort]

### Run Associations ###

catalog_array = ['McConnachie12', 'Harris96', 'Corwen04', 'Nilson73', 'Webbink85', 'Kharchenko13', 'WEBDA14','ExtraDwarfs','ExtraClusters']
catalog = ugali.candidate.associate.SourceCatalog()
for catalog_name in catalog_array:
    catalog += ugali.candidate.associate.catalogFactory(catalog_name)
association_array = []
association_angsep_array = numpy.tile(180., len(sig_array))
for ii in range(0, len(sig_array)):
    glon, glat = ugali.utils.projector.celToGal(ra_array[ii], dec_array[ii])
    idx1, idx2, angsep = catalog.match(glon, glat, tol=0.5, nnearest=1)
    match = catalog[idx2]
    if len(match) > 0:
        association_array.append(match[0]['name'].replace(' ', '_'))
        association_angsep_array[ii] = angsep
    else:
        association_array.append('NONE')
association_array = numpy.array(association_array)
    

#cut = ((r_array < 0.21) | (r_array > 0.31)) & (sig_array > 5.9) & (distance_modulus_array < 23.)# & (distance_modulus_array > 20.)
#cut = (r_array < 0.2) & (sig_array > 7.) & (dec_array < -40.)
#cut = (r_array < 0.2) & (sig_array > 5.6) & (sig_array < 5.7)
#cut = (r_array > 0.2) & (sig_array > 10)
#cut = (r_array > 0.2) & (r_array < 0.28) & (sig_array > 9.)# & (sig_array < 7.)
#cut = (r_array < 0.2) & (sig_array > 7.)
#cut = (r_array < 0.2) & (sig_array > 6.) & (sig_array < 7.)
#cut = (r_array < 0.2) & (sig_array > 5.5) & (sig_array < 6.)
cut  = numpy.logical_not((r_array > 0.2) | \
                         (((numpy.char.count(association_array, 'NGC_') > 0) | \
                           (numpy.char.count(association_array, 'UGC_') > 0) | \
                           (numpy.char.count(association_array, 'IC_') > 0)) & (association_angsep_array < 0.2))) & (sig_array > 5.5)

#cut = ugali.utils.projector.angsep(ra_array, dec_array, 55.20, -37.50) < 0.1
#cut = ugali.utils.projector.angsep(ra_array, dec_array, 56.36, -60.44) < 0.1
#cut = ugali.utils.projector.angsep(ra_array, dec_array, 67.01, -44.34) < 0.1

sig_cut_array = [5.5, 6., 6.5, 7., 10.]
for sig_cut in sig_cut_array:
    print(sig_cut, numpy.sum((sig_array > sig_cut) & (r_array < 0.2)))

for ii in range(0, numpy.sum(cut)):
    print(ra_array[cut][ii], dec_array[cut][ii], sig_array[cut][ii], association_array[cut][ii], association_angsep_array[cut][ii]) 

for ra, dec, distance_modulus in zip(ra_array[cut], dec_array[cut], distance_modulus_array[cut]):
    outfile = 'candidate_%.2f_%.2f.png'%(ra, dec)
    if distance_modulus < 30.:
        pass
        os.system('cp figs_midway_v8/%s figs_temp/.'%(outfile))
        #os.system('open figs_midway_v1/%s'%(outfile))

#sys.exit('DONE')

"""
pylab.figure()
#pylab.scatter(ra_array[cut], dec_array[cut], c=sig_array[cut])
pylab.scatter(ra_array, dec_array, c=sig_array, edgecolor='none')
colorbar = pylab.colorbar()
colorbar.set_label(r'Significance ($\sigma$)')
"""

gc_catalog = ugali.candidate.associate.catalogFactory('Harris96')
sat_catalog = ugali.candidate.associate.catalogFactory('McConnachie12')

ra_ngc, dec_ngc = list(zip(*[[54.63, -35.45],
                        [11.78, -20.76],
                        [51.59, -21.34],
                        [28.25, -13.74],
                        [50.59, -37.26],
                        [51.62, -35.72],
                        [319.12, -48.36],
                        [45.40, -14.85],
                        [66.93, -55.02]]))

ra_des, dec_des, name_des = list(zip(*[[53.92, -54.05, 'Ret II'],
                                  [56.09, -43.53, 'Eri II'],
                                  [343.06, -58.57, 'Tuc II'],
                                  [43.87, -54.11, 'Hor I'],
                                  [317.20, -51.16, 'Kim 2'],
                                  [70.95, -50.28, 'Pic I'],
                                  [354.99, -54.41, 'Phe II'],
                                  [35.69, -52.28, 'Eri III'],
                                  [344.18, -50.16, 'Gru I'],
                                  [49.13, -50.02, 'Hor II'],
                                  [8.51, -49.04, 'Luque 1']]))

ra_des_new, dec_des_new, name_des_new = list(zip(*[[331.06, -46.47, 'Gru II'],
                                              [359.04, -59.63, 'Tuc III'],
                                              [82.85, -28.03, 'Col I'], # Rock solid
                                              #[32.29, -12.17, 'Cet II'],
                                              [0.668, -60.888, 'Tuc IV'],
                                              [354.37, -63.26, 'Tuc V'], # Also solid
                                              #[70.95, -50.29, 'New'], # Pic I
                                              #[94.85, -50.33, 'New'],
                                              #[67.01, -44.34, 'New'],
                                              [56.36, -60.44, 'Many faint stars'],
                                              [30.02, 3.35, 'Wide near S82']]))
                                              #[8.51, -49.04, 'Luque 1'],
                                              #[344.19, -50.16, 'New'], # Grus I
                                              #[317.21, -51.16, 'New'], # Ind I
                                              #[55.20, -37.50, 'New'],
                                              #[14.08, -18.36, 'New'],
                                              #[348.73, -42.56, 'New']])

# Cannot immediately rule out
#55.20, -37.50 # Looks OK, but there is a somewhat bright neaby galaxy in Aladin viewer, needs further investigation
#317.21, -51.16 # Very compact, very few stars, no association, clean in Aladin viewer, Ind I
#81.18, -54.69 # Not super compelling, but possibly a low surface brightness overdensity
#14.08, -18.36 # Very sparse if there at all, but stars line up nicely along isochrone, overdensity visible in map, no association, clean in Aladin viewer
#21.76, -6.04 # Right on edge of footprint, so a bit hard to tell
#24.37, -34.44 # Nothing obviously wrong, but not many stars at MSTO
#26.83, -51.93 # Spotty
#30.37, -3.26 # Whiting 1
#344.19, -50.16 # Looks pretty good, no association, clean in Aladin viewer, Grus I
#348.73, -42.56 # A good example of something at a distance that makes very difficult to distinguish from foreground, but an apparent overdensity
#38.80, 3.84 # Sketchy
#43.90, -60.05 # Overdensity is somewhat visible in the map, MSTO just below g ~ 23
#56.36, -60.44 # Clear overdensity, no association, clean in Aladin viewer 
#58.76, -49.61 # AM 1
#64.70, -24.09 # Sparse, but slight overdensity visible
#66.19, -21.19 # Eri
#67.01, -44.34 # Clearly visible overdensity, very sparse, but stars line up on isochrone, no association, clean in Aladin viewer
#70.95, -50.29 # Looks very good, no association, clean in Aladin viewer, Pic I
#94.85, -50.33 # Possibly very near, no association, clean in Aladin viewer

#[343.04, -58.59, 'New 2'], Tuc II
#[43.89, -54.12, 'New 3'], Hor I

#ra_des, dec_des, name_des = zip(*[[359.04, -59.63, 'Tuc III']])

#m_256 = numpy.load('density_map_nside_512.npy')
m_256 = numpy.load('/Users/keithbechtol/Documents/DES/projects/mw_substructure/des/y2n/skymap/density_map_nside_1024_v6.npy')
m_256[m_256 == 0.] = healpy.UNSEEN

pylab.figure(num='map1', figsize=(16, 10))
healpy.mollview(m_256, 
                fig='map1', cmap='binary', xsize=6400, min=0.5 * numpy.median(m_256[m_256 > 0.]), max=4. * numpy.median(m_256[m_256 > 0.]),
                unit='Stellar Density',
                title='Inclusive')
pylab.xlim(-1, 0.55)
pylab.ylim(-0.85, 0.2)
healpy.projscatter(ra_array, dec_array, c=sig_array, edgecolor='none', lonlat=True, vmin=5., vmax=25.)
if save:
    pylab.savefig('compile_map_inclusive.png', dpi=150, bbox_inches='tight')

pylab.figure(num='map2', figsize=(16, 10))
healpy.mollview(m_256, 
                fig='map2', cmap='binary', xsize=6400, min=0.5 * numpy.median(m_256[m_256 > 0.]), max=4. * numpy.median(m_256[m_256 > 0.]),
                unit='Stellar Density',
                title=r'Cleaned: size < 0.2 deg & significance > 6 $\sigma$ & m - M < 23.')
pylab.xlim(-1, 0.55)
pylab.ylim(-0.85, 0.2)
healpy.projscatter(ra_array[cut], dec_array[cut], c=sig_array[cut], edgecolor='none', lonlat=True, vmin=5., vmax=25.)
#healpy.projscatter(ra_array, dec_array, c=sig_array, edgecolor='none', lonlat=True, vmin=5., vmax=25.)
healpy.projscatter(ra_des, dec_des, edgecolor='red', facecolor='none', marker='o', s=40, lonlat=True)
healpy.projscatter(ra_des_new, dec_des_new, edgecolor='blue', facecolor='none', marker='o', s=50, lonlat=True)
healpy.projscatter(gc_catalog.data['ra'], gc_catalog.data['dec'], edgecolor='red', facecolor='none', marker='x', s=40, lonlat=True)
healpy.projscatter(sat_catalog.data['ra'], sat_catalog.data['dec'], edgecolor='blue', facecolor='none', marker='x', s=40, lonlat=True)
healpy.projscatter(ra_ngc, dec_ngc, edgecolor='green', facecolor='none', marker='x', s=40, lonlat=True)
if save:
    pylab.savefig('compile_map_clean.png', dpi=150, bbox_inches='tight')


#pylab.figure()
#pylab.scatter(ra_array, dec_array, c=sig_array, edgecolor='none', vmin=5., vmax=25.)
#pylab.scatter(ra_array[cut], dec_array[cut], c=sig_array[cut], edgecolor='none', vmin=5., vmax=25.)
#pylab.colorbar()
#pylab.scatter(ra_des, dec_des, c='black', edgecolor='black', marker='x')


pylab.figure()
#pylab.hist(distance_modulus_array[(r_array < 0.2) & (sig_array > 8.)], bins=numpy.arange(16.25, 24.25, 0.5))
pylab.hist(distance_modulus_array, bins=numpy.arange(16.25, 24.25, 0.5))
pylab.xlabel('m - M (mag)')
pylab.ylabel('N')
pylab.savefig('compile_hist_distance_modulus.png', dpi=150, bbox_inches='tight')

pylab.figure()
pylab.hist(r_array, bins=numpy.arange(0.005, 0.315, 0.01))
pylab.xlabel('Size (deg)')
pylab.ylabel('N')
pylab.xlim(0., 0.30)
if save:
    pylab.savefig('compile_hist_size.png', dpi=150, bbox_inches='tight')

cut_dirty = (r_array > 0.2) | \
            (((numpy.char.count(association_array, 'NGC_') > 0) | \
              (numpy.char.count(association_array, 'UGC_') > 0) | \
              (numpy.char.count(association_array, 'IC_') > 0)) & (association_angsep_array < 0.2))

pylab.figure()
pylab.yscale('log', nonposy='clip')
pylab.ylim(0.1, 1.e3)
pylab.hist(sig_array, bins=numpy.linspace(5, 25, 41), color='green', alpha=0.5, label='All')
pylab.hist(sig_array[~cut_dirty], bins=numpy.linspace(5, 25, 41), color='green', label='r < 0.2 deg & no large galaxy association')
pylab.xlabel(r'Significance ($\sigma$)')
pylab.ylabel('N')
pylab.legend(loc='upper right', frameon=False)
if save:
    pylab.savefig('compile_hist_significance.png', dpi=150, bbox_inches='tight')
    #pylab.savefig('figs_summary_v8/compile_hist_significance.png', dpi=150, bbox_inches='tight')

pylab.figure()
pylab.scatter(sig_array, r_array, c=distance_modulus_array, edgecolor='none')
colorbar = pylab.colorbar()
colorbar.set_label('m - M (mag)')
pylab.xlabel(r'Significance ($\sigma$)')
pylab.ylabel('Size (deg)')
pylab.xlim(4., 26.)
pylab.ylim(0., 0.32)
if save:
    pylab.savefig('compile_scatter.png', dpi=150, bbox_inches='tight')

for ii in range(0, len(ra_des)):
    angsep = ugali.utils.projector.angsep(ra_des[ii], dec_des[ii], ra_array, dec_array)
    outfile = 'candidate_%.2f_%.2f.png'%(ra_array[numpy.argmin(angsep)], dec_array[numpy.argmin(angsep)])
    #print name_des[ii], sig_array[numpy.argmin(angsep)], outfile
    print('| %s | %.2f, %.2f | %.2f | {{thumbnail(%s, size=400)}} |'%(name_des[ii], ra_des[ii], dec_des[ii], sig_array[numpy.argmin(angsep)], outfile))
    os.system('cp figs_midway_v8/%s figs_reported/.'%(outfile))

for ii in range(0, len(ra_des_new)):
    angsep = ugali.utils.projector.angsep(ra_des_new[ii], dec_des_new[ii], ra_array, dec_array)
    outfile = 'candidate_%.2f_%.2f.png'%(ra_array[numpy.argmin(angsep)], dec_array[numpy.argmin(angsep)])
    #print name_des_new[ii], ra_des_new[ii], dec_des_new[ii], sig_array[numpy.argmin(angsep)], outfile
    print('| %s | %.2f, %.2f | %.2f | {{thumbnail(%s, size=400)}} |'%(name_des_new[ii], ra_des_new[ii], dec_des_new[ii], sig_array[numpy.argmin(angsep)], outfile))
    os.system('cp figs_midway_v8/%s figs_seed/.'%(outfile))


outfile = 'simple_binner_compiled_v8.csv'
writer = open(outfile, 'w')
writer.write('sig, ra, dec, distance_modulus, r, association, association_angsep\n')
for ii in range(0, len(sig_array)):
    writer.write('%10.2f, %10.2f, %10.2f, %10.2f, %10.2f, %30s, %10.2f\n'%(sig_array[ii], 
                                                                           ra_array[ii], dec_array[ii], distance_modulus_array[ii], 
                                                                           r_array[ii],
                                                                           association_array[ii], association_angsep_array[ii]))
writer.close()
