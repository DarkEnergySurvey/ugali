import glob
import numpy
import healpy
import pyfits
import pylab

import ugali.utils.healpix

pylab.ion()

#datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/hpx/cat'
#datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/v4/hpx'
datadir = '/project/kicp/bechtol/des/mw_substructure/y2n/data/catalog/v6/hpx'
infiles = glob.glob('%s/cat_hpx_*.fits'%(datadir))

nside = 2**10
npix = healpy.nside2npix(nside)

pix_array = []
for infile in infiles:
    print(infile)
    reader = pyfits.open(infile)
    cut = (reader[1].data['FLAGS_G'] < 4) & (reader[1].data['FLAGS_R'] < 4) \
          & (reader[1].data['QSLR_FLAG_G'] == 0) & (reader[1].data['QSLR_FLAG_R'] == 0) \
          & (reader[1].data['WAVG_MAG_PSF_G'] < 23.) \
          & ((reader[1].data['WAVG_MAG_PSF_G'] - reader[1].data['WAVG_MAG_PSF_R']) < 1.) \
          & (numpy.fabs(reader[1].data['WAVG_SPREAD_MODEL_R']) < 0.003 + reader[1].data['SPREADERR_MODEL_R'])
    pix_array.append(ugali.utils.healpix.angToPix(nside, reader[1].data.field('RA')[cut], reader[1].data.field('DEC')[cut]))
    reader.close()

pix_array = numpy.concatenate(pix_array)

m = numpy.histogram(pix_array, numpy.arange(npix + 1))[0].astype(float)
#m[m == 0] = healpy.UNSEEN

#numpy.save('density_map_nside_%i_v4.npy'%(nside), m)
numpy.save('density_map_nside_%i_v6.npy'%(nside), m)

#m = numpy.tile(healpy.UNSEEN)
    
