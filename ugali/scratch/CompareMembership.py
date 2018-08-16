#!/usr/bin/env python
import astropy.io.fits as pyfits
import numpy as np
import pylab as plt

from ugali.utils.projector import match
from ugali.utils.plotting import plotMembership
from ugali.utils.config import Config

import numpy.lib.recfunctions as recfuncs

specfiles = [
    #'/u/ki/kadrlica/sdss/spec/members_CB.dat',
    #'/u/ki/kadrlica/sdss/spec/members_CVnII.dat',
    #'/u/ki/kadrlica/sdss/spec/members_Herc.dat',
    '/u/ki/kadrlica/sdss/spec/members_booI.dat',
]

photfiles = [
    'coma_berenices_mcmc.fits',
    'canes_venatici_ii_mcmc.fits',
    'hercules_mcmc.fits',
    'bootes_i_mcmc.fits',
]    

dtype=[('objid','S12'),('ra',float),('dec',float),('rmag',float),('pluis',float),('member','S5')]
spec = np.empty(0,dtype=dtype)

for f in specfiles:
    s = np.genfromtxt(f,delimiter=[11,12,12,8,8,5],skiprows=1,dtype=dtype)
    spec = recfuncs.stack_arrays([spec,s],usemask=False,asrecarray=True)

isMember = np.array(['N' not in x for x in spec['member']])

#phot = pyfits.open(photfiles[0])[1].data
#for f in photfiles[1:]:
#    p = pyfits.open(f)[1].data
#    phot = recfuncs.stack_arrays([phot,p],usemask=False,asrecarray=True)


photfiles = [
    'bootes_i_hb_0.0.fits',    
    'bootes_i_hb_0.1.fits',    
    'bootes_i_hb_0.2.fits',    
    'bootes_i_hb_0.3.fits',    
    'bootes_i_hb_0.4.fits',    
    'bootes_i_hb_0.5.fits',    
]
for f in photfiles:
    phot = pyfits.open(f)[1].data
    m = match(spec['ra'],spec['dec'],phot['RA'],phot['DEC'],tol=1e-3)

    config = Config('config_dr10_gal.yaml')
    plotMembership(phot[m[1]][isMember[m[0]]],config, distance_modulus=19.)
    plt.suptitle(f.replace('.fits',''))
    plt.savefig(f.replace('.fits','.png'))

#plt.hist(phot['PROB'][m[1]][isMember[m[0]]],bins=25)
#plt.hist(phot['PROB'][m[1]][~isMember[m[0]]],bins=25)
