#!/usr/bin/env python
import pyfits
import numpy as np
import pylab as plt

from ugali.utils.projector import match

import numpy.lib.recfunctions as recfuncs

specfiles = [
    'members_CB.dat',
    'members_CVnII.dat',
    'members_Herc.dat',
    'members_booI.dat',
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

phot = pyfits.open(photfiles[0])[1].data
for f in photfiles[1:]:
    p = pyfits.open(f)[1].data
    phot = recfuncs.stack_arrays([phot,p],usemask=False,asrecarray=True)

m = match(spec['ra'],spec['dec'],phot['RA'],phot['DEC'],tol=1e-3)

plt.hist(phot['PROB'][m[1]][isMember[m[0]]],bins=25)
plt.hist(phot['PROB'][m[1]][~isMember[m[0]]],bins=25)


if __name__ == "__main__":
    import argparse
    description = "python script"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opts = parser.parse_args(); args = opts.args
